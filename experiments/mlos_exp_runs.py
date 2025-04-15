import logging

import ConfigSpace as CS
import pandas as pd
from mlos_core.optimizers import SmacOptimizer

from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload

from .infra import AxeResultDB

NUM_ROUNDS = 100
NUM_TRIALS = 50

LOG_ROUND_QUERY = {
    Policy.Classic: """
    INSERT INTO tunings (
        env_id,
        trial,
        round,
        bits_per_elem,
        size_ratio,
        policy,
        cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    Policy.Fluid: """
    INSERT INTO tunings (
        env_id,
        trial,
        round,
        bits_per_elem,
        size_ratio,
        y_val, z_val,
        cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    Policy.Kapacity: """
    INSERT INTO tunings (
        env_id,
        trial,
        round,
        bits_per_elem,
        size_ratio,
        kap0, kap1, kap2, kap3, kap4,
        kap5, kap6, kap7, kap8, kap9,
        kap10, kap11, kap12, kap13, kap14,
        kap15, kap16, kap17, kap18, kap19,
        cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
}


class ExperimentMLOS:
    def __init__(self, config: dict) -> None:
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        self.bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.policy = getattr(Policy, config["lsm"]["policy"])
        self.cf: Cost = Cost(self.bounds.max_considered_levels)
        self.db = AxeResultDB(config)
        self.config = config

    def _create_table_str(self):
        tunings_cols_comm = """
            idx INTEGER PRIMARY KEY AUTOINCREMENT,
            env_id INTEGER,
            trial INTEGER,
            round INTEGER,
            bits_per_elem REAL,
            size_ratio INTEGER,
            cost REAL
        """

        if self.policy == Policy.Classic:
            policy_field = "policy TEXT"
        elif self.policy == Policy.Fluid:
            policy_field = "y_val INTEGER, z_val INTEGER"
        elif self.policy == Policy.QHybrid:
            raise NotImplementedError
        else:  # self.policy == Policy.Kapacity:
            kap_fields = ", ".join([f"kap{i} REAL" for i in range(20)])
            policy_field = kap_fields
        key_string = "FOREIGN KEY (env_id) REFERENCES workloads(env_id)"
        create_tunings_table_query = f"""
            CREATE TABLE IF NOT EXISTS tunings (
                {tunings_cols_comm}, {policy_field}, 
                {key_string}
            );
        """

        return create_tunings_table_query

    def _log_round(
        self,
        workload_id: int,
        trial: int,
        round: int,
        design: LSMDesign,
        cost: float,
    ):
        args = (workload_id, trial, round, design.bits_per_elem, design.size_ratio)
        if self.policy == Policy.Classic:
            args += (str(design.policy),)
        elif self.policy == Policy.Fluid:
            args += (int(design.kapacity[0]), int(design.kapacity[1]))
        else:  # self.policy == Policy.Kapacity
            args += tuple(design.kapacity)
        args += (cost,)
        sql_query = LOG_ROUND_QUERY.get(self.policy, None)
        if sql_query is None:
            raise NotImplementedError
        self.db.run_sql(sql_query, args)

    def _suggest_to_design(self, suggestion: pd.DataFrame) -> LSMDesign:
        bits_per_element: float = suggestion["bits_per_element"].values[0]
        size_ratio: int = suggestion["size_ratio"].values[0]
        if self.policy == Policy.Classic:
            pol_val: int = suggestion["pol_val"].values[0]
            policy = Policy.Tiering if pol_val == 0 else Policy.Leveling
            kapacity = ()
        elif self.policy == Policy.Fluid:
            kapacity = (suggestion["y_val"].values[0], suggestion["z_val"].values[0])
            policy = Policy.Fluid
        elif self.policy == Policy.QHybrid:
            raise NotImplementedError
        else:  # self.model_type == Policy.Kapacity:
            policy = Policy.Kapacity
            kapacity = suggestion[[f"kap_{idx}" for idx in range(20)]].values[0]

        return LSMDesign(
            bits_per_elem=bits_per_element,
            size_ratio=size_ratio,
            policy=policy,
            kapacity=kapacity,
        )

    def _create_parameter_space(self, system: System) -> CS.ConfigurationSpace:
        parameters = [
            CS.UniformFloatHyperparameter(
                name="bits_per_element",
                lower=self.bounds.bits_per_elem_range[0],
                upper=system.mem_budget - 0.1,
            ),
            CS.UniformIntegerHyperparameter(
                name="size_ratio",
                lower=self.bounds.size_ratio_range[0],
                upper=self.bounds.size_ratio_range[1] - 1,  # ConfigSpace is inclusive
            ),
        ]
        if self.policy == Policy.Classic:
            parameters += [
                CS.UniformIntegerHyperparameter(name="pol_val", lower=0, upper=1)
            ]
        elif self.policy == Policy.Fluid:
            yz_params = [
                CS.UniformIntegerHyperparameter(
                    name="y_val",
                    lower=1,
                    upper=self.bounds.size_ratio_range[1] - 1,
                ),
                CS.UniformIntegerHyperparameter(
                    name="z_val",
                    lower=1,
                    upper=self.bounds.size_ratio_range[1] - 1,
                ),
            ]
            parameters += yz_params
        elif self.policy == Policy.QHybrid:
            raise NotImplementedError
        else:  # self.model_type == Policy.KHybrid:
            kap_params = [
                CS.UniformIntegerHyperparameter(
                    name=f"kap_{i}", lower=1, upper=self.bounds.size_ratio_range[1] - 1
                )
                for i in range(self.bounds.max_considered_levels)
            ]
            parameters += kap_params
        parameter_space = CS.ConfigurationSpace(seed=0)
        parameter_space.add(parameters)

        return parameter_space

    def _create_optimizer(self, parameter_space: CS.ConfigurationSpace):
        return SmacOptimizer(
            parameter_space=parameter_space,
            optimization_targets=["cost"],
            n_random_init=1,
        )

    def _train_model(
        self,
        wl_id: int,
        trial: int,
        workload: Workload,
        system: System,
        num_rounds: int = NUM_ROUNDS,
    ) -> None:
        parameter_space = self._create_parameter_space(system)
        optimizer = self._create_optimizer(parameter_space)
        for round in range(num_rounds):
            suggestion, _ = optimizer.suggest()
            assert isinstance(suggestion, pd.DataFrame)
            design = self._suggest_to_design(suggestion)
            cost = self.cf.calc_cost(design, system, workload)
            optimizer.register(
                configs=suggestion, scores=pd.DataFrame([{"cost": cost}])
            )
            self.log.info(f"[ID {wl_id}][Trial {trial}][Round {round}] Cost: {cost}")
            self._log_round(wl_id, trial, round, design, cost)

        return

    def run(self) -> None:
        system = System()
        table_creation_sql = self._create_table_str()
        self.db.run_sql(table_creation_sql)
        environments = self.db.get_env_table()
        for env in environments.to_dicts():
            row_id = env["env_id"]
            workload = Workload(
                z0=env["empty_reads"],
                z1=env["non_empty_reads"],
                q=env["range_queries"],
                w=env["writes"],
            )
            system = System(
                entry_size=env["entry_size"],
                selectivity=env["selectivity"],
                entries_per_page=env["entries_per_page"],
                num_entries=env["num_entries"],
                phi=env["read_write_asym"],
            )
            self.log.info(f"Workload: {workload}")
            self.log.info(f"System: {system}")
            for trial in range(NUM_TRIALS):
                self.log.info(f"(Workload ID, Trial): ({row_id}, {trial})")
                self._train_model(row_id, trial, workload, system)

        return
