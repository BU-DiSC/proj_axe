import logging
import sqlite3

import ConfigSpace as CS
import numpy as np
import pandas as pd
from axe.lcm.data.generator import KHybridGenerator, ClassicGenerator, YZCostGenerator
from axe.lsm.cost import EndureCost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload
from mlos_core.optimizers import SmacOptimizer

NUM_ROUNDS = 100
NUM_TRIALS = 100


class ExperimentMLOS:
    def __init__(self, config: dict) -> None:
        self.log: logging.Logger = logging.getLogger(config["log"]["name"])
        self.bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.model_type = getattr(Policy, config["lsm"]["design"])
        self.cf: EndureCost = EndureCost(self.bounds.max_considered_levels)
        self.db = MLOSDatabase(config)
        self.config = config
        if self.model_type == Policy.Classic:
            self.gen: ClassicGenerator = ClassicGenerator(self.bounds)
        elif self.model_type == Policy.YZHybrid:
            self.gen: YZCostGenerator = YZCostGenerator(self.bounds)
        else:
            self.gen: KHybridGenerator = KHybridGenerator(self.bounds)

    def _suggest_to_design(self, suggestion: pd.DataFrame) -> LSMDesign:
        bits_per_element: float = suggestion["bits_per_element"].values[0]
        size_ratio: int = suggestion["size_ratio"].values[0]
        if self.model_type == Policy.Classic:
            pol_val: int = suggestion["pol_val"].values[0]
            if pol_val == 0:
                policy = Policy.Tiering
            else:
                policy = Policy.Leveling
            design = LSMDesign(h=bits_per_element, T=size_ratio, policy=policy)
        elif self.model_type == Policy.YZHybrid:
            y_val: int = suggestion["y_val"].values[0]
            z_val: int = suggestion["z_val"].values[0]
            design = LSMDesign(
                h=bits_per_element,
                T=size_ratio,
                policy=Policy.YZHybrid,
                Y=y_val,
                Z=z_val,
            )
        elif self.model_type == Policy.QFixed:
            raise NotImplementedError
        else: # self.model_type == Policy.KHybrid:
            kaps: np.ndarray = suggestion[[f"kap_{idx}" for idx in range(20)]].values[0]
            design = LSMDesign(
                h=bits_per_element, T=size_ratio, policy=Policy.KHybrid, K=kaps.tolist()
            )

        return design

    def _create_parameter_space(self, system: System) -> CS.ConfigurationSpace:
        norm_params = [
            CS.UniformFloatHyperparameter(
                name="bits_per_element",
                lower=self.bounds.bits_per_elem_range[0],
                upper=system.H - 0.1,
            ),
            CS.UniformIntegerHyperparameter(
                name="size_ratio",
                lower=self.bounds.size_ratio_range[0],
                upper=self.bounds.size_ratio_range[1] - 1,  # ConfigSpace is inclusive
            ),
        ]
        if self.model_type == Policy.Classic:
            classic_params = [
                CS.UniformIntegerHyperparameter(
                    name="pol_val",
                    lower=0,
                    upper=1,
                )
            ]
            parameters = norm_params + classic_params
        if self.model_type == Policy.YZHybrid:
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
            parameters = norm_params + yz_params
        elif self.model_type == Policy.QFixed:
            raise NotImplementedError
        else: # self.model_type == Policy.KHybrid:
            kap_params = [
                CS.UniformIntegerHyperparameter(
                    name=f"kap_{i}", lower=1, upper=self.bounds.size_ratio_range[1] - 1
                )
                for i in range(self.bounds.max_considered_levels)
            ]
            parameters = norm_params + kap_params
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
            cost = self.cf.calc_cost(
                design, system, workload.z0, workload.z1, workload.q, workload.w
            )
            optimizer.register(
                configs=suggestion, scores=pd.DataFrame([{"cost": cost}])
            )
            self.log.info(f"[ID {wl_id}][Trial {trial}][Round {round}] Cost: {cost}")
            self.db.log_round(wl_id, trial, round, design, cost)

        return

    def run(self) -> None:
        system = System()
        self.db.create_tables()
        for rep_wl in self.config["workloads"]:
            workload = Workload(
                z0=rep_wl["z0"],
                z1=rep_wl["z1"],
                q=rep_wl["q"],
                w=rep_wl["w"],
            )
            row_id = self.db.log_workload(workload, system)
            self.log.info(f"Workload: {workload}")
            self.log.info(f"System: {system}")
            for trial in range(NUM_TRIALS):
                self.log.info(f"(Workload ID, Trial): ({row_id}, {trial})")
                self._train_model(row_id, trial, workload, system)

        return


class MLOSDatabase:
    def __init__(self, config: dict, db_path: str = "testing_yz.db") -> None:
        self.log: logging.Logger = logging.getLogger(config["log"]["name"])
        self.connector = sqlite3.connect(db_path)
        self.db_path = db_path
        self.config = config
        self.model_type = getattr(Policy, config["lsm"]["design"])

    def create_tables(self) -> None:
        cursor = self.connector.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS environments (
                env_id INTEGER PRIMARY KEY AUTOINCREMENT,
                empty_reads REAL,
                non_empty_reads REAL,
                range_queries REAL,
                writes REAL,
                entry_size INT,
                selectivity REAL,
                entries_per_page INT,
                num_elmement INT,
                bits_per_elem_max REAL,
                read_write_asym FLOAT
            );
            """
        )

        tunings_cols_comm = """
            idx INTEGER PRIMARY KEY AUTOINCREMENT,
            env_id INTEGER,
            trial INTEGER,
            round INTEGER,
            bits_per_elem REAL,
            size_ratio INTEGER,
            cost REAL"""

        if self.model_type == Policy.Classic:
            policy_field = "policy TEXT"
        elif self.model_type == Policy.YZHybrid:
            policy_field = "y_val INTEGER, z_val INTEGER"
        else:
            kap_fields = ", ".join([f"kap{i} REAL" for i in range(20)])
            policy_field = kap_fields
        key_string = "FOREIGN KEY (env_id) REFERENCES workloads(env_id)"
        create_tunings_table_query = f"""
            CREATE TABLE IF NOT EXISTS tunings (
                {tunings_cols_comm}, {policy_field}, 
                {key_string}
            );
        """

        cursor.execute(create_tunings_table_query)
        self.connector.commit()
        cursor.close()

        return

    def log_workload(self, workload: Workload, system: System) -> int:
        cursor = self.connector.cursor()
        cursor.execute(
            """
            INSERT INTO environments (
                empty_reads,
                non_empty_reads,
                range_queries,
                writes,
                entry_size,
                selectivity,
                entries_per_page,
                num_elmement,
                bits_per_elem_max,
                read_write_asym
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workload.z0,
                workload.z1,
                workload.q,
                workload.w,
                int(system.E),
                system.s,
                system.B,
                system.N,
                system.H,
                system.phi,
            ),
        )
        self.connector.commit()

        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def log_round(
        self,
        workload_id: int,
        trial: int,
        round: int,
        design: LSMDesign,
        cost: float,
    ) -> None:
        cursor = self.connector.cursor()
        if self.model_type == Policy.Classic:
            cursor.execute(
                """
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
                (workload_id, trial, round, design.h, int(design.T), str(design.policy))
                + (cost,),
            )
        elif self.model_type == Policy.YZHybrid:
            cursor.execute(
                """
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
                (
                    workload_id,
                    trial,
                    round,
                    design.h,
                    int(design.T),
                    int(design.Y),
                    int(design.Z),
                )
                + (cost,),
            )
        else:
            cursor.execute(
                """
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
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                """,
                (
                    workload_id,
                    trial,
                    round,
                    design.h,
                    int(design.T),
                )
                + tuple(design.K)
                + (cost,),
            )

        self.connector.commit()

        return
