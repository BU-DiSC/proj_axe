import logging
import os

import polars as pl
import toml
import torch
from torch import Tensor
from tqdm.auto import tqdm

from axe.lsm.cost import Cost
from axe.lsm.solver import get_solver_from_policy
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload
from axe.ltuner.data.schema import LTunerDataSchema
from axe.ltuner.model.builder import LTuneModelBuilder
from experiments.infra import AxeResultDB


class ExpLTunerEvaluate:
    def __init__(self, config: dict) -> None:
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        path = config["experiments"]["ltuner_path"]
        model = config["experiments"]["ltuner_model"]
        self.evaluator = LTunerEvaluator(config, model_path=path, model_name=model)
        self.model_path = path
        self.config = config

    def run(self):
        self.log.info(f"Running LTuner Evaluation: {self.model_path}")
        db = AxeResultDB(self.config)
        # okay because apparently polars does not support writing out to sqlite3, we
        # will do a polars -> pandas -> write sqlite3 line
        # This should be fine as polars already uses pandas underneath the hood to write
        # to sqlite with sqlalchemy
        env_table = self.evaluator.evaluate(db.get_env_table())
        env_table = env_table.to_pandas()
        try:
            self.log.info(f"Writing table ltune_eval_rep to {db.db_path}")
            env_table.to_sql(
                name="ltune_eval_rep", con=db.con, if_exists="fail", index=False
            )
        except ValueError as err:
            self.log.warning(f"Error writing table: {err}")
            self.log.warning("Fall back to write csv to 'error_ltune_eval_pred.csv'")
            env_table.to_csv("error_ltune_eval_pred.csv")

        rand_table = self.evaluator.evaluate(self.evaluator.generate_test_data())
        rand_table = rand_table.to_pandas()
        try:
            rand_table.to_sql(
                name="ltune_eval_rand", con=db.con, if_exists="fail", index=False
            )
        except ValueError as err:
            self.log.warning(f"Error writing table: {err}")
            self.log.warning("Fall back to write csv to 'error_ltune_eval_rand.csv'")
            env_table.to_csv("error_ltune_eval_rand.csv")

        return


class LTunerEvaluator:
    def __init__(
        self, config: dict, model_path: str, model_name: str = "best_model.model"
    ) -> None:
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        seed = config["app"]["random_seed"]
        # config is the current applications config file
        # cfg will be the model's last saved config file
        cfg = toml.load(os.path.join(model_path, "axe.toml"))
        policy: Policy = getattr(Policy, cfg["lsm"]["policy"])
        bounds: LSMBounds = LSMBounds(**cfg["lsm"]["bounds"])
        self.schema: LTunerDataSchema = LTunerDataSchema(
            policy=policy, bounds=bounds, seed=seed
        )
        # load in model
        self.model = LTuneModelBuilder(self.schema, **cfg["ltuner"]["model"]).build()

        model_data = torch.load(os.path.join(model_path, model_name), weights_only=True)
        self._load_status = self.model.load_state_dict(model_data["model_state_dict"])
        self.model.eval()
        self.cost_fn = Cost(bounds.max_considered_levels)
        solver_cls = get_solver_from_policy(policy)
        self.solver = solver_cls(bounds)
        self.policy = policy
        self.cfg = cfg

    @property
    def load_status(self):
        return self._load_status

    @property
    def size_ratio_range(self) -> int:
        return (
            self.schema.bounds.size_ratio_range[1]
            - self.schema.bounds.size_ratio_range[0]
        )

    def _klsm_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.size_ratio_range
        h = out[0].item()
        caps = out[1:].reshape(-1, cap_range)
        t = torch.argmax(caps[0]).item() + 2
        k = tuple(torch.argmax(x).item() + 1 for x in caps[1:])

        return LSMDesign(
            bits_per_elem=h, size_ratio=t, policy=Policy.Kapacity, kapacity=k
        )

    def _classic_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.size_ratio_range
        h = out[0].item()
        t = torch.argmax(out[1 : cap_range + 1]).item() + 2
        policy_val = torch.argmax(out[cap_range + 1 :]).item()
        if policy_val:
            policy = Policy.Leveling
        else:
            policy = Policy.Tiering

        return LSMDesign(bits_per_elem=h, size_ratio=t, policy=policy, kapacity=())

    def convert_ltune_out_to_design(self, output: Tensor):
        if self.policy == Policy.QHybrid:
            raise NotImplementedError
        elif self.policy == Policy.Kapacity:
            design = self._klsm_convert(output)
        elif self.policy == Policy.Fluid:
            raise NotImplementedError
        else:  # self.design_type == Policy.Classic
            raise NotImplementedError

        return design

    def generate_test_data(self, num_samples: int = 10000) -> pl.DataFrame:
        self.log.info(f"Generating test data: size={num_samples}")
        table = [
            self.schema.sample_row_dict()
            for _ in tqdm(range(num_samples), ncols=80, desc="Test Data")
        ]
        table = pl.DataFrame(table)

        return table

    def row_to_objs(self, row: dict):
        workload = Workload(
            row["empty_reads"],
            row["non_empty_reads"],
            row["range_queries"],
            row["writes"],
        )
        system = System(
            entries_per_page=row["entries_per_page"],
            selectivity=row["selectivity"],
            entry_size=row["entry_size"],
            mem_budget=row["mem_budget"],
            num_entries=row["num_entries"],
        )
        return workload, system

    def get_ltune_designs(self, table: pl.DataFrame) -> list[LSMDesign]:
        input_dataset = table.to_torch(
            return_type="dataset",
            features=self.schema.feat_cols(),
            label=self.schema.label_cols(),
            dtype=pl.Float32,
        )
        self.log.info("Querying AXE for designs")
        ltune_designs = []
        for feat, _ in tqdm(  # pyright: ignore
            input_dataset,  # pyright: ignore
            desc="Designs",
            ncols=80,
        ):
            out = self.model(feat.unsqueeze(0))
            design = self.convert_ltune_out_to_design(out)
            ltune_designs.append(design)

        return ltune_designs

    def evaluate(self, table: pl.DataFrame):
        ltune_designs: list[LSMDesign] = self.get_ltune_designs(table)
        ltuner_table = []
        solver_table = []
        self.log.info("Evaluating all designs")
        for row, ltune_design in tqdm(
            list(zip(table.to_dicts(), ltune_designs)),  # convert to list for tqdm bar
            desc="Eval",
            ncols=80,
        ):
            workload, system = self.row_to_objs(row)
            row = LTunerDataSchema.design_to_dict(ltune_design)
            row["cost"] = self.cost_fn.calc_cost(ltune_design, system, workload)
            ltuner_table.append(row)

            design, _ = self.solver.get_nominal_design(system=system, workload=workload)
            row = LTunerDataSchema.design_to_dict(design)
            row["cost"] = self.cost_fn.calc_cost(design, system, workload)
            solver_table.append(row)
        ltuner_table = pl.concat((table, pl.DataFrame(ltuner_table)), how="horizontal")
        solver_table = pl.concat((table, pl.DataFrame(solver_table)), how="horizontal")
        out_table = ltuner_table.join(solver_table, on=table.columns, suffix="_solver")

        return out_table
