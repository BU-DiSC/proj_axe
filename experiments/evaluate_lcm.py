import logging
import os

import polars as pl
import toml
import torch
from tqdm import tqdm

from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model import LearnedCostModelBuilder
from axe.lcm.model.wrapper import LCMWrapper
from axe.lsm.types import LSMBounds, Policy
from experiments.infra import AxeResultDB


class ExpLCMEvaluate:
    def __init__(self, config: dict) -> None:
        path = config["experiments"]["lcm_path"]
        model = config["experiments"]["lcm_model"]
        self.evaluator = LCMEvaluator(config, path=path, model=model)
        self.config = config

    def run(self):
        table = self.evaluator.evaluate()
        self.db = AxeResultDB(self.config)
        # okay because apparently polars does not support writing out to sqlite3, we
        # will do a polars -> pandas -> write sqlite3 line
        # This should be fine as polars already uses pandas underneath the hood to write
        # to sqlite with sqlalchemy
        table = table.to_pandas()
        try:
            table.to_sql(name="lcm_evaluation", con=self.db.con, if_exists="fail")
        except ValueError as err:
            print(f"Error writing table: {err}")
            print("Fall back to write csv to 'error_lcm_eval_backup.csv'")
            table.to_csv('error_lcm_eval_backup.csv')

        return


class LCMEvaluator:
    def __init__(
        self, config: dict, path: str, model: str = "best_model.model"
    ) -> None:
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        seed = config["app"]["random_seed"]
        cfg = toml.load(os.path.join(path, "axe.toml"))
        policy: Policy = getattr(Policy, cfg["lsm"]["policy"])
        bounds: LSMBounds = LSMBounds(**cfg["lsm"]["bounds"])
        self.schema: LCMDataSchema = LCMDataSchema(
            policy=policy, bounds=bounds, seed=seed
        )

        # load in model
        model_name = os.path.join(path, model)
        self.model = LearnedCostModelBuilder(self.schema, **cfg["lcm"]["model"]).build()

        model_data = torch.load(model_name, weights_only=True)
        self._load_status = self.model.load_state_dict(model_data["model_state_dict"])
        self.model.eval()
        self.wrapper = LCMWrapper(self.model, self.schema)
        self.cfg = cfg

    @property
    def load_status(self):
        return self._load_status

    def generate_test_data(self, num_samples: int = 10000) -> pl.DataFrame:
        self.log.info(f"Generating test data: size={num_samples}")
        table = [
            self.schema.sample_row_dict() for _ in tqdm(range(num_samples), ncols=80)
        ]
        table = pl.DataFrame(table)

        return table

    def evaluate(self, num_samples: int = 10000):
        test_data = self.generate_test_data(num_samples)
        test_data_evaluation_format = self.schema._preprocess_table(test_data)
        dataset = test_data_evaluation_format.to_torch(
            return_type="dataset",
            features=self.schema.feat_cols(),
            label=self.schema.label_cols(),
            dtype=pl.Float32,
        )
        df = []
        self.log.info("Running model over test set")
        for feat, label in tqdm(dataset, ncols=80):  # pyright: ignore
            pred = self.model(feat.unsqueeze(0))
            df.append({"cost_acm": label.sum().item(), "cost_lcm": pred.sum().item()})
        df = pl.concat([test_data, pl.DataFrame(df)], how="horizontal")
        df = df.with_columns(
            delta=(pl.col("cost_acm") - pl.col("cost_lcm")),
            norm_delta=((pl.col("cost_acm") - pl.col("cost_lcm")) / pl.col("cost_acm")),
        )

        return df
