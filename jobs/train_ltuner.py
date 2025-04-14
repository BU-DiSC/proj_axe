#!/usr/bin/env python
import argparse
import csv
import logging
import os
from typing import Optional, Tuple

import polars as pl
import toml
import torch
from axe.lsm.types import LSMBounds, Policy
from axe.ltuner.data.schema import LTunerDataSchema
from axe.ltuner.loss import LearnedCostModelLoss
from axe.ltuner.model.builder import LTuneModelBuilder
from axe.util.lr_scheduler import LRSchedulerBuilder
from axe.util.optimizer import OptimizerBuilder
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class TrainLTuner:
    def __init__(self, cfg: dict) -> None:
        self.log: logging.Logger = logging.getLogger(cfg["app"]["name"])
        self.disable_tqdm: bool = cfg["app"]["disable_tqdm"]
        self.use_gpu = cfg["job"]["use_gpu_if_avail"]
        self.device = torch.device("cpu")
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.policy: Policy = getattr(Policy, cfg["lsm"]["policy"])
        self.bounds: LSMBounds = LSMBounds(**cfg["lsm"]["bounds"])
        self.schema = LTunerDataSchema(self.policy, self.bounds)
        self.jcfg = cfg["job"]["train_ltuner"]
        self.cfg = cfg

        # Build everything we need for training
        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        torch.set_float32_matmul_precision("high")
        self.training_data, self.validate_data = self._build_data()
        self.train_kwargs = {"temp": 10, "hard": False}
        self.validate_kwargs = {"temp": 0.01, "hard": True}

    def _build_loss_fn(self) -> torch.nn.Module:
        loss = LearnedCostModelLoss(self.cfg, self.jcfg["loss_fn_path"]).to(self.device)
        loss.lcm.compile()

        return loss

    def _build_model(self) -> torch.nn.Module:
        model = LTuneModelBuilder(
            schema=self.schema, **self.cfg["ltuner"]["model"]
        ).build()
        # model.compile()
        model.to(self.device)

        return model

    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        return OptimizerBuilder(self.cfg["optimizer"]).build(
            optimizer_choice=self.jcfg["optimizer"], model=model
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return LRSchedulerBuilder(self.cfg["scheduler"]).build(
            optimizer, self.jcfg["lr_scheduler"]
        )

    def _build_data(self) -> Tuple[DataLoader, DataLoader]:
        table = pl.read_parquet(self.jcfg["data_dir"])
        dataset = table.to_torch(
            return_type="dataset",
            features=self.schema.feat_cols(),
            label=self.schema.label_cols(),
            dtype=pl.Float32,
        )
        train_len = int(len(dataset) * self.jcfg["data_split"])
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        training_data = DataLoader(
            dataset=train_set,
            batch_size=self.jcfg["batch_size"],
            num_workers=self.jcfg["num_workers"],
            shuffle=True,
        )
        validate_data = DataLoader(
            dataset=val_set,
            batch_size=8 * self.jcfg["batch_size"],
            num_workers=self.jcfg["num_workers"],
        )

        return training_data, validate_data

    def _make_save_dir(self) -> None:
        self.log.info(f"Saving tuner in {self.jcfg['save_dir']}")
        os.makedirs(self.jcfg["save_dir"], exist_ok=False)
        if not self.jcfg["no_checkpoint"]:
            os.makedirs(os.path.join(self.jcfg["save_dir"], "checkpoints"))
        with open(os.path.join(self.jcfg["save_dir"], "axe.toml"), "w") as fid:
            toml.dump(self.cfg, fid)

    def temp_step(self, decay_rate: float = 0.95, floor: float = 1):
        self.train_kwargs["temp"] *= decay_rate
        if self.train_kwargs["temp"] < floor:
            self.train_kwargs["temp"] = floor

    def train_step(self, feats: Tensor, labels: Tensor, **kwargs) -> float:
        label = labels.to(self.device)
        feats = feats.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(feats, **kwargs)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_loop(self) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.training_data, ncols=80, disable=self.disable_tqdm)
        for batch, (feats, labels) in enumerate(pbar):
            loss = self.train_step(feats, labels, **self.train_kwargs)
            if batch % (25) == 0:
                pbar.set_description(f"training loss {loss:e}")
            total_loss += loss
            if self.scheduler is not None:
                self.scheduler.step()

        return total_loss / len(self.training_data)

    def validate_step(self, feats: Tensor, labels: Tensor, **kwargs) -> float:
        with torch.no_grad():
            labels = labels.to(self.device)
            feats = feats.to(self.device)
            pred = self.model(feats, **kwargs)
            validate_loss = self.loss_fn(pred, labels).item()

        return validate_loss

    def validate_loop(self) -> float:
        self.model.eval()
        test_loss = 0
        pbar = tqdm(self.validate_data, ncols=80, disable=self.disable_tqdm)
        for feats, labels in pbar:
            loss = self.validate_step(feats, labels, **self.validate_kwargs)
            pbar.set_description(f"validate loss {loss:e}")
            test_loss += loss

        return test_loss / len(self.validate_data)

    def save_model(self, fname: str, **kwargs) -> None:
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        save_dict.update(kwargs)
        torch.save(save_dict, os.path.join(self.jcfg["save_dir"], fname))

    def run(self):
        self.log.info("[Job] Training LCM")
        self._make_save_dir()

        loss_file = os.path.join(self.jcfg["save_dir"], "losses.csv")
        with open(loss_file, "w") as fid:
            loss_csv_write = csv.writer(fid)
            loss_csv_write.writerow(["epoch", "train_loss", "test_loss"])

        max_epochs = self.jcfg["max_epochs"]
        loss_min = self.validate_loop()
        for epoch in range(max_epochs):
            self.log.info(f"Epoch: [{epoch+1}/{max_epochs}]")
            self.log.debug(f"{self.train_kwargs=}")
            train_loss = self.train_loop()
            curr_loss = self.validate_loop()
            self.log.info(f"Training loss: {train_loss}")
            self.log.info(f"Validate loss: {curr_loss}")
            if not self.jcfg["no_checkpoint"]:
                self.save_model(f"checkpoints/epoch{epoch:02d}.model", loss=curr_loss)

            if curr_loss < loss_min:
                loss_min = curr_loss
                self.log.info("New minmum loss saving best model")
                self.save_model("best_model.model", loss=loss_min, epoch=epoch)
            with open(loss_file, "a") as fid:
                write = csv.writer(fid)
                write.writerow([epoch + 1, train_loss, curr_loss])
            self.temp_step()

        self.log.info("Training finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()
    config = toml.load(args.config)
    logging.basicConfig(**config["log"])
    log: logging.Logger = logging.getLogger(config["app"]["name"])
    log.info(f"Log level: {logging.getLevelName(log.getEffectiveLevel())}")

    TrainLTuner(config).run()


if __name__ == "__main__":
    main()
