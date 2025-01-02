#!/usr/bin/env python
from typing import Any, Optional
import logging
import os

from torch.utils.data import DataLoader
import toml
import torch
import torch.optim as TorchOpt

from axe.lcm.data.dataset import LCMDataSet
from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import LSMBounds, Policy
from axe.util.losses import LossBuilder
from axe.util.lr_scheduler import LRSchedulerBuilder
from axe.util.optimizer import OptimizerBuilder
from axe.util.trainer import Trainer


class LCMTrainJob:
    def __init__(self, config: dict[str, Any]) -> None:
        self.log = logging.getLogger(config["log"]["name"])
        self.log.info("Running Training Job")
        self.design = getattr(Policy, config["lsm"]["design"])
        self.use_gpu = config["job"]["use_gpu_if_avail"]
        self.bounds = LSMBounds(**config["lsm"]["bounds"])
        self.loss_builder = LossBuilder(config["loss"])
        self.opt_builder = OptimizerBuilder(config["optimizer"])
        self.schedule_builder = LRSchedulerBuilder(config["scheduler"])
        self.model_builder = LearnedCostModelBuilder(
            size_ratio_range=self.bounds.size_ratio_range,
            max_levels=self.bounds.max_considered_levels,
            **config["lcm"]["model"],
        )

        self.config = config
        self.jconfig = config["job"]["LCMTrain"]

    def _build_loss_fn(self) -> torch.nn.Module:
        choice = self.jconfig["loss_fn"]
        loss = self.loss_builder.build(choice)
        self.log.info(f"Loss function: {choice}")
        if loss is None:
            self.log.warning(f"Invalid loss function: {choice}")
            raise KeyError
        if self.use_gpu and torch.cuda.is_available():
            loss.to("cuda")

        return loss

    def _build_model(self) -> torch.nn.Module:
        model = self.model_builder.build_model(self.design)
        if self.use_gpu and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_optimizer(self, model) -> TorchOpt.Optimizer:
        choice = self.jconfig["optimizer"]

        return self.opt_builder.build_optimizer(choice, model)

    def _build_scheduler(
        self, optimizer: TorchOpt.Optimizer
    ) -> Optional[TorchOpt.lr_scheduler._LRScheduler]:
        choice = self.jconfig["lr_scheduler"]

        return self.schedule_builder.build_scheduler(optimizer, choice)

    def _build_train(self) -> DataLoader:
        train_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.jconfig["train"]["dir"],
        )
        self.log.info(f"Train data dir: {train_dir}")
        train_data = LCMDataSet(
            folder=train_dir,
            lsm_design=self.design,
            bounds=self.bounds,
            shuffle=self.jconfig["train"]["shuffle"],
        )
        train = DataLoader(
            train_data,
            batch_size=self.jconfig["train"]["batch_size"],
            drop_last=self.jconfig["train"]["drop_last"],
            num_workers=self.jconfig["train"]["num_workers"],
            pin_memory=True,
        )

        return train

    def _build_test(self) -> DataLoader:
        test_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.jconfig["test"]["dir"],
        )
        self.log.info(f"Test data: {test_dir}")
        test_data = LCMDataSet(
            folder=test_dir,
            lsm_design=self.design,
            bounds=self.bounds,
            shuffle=self.jconfig["test"]["shuffle"],
            test=True,
        )
        test = DataLoader(
            test_data,
            batch_size=self.jconfig["test"]["batch_size"],
            drop_last=self.jconfig["test"]["drop_last"],
            num_workers=self.jconfig["test"]["num_workers"],
            pin_memory=True,
        )

        return test

    def _make_save_dir(self) -> Optional[str]:
        self.log.info(f"Saving tuner in {self.jconfig['save_dir']}")
        save_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.jconfig["save_dir"],
        )
        try:
            os.makedirs(save_dir, exist_ok=False)
        except FileExistsError:
            return None

        with open(os.path.join(save_dir, "axe.toml"), "w") as fid:
            toml.dump(self.config, fid)

        return save_dir

    def run(self) -> Optional[Trainer]:
        model_dir = self._make_save_dir()
        if model_dir is None:
            self.log.info("Model directory already exists, exiting...")
            return None

        model = self._build_model()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        train_data = self._build_train()
        test_data = self._build_test()
        loss_fn = self._build_loss_fn()
        disable_tqdm = self.config["log"]["disable_tqdm"]

        trainer = Trainer(
            log=self.log,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            test_data=test_data,
            scheduler=scheduler,
            max_epochs=self.jconfig["max_epochs"],
            use_gpu_if_avail=self.use_gpu,
            model_dir=model_dir,
            disable_tqdm=disable_tqdm,
            no_checkpoint=self.jconfig["no_checkpoint"],
        )
        trainer.run()

        return trainer


if __name__ == "__main__":
    from axe.data.io import Reader

    config = Reader.read_config("axe.toml")
    logging.basicConfig(
        format=config["log"]["format"],
        datefmt=config["log"]["datefmt"],
    )
    log = logging.getLogger(config["log"]["name"])
    log.setLevel(config["log"]["level"])

    job = LCMTrainJob(config)
    job.run()
