#!/usr/bin/env python
import os
import torch
import logging
import toml
from typing import Any, Callable, Optional

import torch.optim as Opt
from torch.utils.data import DataLoader

from axe.lsm.types import LSMBounds, Policy
from axe.ltune.data.dataset import LTuneDataSet
from axe.ltune.robust_loss import LearnedRobustLoss
from axe.ltune.model.builder import LTuneModelBuilder
from axe.util.lr_scheduler import LRSchedulerBuilder
from axe.util.optimizer import OptimizerBuilder
from axe.util.trainer import Trainer


class LTuneRobustTrainJob:
    def __init__(self, config: dict[str, Any]) -> None:
        self.log = logging.getLogger(config["log"]["name"])
        self.log.info("Running Robust Training Job")
        self.use_gpu = config["job"]["use_gpu_if_avail"]
        self.save_dir = os.path.join(
            config["io"]["data_dir"],
            config["job"]["LTuneRobustTrain"]["save_dir"],
        )
        self.design = getattr(Policy, config["lsm"]["design"])
        self.bounds = LSMBounds(**config["lsm"]["bounds"])
        self.opt_builder = OptimizerBuilder(config["optimizer"])
        self.schedule_builder = LRSchedulerBuilder(config["scheduler"])
        self.model_builder = LTuneModelBuilder(
            size_ratio_range=self.bounds.size_ratio_range,
            max_levels=self.bounds.max_considered_levels,
            **config["ltune"]["model"],
        )

        self.config = config
        self.jconfig = config["job"]["LTuneRobustTrain"]

    def _build_loss_fn(self) -> torch.nn.Module:
        model = LearnedRobustLoss(self.config, self.jconfig["loss_fn_path"])
        if self.use_gpu and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_model(self) -> torch.nn.Module:
        model = self.model_builder.build_robust_model()
        if self.use_gpu and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_optimizer(self, model: torch.nn.Module) -> Opt.Optimizer:
        choice = self.jconfig["optimizer"]

        return self.opt_builder.build_optimizer(choice, model)

    def _build_scheduler(
        self, optimizer: Opt.Optimizer
    ) -> Optional[Opt.lr_scheduler._LRScheduler]:
        choice = self.jconfig["lr_scheduler"]

        return self.schedule_builder.build_scheduler(optimizer, choice)

    def _build_train(self) -> DataLoader:
        train_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.jconfig["train"]["dir"],
        )
        train_data = LTuneDataSet(
            folder=train_dir,
            shuffle=self.jconfig["train"]["shuffle"],
            robust=True,
        )
        train = DataLoader(
            train_data,
            batch_size=self.jconfig["train"]["batch_size"],
            drop_last=self.jconfig["train"]["drop_last"],
            num_workers=self.jconfig["train"]["num_workers"],
            pin_memory=True,
            prefetch_factor=10,
        )

        return train

    def _build_test(self) -> DataLoader:
        test_dir = os.path.join(
            self.config["io"]["data_dir"],
            self.jconfig["test"]["dir"],
        )
        test_data = LTuneDataSet(
            folder=test_dir,
            shuffle=self.jconfig["test"]["shuffle"],
            robust=True,
        )
        test = DataLoader(
            test_data,
            batch_size=self.jconfig["test"]["batch_size"],
            drop_last=self.jconfig["test"]["drop_last"],
            num_workers=self.jconfig["test"]["num_workers"],
        )

        return test

    def _make_save_dir(self) -> bool:
        self.log.info(f"Saving model in: {self.save_dir}")
        try:
            os.makedirs(self.save_dir, exist_ok=False)
        except FileExistsError:
            return False

        with open(os.path.join(self.save_dir, "axe.toml"), "w") as fid:
            toml.dump(self.config, fid)

        return True

    @staticmethod
    def gumbel_temp_schedule(
        train_kwargs: dict,
        decay_rate: float = 0.95,
        floor: float = 0.01,
    ) -> dict:
        train_kwargs["temp"] *= decay_rate
        if train_kwargs["temp"] < floor:
            train_kwargs["temp"] = floor

        return train_kwargs

    @staticmethod
    def reinmax_temp_schedule(
        train_kwargs: dict,
        decay_rate: float = 0.9,
        floor: float = 1,
    ) -> dict:
        train_kwargs["temp"] *= decay_rate
        if train_kwargs["temp"] < floor:
            train_kwargs["temp"] = floor

        return train_kwargs

    def get_train_callback(self) -> Optional[Callable[[dict], dict]]:
        if self.config["ltune"]["model"]["categorical_mode"] == "reinmax":
            return lambda train_kwargs: self.reinmax_temp_schedule(train_kwargs)
        # default train_callback will be gumbel softmax
        return lambda train_kwargs: self.gumbel_temp_schedule(train_kwargs)

    def run(self) -> Optional[Trainer]:
        dir_success = self._make_save_dir()
        if not dir_success:
            self.log.info("Model directory already exists, exiting...")
            return None

        model = self._build_model()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        train_data = self._build_train()
        test_data = self._build_test()
        loss_fn = self._build_loss_fn()
        disable_tqdm = self.config["log"]["disable_tqdm"]
        callback = self.get_train_callback()

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
            model_dir=self.save_dir,
            model_train_kwargs=self.config["ltune"]["train_kwargs"],
            model_test_kwargs=self.config["ltune"]["test_kwargs"],
            disable_tqdm=disable_tqdm,
            no_checkpoint=self.jconfig["no_checkpoint"],
            train_callback=callback,
        )
        trainer.run()

        return trainer


def main():
    from axe.data.io import Reader

    config = Reader.read_config("axe.toml")
    logging.basicConfig(
        format=config["log"]["format"], datefmt=config["log"]["datefmt"]
    )
    log = logging.getLogger(config["log"]["name"])
    log.setLevel(config["log"]["level"])

    job = LTuneRobustTrainJob(config)
    job.run()


if __name__ == "__main__":
    main()
