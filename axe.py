#!/usr/bin/env python
import logging
import os
import sys
import toml
from typing import Any

from jobs.lcm_train import LCMTrainJob
from jobs.data_gen import DataGenJob
from jobs.ltune_train import LTuneTrainJob
# from jobs.botorch_bo import BayesianPipeline
# from jobs.mlos_bo import BayesianPipelineMlos
from jobs.mlos_exp_runs import ExperimentMLOS
from jobs.ltune_robust_train import LTuneRobustTrainJob


class AxeDriver:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        logging.basicConfig(
            format=config["log"]["format"], datefmt=config["log"]["datefmt"]
        )
        self.log: logging.Logger = logging.getLogger(config["log"]["name"])
        self.log.setLevel(getattr(logging, config["log"]["level"]))
        log_level = logging.getLevelName(self.log.getEffectiveLevel())
        self.log.debug(f"Log level: {log_level}")

    def run(self):
        self.log.info(f'Staring app {self.config["app"]["name"]}')

        jobs = {
            "DataGen": DataGenJob,
            "LCMTrain": LCMTrainJob,
            "LTuneTrain": LTuneTrainJob,
            # "BayesianPipelineBoTorch": BayesianPipeline,
            # "BayesianPipelineMLOS": BayesianPipelineMlos,
            "ExperimentMLOS": ExperimentMLOS,
            "LTuneRobustTrain": LTuneRobustTrainJob,
        }
        jobs_list = self.config["app"]["run"]
        for job_name in jobs_list:
            job = jobs.get(job_name, None)
            if job is None:
                self.log.warning(f"No job associated with {job_name}")
                continue
            job = job(config)
            _ = job.run()

        self.log.info("All jobs finished, exiting")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        file_dir = os.path.dirname(__file__)
        config_path = os.path.join(file_dir, "axe.toml")

    with open(config_path) as fid:
        config = toml.load(fid)

    driver = AxeDriver(config)
    driver.run()
