#!/usr/bin/env python
import logging
import os
import sys
import toml
from typing import Any

from jobs.create_lcm_data import CreateLCMData
from jobs.create_ltuner_data import CreateLTunerData
# from jobs.mlos_exp_runs import ExperimentMLOS
from jobs.train_lcm import TrainLCM
from jobs.train_ltuner import TrainLTuner

class AxeDriver:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        logging.basicConfig(
            format=config["log"]["format"], datefmt=config["log"]["datefmt"]
        )
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        self.log.setLevel(getattr(logging, config["log"]["level"]))
        log_level = logging.getLevelName(self.log.getEffectiveLevel())
        self.log.debug(f"Log level: {log_level}")

    def run(self):
        jobs = {
            "create_lcm_data": CreateLCMData,
            "train_lcm": TrainLCM,
            "create_ltuner_data": CreateLTunerData,
            "train_ltuner": TrainLTuner,
            # "ExperimentMLOS": ExperimentMLOS,
        }
        jobs_list = self.config["app"]["run"]
        self.log.info(f"Jobs to run: {jobs_list}")
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
