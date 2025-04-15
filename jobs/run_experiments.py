#!/usr/bin/env python
import argparse
import logging

import toml

from experiments.mlos_exp_runs import ExperimentMLOS


class RunExperiments:
    def __init__(self, cfg: dict) -> None:
        self.log: logging.Logger = logging.getLogger(cfg["app"]["name"])

        jcfg = cfg["job"]["run_experiments"]
        self.exp_list = jcfg["exp_list"]
        self.cfg = cfg

    def run(self) -> None:
        experiments = {"ExperimentMLOS": ExperimentMLOS}
        self.log.info(f"Jobs to run: {self.exp_list}")
        for exp_name in self.exp_list:
            experiment = experiments.get(exp_name, None)
            if experiment is None:
                self.log.warning(f"No job associated with {exp_name}")
                continue
            exp = experiment(self.cfg)
            _ = exp.run()

        self.log.info("All experiments finished, exiting")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()
    config = toml.load(args.config)
    logging.basicConfig(**config["log"])
    log: logging.Logger = logging.getLogger(config["app"]["name"])
    log.info(f"Log level: {logging.getLevelName(log.getEffectiveLevel())}")

    RunExperiments(toml.load(args.config)).run()


if __name__ == "__main__":
    main()
