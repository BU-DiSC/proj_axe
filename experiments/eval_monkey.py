import argparse
import logging
import subprocess

import polars as pl
import toml

from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload
from axe.ltuner.data.schema import LTunerDataSchema
from experiments.infra import AxeResultDB

MONKEY_BIN = "../../rocksdb-dosto/examples/monkey_experiments/throughput_exp_runner"
T_IDX = 12
H_IDX = 18
P_IDX = 15


class ExpMonkeyEvaluation:
    def __init__(self, config: dict, monkey_bin_path: str = MONKEY_BIN) -> None:
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        self.db = AxeResultDB(config)
        self.monkey_bin_path = monkey_bin_path
        bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.cost_fn = Cost(bounds.max_considered_levels)

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
            mem_budget=row["bits_per_elem_max"],
            num_entries=row["num_entries"],
        )
        return workload, system

    def get_monkey_tuning(self, system: System, workload: Workload):
        cmd = " ".join(
            [
                self.monkey_bin_path,
                "--mock_run",
                "--path=/tmp/db",
                "--num_elements=100000000",
                "--hol_opt=2",
                f"--bits-per-entry={system.mem_budget}",
                f"--entry_size={system.entry_size}",
                f"--hol_z={workload.z0 + workload.z1}",
                f"--hol_q={workload.q}",
                f"--hol_w={workload.w}",
                f"--hol_v={workload.z1 / (workload.z0 + workload.z1)}",
            ]
        )
        proc_out, _ = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        ).communicate()
        proc_out = proc_out.split(",")
        bits_per_elem = float(proc_out[H_IDX])
        size_ratio = int(proc_out[T_IDX])
        policy = Policy.Leveling if proc_out[P_IDX] == "l" else Policy.Tiering

        return LSMDesign(
            bits_per_elem=bits_per_elem,
            size_ratio=size_ratio,
            policy=policy,
            kapacity=(),
        )

    def run(self):
        env_table: pl.DataFrame = self.db.get_env_table()
        table = []
        for env in env_table.to_dicts():
            self.log.info(f"{env=}")
            workload, system = self.row_to_objs(env)
            design = self.get_monkey_tuning(system, workload)
            row = LTunerDataSchema.design_to_dict(design)
            row["cost"] = self.cost_fn.calc_cost(design, system, workload)
            table.append(row)
        table = pl.concat((env_table, pl.DataFrame(table)), how="horizontal")

        return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--monkey_bin", type=str, help="path to monkey bin")
    args = parser.parse_args()
    config = toml.load(args.config)
    logging.basicConfig(**config["log"])
    log: logging.Logger = logging.getLogger(config["app"]["name"])
    log.info(f"Log level: {logging.getLevelName(log.getEffectiveLevel())}")

    ExpMonkeyEvaluation(config, monkey_bin_path=args.monkey_bin).run()


if __name__ == "__main__":
    main()
