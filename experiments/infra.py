import logging
import os
import sqlite3

import polars as pl

from axe.lsm.types import System, Workload

workloads = [
    Workload(0.25, 0.25, 0.25, 0.25),
    Workload(0.97, 0.01, 0.01, 0.01),
    Workload(0.01, 0.97, 0.01, 0.01),
    Workload(0.01, 0.01, 0.97, 0.01),
    Workload(0.01, 0.01, 0.01, 0.97),
    Workload(0.49, 0.49, 0.01, 0.01),
    Workload(0.49, 0.01, 0.49, 0.01),
    Workload(0.49, 0.01, 0.01, 0.49),
    Workload(0.01, 0.49, 0.49, 0.01),
    Workload(0.01, 0.49, 0.01, 0.49),
    Workload(0.01, 0.01, 0.49, 0.49),
    Workload(0.33, 0.33, 0.33, 0.01),
    Workload(0.33, 0.33, 0.01, 0.33),
    Workload(0.33, 0.01, 0.33, 0.33),
    Workload(0.01, 0.33, 0.33, 0.33),
]


class AxeResultDB:
    def __init__(self, config: dict) -> None:
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        self.db_path = os.path.join(config["io"]["data_dir"], config["io"]["database"])
        self.con = sqlite3.connect(self.db_path)
        self.build_environments_table()

    def build_environments_table(self):
        cursor = self.con.cursor()
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='environments'"
        )
        result = cursor.fetchone()[0]
        if result:
            self.log.info("Environments table already created, skipping")
            return

        # TODO: Honestly Andy should probably just change the table to include both the
        # representative benchmark suite (workloads array above) and a handful of
        # randomly generated ones into one table. Add a simple TEXT tag "bench" and
        # "rand" to distinguish between the two. Should make experiments a little easier
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
                num_entries INT,
                mem_budget REAL,
                read_write_asym FLOAT
            );
            """
        )
        insert_sql = """
        INSERT INTO environments (
            empty_reads,
            non_empty_reads,
            range_queries,
            writes,
            entry_size,
            selectivity,
            entries_per_page,
            num_entries,
            mem_budget,
            read_write_asym
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        system = System()
        for workload in workloads:
            args = (
                workload.z0,
                workload.z1,
                workload.q,
                workload.w,
                system.entry_size,
                system.selectivity,
                system.entries_per_page,
                system.num_entries,
                system.mem_budget,
                system.phi,
            )
            cursor.execute(insert_sql, args)

        self.con.commit()
        cursor.close()

    def get_env_table(self) -> pl.DataFrame:
        env_table = pl.read_database("SELECT * FROM environments;", self.con)

        return env_table

    def run_sql(self, sql_query: str, args: tuple = ()):
        cursor = self.con.cursor()
        cursor.execute(sql_query, args)
        self.con.commit()
        cursor.close()

        return
