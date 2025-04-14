import random

import axe.lsm.data_generator as DataGen
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, Policy, System, Workload

kSYSTEM_HEADER = ["entry_p_page", "selec", "entry_size", "mem_budget", "num_elem"]
kWORKLOAD_HEADER = ["z0", "z1", "q", "w"]


class LTunerDataSchema:
    def __init__(
        self,
        policy: Policy,
        bounds: LSMBounds,
        precision: int = 3,
        seed: int = 0,
        robust: bool = False,
    ) -> None:
        self.cost: Cost = Cost(bounds.max_considered_levels)
        self.gen = DataGen.build_data_gen(policy=policy, bounds=bounds, seed=seed)
        self.policy: Policy = policy
        self.bounds: LSMBounds = bounds
        self.precision: int = precision
        self.robust: bool = robust

    def label_cols(self) -> list[str]:
        return self.feat_cols()

    def feat_cols(self) -> list[str]:
        cols = kWORKLOAD_HEADER + kSYSTEM_HEADER
        if self.robust:
            cols += ["rho"]
        return cols

    def get_column_names(self) -> list[str]:
        return kWORKLOAD_HEADER + kSYSTEM_HEADER + ["rho"]

    def sample_row(self) -> list:
        workload: Workload = self.gen.sample_workload()
        system: System = self.gen.sample_system()
        rho: float = self.gen.rng.uniform(low=0, high=2)

        line = [
            workload.z0,
            workload.z1,
            workload.q,
            workload.w,
            system.entries_per_page,
            system.selectivity,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
            rho,
        ]

        return line

    def sample_row_dict(self) -> dict:
        column_names = self.get_column_names()
        row = self.sample_row()
        line = {}
        for key, val in zip(column_names, row):
            line[key] = val

        return line
