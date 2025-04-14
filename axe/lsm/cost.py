import numpy as np
import axe.lsm.lsm_cost_model as CostModel
from axe.lsm.types import Policy, System, LSMDesign, Workload


class Cost:
    def __init__(self, max_levels: int) -> None:
        super().__init__()
        self.max_levels = max_levels

    def L(self, design: LSMDesign, system: System, ceil=False):
        level = CostModel.calc_level(
            design.bits_per_elem,
            design.size_ratio,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
            ceil,
        )

        return level

    def mbuff(self, design: LSMDesign, system: System):
        return CostModel.calc_mbuff(
            design.bits_per_elem, system.mem_budget, system.num_entries
        )

    def create_k_list(self, design: LSMDesign, system: System) -> np.ndarray:
        assert design.kapacity is not None
        if design.policy is Policy.Kapacity:
            kapacities = np.array(design.kapacity)
        elif design.policy is Policy.Tiering:
            kapacities = np.full(self.max_levels, design.size_ratio - 1)
        elif design.policy is Policy.Leveling:
            kapacities = np.ones(self.max_levels)
        elif design.policy is Policy.Fluid:
            levels = CostModel.calc_level(
                design.bits_per_elem,
                design.size_ratio,
                system.entry_size,
                system.mem_budget,
                system.num_entries,
                True,
            )
            levels = int(levels)
            kapacities = np.full(levels - 1, design.kapacity[0])
            kapacities = np.concatenate((kapacities, [design.kapacity[1]]))
            kapacities = np.pad(
                kapacities,
                (0, self.max_levels - len(kapacities)),
                "constant",
                constant_values=(1.0, 1.0),
            )
        elif design.policy is Policy.QHybrid:
            kapacities = np.full(self.max_levels, design.kapacity[0])
        else:
            kapacities = np.ones(self.max_levels)

        return kapacities

    def Z0(self, design: LSMDesign, system: System) -> float:
        kapacities = self.create_k_list(design, system)
        cost = CostModel.empty_op(
            design.bits_per_elem,
            design.size_ratio,
            kapacities,
            system.num_entries,
            system.entry_size,
            system.mem_budget,
        )

        return cost

    def Z1(self, design: LSMDesign, system: System) -> float:
        kapacities = self.create_k_list(design, system)
        cost = CostModel.non_empty_op(
            design.bits_per_elem,
            design.size_ratio,
            kapacities,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
        )

        return cost

    def Q(self, design: LSMDesign, system: System) -> float:
        kapacities = self.create_k_list(design, system)
        cost = CostModel.range_op(
            design.bits_per_elem,
            design.size_ratio,
            kapacities,
            system.entries_per_page,
            system.selectivity,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
        )

        return cost

    def W(self, design: LSMDesign, system: System) -> float:
        kapacities = self.create_k_list(design, system)
        cost = CostModel.write_op(
            design.bits_per_elem,
            design.size_ratio,
            kapacities,
            system.entries_per_page,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
            system.phi,
        )

        return cost

    def calc_cost(
        self,
        design: LSMDesign,
        system: System,
        workload: Workload,
    ):
        kapacities = self.create_k_list(design, system)
        cost = CostModel.calc_cost(
            design.bits_per_elem,
            design.size_ratio,
            kapacities,
            workload.z0,
            workload.z1,
            workload.q,
            workload.w,
            system.entries_per_page,
            system.selectivity,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
            system.phi,
        )

        return cost
