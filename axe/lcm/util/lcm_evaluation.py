import torch

from .util import eval_lcm_impl
from axe.lcm.data.schema import LCMDataSchema
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload


class LCMEvalUtil:
    def __init__(
        self,
        min_size_ratio: int,
        max_size_ratio: int,
        max_levels: int,
        model: torch.nn.Module,
        schema: LCMDataSchema,
        bounds: LSMBounds,
    ) -> None:
        self.cf = Cost(bounds.max_considered_levels)

        self.model = model
        self.schema = schema
        self.bounds = bounds

    def eval_lcm(
        self,
        design: LSMDesign,
        system: System,
        workload: Workload,
    ) -> float:
        return eval_lcm_impl(
            design,
            system,
            workload,
            self.model,
            self.bounds.size_ratio_range[0],
            self.bounds.size_ratio_range[1],
        )

    # def gen_random_sample(self):
    #     row = {}
    #     workload = self.scehma.gen.sample_workload()
    #     system: System = self.schema.gen.sample_system()
    #     design: LSMDesign = self.schema.gen.sample_design(system)
    #     cost_lcm = self.eval_lcm(design, system, workload)
    #     cost_acm = self.cf.calc_cost(design, system, workload)
    #     row = {
    #         "z0": workload.z0,
    #         "z1": workload.z1,
    #         "q": workload.q,
    #         "w": workload.w,
    #         "B": system.B,
    #         "s": system.s,
    #         "E": system.E,
    #         "H": system.H,
    #         "N": system.N,
    #         "h": design.h,
    #         "T": design.T,
    #     }
    #     if design.policy in (Policy.Tiering, Policy.Leveling):
    #         row["policy"] = design.policy.value
    #     elif design.policy == Policy.KHybrid:
    #         for idx, k in enumerate(design.K):
    #             row[f"K_{idx}"] = k
    #     elif design.policy == Policy.QFixed:
    #         row["Q"] = design.Q
    #     elif design.policy == Policy.YZHybrid:
    #         row["Y"] = design.Y
    #         row["Z"] = design.Z
    #     row["cost_lcm"] = cost_lcm
    #     row["cost_acm"] = cost_acm
    #
    #     return row, design, system
