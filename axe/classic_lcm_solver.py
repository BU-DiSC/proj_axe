from typing import Optional, Callable, Tuple, List
import torch

import numpy as np
import scipy.optimize as SciOpt

from axe.lcm.model.wrapper import LCMWrapper
from axe.lsm.cost import Cost
from axe.lsm.types import LSMDesign, Policy, System, LSMBounds, Workload
from axe.lsm.solver.util import get_bounds

H_DEFAULT = 3
T_DEFAULT = 5
LAMBDA_DEFAULT = 10
ETA_DEFAULT = 10


class ClassicLCMSolver:
    def __init__(
        self,
        bounds: LSMBounds,
        lcm: LCMWrapper,
        policies: Optional[List[Policy]] = None,
    ) -> None:
        self.bounds = bounds
        self.costfunc = Cost(bounds.max_considered_levels)
        if policies is None:
            policies = [Policy.Tiering, Policy.Leveling]
        self.policies = policies
        self.lcm = lcm

    def robust_objective(
        self,
        x: np.ndarray,
        policy: Policy,
        system: System,
        workload: Workload,
        rho: float,
    ) -> float:
        eta, lamb, h, T = x
        wl = torch.Tensor([workload.z0, workload.z1, workload.q, workload.w]).clamp(
            min=1e-6
        )
        design = LSMDesign(bits_per_elem=h, size_ratio=T, policy=policy, kapacity=())
        cost = self.lcm(design, system, workload).flatten()
        cost = cost / wl
        cost = (cost - eta) / max(lamb, 1)
        cost = torch.exp(cost) - 1
        cost = cost * wl
        cost = eta + (lamb * rho) + (lamb * cost.sum().item())

        return cost

    def nominal_objective(
        self,
        x: np.ndarray,
        policy: Policy,
        system: System,
        workload: Workload,
    ) -> float:
        h, T = x
        design = LSMDesign(bits_per_elem=h, size_ratio=T, policy=policy, kapacity=())
        cost = self.lcm(design, system, workload).sum().item()

        return cost

    def get_robust_design(
        self,
        system: System,
        workload: Workload,
        rho: float,
        init_args: np.ndarray = np.array(
            [ETA_DEFAULT, LAMBDA_DEFAULT, H_DEFAULT, T_DEFAULT]
        ),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        design = None
        solution = None

        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                system=system,
                robust=True,
            ),
            "options": {"ftol": 1e-12, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        min_sol = np.inf
        assert len(self.policies) > 0
        for policy in self.policies:
            sol = SciOpt.minimize(
                fun=lambda x: self.robust_objective(x, policy, system, workload, rho),
                x0=init_args,
                callback=callback_fn,
                **default_kwargs
            )
            if sol.fun < min_sol or (design is None and solution is None):
                min_sol = sol.fun
                design = LSMDesign(
                    bits_per_elem=sol.x[2],
                    size_ratio=sol.x[3],
                    policy=policy,
                    kapacity=(),
                )
                solution = sol
        assert design is not None
        assert solution is not None

        return design, solution

    def get_nominal_design(
        self,
        system: System,
        workload: Workload,
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT]),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-12, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        design, solution = None, None
        min_sol = np.inf
        for policy in self.policies:
            sol = SciOpt.minimize(
                fun=lambda x: self.nominal_objective(x, policy, system, workload),
                x0=init_args,
                callback=callback_fn,
                **default_kwargs
            )
            if sol.fun < min_sol or (design is None and solution is None):
                min_sol = sol.fun
                design = LSMDesign(
                    bits_per_elem=sol.x[0],
                    size_ratio=sol.x[1],
                    policy=policy,
                    kapacity=(),
                )
                solution = sol
        assert design is not None
        assert solution is not None

        return design, solution
