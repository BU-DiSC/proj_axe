from typing import Callable, Optional, Tuple

import numpy as np
import scipy.optimize as SciOpt
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload

from .util import (
    ETA_DEFAULT,
    H_DEFAULT,
    K_DEFAULT,
    LAMBDA_DEFAULT,
    T_DEFAULT,
    get_bounds,
    kl_div_con,
)


class KLSMSolver:
    def __init__(self, bounds: LSMBounds):
        self.bounds = bounds
        self.costfunc = Cost(bounds.max_considered_levels)

    def robust_objective(
        self,
        x: np.ndarray,
        system: System,
        rho: float,
        workload: Workload,
    ) -> float:
        lamb, eta = x[0:2]
        h, t = x[2:4]
        kaps = x[4:].tolist()
        design = LSMDesign(
            bits_per_elem=h, size_ratio=t, kapacity=kaps, policy=Policy.Kapacity
        )
        query_cost = 0
        query_cost += workload.z0 * kl_div_con(
            (self.costfunc.Z0(design, system) - eta) / lamb
        )
        query_cost += workload.z1 * kl_div_con(
            (self.costfunc.Z1(design, system) - eta) / lamb
        )
        query_cost += workload.q * kl_div_con(
            (self.costfunc.Q(design, system) - eta) / lamb
        )
        query_cost += workload.w * kl_div_con(
            (self.costfunc.W(design, system) - eta) / lamb
        )
        cost = eta + (rho * lamb) + (lamb * query_cost)

        return cost

    def nominal_objective(
        self,
        x: np.ndarray,
        system: System,
        workload: Workload,
    ) -> float:
        h, t = x[0:2]
        kaps = x[2:].tolist()
        design = LSMDesign(
            bits_per_elem=h, size_ratio=t, kapacity=kaps, policy=Policy.Kapacity
        )
        cost = self.costfunc.calc_cost(design, system, workload)

        return cost

    def get_robust_design(
        self,
        system: System,
        workload: Workload,
        rho: float,
        init_args: np.ndarray = np.array(
            [LAMBDA_DEFAULT, ETA_DEFAULT, H_DEFAULT, T_DEFAULT, K_DEFAULT]
        ),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        max_levels = self.bounds.max_considered_levels
        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                policy=Policy.Kapacity,
                system=system,
                robust=True,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        kap_val = init_args[-1]
        init_args = np.concatenate(
            (init_args[0:4], np.array([kap_val for _ in range(max_levels)]))
        )
        solution = SciOpt.minimize(
            fun=lambda x: self.robust_objective(x, system, rho, workload),
            x0=init_args,
            callback=callback_fn,
            **default_kwargs
        )
        design = LSMDesign(
            bits_per_elem=solution.x[2],
            size_ratio=solution.x[3],
            policy=Policy.Kapacity,
            kapacity=solution.x[4:].tolist(),
        )

        return design, solution

    def get_nominal_design(
        self,
        system: System,
        workload: Workload,
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT, K_DEFAULT]),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        max_levels = self.bounds.max_considered_levels

        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                policy=Policy.Kapacity,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)
        kap_val = init_args[-1]
        init_args = np.concatenate(
            (init_args[0:2], np.array([kap_val for _ in range(max_levels)]))
        )

        solution = SciOpt.minimize(
            fun=lambda x: self.nominal_objective(x, system, workload),
            x0=init_args,
            callback=callback_fn,
            **default_kwargs
        )
        design = LSMDesign(
            bits_per_elem=solution.x[0],
            size_ratio=solution.x[1],
            kapacity=solution.x[2:],
            policy=Policy.Kapacity,
        )

        return design, solution
