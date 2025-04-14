from typing import Callable, Optional, Tuple

import numpy as np
import scipy.optimize as SciOpt
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload

from .util import (
    ETA_DEFAULT,
    H_DEFAULT,
    LAMBDA_DEFAULT,
    T_DEFAULT,
    Y_DEFAULT,
    Z_DEFAULT,
    get_bounds,
    kl_div_con,
)


class FluidLSMSolver:
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
        h, t, y, z, lamb, eta = x
        design = LSMDesign(
            bits_per_elem=h, size_ratio=t, policy=Policy.Fluid, kapacity=(y, z)
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
        h, t, y, z = x
        design = LSMDesign(
            bits_per_elem=h, size_ratio=t, policy=Policy.Fluid, kapacity=(y, z)
        )
        cost = self.costfunc.calc_cost(design, system, workload)

        return cost

    def get_robust_design(
        self,
        system: System,
        rho: float,
        z0: float,
        z1: float,
        q: float,
        w: float,
        init_args: np.ndarray = np.array(
            [H_DEFAULT, T_DEFAULT, LAMBDA_DEFAULT, ETA_DEFAULT]
        ),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        raise NotImplementedError

    def get_nominal_design(
        self,
        system: System,
        workload: Workload,
        init_args: np.ndarray = np.array([H_DEFAULT, T_DEFAULT, Y_DEFAULT, Z_DEFAULT]),
        minimizer_kwargs: dict = {},
        callback_fn: Optional[Callable] = None,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                policy=Policy.Fluid,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
        }
        default_kwargs.update(minimizer_kwargs)

        solution = SciOpt.minimize(
            fun=lambda x: self.nominal_objective(x, system, workload),
            x0=init_args,
            callback=callback_fn,
            **default_kwargs
        )
        design = LSMDesign(
            bits_per_elem=solution.x[0],
            size_ratio=solution.x[1],
            policy=Policy.Fluid,
            kapacity=(solution.x[2], solution.x[3]),
        )

        return design, solution
