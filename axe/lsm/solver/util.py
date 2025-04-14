from typing import Optional, Tuple

import numpy as np
import scipy.optimize as SciOpt

from axe.lsm.types import Policy, System, LSMBounds

H_DEFAULT = 3
T_DEFAULT = 3
Z_DEFAULT = 1
Y_DEFAULT = 1
Q_DEFAULT = 1
K_DEFAULT = 1
LAMBDA_DEFAULT = 1
ETA_DEFAULT = 1


def kl_div_con(input: float):
    return np.exp(input) - 1


def get_lambda_bounds() -> Tuple[float, float]:
    return 0.1, np.inf


def get_eta_bounds() -> Tuple[float, float]:
    return -np.inf, np.inf


def get_bounds(
    bounds: LSMBounds,
    policy: Policy = Policy.Leveling,
    system: Optional[System] = None,
    robust: bool = False,
) -> SciOpt.Bounds:
    t_bounds = bounds.size_ratio_range
    if system is None:
        h_bounds = (bounds.bits_per_elem_range[0], bounds.bits_per_elem_range[1] - 0.1)
    else:
        h_bounds = (bounds.bits_per_elem_range[0], system.mem_budget - 0.1)

    lb = (h_bounds[0], t_bounds[0])
    ub = (h_bounds[1], t_bounds[1] - 1)
    if policy == Policy.QHybrid:
        lb += (t_bounds[0] - 1,)
        ub += (t_bounds[1] - 2,)
    elif policy == Policy.Fluid:
        lb += (t_bounds[0] - 1, t_bounds[0] - 1)
        ub += (t_bounds[1] - 2, t_bounds[1] - 2)
    elif policy == Policy.Kapacity:
        max_levels: int = bounds.max_considered_levels
        lb += tuple(t_bounds[0] - 1 for _ in range(max_levels))
        ub += tuple(t_bounds[1] - 1 for _ in range(max_levels))
    elif policy in (Policy.Tiering, Policy.Leveling):
        pass  # No need to add more items for classic policy

    if robust:
        lambda_bounds = get_lambda_bounds()
        eta_bounds = get_eta_bounds()
        lb = (eta_bounds[0], lambda_bounds[0]) + lb
        ub = (eta_bounds[1], lambda_bounds[1]) + ub

    return SciOpt.Bounds(lb=lb, ub=ub, keep_feasible=True)  # type: ignore


def get_default_decision_vars(policy: Policy, max_levels: int) -> np.ndarray:
    out = [H_DEFAULT, T_DEFAULT]
    if policy == Policy.Kapacity:
        out += [K_DEFAULT for _ in range(max_levels)]
    elif policy == Policy.Fluid:
        out += [Y_DEFAULT, Z_DEFAULT]
    elif policy == Policy.QHybrid:
        out += [Q_DEFAULT]

    return np.array(out)
