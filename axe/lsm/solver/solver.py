import numpy as np
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, Policy, System, Workload

from .util import (
    get_bounds,
    get_default_decision_vars,
)


class BlackBoxSolver:
    def __init__(self, bounds: LSMBounds, policy: Policy):
        self.bounds: LSMBounds = bounds
        self.policy: Policy = policy
        self.costfunc: Cost = Cost(bounds.max_considered_levels)

    def nominal_objective(
        self,
        decision_vars: np.ndarray,
        system: System,
        workload: Workload,
    ) -> float:
        return 0

    def get_nominal_design(
        self,
        system: System,
        workload: Workload,
        minimizer_kwargs: dict = {},
    ):
        default_kwargs = {
            "method": "SLSQP",
            "bounds": get_bounds(
                bounds=self.bounds,
                policy=self.policy,
                system=system,
                robust=False,
            ),
            "options": {"ftol": 1e-6, "disp": False, "maxiter": 1000},
            "x0": get_default_decision_vars(
                self.policy, self.bounds.max_considered_levels
            ),
        }
        default_kwargs.update(minimizer_kwargs)

        return
