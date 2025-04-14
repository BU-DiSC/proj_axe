from typing import Type
from axe.lsm.types import Policy
from .classic_solver import ClassicSolver
from .qlsm_solver import QLSMSolver
from .klsm_solver import KLSMSolver
from .fluidlsm_solver import FluidLSMSolver


def get_solver_from_policy(
    choice: Policy,
) -> Type[ClassicSolver | QLSMSolver | KLSMSolver | FluidLSMSolver]:
    choices = {
        Policy.Tiering: ClassicSolver,
        Policy.Leveling: ClassicSolver,
        Policy.Classic: ClassicSolver,
        Policy.QHybrid: QLSMSolver,
        Policy.Fluid: FluidLSMSolver,
        Policy.Kapacity: KLSMSolver,
    }
    solver = choices.get(choice, None)
    if solver is None:
        raise KeyError

    return solver
