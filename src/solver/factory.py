from enum import Enum
from typing import Type

from .discrete import Discrete
from solver import Solver
from .rk4 import RK4


class Integrators(Enum):
    """The enumeration of available integrators."""

    RK4 = "rk4"
    DISCRETE = "discrete"


def get_solver(solver: str) -> Type[Solver]:
    """It gets the solver based on the selected input."""

    match Integrators(solver):
        case Integrators.RK4:
            return RK4
        case Integrators.DISCRETE:
            return Discrete
    raise NotImplementedError("Unknown integrator.")
