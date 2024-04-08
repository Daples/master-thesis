from enum import Enum
from typing import Callable

from .discrete import discrete
from .euler import euler
from .rk4 import rk4


class Integrators(Enum):
    """The enumeration of available integrators."""

    EULER = "euler"
    RK4 = "rk4"
    DISCRETE = "discrete"


def get_solver(solver: str) -> Callable:
    """It gets the solver based on the selected input."""

    match Integrators(solver):
        case Integrators.EULER:
            return euler
        case Integrators.RK4:
            return rk4
        case Integrators.DISCRETE:
            return discrete
    raise NotImplementedError("Unknown integrator.")
