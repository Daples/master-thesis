from numpy.typing import NDArray
from utils._typing import State, Time, SystemDynamics, InputFunction
from solver import Solver


class Discrete(Solver):
    """Forward a discrete model in time."""

    @classmethod
    def step(
        cls,
        f: SystemDynamics,
        t: Time,
        x: State,
        _: Time,
        b: InputFunction,
    ) -> NDArray:
        """A step of the discrete model."""

        return f(t, x).squeeze() + b(t, x)
