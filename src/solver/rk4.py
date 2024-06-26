from numpy.typing import NDArray
from utils._typing import State, InputFunction, Time, SystemDynamics
from solver import Solver


class RK4(Solver):
    """Solve an explicit ODE using 4th-order Runge-Kutta method (RK4)."""

    @classmethod
    def step(
        cls,
        f: SystemDynamics,
        t: Time,
        x: State,
        h: float,
        b: InputFunction,
    ) -> NDArray:
        """Integration step for the RK4."""

        k1 = h * f(t, x).squeeze()
        k2 = h * f(t + h / 2, x + k1 / 2).squeeze()
        k3 = h * f(t + h / 2, x + k2 / 2).squeeze()
        k4 = h * f(t + h, x + k3).squeeze()
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 + b(t, x)
