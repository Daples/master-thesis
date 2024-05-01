import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model import ODEModel
from utils._typing import DynamicMatrix


class Lorenz96(ODEModel):
    """A class to represent the Lorenz-96 model.

    Attributes
    ----------
    forcing: float
        The forcing parameters of the model.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        time_step: float,
        n_states: int,
        forcing: float,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        solver: str = "rk4",
    ) -> None:
        super().__init__(
            initial_condition,
            time_step,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
        )
        self.n_states: int = n_states
        self.forcing: float = forcing

    def f(self, _: float, state: NDArray) -> NDArray:
        """The right-hand side of the model."""

        x = state
        n_states = x.shape[0]
        vec = np.zeros(n_states)
        for i in range(n_states):
            vec[i] = (x[(i + 1) % n_states] - x[i - 2]) * x[i - 1] - x[i] + self.forcing
        return vec

    def _observe(self, state: NDArray) -> NDArray:
        """All states are observable."""

        return state
