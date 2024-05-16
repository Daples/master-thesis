import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model import ODEModel
from model.parameter import Parameter
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
        forcing: Parameter,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        solver: str = "rk4",
    ) -> None:
        self.n_states: int = n_states
        self.forcing: float = forcing.init_value
        super().__init__(
            initial_condition,
            [forcing],
            time_step,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
        )

    def f(self, _: float, state: NDArray) -> NDArray:
        """The right-hand side of the model."""

        x = state
        n_states = x.shape[0]
        vec = np.zeros(n_states)
        for i in range(n_states):
            forcing = self.parameters[0].current_value
            vec[i] = (x[(i + 1) % n_states] - x[i - 2]) * x[i - 1] - x[i] + forcing
        return vec

    def _observe(self, state: NDArray) -> NDArray:
        """All states are observable."""

        return state
