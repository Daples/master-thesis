import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model import ExplicitModel
from model.parameter import Parameter
from utils._typing import DynamicMatrix, State, Input


class Lorenz96(ExplicitModel):
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
        bias: Parameter,
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        solver: str = "rk4",
        stochastic_propagation: bool = True,
        stochastic_integration: bool = False,
    ) -> None:
        self.n_states: int = n_states
        self.forcing: float = forcing.init_value
        super().__init__(
            initial_condition,
            [forcing, bias],
            time_step,
            H,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
            stochastic_integration=stochastic_integration,
            stochastic_propagation=stochastic_propagation,
        )

    def f(self, _: float, state: State, __: Input) -> NDArray:
        """The right-hand side of the model."""

        x = state
        n_states = x.shape[0]
        vec = np.zeros(n_states)
        for i in range(n_states):
            forcing = (
                self.parameters[0].current_value + self.parameters[1].current_value
            )
            vec[i] = (x[(i + 1) % n_states] - x[i - 2]) * x[i - 1] - x[i] + forcing
        return vec

    def _observe(self, state: State) -> NDArray:
        """All states are observable."""

        return state
