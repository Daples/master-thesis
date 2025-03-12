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
        forcing: Parameter,
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        solver: str = "rk4",
        type_f: int = 2,
        stochastic_propagation: bool = True,
        stochastic_integration: bool = False,
    ) -> None:
        self.forcing: float = forcing.init_value
        self.type_f: int = type_f
        super().__init__(
            initial_condition,
            [forcing],
            time_step,
            H,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
            stochastic_integration=stochastic_integration,
            stochastic_propagation=stochastic_propagation,
        )

    def f(self, time: float, state: NDArray, input: NDArray) -> NDArray:
        return getattr(self, f"f_{self.type_f}")(time, state, input)

    def f_0(self, _: float, state: State, __: Input) -> NDArray:
        """The right-hand side of the model, direct calculation."""

        x = state
        n_states = x.shape[0]
        vec = np.zeros(n_states)
        forcing = self.parameters[0].current_value
        for i in range(n_states):
            vec[i] = (x[(i + 1) % n_states] - x[i - 2]) * x[i - 1] - x[i] + forcing
        return vec

    def f_1(self, _: float, state: State, __: Input) -> NDArray:
        """The right-hand side of the model, rolling the arrays."""

        x = state
        F = self.parameters[0].current_value
        x1 = np.roll(x, -1)
        x_1 = np.roll(x, 1)
        x_2 = np.roll(x, 2)

        return (x1 - x_2) * x_1 - x + F

    def f_2(self, _: float, state: State, __: Input) -> NDArray:
        """The right-hand side of the model, concatenating splitting of arrays."""

        F = self.parameters[0].current_value
        x_p1 = np.concatenate((state[1:], [state[0]]))
        x = state
        x_m1 = np.concatenate(([state[-1]], state[:-1]))
        x_m2 = np.concatenate((state[-2:], state[:-2]))

        return (x_p1 - x_m2) * x_m1 - x + F

    def _observe(self, state: State) -> NDArray:
        """All states are observable."""

        return state
