from numpy.typing import NDArray
from numpy.random import Generator
from model import ExplicitModel
from utils._typing import DynamicMatrix, InputFunction, Input, State

import numpy as np


class ARModel(ExplicitModel):
    """An auto-regressive model.

    Attributes
    ----------
    A: DynamicMatrix
        The coefficient matrix for the multivariate AR process.
    add_noise: bool
        If the model should be propagated with added noise.
    """

    def __init__(
        self,
        A: DynamicMatrix,
        H: DynamicMatrix,
        initial_condition: State,
        time_step: float,
        system_cov: DynamicMatrix,
        generator: Generator,
        stochastic_propagation: bool = False,
        stochastic_integration: bool = False,
        input: InputFunction | None = None,
    ) -> None:
        parameters = []
        observation_cov = lambda _: 0 * np.eye(H(0).shape[0])
        super().__init__(
            initial_condition,
            parameters,
            time_step,
            H,
            system_cov,
            observation_cov,
            generator,
            stochastic_propagation=stochastic_propagation,
            stochastic_integration=stochastic_integration,
            solver="discrete",
            input=input,
        )
        self.A: DynamicMatrix = A

    def f(self, time: float, state: State, input: Input) -> NDArray:
        """AR model with affine input."""

        return (self.A(time) @ state[:, np.newaxis]).squeeze() + input
