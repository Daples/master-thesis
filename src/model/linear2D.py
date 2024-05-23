import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model import ODEModel
from model.parameter import Parameter
from utils._typing import DynamicMatrix


class Linear2D(ODEModel):
    """A class to represent a 2D linear system.

    Attributes
    ----------
    H: DynamicMatrix
        The observation model.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        time_step: float,
        parameters: list[Parameter],
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        solver: str = "rk4",
    ) -> None:
        super().__init__(
            initial_condition,
            parameters,
            time_step,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
        )
        self.H: DynamicMatrix = H

    def f(self, _: float, state: NDArray) -> NDArray:
        """The right-hand side of the model."""

        return (
            np.array(
                [
                    [0, 1],
                    [
                        self.parameters[0].current_value,
                        self.parameters[1].current_value,
                    ],
                ]
            )
            @ state
        )

    def _observe(self, state: NDArray) -> NDArray:
        """All states are observable."""

        return self.H(self.current_time) @ state
