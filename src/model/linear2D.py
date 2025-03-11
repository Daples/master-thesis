import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model import ExplicitModel
from model.parameter import Parameter
from utils._typing import DynamicMatrix, Input, State, Time


class Linear2D(ExplicitModel):
    """A class to represent a 2D linear system."""

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
            H,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
        )

    def f(self, _: Time, state: State, __: Input) -> NDArray:
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
