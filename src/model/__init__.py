from abc import ABC, abstractmethod
from typing import Any
from numpy.typing import NDArray

import numpy as np


class Model(ABC):
    """A class to represent an arbitrary numerical forecasting model."""

    def __init__(self, initial_condition: NDArray) -> None:
        self.initial_conditon: NDArray = initial_condition
        self.current_state: NDArray = np.zeros(initial_condition.shape)

    @abstractmethod
    def forward(
        self, state: NDArray, start_time: float, end_time: float, *params: Any
    ) -> NDArray:
        """It runs the numerical model forward from `start_time` to `end_time` using the
        input `state`.

        Parameters
        ----------
        state: NDArray
            The current state vector of the numerical model.
        start_time: float
            The current simulation time.
        end_time: float
            The time to simulate to.
        *params: Any
            Any additional parameters needed to run the forward model.

        Returns
        -------
        NDArray
            The new state vector after simulation.
        """
