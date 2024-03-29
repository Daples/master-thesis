from model import Model
from numpy.typing import NDArray
from typing import Any
from abc import ABC, abstractmethod

import numpy as np


class Filter(ABC):
    """A class that represents a filter (sequential data assimilation) method.

    Attributes
    ----------
    current_time: float
        The current assimilation time.
    init_state: NDArray
        The mean vector of the initial state.
    init_cov: NDArray
        The covariance matrix of the initial state.
    model: model.model.
    """

    def __init__(self, model: Model, init_state: NDArray, init_cov: NDArray) -> None:
        self.current_time: float = 0
        self.init_state: NDArray = init_state
        self.init_cov: NDArray = init_cov
        self.model: Model = model

        self.analysis_state: NDArray = self.init_state
        self.analysis_cov: NDArray = self.init_cov
        self.forecast_state: NDArray = np.zeros_like(self.init_state)
        self.forecast_cov: NDArray = np.zeros_like(self.init_cov)

    def forecast(self, end_time: float, *params: Any) -> None:
        """Forecast step of the filter. Runs the forward model from the current model
        time to `end_time`.

        Parameters
        ----------
        end_time: float
            The time to forecast the model to.
        *params: Any
            Any additional parameters to run the forecast model.
        """

        self.forecast_state = self.model.forward(
            self.analysis_state, self.current_time, end_time, *params
        )
        self.current_time = end_time

    @abstractmethod
    def analysis(self) -> None:
        """Analysis step of the filter. It corrects the forecasted state using
        observations and (potentially) bias estimation.
        """
