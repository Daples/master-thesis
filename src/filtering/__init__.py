from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from model import Model
from utils import default_generator


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
    model: Model
        The forward model.
    analysis_state: NDArray
        The analysis state vector, after assimilation.
    analysis_cov: NDArray
        The analysis covariance matrix.
    forecast_state: NDArray
        The forecast state vector, prior to assimilation.
    forecast_cov: NDArray
        The forecast state covariance matrix.
    generator: Generator
        The random number generator.
    """

    def __init__(
        self,
        model: Model,
        init_state: NDArray,
        init_cov: NDArray,
        generator: Generator | None = None,
    ) -> None:
        self.current_time: float = 0
        self.init_state: NDArray = init_state
        self.init_cov: NDArray = init_cov
        self.model: Model = model

        if generator is None:
            generator = default_generator
        self.generator: Generator = generator

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
        self.current_time = self.model.current_time

    @abstractmethod
    def analysis(self, observation: NDArray) -> None:
        """Analysis step of the filter. It corrects the forecasted state using
        observations.

        Parameters
        ----------
        observation: NDArray
            The observation to assimilate.
        """

    @abstractmethod
    def compute_gain(self) -> NDArray:
        """It computes the gain matrix for the analysis step.

        Returns
        -------
        NDArray
            The gain matrix.
        """

    def filter(
        self, times: list[float], observations: list[NDArray]
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Apply the filter to a set of observations.

        Parameters
        ----------
        times: list[float]
            The list of assimilation times.
        observations: NDArray
            The list of observations to assimilate.

        Returns
        -------
        list[NDArray]
            The list of corrected (analysis) states.
        list[NDArray]
            The list of estimated (analysis) covariance matrices.
        """

        if len(times) != len(observations):
            raise IndexError("Observation times and values have different length.")

        estimated_states = []
        estimated_covs = []
        for k, _ in enumerate(times):
            self.forecast(times[k])
            self.analysis(observations[k])

            estimated_states.append(self.analysis_state)
            estimated_covs.append(self.analysis_cov)
        return estimated_states, estimated_covs


class EnsembleFilter(Filter):
    """A class to represent ensemble-based filters.

    Attributes
    ----------
    ensemble_size: int
        The number of ensemble members.
    ensembles_forecast: NDArray
        The matrix of ensemble states.
    """

    def __init__(
        self, model: Model, init_state: NDArray, init_cov: NDArray, ensemble_size: int
    ) -> None:
        super().__init__(model, init_state, init_cov)

        self.ensemble_size = ensemble_size
        self.ensemble_analysis: NDArray = self.generator.multivariate_normal(
            self.init_state, self.init_cov, self.ensemble_size
        ).T
        self.ensemble_forecast: NDArray = np.zeros_like(self.ensemble_analysis)

    def forecast(self, end_time: float, *params: Any) -> None:
        """Forecast all ensembles and compute the forecast state statistics.
        TODO: Ensemble forecast cov may not be needed, skip its calculation? Too expensive in
        memory in general.
        """

        for i in range(self.ensemble_size):
            new_state = self.model.forward(
                self.ensemble_analysis[:, i], self.current_time, end_time, *params
            )
            # TODO: end_time or current_time
            noise = self.generator.multivariate_normal(
                np.zeros_like(self.init_state), self.model.system_cov(end_time)
            )
            self.ensemble_forecast[:, i] = new_state + noise

        self.forecast_state = self.ensemble_forecast.mean(axis=1)
        self.forecast_cov = np.cov(self.ensemble_forecast, ddof=1)
        self.current_time = self.model.current_time

    def analysis(self, observation: NDArray) -> None:
        """Perform the analysis for all ensembles."""

        for i in range(self.ensemble_size):
            state = self.ensemble_forecast[:, i]
            observation_noise = self.generator.multivariate_normal(
                np.zeros_like(observation),
                self.model.observation_cov(self.current_time),
            )
            model_output = self.model.observe(state)
            innovation = observation - model_output - observation_noise
            K = self.compute_gain()

            self.ensemble_analysis[:, i] = state + K @ innovation
            self.analysis_state = self.ensemble_analysis.mean(axis=1)
            self.analysis_cov = np.cov(self.ensemble_analysis, ddof=1)
