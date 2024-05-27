from abc import ABC, abstractmethod
from typing import Any

import copy
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from model import Model
from utils.results import FilteringResults
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
    generator: Generator
        The random number generator.
    correct: bool
        If assimilation should be performed. This set the gain to 0 for the analysis.
    n_states: int
        The number of states.
    n_outputs: int
        The number of outputs.
    full_init_state: NDArray
        The augmented initial state.
    full_init_cov: NDArray
        The covariance of the augmented state.
    full_analysis_state: NDArray
        The augmented analysis state.
    full_analysis_cov: NDArray
        The covariance of the augmented analysis state.
    full_forecast_state: NDArray
        The augmented forecast state.
    full_forecast_cov: NDArray
        The covariance of the augmented forecast state.

    Properties
    ----------
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
    correct: bool
        If assimilation should be performed. This set the gain to 0 for the analysis.
    n_states: int
        The number of states.
    n_outputs: int
        The number of outputs.
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
        self.correct: bool = True
        self.n_states: int = len(self.init_state)
        self.n_outputs: int = len(self.model.observe(self.init_state))

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
        self,
        times: NDArray,
        observations: NDArray,
        cut_off_time: float | None = None,
        run_id: str | None = None,
    ) -> FilteringResults:
        """Apply the filter to a set of observations.

        Parameters
        ----------
        times: NDArray
            The array of assimilation times.
        observations: NDArray
            The observations to assimilate. Shape: `(n_output, analysis_times)`
        cut_off_time: float | None, optional
            A time to stop the assimilation (emulates forecasting). Default: None
        run_id: str | None, optional
            An ID for the filtering run/experiment performed. Default: None

        Returns
        -------
        FilteringResults
            The filtering results object.
        """

        if cut_off_time is None:
            cut_off_time = times[-1]

        _, analysis_times = observations.shape
        if len(times) != analysis_times:
            raise IndexError("Observation times and values have different length.")

        estimated_states = np.zeros((self.n_states, analysis_times))
        estimated_covs = np.zeros((*self.init_cov.shape, analysis_times))
        for k, t in enumerate(tqdm(times)):
            if t > cut_off_time:
                self.correct = False

            self.forecast(times[k])
            self.analysis(observations[:, k])

            estimated_states[:, k] = self.analysis_state
            estimated_covs[:, :, k] = self.analysis_cov

        self.correct = True

        return FilteringResults(
            copy.deepcopy(self.model),
            times,
            observations,
            estimated_states,
            estimated_covs,
            self.model.times,
            run_id=run_id,
        )


class EnsembleFilter(Filter):
    """A class to represent ensemble-based filters.

    Attributes
    ----------
    ensemble_size: int
        The number of ensemble members.
    ensembles: list[Model]
        The list of independent copies of the reference model.
    ensembles_forecast: NDArray
        The matrix of ensemble states.

    Properties
    ----------
    forecast_ensemble: bool
        If the ensemble should be propagated (i.e. with noise).
    """

    def __init__(
        self,
        model: Model,
        init_state: NDArray,
        init_cov: NDArray,
        ensemble_size: int,
        generator: Generator | None,
    ) -> None:
        super().__init__(model, init_state, init_cov, generator=generator)

        self.ensemble_size: int = ensemble_size
        self.ensembles: list[Model] = []
        self.ensemble_analysis: NDArray = self.generator.multivariate_normal(
            self.init_state, self.init_cov, self.ensemble_size
        ).T
        self.ensemble_forecast: NDArray = np.zeros_like(self.ensemble_analysis)
        self.__init_ensembles__()

    @property
    def forecast_ensembles(self) -> bool:
        """If the ensembles should be propagated with nosie.

        Returns
        -------
        bool
            The logic value.
        """

        return self.correct

    def __init_ensembles__(self) -> None:
        """Initialize each ensemble as an independent model instance."""

        for _ in range(self.ensemble_size):
            self.ensembles.append(copy.deepcopy(self.model))

    def forecast(self, end_time: float, *params: Any) -> None:
        """Forecast all ensembles and compute the forecast state statistics.
        TODO: Ensemble forecast cov may not be needed, skip its calculation? Too expensive in
        memory in general.
        """

        # Propagate reference model (proxi to estimated state)
        self.model.forward(self.analysis_state, self.current_time, end_time, *params)

        # Propagate each ensemble
        noises = self.generator.multivariate_normal(
            np.zeros_like(self.init_state),
            self.model.system_cov(end_time),
            self.ensemble_size,
        )
        for i, ensemble_model in enumerate(self.ensembles):
            new_state = self.model.current_state
            if self.forecast_ensembles:
                new_state = ensemble_model.forward(
                    self.ensemble_analysis[:, i], self.current_time, end_time, *params
                )
                new_state += noises[i, :]
            self.ensemble_forecast[:, i] = new_state

        self.forecast_state = self.ensemble_forecast.mean(axis=1)
        self.forecast_cov = np.cov(self.ensemble_forecast, ddof=1)
        self.current_time = self.model.current_time

    def analysis(self, observation: NDArray) -> None:
        """Perform the analysis for all ensembles."""

        for i, ensemble_model in enumerate(self.ensembles):
            state = self.ensemble_forecast[:, i]
            model_output = ensemble_model.observe(state, add_noise=True)
            innovation = observation - model_output
            K = self.compute_gain()

            self.ensemble_analysis[:, i] = state + K @ innovation
            self.analysis_state = self.ensemble_analysis.mean(axis=1)
            self.analysis_cov = np.cov(self.ensemble_analysis, ddof=1)
