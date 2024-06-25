from abc import ABC, abstractmethod
from typing import Any, Self

import copy
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from model import StochasticModel
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
    current_member: StochasticModel
        The current ensemble member.
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
    full_init_state: NDArray
        The augmented initial state.

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
    param_analysis: NDArray
        The estimated parameter vector, after assimilation.
    param_analysis_cov: NDArray
        The covariance matrix of estimated parameters.
    param_forecast: NDArray
        The propagated parameter vector, before assimilation.
    param_forecast_cov: NDArray
        The covariance of the propagated parameters.
    param_values: NDArray
        The current parameter values.
    """

    def __init__(
        self,
        model: StochasticModel,
        init_state: NDArray,
        init_cov: NDArray,
        generator: Generator | None = None,
        **_: Any,
    ) -> None:
        self.current_time: float = 0
        self.init_state: NDArray = init_state
        self.init_cov: NDArray = init_cov
        self.model: StochasticModel = model
        self.current_model: StochasticModel = self.model

        if generator is None:
            generator = default_generator
        self.generator: Generator = generator
        self.n_params: int = len(self.model.uncertain_parameters)
        self.n_states: int = len(self.init_state)
        self.n_outputs: int = len(self.model.observe(self.init_state))
        self.n_aug: int = self.n_states + self.n_params
        self.correct: bool = True

        self.full_init_state: NDArray = np.zeros(self.n_aug)
        self.full_init_cov: NDArray = np.zeros([self.n_aug] * 2)
        self.full_analysis_state: NDArray = np.zeros_like(self.full_init_state)
        self.full_analysis_cov: NDArray = np.zeros_like(self.full_init_cov)
        self.full_forecast_state: NDArray = np.zeros_like(self.full_analysis_state)
        self.full_forecast_cov: NDArray = np.zeros_like(self.full_analysis_cov)
        self.__augment_state__()

    def __augment_state__(self) -> None:
        """Initialize augmented objects."""

        self.full_init_state[: self.n_states] = self.init_state
        self.full_init_state[self.n_states :] = [
            param.init_value for param in self.model.uncertain_parameters
        ]
        self.full_init_cov[: self.n_states, : self.n_states] = self.init_cov
        self.full_init_cov[self.n_states :, self.n_states :] = np.diag(
            [param.uncertainty**2 for param in self.model.uncertain_parameters]
        )
        self.full_analysis_state = self.full_init_state.copy()
        self.full_analysis_cov = self.full_init_cov.copy()

    @property
    def analysis_state(self) -> NDArray:
        """Extract model state.

        Returns
        -------
        NDArray
            The analysis state vector.
        """

        return self.full_analysis_state[: self.n_states]

    @property
    def analysis_cov(self) -> NDArray:
        """Extract the analysis state covariance.

        Returns
        -------
        NDArray
            The analysis covariance matrix.
        """

        return self.full_analysis_cov[: self.n_states, : self.n_states]

    @property
    def param_analysis(self) -> NDArray:
        """Extract the estimated parameters.

        Returns
        -------
        NDArray
            The vector of estimated parameters.
        """

        return self.full_analysis_state[self.n_states :]

    @property
    def param_analysis_cov(self) -> NDArray:
        """Extract the estimated parameters covariance.

        Returns
        -------
        NDArray
            The covariance matrix of the estimated parameters.
        """

        return self.full_analysis_cov[self.n_states :, self.n_states]

    @property
    def param_forecast(self) -> NDArray:
        """Extract the propagated parameters.

        Returns
        -------
        NDArray
            The vector of propagated parameters.
        """

        return self.full_forecast_state[self.n_states :]

    @property
    def param_forecast_cov(self) -> NDArray:
        """Extract the propagated parameters covariance.

        Returns
        -------
        NDArray
            The covariance matrix of the propagated parameters.
        """

        return self.full_forecast_cov[self.n_states :, self.n_states]

    @property
    def forecast_state(self) -> NDArray:
        """Extract model state.

        Returns
        -------
        NDArray
            The forecast state vector.
        """

        return self.full_forecast_state[: self.n_states]

    @property
    def forecast_cov(self) -> NDArray:
        """Extract the analysis state covariance.

        Returns
        -------
        NDArray
            The analysis covariance matrix.
        """

        return self.full_forecast_cov[: self.n_states, : self.n_states]

    @property
    def param_values(self) -> NDArray:
        """The current values of the parameters in the model.

        Returns
        -------
        NDArray
            The array of parameters.
        """

        return np.array(
            [param.current_value for param in self.model.uncertain_parameters]
        )

    def update_parameters(self) -> None:
        """Update the reference model parameters with the latest estimation."""

        self.model.uncertain_parameters = self.param_analysis

    def clone(self, **kwargs: Any) -> Self:
        """Returns a new instance of the filter object with a different model.

        Parameters
        ----------
        **kwargs
            The modified arguments for the new filter.

        Returns
        -------
        Filter
            The newly initialized filter.
        """

        attrs = self.__dict__
        attrs |= kwargs
        return type(self)(**attrs)

    def forecast(self, end_time: float, *params: Any) -> None:
        """Forecast step of the filter for the state. Runs the forward model from the
        current model time to `end_time`.

        Parameters
        ----------
        end_time: float
            The time to forecast the model to.
        stoch_params: bool, optional
            Whether the parameters should be propagated stochastically. Default: False
        *params: Any
            Any additional parameters to run the forecast model.
        """

        # Propagate model with analysis state
        forecast_state = self.model.forward(
            self.analysis_state, self.current_time, end_time, *params
        )

        # Save to filter attributes
        self.full_forecast_state = np.hstack((forecast_state, self.param_values))
        self.current_time = self.model.current_time

    def filter(
        self,
        times: NDArray,
        observations: NDArray,
        spin_up_time: float | None = None,
        cut_off_time: float | None = None,
        run_id: str | None = None,
        show_progress: bool = False,
    ) -> FilteringResults:
        """Apply the filter to a set of observations.

        Parameters
        ----------
        times: NDArray
            The array of assimilation times.
        observations: NDArray
            The observations to assimilate. Shape: `(n_output, analysis_times)`
        TODO: spin_up_time: float | None, optional
            Spin up time for the forward model before assimilation.
        cut_off_time: float | None, optional
            A time to stop the assimilation (emulates forecasting). Default: None
        run_id: str | None, optional
            An ID for the filtering run/experiment performed. Default: None
        show_progress: bool, optional
            Show progress bar for assimilation times. Default: False

        Returns
        -------
        FilteringResults
            The filtering results object.
        """

        # Reset previously computed model states (TODO: maybe fix?)
        self.model.reset_model(self.init_state)

        if cut_off_time is None:
            cut_off_time = times[-1]

        _, analysis_times = observations.shape
        if len(times) != analysis_times:
            raise IndexError("Observation times and values have different length.")

        estimated_states = np.zeros((self.n_aug, analysis_times))
        estimated_covs = np.zeros((*self.full_init_cov.shape, analysis_times))

        # TODO: Run spin-up time

        # Run assimilation
        array = times
        if show_progress:
            array = tqdm(times)
        for k, t in enumerate(array):
            if t > cut_off_time:
                self.correct = False

            self.forecast(t)
            self.analysis(observations[:, k])

            # Update model parameters
            self.update_parameters()

            estimated_states[:, k] = self.full_analysis_state
            estimated_covs[:, :, k] = self.full_analysis_cov

        self.correct = True

        return FilteringResults(
            model=copy.deepcopy(self.model),
            simulation_times=self.model.times,
            assimilation_times=times,
            observations=observations,
            estimated_states=estimated_states[: self.n_states, :],
            estimated_covs=estimated_covs[: self.n_states, : self.n_states, :],
            estimated_params=estimated_states[self.n_states :, :],
            estimated_params_covs=estimated_covs[self.n_states :, self.n_states :, :],
            full_estimated_states=estimated_states,
            full_estimated_covs=estimated_covs,
            run_id=run_id,
        )

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


class EnsembleFilter(Filter):
    """A class to represent ensemble-based filters.

    Attributes
    ----------
    ensemble_size: int
        The number of ensemble members.
    ensembles: list[StochasticModel]
        The list of independent copies of the reference model.
    full_ensembles_forecast: NDArray
        The forecast of the augmented state.
    full_ensemble_analysis: NDArray
        The analysis of the augmented state.

    Properties
    ----------
    forecast_ensemble: bool
        If the ensemble should be propagated (i.e. with noise).
    ensemble_analysis: NDArray
        The analysis state of all ensembles.
    ensemble_params_analysis: NDArray
        The ensemble of estimated parameters.
    ensemble_forecast: NDArray
        The ensemble forecast of states.
    ensemble_params_forecast: NDArray
        The propagated parameters for all ensembles.
    """

    def __init__(
        self,
        model: StochasticModel,
        init_state: NDArray,
        init_cov: NDArray,
        ensemble_size: int,
        generator: Generator | None,
        **_: Any,
    ) -> None:
        super().__init__(model, init_state, init_cov, generator=generator)

        self.ensemble_size: int = ensemble_size
        self.ensembles: list[StochasticModel] = []

        self.full_ensemble_analysis: NDArray = self.generator.multivariate_normal(
            self.full_init_state, self.full_init_cov, self.ensemble_size
        ).T
        self.full_ensemble_forecast: NDArray = np.zeros_like(
            self.full_ensemble_analysis
        )
        self.__init_ensembles__()

    def __init_ensembles__(self) -> None:
        """Initialize each ensemble as an independent model instance."""

        for i in range(self.ensemble_size):
            model = copy.deepcopy(self.model)
            model.name = f"{model.name}_{i}"
            self.ensembles.append(model)

    @property
    def forecast_ensembles(self) -> bool:
        """If the ensembles should be propagated with nosie.

        Returns
        -------
        bool
            The logic value.
        """

        return self.correct

    @property
    def ensemble_analysis(self) -> NDArray:
        """Extract analysis states.

        Returns
        -------
        NDArray
            The matrix of analysis states for all ensembles.
        """

        return self.full_ensemble_analysis[: self.n_states, :]

    @property
    def ensemble_params_analysis(self) -> NDArray:
        """Extract analysis states.

        Returns
        -------
        NDArray
            The matrix of analysis states for all ensembles.
        """

        return self.full_ensemble_analysis[self.n_states :, :]

    @property
    def ensemble_forecast(self) -> NDArray:
        """Extract the forecasted states.

        Returns
        -------
        NDArray
            The matrix of forecast states for all ensembles.
        """

        return self.full_ensemble_forecast[: self.n_states, :]

    @property
    def ensemble_params_forecast(self) -> NDArray:
        """Extract analysis states.

        Returns
        -------
        NDArray
            The matrix of analysis states for all ensembles.
        """

        return self.full_ensemble_forecast[self.n_states :, :]

    def forecast(self, end_time: float, *params: Any) -> None:
        """Forecast all ensembles and compute the forecast state statistics.
        TODO: Ensemble forecast cov may not be needed, skip its calculation? Too expensive in memory in general.
        """

        # Propagate reference model (proxi to estimated state)
        self.current_model = self.model
        self.model.forward(self.analysis_state, self.current_time, end_time, *params)

        # Generate masked noises
        noises = np.multiply(
            self.model.noise_mask,
            self.model.system_error(end_time, n_samples=self.ensemble_size),
        )

        # Propagate each ensemble
        for i, ensemble_model in enumerate(self.ensembles):
            self.current_model = ensemble_model
            new_state = self.model.current_state
            if self.forecast_ensembles:
                new_state = ensemble_model.forward(
                    self.ensemble_analysis[:, i], self.current_time, end_time, *params
                )

                # Add masked noise
                new_state += noises[:, i]

            self.full_ensemble_forecast[: self.n_states, i] = new_state
            self.full_ensemble_forecast[self.n_states :, i] = (
                self.full_ensemble_analysis[self.n_states :, i]
            )

        self.full_forecast_state = self.full_ensemble_forecast.mean(axis=1)
        self.full_forecast_cov = np.cov(self.full_ensemble_forecast, ddof=1)
        self.current_time = self.model.current_time

    def analysis(self, observation: NDArray) -> None:
        """Perform the analysis for all ensembles."""

        for i, ensemble_model in enumerate(self.ensembles):
            state = self.ensemble_forecast[:, i]
            model_output = ensemble_model.observe(state, add_noise=True)
            innovation = observation - model_output
            K = self.compute_gain()
            self.full_ensemble_analysis[:, i] = (
                self.full_ensemble_forecast[:, i] + K @ innovation
            )

        self.full_analysis_state = self.full_ensemble_analysis.mean(axis=1)
        self.full_analysis_cov = np.cov(self.full_ensemble_analysis, ddof=1)

    def update_parameters(self) -> None:
        """Update model parameters for reference model and ensemble."""

        super().update_parameters()
        for i, ensemble_model in enumerate(self.ensembles):
            ensemble_model.uncertain_parameters = self.ensemble_params_analysis[:, i]

    def filter(
        self,
        times: NDArray,
        observations: NDArray,
        store_ensemble: bool = True,
        spin_up_time: float | None = None,
        cut_off_time: float | None = None,
        run_id: str | None = None,
    ) -> FilteringResults:
        """Wrapper for filtering. Adds the ensemble data to the filtering results.

        Parameters
        ----------
        times: NDArray
            The array of assimilation times.
        observations: NDArray
            The observations to assimilate. Shape: `(n_output, analysis_times)`
        TODO: spin_up_time: float | None, optional
            Spin up time for the forward model before assimilation.
        cut_off_time: float | None, optional
            A time to stop the assimilation (emulates forecasting). Default: None
        run_id: str | None, optional
            An ID for the filtering run/experiment performed. Default: None

        Returns
        -------
        FilteringResults
            The filtering results object with the added ensemble.
        """

        results = super().filter(
            times, observations, spin_up_time, cut_off_time, run_id
        )
        if store_ensemble:
            results.ensembles = self.ensembles
        return results
