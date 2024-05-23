from model import StochasticModel
from utils.plotter import Plotter

from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.pyplot import axhline
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d
from typing import Any


import numpy as np


@dataclass
class FilteringResults:
    """A class to handle and process filtering results.

    Attributes
    ----------
    model: StochasticModel
        The forward model.
    simulation_times: NDArray
        The forward model simulation time (in simulation time steps).
    assimilation_times: NDArray
        The array of times where observations were assimilated.
    observations: NDArray
        The observations.
    estimated_states: NDArray
        The estimated (filtered) states at the assimilation times.
    estimated_covs: NDArray
        The estimated (filtered) covariances at the assimilation times.
    estimated_params: NDArray
        The estimated parameters (if available) at each assimilation time.
    estimated_params_covs: NDArray
        The estimated covariance of the parameters at each assimilation time.
    full_estimated_states: NDArray
        The augmented estimated states at each assimilation time.
    full_estimated_covs: NDArray
        The covariance of the augmented state at each assimilation time.
    run_id: str
        The experiment ID. Default: None

    Properties
    ----------
    ensembles: list[StochasticModel]
        The list of ensemble models if available.
    estimated_ensemble: NDArray
        The analysis states for all ensembles at each assimilation time.
    true_times: NDArray
        Time instants of true data.
    true_states: NDArray
        The true states (if available).
    innovations: NDArray
        The array of innovations (observation - forecast)
    rmses: NDArray
        The array of advancing RMSEs.
    """

    model: StochasticModel
    simulation_times: NDArray
    assimilation_times: NDArray
    observations: NDArray
    estimated_states: NDArray
    estimated_covs: NDArray
    estimated_params: NDArray
    estimated_params_covs: NDArray
    full_estimated_states: NDArray
    full_estimated_covs: NDArray
    run_id: str | None = None

    _estimated_ensemble: NDArray | None = None
    _ensembles: list[StochasticModel] | None = None
    _estimated_observations: NDArray | None = None
    _true_times: NDArray | None = None
    _true_states: NDArray | None = None
    _innovations: NDArray | None = None
    _rmses: NDArray | None = None

    @property
    def estimated_ensemble(self) -> NDArray:
        """The estimated states for all ensemble members at each assimilation time.

        Returns
        -------
        NDArray
            The estimated ensembles.
        """

        if self._estimated_ensemble is None:
            n_states, n_times = self.estimated_states.shape
            n_ensembles = len(self.ensembles)
            self._estimated_ensemble = np.zeros((n_states, n_ensembles, n_times))
            for i, ensemble_model in enumerate(self.ensembles):
                self._estimated_ensemble[:, i, :] = ensemble_model.states
        return self._estimated_ensemble

    @property
    def ensembles(self) -> list[StochasticModel]:
        """The list of models in the ensemble.

        Returns
        -------
        list[StochasticModel]
            The list of stochastic models.

        Raises
        ------
        ValueError
            When no ensemble has been set.
        """

        if self._ensembles is None:
            raise ValueError("No ensemble available.")
        return self._ensembles

    @ensembles.setter
    def ensembles(self, models: list[StochasticModel]) -> None:
        """Store the models in the ensemble.

        Parameters
        ----------
        models: list[StochasticModel]
            The list of stochastic models to store.
        """

        self._ensembles = models

    @property
    def true_times(self) -> NDArray:
        """It stores the true times if available.

        Returns
        -------
        NDArray
            The matrix of true times through time.

        Raises
        ------
        ValueError
            When no ground truth is available.
        """

        if self._true_times is None:
            raise ValueError("No true times available.")
        return self._true_times

    @true_times.setter
    def true_times(self, array: NDArray) -> None:
        """Set the true times when available.

        Parameters
        ----------
        array: NDArray
            The array of true times.
        """

        self._true_times = array

    @property
    def true_states(self) -> NDArray:
        """It stores the true states if available.

        Returns
        -------
        NDArray
            The matrix of true states through time.

        Raises
        ------
        ValueError
            When no ground truth is available.
        """

        if self._true_states is None:
            raise ValueError("No true states available.")
        return self._true_states

    @true_states.setter
    def true_states(self, array: NDArray) -> None:
        """Set the true states if available.

        Parameters
        ----------
        array: NDArray
            The array of true states.
        """

        self._true_states = array

    @property
    def innovations(self) -> NDArray:
        """It calculates the innvoations if required.

        Returns
        -------
        NDArray
            The array of innovations at assimilation times.
        """

        if self._innovations is None:
            self._innovations = self.observations - self.estimated_observations
        return self._innovations

    @property
    def estimated_observations(self) -> NDArray:
        """It calculates the innovations if required.

        Returns
        -------
        NDArray
            The array of innovations at assimilation times.
        """

        if self._estimated_observations is None:
            self._estimated_observations = self.model.observe(self.estimated_states)
        return self._estimated_observations

    @property
    def rmses(self) -> NDArray:
        """It calculates the RMSE through time.

        Returns
        -------
        NDArray
            The array of RMSEs at assimilation times.
        """

        if self._rmses is None:
            self._rmses = np.zeros_like(self.estimated_observations)
            for k in range(1, len(self.assimilation_times) + 1):
                mse = (
                    (self.observations[:, :k] - self.estimated_observations[:, :k]) ** 2
                ).mean(axis=1)
                self._rmses[:, k] = np.sqrt(mse)
        return self._rmses

    @property
    def rmse(self) -> NDArray:
        """It returns the RMSE at final time for each state.

        Returns
        -------
        NDArray
            The array of RMSE for each state.
        """

        return self.rmses[:, -1]

    def get_label(self, label: str) -> str:
        """It returns the label with the run ID.

        Parameters
        ----------
        label: str
            The line label.

        Returns
        -------
        str
            The line label with added run ID.
        """

        if self.run_id is not None:
            label += f" ({self.run_id})"
        return label

    def plot_innovations(
        self,
        state_idx: int,
        window: int | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Show the innovation process after assimilation.

        Parameters
        ----------
        state_idx: int
            The state index to present the results from.
        window: int | None, optional
            The moving average window if required. Default: None
        path: str | None, optional
            The filename to output the figure. Default: None
        **kwargs: Any
            Extra keyword arguments for the plotting function.

        Returns
        -------
        Axes
            The axis handle.
        """

        kwargs |= {"figsize": "horizontal"}
        ax = kwargs.pop("ax", None)

        innovations = self.innovations[state_idx, :]
        if window is not None:
            averaged = uniform_filter1d(innovations, size=window)
            ax = Plotter.plot(
                self.assimilation_times,
                averaged,
                "r",
                label=self.get_label(f"Averaged w={window}"),
                ax=ax,
                **kwargs,
            )
        ax = Plotter.plot(
            self.assimilation_times,
            innovations,
            "k",
            alpha=0.7,
            zorder=-1,
            xlabel="$t$",
            ylabel="",
            label=self.get_label("Innovations"),
            path=path,
            ax=ax,
            **kwargs,
        )
        return ax

    def plot_filtering(
        self,
        state_idx: int,
        path: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """It shows the result from the filtering.
        TODO: This function works assuming H = I in Lorenz96. Fix?

        Parameters
        ----------
        state_idx: int
            The state index to present the results from.
        path: str | None, optional
            The filename to output the figure. Default: None
        **kwargs: Any
            Extra keyword arguments for the plotting function.

        Returns
        -------
        Axes
            The axis handle.
        """

        kwargs |= {"figsize": "horizontal"}
        ax = kwargs.pop("ax", None)

        ax = Plotter.plot(
            self.simulation_times,
            self.model.states[state_idx, :],
            "k",
            label="Assimilation",
            alpha=0.2,
            ax=ax,
            **kwargs,
        )
        if self.true_times is not None and self.true_states is not None:
            ax = Plotter.plot(
                self.true_times,
                self.true_states[state_idx, :],
                "b",
                alpha=0.5,
                label="Truth",
                ax=ax,
                **kwargs,
            )
        Plotter.plot(
            self.assimilation_times,
            self.observations[state_idx, :],
            "kx",
            label="Observations",
            ax=ax,
            **kwargs,
        )
        Plotter.plot(
            self.assimilation_times,
            self.full_estimated_states[state_idx, :],
            "r*",
            label="Estimates",
            ylabel=f"$x_{{{state_idx}}}$",
            xlabel="$t$",
            path=path,
            ax=ax,
            **kwargs,
        )
        return ax

    def plot_params(
        self,
        param_indices: list[int],
        ref_params: list[float] | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot the estimated parameters and optionally show the reference value.

        Parameters
        ----------
        param_indices: list[int]
            The list of indices of parameters to show on the same plot.
        """

        ax = kwargs.pop("ax", None)
        colors = ["r", "k", "b", "g"]
        n_states = self.estimated_states.shape[0]
        for i in param_indices:
            param = self.model.parameters[i]
            ax = Plotter.plot(
                self.assimilation_times,
                self.full_estimated_states[n_states + i, :],
                f"{colors[i]}-o",
                markersize=3,
                drawstyle="steps-post",
                xlabel=Plotter.t_label,
                ylabel="",
                label=param.name,
                ax=ax,
            )
            if ref_params is not None:
                ax = Plotter.hline(ref_params[i], color=colors[i], path=path, ax=ax)
        return ax
