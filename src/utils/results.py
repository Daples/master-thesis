from model import StochasticModel
from utils.plotter import Plotter
from utils import state_str, bias_str

from dataclasses import dataclass
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d
from typing import Any

import matplotlib.pyplot as plt
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
    is_bias_aware: bool
    cut_off_time: float | None
    cut_off_index: int | None
    run_id: str | None = None
    true_times: NDArray | None = None
    true_states: NDArray | None = None
    full_ensemble_states: NDArray | None = None

    _var_names: list[str] | None = None
    _estimated_ensemble: NDArray | None = None
    _ensembles: list[StochasticModel] | None = None
    _estimated_observations: NDArray | None = None
    _innovations: NDArray | None = None
    _rmses: NDArray | None = None

    figsize: str = "standard"

    @property
    def var_names(self) -> list[str]:
        """Add variable names."""

        if self._var_names is None:
            nx = self.model.n_states
            np = len(self.model.uncertain_parameters)

            self._var_names = [""] * (nx + np)
            self._var_names[:nx] = [f"${{{state_str}}}_{{{i}}}$" for i in range(nx)]
            self._var_names[nx:] = [
                param.name for param in self.model.uncertain_parameters
            ]

            if self.is_bias_aware:
                nb = int(nx / 2)
                for i in range(nb):
                    self._var_names[nb + i] = f"${{{bias_str}}}_{{{i}}}$"
        return self._var_names

    @property
    def estimated_ensemble(self) -> NDArray:
        """The estimated states for all ensemble members at each assimilation time.

        Returns
        -------
        NDArray
            The estimated ensembles.
        """

        if self._estimated_ensemble is None:
            n_states, n_times = self.model.states.shape
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
            self._estimated_observations = np.zeros_like(self.observations)
            for i, _ in enumerate(self.assimilation_times):
                self._estimated_observations[:, i] = self.model.observe(
                    self.estimated_states[:, i]
                )
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
        alpha: float = 1,
        shift: float = 0.0,
        color: str | None = None,
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

        kwargs |= {"figsize": "standard"}
        # kwargs |= {"figsize": self.figsize}
        ax = kwargs.pop("ax", None)
        if color is None:
            color = Plotter.color

        if self.cut_off_time is not None:
            ax = Plotter.vline(
                self.cut_off_time, ax=ax, color=Plotter.cut_color, **kwargs
            )
        ax = Plotter.hline(0, color="k", ax=ax)
        innovations = self.innovations[state_idx, :]
        if window is not None:
            averaged = uniform_filter1d(innovations, size=window)
            ax = Plotter.plot(
                self.assimilation_times,
                averaged,
                "--",
                color=color,
                alpha=alpha / 3,
                zorder=np.inf,
                label=self.get_label(f"Averaged w={window}"),
                ax=ax,
                **kwargs,
            )
        ax = Plotter.stem(
            self.assimilation_times + shift,
            innovations,
            cut_index=self.cut_off_index,
            color=color,
            alpha=alpha,
            xlabel=Plotter.t_label,
            ylabel="",
            label=self.get_label("Innovations"),
            path=path,
            ax=ax,
        )
        return ax

    def plot_filtering(
        self,
        state_idx: int,
        plot_ensemble: bool,
        plot_bands: bool,
        path: str | None = None,
        only_state: bool = False,
        color: str | None = None,
        legend: bool = True,
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

        if color is None:
            color = Plotter.color
        ax = kwargs.pop("ax", None)
        if ax is None:
            Plotter.setup()
            _, ax = Plotter.subplots(1, 1, figsize=self.figsize)

        if self.cut_off_time is not None:
            ax = Plotter.vline(self.cut_off_time, ax=ax, color=Plotter.cut_color)

        add_label = lambda s: s if legend else None
        if plot_ensemble:
            for ensemble in self.ensembles:
                ax = Plotter.plot(
                    ensemble.times,
                    ensemble.states[state_idx, :],
                    color,
                    ax=ax,
                    alpha=Plotter.ensemble_alpha,
                    zorder=-1,
                    linewidth=Plotter.ensemble_width,
                )

        # Plot ensemble spread
        if plot_bands:
            times = self.simulation_times
            ensembles_states = np.array([m.states[state_idx] for m in self.ensembles])
            estimations = self.model.states[state_idx]
            stds = ensembles_states.std(ddof=1, axis=0)
            ax = Plotter.bands(
                times,
                estimations,
                stds,
                ax=ax,
                color=color,
            )
        if self.true_times is not None and self.true_states is not None:
            ax = Plotter.plot(
                self.true_times,
                self.true_states[state_idx, :],
                "k--",
                alpha=Plotter.truth_alpha,
                label=add_label("Truth"),
                ax=ax,
                **kwargs,
                zorder=2,
            )
        if not only_state:
            Plotter.plot(
                self.assimilation_times,
                self.observations[state_idx, :],
                "kx",
                markersize=4,
                alpha=1,
                label=add_label("Observations"),
                ax=ax,
                zorder=4,
                **kwargs,
            )
        ax = Plotter.plot(
            self.simulation_times,
            self.model.states[state_idx, :],
            color,
            label=add_label("Assimilation"),
            ylabel=self.var_names[state_idx],
            xlabel="$t$",
            alpha=1,
            ax=ax,
            path=path,
            zorder=3,
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

        kwargs |= {"figsize": self.figsize}
        ax = kwargs.pop("ax", None)
        if self.cut_off_time is not None:
            ax = Plotter.vline(self.cut_off_time, ax=ax, color=Plotter.cut_color)

        n_states = self.model.n_states
        for j, i in enumerate(param_indices):
            param = self.model.parameters[i]
            x = np.hstack((0, self.assimilation_times))
            y = np.hstack(
                (param.init_value, self.full_estimated_states[n_states + i, :])
            )

            ax = Plotter.plot(
                x,
                y,
                "-o",
                color=Plotter.colors[i],
                markersize=3,
                drawstyle="steps-post",
                xlabel=Plotter.t_label,
                ylabel="",
                label=param.name,
                ax=ax,
                **kwargs,
            )
            s = np.sqrt(self.estimated_params_covs[i, i, :])
            s = np.hstack((param.uncertainty, s))
            ax = Plotter.bands(x, y, s, ax=ax, color=Plotter.colors[i])
            if ref_params is not None:
                ax = Plotter.hline(
                    ref_params[j], color=Plotter.colors[i], path=path, ax=ax
                )
        return ax

    def plot_av_innovation(self, path: str | None = None, **kwargs: Any) -> Axes:
        """"""

        averages = []
        for i in range(self.model.n_states):
            averages.append(self.innovations[i, :].mean())
        return Plotter.hist(
            averages,
            bins=int(self.model.n_states / 2),
            xlabel="Average innovations per state",
            ylabel="Frequency",
            path=path,
            **kwargs,
        )
