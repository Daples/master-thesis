from utils.plotter import Plotter, setup_save
from utils.results import FilteringResults
from utils import tud_blue, purple

from dataclasses import dataclass
from matplotlib.axes import Axes
from scipy.ndimage import uniform_filter1d
from typing import Any

import numpy as np


@dataclass
class Comparator:
    """"""

    results: list[FilteringResults]
    labels: list[str]
    colors: tuple[Any, ...] = (tud_blue, purple)
    figsize: str = "standard"
    inn_offset: float = 0.07

    @setup_save
    def compare_filtering(
        self,
        state_idx: int,
        plot_ensemble: bool,
        plot_bands: bool,
        only_state: bool = False,
        path: str | None = None,
        ax: Axes | None = None,
        figsize: str | None = None,
        legend: bool = True,
    ) -> Axes:
        """"""

        if figsize is None:
            figsize = self.figsize
        if ax is None:
            _, ax = Plotter.subplots(1, 1, figsize)

        cut_time = self.results[0].cut_off_time
        if cut_time is not None:
            ax = Plotter.vline(cut_time, ax=ax, color=Plotter.cut_color)

        # Plot truth
        if not only_state:
            for results in self.results:
                if results.true_times is not None and results.true_states is not None:
                    ax = Plotter.plot(
                        results.true_times,
                        results.true_states[state_idx, :],
                        "k--",
                        alpha=Plotter.truth_alpha,
                        label="Truth",
                        ax=ax,
                        path=path,
                    )
                    break

        for i, results in enumerate(self.results):
            color = self.colors[i]
            label = self.labels[i] if legend else None

            # Plot ensembles
            if plot_ensemble:
                for ensemble in results.ensembles:
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
                times = results.simulation_times
                ensembles_states = np.array(
                    [m.states[state_idx] for m in results.ensembles]
                )
                estimations = results.model.states[state_idx]
                stds = ensembles_states.std(ddof=1, axis=0)
                ax = Plotter.bands(
                    times,
                    estimations,
                    stds,
                    ax=ax,
                    color=color,
                )

            # Plot filter results
            ax = Plotter.plot(
                results.simulation_times,
                results.model.states[state_idx, :],
                color,
                label=label,
                ax=ax,
            )

        # Plot observations
        if not only_state:
            Plotter.plot(
                results.assimilation_times,
                results.observations[state_idx, :],
                "kx",
                markersize=4,
                alpha=1,
                label="Observations",
                ylabel=results.var_names[state_idx],
                xlabel=Plotter.t_label,
                ax=ax,
                # zorder=-1,
            )

        return ax

    def compare_innovations(
        self,
        state_idx: int,
        alpha: float = 1,
        window: int | None = None,
        path: str | None = None,
        ax: Axes | None = None,
        figsize: str | None = None,
        legend: bool = True,
    ) -> Axes:
        """"""

        if figsize is None:
            figsize = self.figsize
        if ax is None:
            _, ax = Plotter.subplots(1, 1, figsize)

        cut_time = self.results[0].cut_off_time
        if cut_time is not None:
            ax = Plotter.vline(cut_time, ax=ax, color=Plotter.cut_color)

        for i, results in enumerate(self.results):
            color = self.colors[i]
            label = self.labels[i] if legend else None
            innovations = results.innovations[state_idx, :]
            shift = self.inn_offset * i * 0

            if window is not None:
                averaged = uniform_filter1d(innovations, size=window)
                ax = Plotter.plot(
                    results.assimilation_times,
                    averaged,
                    "--",
                    color=color,
                    alpha=alpha / 3,
                    zorder=np.inf,
                    ax=ax,
                )
            ax = Plotter.stem(
                results.assimilation_times + shift,
                innovations,
                cut_index=self.results[0].cut_off_index,
                color=color,
                alpha=alpha,
                xlabel=Plotter.t_label,
                ylabel="",
                # ylabel="Innovations",
                label=label,
                path=path,
                ax=ax,
            )
        return ax

    def compare_av_innovations(
        self,
        path: str | None = None,
        ax: Axes | None = None,
        figsize: str | None = None,
        legend: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Compare histograms of avarage innovations for all states."""

        if figsize is None:
            figsize = self.figsize
        if ax is None:
            _, ax = Plotter.subplots(1, 1, figsize)

        for i, result in enumerate(self.results):
            label = self.labels[i] if legend else None
            color = self.colors[i]
            averages = result.innovations.mean(axis=1)
            ax = Plotter.hist(
                averages,
                bins=None,
                xlabel="Average innovations per state",
                ylabel="Frequency",
                label=label,
                ax=ax,
                path=path,
                color=color,
                **kwargs,
            )
            ax.axvline(x=0, color="k", alpha=0.8, linestyle=":", zorder=-1)
        return ax

    def compare_params(
        self,
        param_indices: list[int],
        ref_params: list[float] | None = None,
        path: str | None = None,
        ax: Axes | None = None,
        figsize: str | None = None,
        legend: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Compared estimated parameters."""

        if figsize is None:
            figsize = self.figsize
        if ax is None:
            _, ax = Plotter.subplots(1, 1, figsize)

        for j, results in enumerate(self.results):
            label = self.labels[j]
            color = self.colors[j]
            if results.cut_off_time is not None:
                ax = Plotter.vline(results.cut_off_time, ax=ax, color=Plotter.cut_color)

            n_states = results.model.n_states
            for i in param_indices:
                param = results.model.parameters[i]
                x = np.hstack((0, results.assimilation_times))
                y = np.hstack(
                    (param.init_value, results.full_estimated_states[n_states + i, :])
                )

                ax = Plotter.plot(
                    x,
                    y,
                    "-o",
                    color=color,
                    markersize=3,
                    drawstyle="steps-post",
                    xlabel=Plotter.t_label,
                    ylabel="",
                    label=f"{param.name} - {label}",
                    ax=ax,
                    **kwargs,
                )
                s = np.sqrt(results.estimated_params_covs[i, i, :])
                s = np.hstack((param.uncertainty, s))
                ax = Plotter.bands(x, y, s, ax=ax, color=color)
                if ref_params is not None:
                    ax = Plotter.hline(ref_params[i], color=color, path=path, ax=ax)
        return ax
