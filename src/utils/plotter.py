from numpy.typing import ArrayLike

from typing import Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os


class Plotter:
    """A class to wrap the plotting functions.

    (Static) Attributes
    -------------------
    _folder: str
        The folder to store the output figures.
    args: list[Any]
        The additional arguments for all plots.
    kwargs: dict[str, Any]
        The keyword arguments for all plots.
    """

    _folder: str = os.path.join(os.getcwd(), "figs")
    args: list[Any] = ["k-"]
    kwargs: dict[str, Any] = {"markersize": 3}
    figsize_standard: tuple[int, int] = (8, 5)
    figsize_horizontal: tuple[int, int] = (16, 5)
    figsize_vertical: tuple[int, int] = (8, 10)
    font_size: int = 18
    bands_alpha: float = 0.2
    h_label: str = "$h\ (\mathrm{m})$"
    u_label: str = "$u\ (\mathrm{m})$"
    x_label: str = "$x\ (\mathrm{km})$"
    c_label: str = "$c\ (\mathrm{m/s}))$"
    t_label: str = "Time"

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()
        plt.close("all")

    @classmethod
    def __setup_config__(cls) -> None:
        """It sets up the matplotlib configuration."""

        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": cls.font_size})

    @classmethod
    def legend(cls, ax: Axes) -> None:
        """It moved the legend outside the plot.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower right",
            ncol=3,
        )

        # Bottom outside (may overlap)
        # # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position(
        #     [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        # )

        # # Put a legend below current axis
        # ax.legend(
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.05),
        #     ncol=5,
        # )

        # Right outside
        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    @staticmethod
    def show() -> None:
        """Display the figure."""

        plt.show()

    @classmethod
    def date_axis(cls, ax: Axes) -> None:
        """It formats the x-axis for dates.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=cls.font_size)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

    @classmethod
    def grid(cls, ax: Axes) -> None:
        """It adds a grid to the axes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        ax.grid(alpha=0.4)

    @classmethod
    def save_fig(cls, path: str | None) -> None:
        """It saves the figure to the default folder if needed.

        Parameters
        ----------
        path: str | None, optional
            The path to save the figure to, if needed. Default: None
        """

        if path is not None:
            if not os.path.exists(cls._folder):
                os.mkdir(cls._folder)
            path = os.path.join(cls._folder, path)
            plt.savefig(path, bbox_inches="tight")

    @classmethod
    def plot(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        path: str | None = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        clear: bool = True,
    ) -> tuple[Figure, Axes]:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on horizontal axis.
        y: utils._typing.DataArray
            The data on vertical axis.
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        clear: bool
            Whether to clear the figure or not. Default: True

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        cls.__setup_config__()
        if clear:
            cls.__clear__()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(x, y, *cls.args, **cls.kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cls.grid(ax)
        cls.save_fig(path)
        return fig, ax

    @classmethod
    def plot3(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        path: str | None = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        zlabel: str = "$z$",
        clear: bool = True,
    ) -> tuple[Figure, Axes]:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: ArrayLike
            The data on the first axis.
        y: ArrayLike
            The data on second axis.
        z: ArrayLike
            The data on the third axis.
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        zlabel: str, optional
            The label on the third axis. Default: "$z$"
        clear: bool
            Whether to clear the figure or not. Default: True

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        cls.__setup_config__()
        if clear:
            cls.__clear__()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(x, y, *cls.args, **cls.kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cls.grid(ax)
        cls.save_fig(path)
        return fig, ax
