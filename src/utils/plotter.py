import os
from typing import Any, Callable, Literal, overload

import functools
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import ArrayLike


def setup_save(func) -> Callable:
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # Clear the figure if needed
        if kwargs.get("clear", True):
            plt.cla()
            plt.clf()
            plt.close("all")

        # Setup matplotlib
        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": Plotter.font_size})

        # Plotting function
        response = func(*args, **kwargs)

        # Save fig
        folder = Plotter._folder
        path = kwargs.get("path", None)
        if path is not None:
            if not os.path.exists(folder):
                os.mkdir(folder)
            path = os.path.join(folder, path)
            plt.savefig(path, bbox_inches="tight")
        return response

    return wrapped


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
    TODO: finish list
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

    @classmethod
    def legend(cls, ax: Axes) -> None:
        """It moves the legend outside the plot.

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
    @overload
    def subplots(
        cls, nrows: int, ncols: int, make_3d: Literal[True] = ...
    ) -> tuple[Figure, Axes3D]: ...

    @classmethod
    @overload
    def subplots(
        cls, nrows: int, ncols: int, make_3d: Literal[False]
    ) -> tuple[Figure, Axes]: ...

    @classmethod
    def subplots(
        cls, nrows: int, ncols: int, make_3d: bool = False
    ) -> tuple[Figure, Axes | Axes3D]:
        """Get matplotlib figure and axes. Mostly for safe typing.

        Parameters
        ----------
        nrows: int
            The number of rows of axes.
        ncols: int
            The number of columns of axes.
        make_3d: bool, optional
            If the plots will be 3D or not. Default: False

        Returns
        -------
        Figure
            The figure handle.
        Axes | Axes3D
            The axes handle, depending on the projection as well.
        """

        kwargs = {}
        if make_3d:
            kwargs |= {"subplot_kw": {"projection": "3d"}}
        return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

    @classmethod
    @setup_save
    def plot(
        cls, x: ArrayLike, y: ArrayLike, xlabel: str = "$x$", ylabel: str = "$y$", **_
    ) -> Axes:
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
        matplotlib.figure.Axes
            The axes handle.
        """

        _, ax = cls.subplots(1, 1)
        ax.plot(x, y, *cls.args, **cls.kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cls.grid(ax)
        return ax

    @classmethod
    @setup_save
    def plot3(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        zlabel: str = "$z$",
        **_
    ) -> Axes:
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
        matplotlib.figure.Axes
            The axes handle.
        """

        _, ax = cls.subplots(1, 1, make_3d=True)
        ax.plot(x, y, z, *cls.args, **cls.kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        cls.grid(ax)
        return ax
