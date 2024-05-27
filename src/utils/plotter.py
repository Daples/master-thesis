import os
from typing import Any, Callable, Literal, overload, cast
from utils._typing import P, T

import functools
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import ArrayLike
from numpy import mean


def setup_save(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator for common functionality around plots. It sets up matplotlib with class
    parameters, saves the figure in the specified location and displays it if needed.

    Parameters
    ----------
    func: (P, T) -> (P, T)
        The plotting function to wrap.

    Returns
    -------
    (P, T) -> (P, T)
        The plotting function with added functionality.
    """

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        # Setup matplotlib
        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": Plotter.font_size})

        # Plotting function
        path = cast(str | None, kwargs.pop("path", None))
        response = func(*args, **kwargs)

        # Save fig
        folder = Plotter._folder
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
    args: tuple[Any] = ("k-",)
    kwargs: dict[str, Any] = {"markersize": 3}
    standard: str = "standard"
    figsize_standard: tuple[int, int] = (8, 5)
    figsize_horizontal: tuple[int, int] = (16, 5)
    figsize_vertical: tuple[int, int] = (8, 10)
    font_size: int = 18
    bands_alpha: float = 0.2
    h_label: str = r"$h\ (\mathrm{m})$"
    u_label: str = r"$u\ (\mathrm{m})$"
    x_label: str = r"$x\ (\mathrm{km})$"
    c_label: str = r"$c\ (\mathrm{m/s}))$"
    t_label: str = "Time"

    @classmethod
    def check_args(
        cls, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Get default class arguments if needed.

        Parameters
        ----------
        args: tuple[Any, ...]
            The input positional arguments.
        kwargs: dict[str, Any]
            The input keyword arguments.

        Returns
        -------
        tuple[Any, ...]
            The output positional arguments.
        dict[str, Any]
            The output keyword arguments.
        """

        if not args:
            args = cls.args
        if not kwargs:
            kwargs = cls.kwargs
        return args, kwargs

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
            ncol=5,
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
    def __clear__(cls) -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

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
        cls, nrows: int, ncols: int, figsize: str, make_3d: Literal[True] = ...
    ) -> tuple[Figure, Axes3D]: ...

    @classmethod
    @overload
    def subplots(
        cls, nrows: int, ncols: int, figsize: str, make_3d: Literal[False]
    ) -> tuple[Figure, Axes]: ...

    @classmethod
    def subplots(
        cls, nrows: int, ncols: int, figsize: str, make_3d: bool = False
    ) -> tuple[Figure, Axes | Axes3D]:
        """Get matplotlib figure and axes. Mostly for safe typing.

        Parameters
        ----------
        nrows: int
            The number of rows of axes.
        ncols: int
            The number of columns of axes.
        figsize: str
            The figure size to use.
        make_3d: bool, optional
            If the plots will be 3D or not. Default: False

        Returns
        -------
        Figure
            The figure handle.
        Axes | Axes3D
            The axes handle, depending on the projection as well.
        """

        aux = "figsize"
        size = getattr(cls, aux + "_" + figsize)
        kwargs = {aux: size}
        if make_3d:
            kwargs |= {"subplot_kw": {"projection": "3d"}}
        return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

    @classmethod
    @setup_save
    def plot(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        *args: Any,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        ax: Axes | None = None,
        figsize: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on horizontal axis.
        y: utils._typing.DataArray
            The data on vertical axis.
        *args: Any
            Any additional arguments for the plot.
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        ax: Axes, optional
            The axis to plot on. Default: None
        figsize: str | None, optional
            The figure size. Default: None
        **kwargs: Any
            The additional keyword arguments for the plot.

        Returns
        -------
        matplotlib.figure.Axes
            The axes handle.
        """

        if figsize is None:
            figsize = cls.standard
        if ax is None:
            _, ax = cls.subplots(1, 1, figsize)
        args, kwargs = cls.check_args(args, kwargs)
        ax.plot(x, y, *args, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cls.grid(ax)
        if "label" in kwargs:
            cls.legend(ax)
        return ax

    @classmethod
    @setup_save
    def stem(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        ax: Axes | None = None,
        figsize: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on horizontal axis.
        y: utils._typing.DataArray
            The data on vertical axis.
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        ax: Axes, optional
            The axis to plot on. Default: None
        figsize: str | None, optional
            The figure size. Default: None
        *args: Any
            Any additional arguments for the plot.
        **kwargs: Any
            The additional keyword arguments for the plot.

        Returns
        -------
        matplotlib.figure.Axes
            The axes handle.
        """

        if figsize is None:
            figsize = cls.standard
        if ax is None:
            _, ax = cls.subplots(1, 1, figsize)
        args, kwargs = cls.check_args(args, kwargs)
        ax.stem(x, y, *args, bottom=mean(y), **kwargs)  # type: ignore
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cls.grid(ax)
        if "label" in kwargs:
            cls.legend(ax)
        return ax

    @classmethod
    @setup_save
    def mplot(
        cls,
        x,
        ys,
        labels: list[str] | None = None,
        path: str | None = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        clear: bool = True,
        is_ax_date: bool = False,
        linewidth: float = 1,
    ) -> tuple[Figure, Axes]:
        """It plots several lines with standard formatting.

        Parameters
        ----------
        x
            The data on horizontal axis.
        ys
            The data sets on vertical axis. Shape: (data, samples)
        labels: list[str] | None
            The labels for each data sample (legend). Default: None
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        clear: bool
            Whether to clear the figure or not. Default: True
        is_ax_date: bool
            Whether the horizontal axis should be date formatted. Default: False
        linewidth: float
            The line width for the plots. Default: 1

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cls.figsize_standard)
        kwargs = {"linewidth": linewidth}
        if labels is not None:
            kwargs = {"label": labels}

        for i in range(ys.shape[1]):
            ax.plot(x, ys[:, i], **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if is_ax_date:
            cls.date_axis(ax)
        cls.grid(ax)
        return ax

    @classmethod
    @setup_save
    def plot3(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        *args: Any,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        zlabel: str = "$z$",
        ax: Axes3D | None = None,
        figsize: str | None = None,
        **kwargs: Any,
    ) -> Axes3D:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: ArrayLike
            The data on the first axis.
        y: ArrayLike
            The data on second axis.
        z: ArrayLike
            The data on the third axis.
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        zlabel: str, optional
            The label on the third axis. Default: "$z$"
        ax: Axes, optional
            The axis to plot on. Default: None
        figsize: str | None, optional
            The figure size. Default: None
        *args: Any
            Any additional arguments for the plot.
        **kwargs: Any
            The additional keyword arguments for the plot.

        Returns
        -------
        matplotlib.figure.Axes
            The axes handle.
        """

        if figsize is None:
            figsize = cls.standard
        if ax is None:
            _, ax = cls.subplots(1, 1, make_3d=True, figsize=figsize)
        args, kwargs = cls.check_args(args, kwargs)
        ax.plot(x, y, z, *args, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        cls.grid(ax)
        return ax

    @classmethod
    @setup_save
    def hline(
        cls,
        y: float,
        ax: Axes | None = None,
        figsize: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """It plots a horizontal axis line at the specified height.

        Parameters
        ----------
        y: utils._typing.DataArray
            The height of the axline.
        ax: Axes, optional
            The axis to plot on. Default: None
        figsize: str | None, optional
            The figure size. Default: None
        *args: Any
            Any additional arguments for the plot.
        **kwargs: Any
            The additional keyword arguments for the plot.

        Returns
        -------
        matplotlib.figure.Axes
            The axes handle.
        """

        if figsize is None:
            figsize = cls.standard
        if ax is None:
            _, ax = cls.subplots(1, 1, figsize)
        args, kwargs = cls.check_args(args, kwargs)

        ax.axhline(y=y, **kwargs, alpha=0.7, linestyle=":")
        cls.grid(ax)
        if "label" in kwargs:
            cls.legend(ax)
        return ax

    @classmethod
    @setup_save
    def hist(
        cls,
        data: ArrayLike,
        bins: int | None,
        normalize: bool = False,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        ax: Axes | None = None,
        figsize: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """It plots a histogram with standard formatting.

        Parameters
        ----------
        TODO:
        data: numpy.ndarray | list
            The data to create the histogram of.
        bins: int | None, optional
            The number of bins to use. Default: None
        path: str | None, optional
            The path to save the figure. Default: None
        xlabel: str | None, optional
            The label for the horizontal axis. Default: None
        ylabel: str | None, optional
            The label for the vertical axis. Default: None
        normalize: bool, optional
            If the histogram should be normalized (density). Default: False
        """

        if figsize is None:
            figsize = cls.standard
        if ax is None:
            _, ax = cls.subplots(1, 1, figsize)

        args, kwargs = cls.check_args(args, kwargs)
        kwargs = {"color": "skyblue", "ec": "white", "lw": 0.3}
        if bins is not None:
            kwargs |= {"bins": bins}
        ax.hist(data, density=normalize, **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        cls.grid(ax)
        if "label" in kwargs:
            cls.legend(ax)
        return ax
