from typing import Callable
from numpy.typing import NDArray

import numpy as np


def rk4(
    f: Callable[[float, NDArray], NDArray],
    init_state: NDArray,
    init_time: float,
    end_time: float,
    time_step: float,
) -> tuple[NDArray, NDArray]:
    """It integrates an ODE between the given `init_time` and `end_time` using
    `init_state` and dynamics `f`, using a 4th order Runge-Kutta scheme.

    Parameters
    ----------
    f: (float, NDArray) -> NDArray
        The right-hand side of the system of ODEs.
    init_state: NDArray
        The initial value of the state.
    init_time: float
        The time of the initial state.
    end_time: float
        The time to integrate the ODE to.
    time_step: float
        The time step for integration.

    Returns
    -------
    NDArray
        The sequence of time points.
    NDArray
        The sequence of simulated states.
    """

    num_steps = int((end_time - init_time) / time_step) + 1
    time_vector = np.linspace(init_time, end_time, num=num_steps)
    states = np.zeros((init_state.shape[0], num_steps))

    states[:, 0] = init_state
    h = time_step
    for i, t in enumerate(time_vector[:-1]):
        k1 = h * f(t, states[:, i])
        k2 = h * f(t + h / 2, states[:, i] + k1 / 2)
        k3 = h * f(t + h / 2, states[:, i] + k2 / 2)
        k4 = h * f(t + h, states[:, i] + k3)
        states[:, i + 1] = states[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return time_vector, states
