from typing import Callable

import numpy as np
from numpy.typing import NDArray


def discrete(
    f: Callable[[float, NDArray], NDArray],
    init_state: NDArray,
    init_time: float,
    end_time: float,
    time_step: float,
) -> tuple[NDArray, NDArray]:
    """It performs forward steps of a discrete dynamical system.

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
    for i, t in enumerate(time_vector[:-1]):
        states[:, i + 1] = f(t, states[:, i]).squeeze()

    return time_vector, states
