import numpy as np
from numpy.typing import NDArray
from utils._typing import State, Time, SystemDynamics


def euler(
    f: SystemDynamics,
    init_state: State,
    init_time: Time,
    end_time: Time,
    time_step: float,
) -> tuple[NDArray, NDArray]:
    """It integrates an ODE between the given `init_time` and `end_time` using
    `init_state` and dynamics `f`, using Euler's method.

    Parameters
    ----------
    f: SystemDynamics
        The right-hand side of the system of ODEs.
    init_state: State
        The initial value of the state.
    init_time: Time
        The time of the initial state.
    end_time: Time
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
        states[:, i + 1] = states[:, i] + time_step * f(t, states[:, i])

    return time_vector, states
