from abc import ABC, abstractmethod

from utils._typing import SystemDynamics, InputFunction, State, Time
from numpy.typing import NDArray

import numpy as np


class Solver(ABC):
    """A class to represent a solver of an explicit dynamic model."""

    @classmethod
    @abstractmethod
    def step(
        cls, f: SystemDynamics, t: Time, x: State, h: float, b: InputFunction
    ) -> NDArray:
        """Perform a step of the solver.

        Parameters
        ----------
        f: SystemDynamics
            The right-hand side of the system of ODEs.
        t: Time
            The current time.
        x: State
            The current state.
        h: float
            Time simulation time step.
        b: Input
            A discrete forcing for the discrete-time system state.

        Returns
        -------
        NDArray
            The new state.
        """

    @classmethod
    def solve(
        cls,
        f: SystemDynamics,
        init_state: State,
        init_time: Time,
        end_time: Time,
        time_step: float,
        discrete_forcing: InputFunction,
    ) -> tuple[NDArray, NDArray]:
        """It integrates an ODE between the given `init_time` and `end_time` using
        `init_state` and dynamics `f.

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
        discrete_forcing: InputFunction
            An input forcing to add at each discrete tiem step.

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
            states[:, i + 1] = cls.step(f, t, states[:, i], time_step, discrete_forcing)
        return time_vector, states
