from abc import ABC, abstractmethod
from typing import Any, Callable
from numpy.typing import NDArray

from utils._typing import DynamicMatrix, Integrator
from solvers import get_solver

import numpy as np


class Model(ABC):
    """A class to represent an arbitrary numerical forecasting model.

    Attributes
    ----------
    current_time: float
        The current time in the model.
    initial_condition: NDArray
        The model's initial condition. Not completely needed, but helps define the
        size of the state.
    current_state: NDArray
        The current state of the system.
    system_cov: DynamicMatrix
        The system error covariance matrix.
    observation_cov: DynamicMatrix
        The observation process error covariance matrix.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
    ) -> None:
        self.current_time: float = 0
        self.initial_condition: NDArray = initial_condition
        self.current_state: NDArray = np.zeros_like(initial_condition)
        self.system_cov: DynamicMatrix = system_cov
        self.observation_cov: DynamicMatrix = observation_cov

    @abstractmethod
    def forward(
        self,
        state: NDArray,
        start_time: float,
        end_time: float,
        *params: Any,
        stochastic: bool = False
    ) -> NDArray:
        """It runs the numerical model forward from `start_time` to `end_time` using the
        input `state`.

        Parameters
        ----------
        state: NDArray
            The current state vector of the numerical model.
        start_time: float
            The current simulation time.
        end_time: float
            The time to simulate to.
        *params: Any
            Any additional parameters needed to run the forward model.
        stochastic: bool, optional
            If the model should be propagated stochastically. Default: False

        Returns
        -------
        NDArray
            The sequence of state vectors through time as a matrix `(n_states, time)`.
        """

    @abstractmethod
    def observe(self, state: NDArray) -> NDArray:
        """It extracts the observed state from the state.

        Parameters
        ----------
        state: NDArray
            The state to observe at `current_time`.

        Returns
        -------
        NDArray
            The observed output.
        """


class ODEModel(Model):
    """A class to represent an explicit ODE model. Mostly to include the solver logic.

    Attributes
    ----------
    time_step: float
        The simulation time step for the integrator.
    integrate: ODEIntegrator
        The ODE The array of complete computed states.
    states: NDArray
        The sequence of all computed states.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        time_step: float,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        solver: str = "rk4",
    ) -> None:
        super().__init__(initial_condition, system_cov, observation_cov)
        self.time_step: float = time_step
        self.states: NDArray = self.initial_condition[:, None]
        self._integrator: Integrator = get_solver(solver)

    def integrate(
        self,
        init_time: float,
        end_time: float,
        initial_condition: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Cast integrator to function.

        Parameters
        ----------
        init_time: float
            The lower time bound.
        end_time: float
            The upper bound.
        initial_condition: NDArray | None, optional
            The initial condition for the integration process. Default: None
        """

        if initial_condition is None:
            initial_condition = self.initial_condition
        return self._integrator(
            self.f, initial_condition, init_time, end_time, self.time_step
        )

    def forward(
        self, state: NDArray, start_time: float, end_time: float, *_: Any
    ) -> NDArray:
        """The forward operator for an ODE model."""

        __, states = self.integrate(start_time, end_time, initial_condition=state)

        self.current_time = end_time
        self.current_state = self.states[:, -1]
        self.states = np.hstack((self.states, states))
        return self.current_state

    @abstractmethod
    def f(self, time: float, state: NDArray) -> NDArray:
        """The right-hand side of the system of ODEs.

        Parameters
        ----------
        time: float
            The current simulation time.
        state: NDArray
            The current state vector.

        Returns
        -------
        NDArray
            The evaluated derivatives.
        """


class LinearModel(ODEModel):
    """A class to represent linear models.

    Attributes
    ----------
    M: DynamicMatrix
        The state-transition matrix.
    H: DynamicMatrix
        The observation model.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        time_step: float,
        M: DynamicMatrix,
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        solver: str = "rk4",
    ) -> None:
        super().__init__(
            initial_condition, time_step, system_cov, observation_cov, solver=solver
        )
        self.M: DynamicMatrix = M
        self.H: DynamicMatrix = H

    def f(self, time: float, state: NDArray) -> NDArray:
        """A linear propagation of the state."""

        return self.M(time) @ state

    def observe(self, state: NDArray) -> NDArray:
        """Observation model for a linear system."""

        return self.H(self.current_time) @ state
