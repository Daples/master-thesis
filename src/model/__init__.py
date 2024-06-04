from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model.parameter import Parameter
from solvers import get_solver
from utils import default_generator
from utils._typing import (
    DynamicMatrix,
    Integrator,
    State,
    Input,
    InputFunction,
    Observation,
    Time,
)

# TODO: Update docstring


class Model(ABC):
    """An abstract class to represent a general model.

    Attributes
    ----------
    current_time: Time
        The current time in the model.
    current_state: State
        The current state of the system.
    initial_condition: State
        The model's initial condition. Not completely needed, but helps define the
        size of the state.
    times: NDArray
        The array of all simulation times.
    states: NDArray
        The sequence of all computed states.
    n_states: int
        The number of states in the model.
    """

    def __init__(
        self,
        initial_condition: State,
    ) -> None:
        self.current_time: Time = 0
        self.initial_condition: State = initial_condition
        self.current_state: State = np.zeros_like(initial_condition)
        self.times: NDArray = np.zeros(0)
        self.states: NDArray = np.zeros((self.initial_condition.shape[0], 0))
        self.n_states: int = len(self.initial_condition)

    def reset_model(self) -> None:
        """It clears the computed states and sets the initial state to zero."""

        self.current_time = 0
        self.current_state = np.zeros_like(self.initial_condition)
        self.times = np.zeros(0)
        self.states = np.zeros((self.initial_condition.shape[0], 0))

    @abstractmethod
    def forward(
        self,
        state: State,
        start_time: Time,
        end_time: Time,
        *params: Any,
    ) -> NDArray:
        """It runs the numerical model forward from `start_time` to `end_time` using the
        input `state`.

        Parameters
        ----------
        state: State
            The current state vector of the numerical model.
        start_time: Time
            The current simulation time.
        end_time: Time
            The time to simulate to.
        *params: Any
            Any additional parameters needed to run the forward model.

        Returns
        -------
        NDArray
            The sequence of state vectors through time as a matrix `(n_states, time)`.
        """

    @abstractmethod
    def _observe(self, state: State) -> Observation:
        """It extracts the observed state from the state.

        Parameters
        ----------
        state: State
            The state to observe at `current_time`.

        Returns
        -------
        Observation
            The observed output.
        """


class StochasticModel(Model, ABC):
    """A class to represent an arbitrary numerical forecasting model.

    Attributes
    ----------
    parameters: list[Parameter]
        The list of model parameters (for possible estimation).
    system_cov: DynamicMatrix
        The system error covariance matrix.
    observation_cov: DynamicMatrix
        The observation process error covariance matrix.
    H: DynamicMatrix
        The linear observation model (assumption for applying Kalman analysis).
    generator: Generator
        The random number generator.
    observation_offset: InputFunction | None
        Offset function for the observation operator.
    """

    def __init__(
        self,
        initial_condition: State,
        parameters: list[Parameter],
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        observation_offset: InputFunction | None = None,
    ) -> None:
        super().__init__(initial_condition)
        self.parameters: list[Parameter] = parameters
        self.system_cov: DynamicMatrix = system_cov
        self.observation_cov: DynamicMatrix = observation_cov
        self.H: DynamicMatrix = H
        self.generator: Generator = generator
        if observation_offset is None:
            observation_offset = lambda _, __: np.zeros(self.system_cov(0).shape[0])
        self.observation_offset: InputFunction = observation_offset

    @property
    def uncertain_parameters(self) -> list[Parameter]:
        """Get the list of parameters to be estimated.

        Returns
        -------
        list[Parameter]
            The list of uncertain parameters to be estimated.
        """

        return [param for param in self.parameters if param.estimate]

    @uncertain_parameters.setter
    def uncertain_parameters(self, new_values: NDArray) -> None:
        """Update the uncertain parameters with the estimated values.

        Parameters
        ----------
        new_values: NDArray
            The array of new values.
        """

        for i, param in enumerate(self.uncertain_parameters):
            param.current_value = new_values[i]

    def system_error(self, time: Time) -> State:
        """Return a realization of the system error.

        Parameters
        ----------
        time: Time
            The time to evaluate the realization.

        Returns
        -------
        State
            The state error at the given time.
        """

        return self.generator.multivariate_normal(
            np.zeros(self.system_cov(0).shape[0]),
            self.system_cov(time),
        )

    def observation_error(self, time: Time) -> Observation:
        """Return a realization of the observation error.

        Parameters
        ----------
        time: Time
            The time to evaluate the realization.

        Returns
        -------
        Observation
            The observation error at the given time.
        """

        return self.generator.multivariate_normal(
            np.zeros(self.observation_cov(0).shape[0]),
            self.observation_cov(time),
        )

    def observe(self, state: State, add_noise: bool = False) -> Observation:
        """It extracts the observed state from the state.

        Parameters
        ----------
        state: State
            The state to observe at `current_time`.
        add_noise: bool, optional
            If noise should be added to the observations.

        Returns
        -------
        Observation
            The observed output.
        """

        observation = self._observe(
            state + self.observation_offset(self.current_time, state)
        ).squeeze()
        if add_noise:
            observation = observation.copy() + self.observation_error(self.current_time)
        return observation

    def _observe(self, state: State) -> Observation:
        """Explicit linear observation operator. (TODO: generalize?)"""

        return (self.H(self.current_time) @ state).squeeze()


class ExplicitModel(StochasticModel, ABC):
    """A class to represent an explicit ODE model. Mostly to include the solver logic.

    Attributes
    ----------
    time_step: float
        The simulation time step for the integrator.
    _integrator: Integrator
        The model integrator. Default: rk4
    input: InputFunction
        A function to add an input to the model dynamics. It defaults to zero.
    offset: InputFunction
        A function to add an offset to the model dynamics. It defaults to zero.
    """

    def __init__(
        self,
        initial_condition: State,
        parameters: list[Parameter],
        time_step: float,
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        input: InputFunction | None = None,
        offset: InputFunction | None = None,
        solver: str = "rk4",
    ) -> None:
        super().__init__(
            initial_condition, parameters, H, system_cov, observation_cov, generator
        )
        self.time_step: float = time_step
        self._integrator: Integrator = get_solver(solver)

        if input is None:
            input = lambda _, __: np.zeros_like(self.initial_condition)
        self.input: InputFunction = input

        if offset is None:
            offset = lambda _, __: np.zeros_like(self.initial_condition)
        self.offset: InputFunction = offset

    def integrate(
        self,
        init_time: Time,
        end_time: Time,
        initial_condition: State | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Wrapper for model integrator.

        Parameters
        ----------
        init_time: Time
            The lower time bound.
        end_time: Time
            The upper bound.
        initial_condition: NDArray | None, optional
            The initial condition for the integration process. Default: None

        Returns
        -------
        NDArray
            The array of simulation time steps.
        NDArray
            The array of simulated states for all time steps.
        """

        if initial_condition is None:
            initial_condition = self.initial_condition

        f = lambda t, x: self.f(t, x, self.input(t, x)) + self.offset(t, x)
        return self._integrator(
            f, initial_condition, init_time, end_time, self.time_step
        )

    def forward(self, state: State, start_time: Time, end_time: Time, *_: Any) -> State:
        """The forward operator for an ODE model."""

        times, states = self.integrate(start_time, end_time, initial_condition=state)

        self.states = np.hstack((self.states, states))
        self.times = np.hstack((self.times, times))
        self.current_time = end_time
        self.current_state = self.states[:, -1]

        return self.current_state

    @abstractmethod
    def f(self, time: Time, state: State, input: Input) -> NDArray:
        """The right-hand side of the system of ODEs.

        Parameters
        ----------
        time: Time
            The current simulation time.
        state: State
            The current state vector.
        input: Input
            An input for the system.

        Returns
        -------
        NDArray
            The evaluated derivatives.
        """


class LinearModel(ExplicitModel):
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
        initial_condition: State,
        time_step: Time,
        M: DynamicMatrix,
        H: DynamicMatrix,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator | None,
        solver: str = "rk4",
    ) -> None:
        if generator is None:
            generator = default_generator
        super().__init__(
            initial_condition,
            [],
            time_step,
            H,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
        )
        self.M: DynamicMatrix = M
        self.H: DynamicMatrix = H

    def f(self, time: Time, state: State, input: Input) -> NDArray:
        """A linear propagation of the state."""

        return (self.M(time) @ state[:, np.newaxis]).squeeze() + input
