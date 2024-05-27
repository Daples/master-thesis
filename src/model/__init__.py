from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

from model.parameter import Parameter
from solvers import get_solver
from utils import default_generator
from utils._typing import DynamicMatrix, Integrator


class Model(ABC):
    """An abstract class to represent a general model.

    Attributes
    ----------
    current_time: float
        The current time in the model.
    current_state: NDArray
        The current state of the system.
    initial_condition: NDArray
        The model's initial condition. Not completely needed, but helps define the
        size of the state.
    times: NDArray
        The array of all simulation times.
    states: NDArray
        The sequence of all computed states.
    """

    def __init__(
        self,
        initial_condition: NDArray,
    ) -> None:
        self.current_time: float = 0
        self.initial_condition: NDArray = initial_condition
        self.current_state: NDArray = np.zeros_like(initial_condition)
        self.times: NDArray = np.zeros(0)
        self.states: NDArray = np.zeros((self.initial_condition.shape[0], 0))
        self.n_states: int = len(self.initial_condition)

    def reset_model(self) -> None:
        """It clears the array of computed states and adds the new initial condition."""

        self.current_time = 0
        self.current_state = np.zeros_like(self.initial_condition)
        self.times = np.zeros(0)
        self.states = np.zeros((self.initial_condition.shape[0], 0))

    @abstractmethod
    def forward(
        self,
        state: NDArray,
        start_time: float,
        end_time: float,
        *params: Any,
        stochastic: bool = False,
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
    def _observe(self, state: NDArray) -> NDArray:
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
    generator: Generator
        The random number generator.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        parameters: list[Parameter],
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
    ) -> None:
        super().__init__(initial_condition)
        self.parameters: list[Parameter] = parameters
        self.system_cov: DynamicMatrix = system_cov
        self.observation_cov: DynamicMatrix = observation_cov
        self.generator: Generator = generator

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

    def observe(self, state: NDArray, add_noise: bool = False) -> NDArray:
        """It extracts the observed state from the state.

        Parameters
        ----------
        state: NDArray
            The state to observe at `current_time`.
        add_noise: bool, optional
            If noise should be added to the observations.

        Returns
        -------
        NDArray
            The observed output.
        """

        observation = self._observe(state).squeeze()
        if add_noise:
            observation = observation.copy() + self.generator.multivariate_normal(
                np.zeros_like(observation),
                self.observation_cov(self.current_time),
            )
        return observation


class ODEModel(StochasticModel, ABC):
    """A class to represent an explicit ODE model. Mostly to include the solver logic.

    Attributes
    ----------
    time_step: float
        The simulation time step for the integrator.
    _integrator: Integrator
        The model integrator. Default: rk4
    model_bias: (float) -> NDArray
        A function to add bias to the model dynamics. It defaults to zero bias.
    """

    def __init__(
        self,
        initial_condition: NDArray,
        parameters: list[Parameter],
        time_step: float,
        system_cov: DynamicMatrix,
        observation_cov: DynamicMatrix,
        generator: Generator,
        model_bias: Callable[[float, NDArray], NDArray] | None = None,
        solver: str = "rk4",
    ) -> None:
        super().__init__(
            initial_condition, parameters, system_cov, observation_cov, generator
        )
        self.time_step: float = time_step
        self._integrator: Integrator = get_solver(solver)

        if model_bias is None:
            model_bias = lambda _, __: np.zeros_like(self.initial_condition)
        self.model_bias: Callable[[float, NDArray], NDArray] = model_bias

    def integrate(
        self,
        init_time: float,
        end_time: float,
        initial_condition: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Wrapper for model integrator.

        Parameters
        ----------
        init_time: float
            The lower time bound.
        end_time: float
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

        f = lambda time, state: self.f(time, state) + self.model_bias(time, state)
        return self._integrator(
            f, initial_condition, init_time, end_time, self.time_step
        )

    def forward(
        self, state: NDArray, start_time: float, end_time: float, *_: Any
    ) -> NDArray:
        """The forward operator for an ODE model."""

        times, states = self.integrate(start_time, end_time, initial_condition=state)

        self.states = np.hstack((self.states, states))
        self.times = np.hstack((self.times, times))
        self.current_time = end_time
        self.current_state = self.states[:, -1]

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
        generator: Generator | None,
        solver: str = "rk4",
    ) -> None:
        if generator is None:
            generator = default_generator
        super().__init__(
            initial_condition,
            [],
            time_step,
            system_cov,
            observation_cov,
            generator,
            solver=solver,
        )
        self.M: DynamicMatrix = M
        self.H: DynamicMatrix = H

    def f(self, time: float, state: NDArray) -> NDArray:
        """A linear propagation of the state."""

        return (self.M(time) @ state[:, np.newaxis]).squeeze()

    def _observe(self, state: NDArray) -> NDArray:
        """Observation model for a linear system."""

        return (self.H(self.current_time) @ state).squeeze()
