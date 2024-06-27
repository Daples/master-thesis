from numpy.typing import NDArray
from numpy.random import Generator
from model import ExplicitModel
from typing import Any
from utils import block_diag
from utils._typing import State, Time

import numpy as np


class MixedDynamicModel(ExplicitModel):
    """A class to represent a model that uses different solvers.

    Attributes
    ----------
    models: list[ExplicitModel]
        The models to join.
    slices: list[slice]
        The slices of each model in the augmented state.
    """

    def __init__(self, models: list[ExplicitModel]) -> None:
        self.models: list[ExplicitModel] = models
        self.slices: list[slice] = []

        count = 0
        for model in self.models:
            self.slices.append(slice(count, count + model.n_states))
            count += model.n_states

        # Initialize augmented model
        initial_condition = np.hstack(
            [model.initial_condition for model in self.models]
        )
        parameters = [param for model in self.models for param in model.parameters]
        time_step = self.models[0].time_step
        H = lambda k: block_diag([model.H(k) for model in self.models])
        system_cov = lambda k: block_diag(
            [model.system_cov(k) for model in self.models]
        )
        observation_cov = lambda k: block_diag(
            [model.observation_cov(k) for model in self.models]
        )
        generator = self.models[0].generator

        super().__init__(
            initial_condition,
            parameters,
            time_step,
            H,
            system_cov,
            observation_cov,
            generator,
        )

    @property
    def noise_mask(self) -> NDArray:
        """Construct the noise mask based on each models'.

        Returns
        -------
        NDArray
            The vstack of noise masks.
        """

        if self._noise_mask is None:
            self._noise_mask = np.vstack([model.noise_mask for model in self.models])
        return self._noise_mask

    def update_generator(self, generator: Generator) -> None:
        """Update the generator for all models."""

        super().update_generator(generator)
        for model in self.models:
            model.update_generator(generator)

    def f(self, *_: Any) -> NDArray:
        return np.zeros_like(self.initial_condition)

    def forward(
        self, state: State, start_time: Time, end_time: float, *_: Any
    ) -> NDArray:
        """Solve all models sequentially (run all models for each time step)."""

        num_steps = int((end_time - start_time) / self.time_step) + 1
        time_vector = np.linspace(start_time, end_time, num=num_steps)
        states = np.zeros((self.n_states, num_steps))

        # Add initial conditions
        states[:, 0] = state
        for model_slice, model in zip(self.slices, self.models):
            current_state = state[model_slice]
            model.states = np.hstack(
                (model.states, current_state.reshape((model.n_states, 1)))
            )
            model.current_state = current_state

        for i, t in enumerate(time_vector[:-1]):
            count = 0
            for model in self.models:
                model_slice = slice(count, count + model.n_states)
                f = model.get_modified_dynamics()
                new_state = model.solver.step(
                    f,
                    t,
                    model.current_state,
                    model.time_step,
                    model.integration_forcing,
                )
                states[model_slice, i + 1] = new_state
                count += model.n_states

                # Update states for each model
                # TODO: this is nasty
                if not i == len(time_vector) - 2:
                    model.states = np.hstack(
                        (model.states, new_state.reshape((model.n_states, 1)))
                    )
                    model.times = np.hstack((model.times, t))
                model.current_time = time_vector[i + 1]
                model.current_state = new_state

        # Update states for mixed model
        self.states = np.hstack((self.states, states[:, :-1]))
        self.times = np.hstack((self.times, time_vector[:-1]))
        self.current_time = end_time
        self.current_state = states[:, -1]

        return self.current_state
