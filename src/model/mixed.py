from numpy.typing import NDArray
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
    """

    def __init__(self, models: list[ExplicitModel]) -> None:
        self.models: list[ExplicitModel] = models

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

    def forward(
        self, state: State, start_time: Time, end_time: float, *_: Any
    ) -> NDArray:
        """Run the forward of each model independently."""

        count = 0
        for model in self.models:
            model.forward(state[count : count + model.n_states], start_time, end_time)
            count += model.n_states

        # Assuming both models run on the same timestep
        self.times = self.models[0].times
        try:
            self.states = np.vstack([model.states for model in self.models])
        except ValueError:
            raise ValueError(
                "The states have different time lengths. Try resetting both models."
            )

        self.current_time = end_time
        self.current_state = self.states[:, -1]
        return self.current_state

    def f(self, *_: Any) -> NDArray:
        return np.zeros_like(self.initial_condition)
