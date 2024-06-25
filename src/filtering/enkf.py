from numpy.typing import NDArray
from numpy.random import Generator
from typing import Any

from filtering import EnsembleFilter
from utils import kalman_gain
from utils._typing import DynamicMatrix

from model import StochasticModel

import numpy as np


class EnKF(EnsembleFilter):
    """A class to represent the Ensemble Kalman Filter.

    Attributes
    ----------
    H: DynamicMatrix
        The (linearized) observation model.
    """

    def __init__(
        self,
        model: StochasticModel,
        init_state: NDArray,
        init_cov: NDArray,
        ensemble_size: int,
        generator: Generator | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            model, init_state, init_cov, ensemble_size, generator=generator
        )

    def observation_matrix(self, time: float) -> NDArray:
        """A wrapper for the augmented observation model.

        Parameters
        ----------
        time: float
            The current time.

        Returns
        -------
        NDArray
            The augmented observation matrix.
        """

        obs_matrix = np.zeros((self.n_outputs, self.n_aug))
        obs_matrix[:, : self.n_states] = self.model.H(time)
        return obs_matrix

    def compute_gain(self) -> NDArray:
        """Kalman gain."""

        t = self.current_time
        K = np.zeros((self.n_aug, self.n_outputs))
        if self.correct:
            K = kalman_gain(
                self.full_forecast_cov,
                self.observation_matrix(t),
                self.model.observation_cov(t),
            )
        return K
