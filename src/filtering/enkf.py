from numpy.typing import NDArray
from numpy.random import Generator

from filtering import EnsembleFilter
from utils import kalman_gain
from utils._typing import DynamicMatrix

from model import Model

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
        model: Model,
        init_state: NDArray,
        init_cov: NDArray,
        ensemble_size: int,
        H: DynamicMatrix,
        generator: Generator | None = None,
    ) -> None:
        super().__init__(
            model, init_state, init_cov, ensemble_size, generator=generator
        )
        self.H: DynamicMatrix = H

    def compute_gain(self) -> NDArray:
        """Kalman gain."""

        t = self.current_time
        if self.correct:
            K = kalman_gain(self.forecast_cov, self.H(t), self.model.observation_cov(t))
        else:
            K = np.zeros((self.n_states, self.n_outputs))
        return K
