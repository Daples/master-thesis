from numpy.typing import NDArray
from ..model import Model
from filtering import EnsembleFilter
from utils._typing import DynamicMatrix
from utils import kalman_gain


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
    ) -> None:
        super().__init__(model, init_state, init_cov, ensemble_size)
        self.H: DynamicMatrix = H

    def compute_gain(self) -> NDArray:
        """Kalman gain."""

        t = self.current_time
        return kalman_gain(self.forecast_cov, self.H(t), self.model.observation_cov(t))