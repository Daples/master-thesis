from typing import Any

import numpy as np
from numpy.typing import NDArray

from filtering import Filter
from model import LinearModel
from utils import kalman_gain


class KF(Filter):
    """A class that represents a Kalman filter.

    Attributes
    ----------
    R: DynamicMatrix
        The observation noise covariance matrix.
    """

    model: LinearModel

    def forecast(self, end_time: float, *params: Any) -> None:
        """Linear forecast with propagation of covariance."""

        super().forecast(end_time, *params)
        M = self.model.M
        t = self.current_time
        Q = self.model.system_cov(t)

        self.forecast_cov = M(t) @ self.analysis_state @ M(t).T + Q

    def compute_gain(self) -> NDArray:
        """Kalman gain."""

        t = self.current_time
        return kalman_gain(
            self.forecast_cov, self.model.H(t), self.model.observation_cov(t)
        )

    def analysis(self, observation: NDArray) -> None:
        """Linear analysis equation (Kalman analysis)."""

        # Extract information
        t = self.current_time
        P = self.forecast_cov
        H = self.model.H(t)
        x = self.forecast_state
        I = np.eye(self.init_state.shape[0])

        # Linear discrete Kalman filter equations
        K = self.compute_gain()
        self.analysis_state = x + K @ (observation - self.model.observe(x))
        self.analysis_cov = (I - K @ H) @ P
