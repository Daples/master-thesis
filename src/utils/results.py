from filtering import Filter
from model import Model

from dataclasses import dataclass
from numpy.typing import NDArray

import numpy as np


@dataclass
class FilteringResults:
    """A class to handle and process filtering results.

    Attributes
    ----------
    model: Model
        The forward model.
    assimilation_times: NDArray
        The array of times where observations were assimilated.
    observations: NDArray
        The observations.
    estimated_states: NDArray
        The estimated (filtered) states at the assimilation times.
    estimated_covs: NDArray
        The estimated (filtered) covariances at the assimilation times.
    simulation_times: NDArray
        The forward model simulation time (in simulation time steps).

    Properties
    ----------
    innovations: NDArray
        The array of innovations (observation - forecast)
    rmses: NDArray
        The array of advancing RMSEs.
    """

    model: Model
    assimilation_times: NDArray
    observations: NDArray
    estimated_states: NDArray
    estimated_covs: NDArray
    simulation_times: NDArray
    _estimated_observations: NDArray
    _innovations: NDArray | None = None
    _rmses: NDArray | None = None

    @property
    def innovations(self) -> NDArray:
        """It calculates the innvoations if required.

        Returns
        -------
        NDArray
            The array of innovations at assimilation times.
        """

        if self._innovations is None:
            self._innovations = self.observations - self.estimated_observations
        return self._innovations

    @property
    def estimated_observations(self) -> NDArray:
        """It calculates the innvoations if required.

        Returns
        -------
        NDArray
            The array of innovations at assimilation times.
        """

        if self._estimated_observations is None:
            self._estimated_observations = self.model.observe(self.estimated_states)
        return self._estimated_observations

    @property
    def rmses(self) -> NDArray:
        """It calculates the RMSE through time.

        Returns
        -------
        NDArray
            The array of RMSEs at assimilation times.
        """

        if self._rmses is None:
            self._rmses = np.zeros_like(self.estimated_observations)
            for k in range(1, len(self.assimilation_times) + 1):
                mse = (
                    (self.observations[:, :k] - self.estimated_observations[:, :k]) ** 2
                ).mean(axis=1)
                self._rmses[:, k] = np.sqrt(mse)
        return self._rmses

    @property
    def rmse(self) -> NDArray:
        """It returns the RMSE at final time for each state.

        Returns
        -------
        NDArray
            The array of RMSE for each state.
        """

        return self.rmses[:, -1]
