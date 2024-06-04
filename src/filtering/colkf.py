from dataclasses import dataclass
from numpy.typing import NDArray
from numpy.random import Generator
from typing import Any, cast

from filtering import Filter
from model import ExplicitModel
from model.autoregressive import ARModel
from model.mixed import MixedDynamicModel
from utils import block_diag
from utils.results import FilteringResults
import numpy as np


@dataclass
class ColKF:
    """An abstract class for bias-aware filters.

    Attributes
    ----------
    bias_model: Model
        The forward model for the bias.
    filter: Filter
        The filtering method.
    init_state: NDArray
        The initial state of the model.
    init_cov: NDArray
        The initial covariance of the model states.
    init_bias: NDArray
        The initial state of the bias.
    init_bias_cov: NDArray
        The initial covariance of the bias.
    feedback: bool
        Whether the filter is with or without feedback.

    Properties
    ----------
    model: Model
        The system forward model.
    forecast_bias: NDArray
        The bias forecast state.
    forecast_bias_cov: NDArray
        The covariance of the forecast bias.
    forecast_state: NDArray
        The forecast state
    forecast_cov: NDArray
        The forecast covariance of states.
    analysis_bias: NDArray
        The bias analysis state.
    anaylsis_bias_cov: NDArray
        The covariance of bias analysis.
    analysis_state: NDArray
        The analysis state.
    analysis_cov: NDArray
        The covariance of the analysis states.
    n_states: int
        The number of states in the system.
    n_aug: int
        The number of states in the filter (augmented state for parameter estimation).
    n_outputs: int
        The number of outputs of the system.
    generator: Generator
        The RNG.
    TODO:
    """

    ar_model: ARModel
    filter_obj: Filter
    init_state: NDArray
    init_cov: NDArray
    init_bias: NDArray
    init_bias_cov: NDArray
    feedback: bool
    augmented_model: MixedDynamicModel | None = None

    def __post_init__(self) -> None:
        """Clone filter with new augmented model"""

        self._add_feedback()
        self.augmented_model = MixedDynamicModel([self.model, self.ar_model])
        kwargs = {}
        kwargs["model"] = self.augmented_model
        kwargs["init_state"] = np.hstack((self.init_state, self.init_bias))
        kwargs["init_cov"] = block_diag([self.init_cov, self.init_bias_cov])
        self.filter_obj = self.filter_obj.clone(**kwargs)

    def _add_feedback(self) -> None:
        """Add AR process to the RHS of the ODE."""

        if self.feedback:
            self.model.offset = lambda *_: -self.ar_model.current_state
        else:
            self.model.observation_offset = lambda *_: -self.model._observe(
                self.ar_model.current_state
            )

    @property
    def state_slice(self) -> slice:
        return slice(None, self.n_states)

    @property
    def bias_slice(self) -> slice:
        return slice(self.n_states, -self.n_params)

    @property
    def params_slice(self) -> slice:
        return slice(-self.n_params, None)

    """"""

    @property
    def model(self) -> ExplicitModel:
        return cast(ExplicitModel, self.filter_obj.model)

    @property
    def generator(self) -> Generator:
        return self.filter_obj.generator

    """"""

    @property
    def forecast_state(self) -> NDArray:
        return self.filter_obj.full_forecast_state[self.state_slice]

    @property
    def forecast_cov(self) -> NDArray:
        return self.filter_obj.full_forecast_cov[self.state_slice, self.state_slice]

    @property
    def analysis_state(self) -> NDArray:
        return self.filter_obj.full_analysis_state[self.state_slice]

    @property
    def analysis_cov(self) -> NDArray:
        return self.filter_obj.full_analysis_cov[self.state_slice, self.state_slice]

    """"""

    @property
    def forecast_bias(self) -> NDArray:
        return self.filter_obj.full_forecast_state[self.bias_slice]

    @property
    def forecast_bias_cov(self) -> NDArray:
        return self.filter_obj.full_forecast_cov[self.bias_slice, self.bias_slice]

    @property
    def analysis_bias(self) -> NDArray:
        return self.filter_obj.full_analysis_state[self.bias_slice]

    @property
    def analysis_bias_cov(self) -> NDArray:
        return self.filter_obj.full_analysis_cov[self.bias_slice, self.bias_slice]

    """"""

    @property
    def param_forecast(self) -> NDArray:
        return self.filter_obj.param_forecast

    @property
    def param_forecast_cov(self) -> NDArray:
        return self.filter_obj.param_forecast_cov

    @property
    def param_analysis(self) -> NDArray:
        return self.filter_obj.param_analysis

    @property
    def param_analysis_cov(self) -> NDArray:
        return self.filter_obj.param_analysis_cov

    """"""

    @property
    def n_states(self) -> int:
        return self.filter_obj.n_states

    @property
    def n_params(self) -> int:
        return self.filter_obj.n_params

    @property
    def n_aug(self) -> int:
        return self.filter_obj.n_aug

    @property
    def n_outputs(self) -> int:
        return self.filter_obj.n_outputs

    """"""

    def forecast(self, end_time: float, *params: Any) -> None:
        """Wrapper for the forecast step of the filter."""

        self.filter_obj.forecast(end_time, *params)

    def analysis(self, observation: NDArray) -> None:
        """Wrapper for the analysis step of the filter."""

        self.filter_obj.analysis(observation)

    def filter(
        self,
        times: NDArray,
        observations: NDArray,
        spin_up_time: float | None = None,
        cut_off_time: float | None = None,
        run_id: str | None = None,
    ) -> FilteringResults:
        """Wrapper for the filtering process."""

        return self.filter_obj.filter(
            times,
            observations,
            spin_up_time=spin_up_time,
            cut_off_time=cut_off_time,
            run_id=run_id,
        )
