from model.linear2D import Linear2D
from model.parameter import Parameter
from model.lorenz96 import Lorenz96
from utils import default_generator
import numpy as np


def linear() -> Linear2D:
    """Return a reference linear model in two dimensions.

    Returns
    -------
    Linear2D
        The linear model instance.
    """

    generator = default_generator

    init_state = np.array([np.pi / 2, 6.5])
    n_states = len(init_state)
    n_obs = 2

    p1 = -2
    p2 = -0.5
    params = [
        Parameter(p1, uncertainty=1, name="$p_1$", estimate=True),
        Parameter(p2, uncertainty=1, name="$p_2$", estimate=True),
    ]

    H = lambda _: np.eye(n_obs)
    system_cov = lambda _: 0.03 * np.eye(n_states)
    obs_cov = lambda _: 0.01 * np.eye(n_obs)

    time_step = 0.1

    return Linear2D(init_state, time_step, params, H, system_cov, obs_cov, generator)


def lorenz96(n_states) -> Lorenz96:
    """Return a reference Lorenz-96 model.

    Returns
    -------
    Lorenz96
        The model instance.
    """

    generator = default_generator
    forcing = Parameter(init_value=8, uncertainty=0.2, name="$F$", estimate=False)
    n_states = 40

    time_step = 0.05

    x0_unperturbed = generator.normal(size=n_states)
    x0 = x0_unperturbed.copy()
    x0[0] += 0.01

    system_cov = lambda _: np.eye(n_states)
    obs_cov = lambda _: 0.09 * np.eye(n_states)

    H = lambda _: np.eye(n_states)
    return Lorenz96(
        x0,
        time_step,
        n_states,
        forcing,
        H,
        system_cov,
        obs_cov,
        generator,
        solver="rk4",
    )
