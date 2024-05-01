import numpy as np

from model import LinearModel
from filtering.enkf import EnKF
from filtering.kf import KF
from filtering.dummy.dummy_EnKF import ensemble_kalman_filter
from filtering.dummy.dummy_KF import kalman_filter
import matplotlib.pyplot as plt

generator = np.random.default_rng(1234)

init_state = np.array([np.pi / 2, 0.5])
init_state_cov = 0.03 * np.eye(len(init_state))
n_states = len(init_state)
n_obs = 2
time_step = 0.1
M = lambda _: np.array([[0, 1], [-1, 0]])
M = lambda _: np.array([[0, 1], [-2, -0.5]])
# H = lambda _: np.array([[1, 0]])
H = lambda _: np.eye(n_obs)
system_cov = lambda _: 0.05 * np.eye(n_states)
obs_cov = lambda _: 0.01 * np.eye(n_obs)

init_time = 0
end_time = 30
model = LinearModel(
    init_state, time_step, M, H, system_cov, obs_cov, generator, solver="rk4"
)
times, states = model.integrate(init_time, end_time)

assimilation_times = np.arange(1, end_time, 0.5)
observed = np.zeros((n_obs, len(assimilation_times)))
observed_true = np.zeros((n_obs, len(assimilation_times)))
for i, t in enumerate(assimilation_times):
    k = int(t / time_step)
    observed[:, i] = model.observe(states[:, k], add_noise=True)
    observed_true[:, i] = model.observe(states[:, k])

# Run Ensemble Kalman Filter
ensemble_size = 80
assimilation_data = observed
# f = KF(model, init_state, init_state_cov, generator)
f = EnKF(model, init_state, init_state_cov, ensemble_size, H)

# Test previous KF implementation
estimates, estimated_covs = f.filter(
    assimilation_times.tolist(),
    observed_true,
    cut_off_time=None,
)
estimates_dummy, estimated_covs_dummy = kalman_filter(
    M,
    H,
    system_cov,
    obs_cov,
    init_state,
    init_state_cov,
    assimilation_data,
)

# Plotting
plt.figure(figsize=(15, 6))
i = 0
plt.plot(times, states[i, :], "b", label="Run without assimilation")
plt.plot(
    model.times,
    model.states[i, :],
    "k",
    label="Assimilated states",
    alpha=0.2,
)
# plt.plot(
#     assimilation_times,
#     observed_true[i, :],
#     "bo",
#     label="True observation" if i == 0 else None,
# )
plt.plot(assimilation_times, assimilation_data[i, :], "kx", label="Observations")
plt.plot(assimilation_times, estimates[i, :], "r*", label="KF Estimates")
plt.plot(assimilation_times, estimates_dummy[i, :], "g*", label="DA Estimates")
plt.legend()
plt.title("Twin Experiment: True State, Observations, and Estimates")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()
