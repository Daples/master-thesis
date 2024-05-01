import numpy as np

from model.lorenz96 import Lorenz96
from filtering.enkf import EnKF
from utils.plotter import Plotter
import matplotlib.pyplot as plt

generator = np.random.default_rng(123)
forcing = 8
n_states = 5

x0_unperturbed = generator.normal(size=n_states)
x0 = x0_unperturbed.copy()
x0[0] += 0.01

time_step = 0.01
init_time = 0
end_time = 15

ensemble_size = 80
initial_state_covariance = np.eye(n_states)

system_cov = lambda _: np.eye(n_states)
obs_cov = lambda _: 0.5 * np.eye(n_states)

# Simulate without noise

# Generate true state
l96 = Lorenz96(x0, time_step, n_states, forcing, system_cov, obs_cov, generator)
times, states = l96.integrate(init_time, end_time)
l96.reset_model()

H = lambda _: np.eye(n_states)

# Generate observations (with added noise)
assimilation_times = np.arange(1, end_time, 0.25)
observed = np.zeros((n_states, len(assimilation_times)))
observed_true = np.zeros((n_states, len(assimilation_times)))
for i, t in enumerate(assimilation_times):
    k = int(t / time_step)
    observed[:, i] = l96.observe(states[:, k], add_noise=True)
    observed_true[:, i] = l96.observe(states[:, k])

# Run Ensemble Kalman Filter
f = EnKF(l96, x0, initial_state_covariance, ensemble_size, H)
estimates, estimated_covs = f.filter(
    assimilation_times.tolist(), observed, cut_off_time=3
)

# Plotting
plt.figure(figsize=(15, 6))
i = 0
plt.plot(times, states[i, :], "b", label="Truth")
plt.plot(
    l96.times,
    l96.states[i, :],
    "k",
    label="Assimilation",
    alpha=0.2,
)
plt.plot(
    assimilation_times,
    observed_true[i, :],
    "bo",
    # label="True observation" if i == 0 else None,
)
plt.plot(assimilation_times, observed[i, :], "kx", label="Observations")
plt.plot(assimilation_times, estimates[i, :], "r*", label="Estimates")
plt.legend()
plt.xlabel("Time")
plt.ylabel("State")
plt.show()
