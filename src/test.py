import numpy as np

from model.lorenz96 import Lorenz96
from utils.plotter import Plotter

forcing = 8
n_states = 5

# Initial equilibrium state with small perturbation
x0 = forcing * np.ones(n_states)
x0[0] += 0.01
time_step = 0.01
init_time = 0
end_time = 30

system_cov = lambda _: 0.5 * np.eye(n_states)
obs_cov = lambda _: 0.05 * np.eye(n_states)

l96 = Lorenz96(x0, time_step, n_states, forcing, system_cov, obs_cov)
times, states = l96.integrate(init_time, end_time)
f = lambda x, t: l96.f(t, x)

i = 1
# Plotter.plot(times, states[i, :], xlabel="$t$", ylabel=f"$x_{i}$")
Plotter.plot3(states[i, :], states[i + 1, :], states[i + 2, :])
Plotter.show()
