from model.autoregressive import ARModel
from model import LinearModel
from filtering.colkf import ColKF
from filtering.enkf import EnKF
from utils.plotter import Plotter
from utils.compare import Comparator

import matplotlib.pyplot as plt
import numpy as np

generator = np.random.default_rng(1234567)

N = 80
T0 = 0
T = 5
dt = 0.01
D = 2

x0 = np.array([np.pi / 2, 0.5])
x0_cov = 0.15 * np.eye(D)

b0 = 0 * np.ones(D)
b0_cov = 0.01 * np.eye(D)

R = lambda _: 0.5 * np.eye(D)
Q_X = lambda _: 0.2 * np.eye(D)
Q_b = lambda _: 0.0001 * np.eye(D)

b_true = -0.1 * np.ones(D)

M = lambda _: np.array([[0, 1], [-2, -0.5]])
H = lambda _: np.eye(D)

A = lambda _: np.diag([1] * D)

plot_ensemble = True
plot_bands = True

linear_model = LinearModel(
    x0,
    dt,
    M,
    H,
    Q_X,
    R,
    generator,
    solver="rk4",
    stochastic_propagation=False,
    stochastic_integration=False,
)
linear_model.discrete_forcing = lambda *_: b_true
times, states = linear_model.integrate(T0, T)
linear_model.reset_model(x0)

# Generate observations with noise
assimilation_step = 40 * dt
assimilation_times = np.arange(
    T0 + assimilation_step,
    T,
    assimilation_step,
)
observed = np.zeros((D, len(assimilation_times)))
for i, t in enumerate(assimilation_times):
    k = int(t / dt)
    observed[:, i] = linear_model.observe(states[:, k], add_noise=True)

linear_model.discrete_forcing = lambda *_: np.zeros(D)
linear_model.stochastic_propagation = True

enkf = EnKF(linear_model, x0, x0_cov, N, generator=generator)
results = enkf.filter(assimilation_times, observed)
results.true_times, results.true_states = times, states

# for i in range(1):
#     ax = results.plot_filtering(i, plot_ensemble, plot_bands)

H_ar = lambda _: np.zeros((0, b0.shape[0]))
ar_model = ARModel(
    A,
    H_ar,
    b0,
    dt,
    Q_b,
    generator,
    stochastic_propagation=True,
    stochastic_integration=False,
)
colkf = ColKF(ar_model, enkf, x0, x0_cov, b0, b0_cov, feedback=True)
results_col = colkf.filter(assimilation_times, observed)
results_col.true_times = times
results_col.true_states = np.vstack((states, -np.diag(b_true) @ np.ones_like(states)))

Plotter.setup()
c = Comparator([results_col, results], ["Bias-aware", "Bias-blind"])
_, axs = Plotter.subplots(3, 1, "tall", make_3d=False, height_ratios=[2, 1, 1])
# c.compare_filtering(
#     state_idx=0,
#     plot_ensemble=True,
#     plot_bands=False,
#     figsize="tall",
#     # ax=axs[0],
# )


c.compare_filtering(
    state_idx=0,
    plot_ensemble=True,
    plot_bands=False,
    figsize="horizontal",
    ax=axs[0],  # type: ignore
)
results_col.plot_filtering(
    state_idx=2,
    plot_ensemble=False,
    plot_bands=True,
    only_state=True,
    ax=axs[2],  # type: ignore
    legend=False,
    color=c.colors[0],
    path="test.pdf",
)
results_col.plot_filtering(3, True, True, only_state=True, color=c.colors[0], ax=axs[1])  # type: ignore
c.compare_innovations(0, ax=axs[1], legend=False, window=3, path="test.pdf")  # type: ignore
plt.show()
