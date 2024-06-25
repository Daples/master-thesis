import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy

np.random.seed(0)

plt.close("all")


# Fourth-order Runge-Kutta scheme
def rk4(Z, fun, t=0, dt=1, nt=1):  # (x0, y0, x, h):
    """
    Parameters
        t       : initial time
        Z       : initial states
        fun     : function to be integrated
        dt      : time step length
        nt      : number of time steps

    """

    # Prepare array for use
    if len(Z.shape) == 1:  # We have only one particle, convert it to correct format
        Z = Z[np.newaxis, :]

    # Go through all time steps
    for i in range(nt):

        # Calculate the RK4 values
        k1 = fun(t + i * dt, Z)
        k2 = fun(t + i * dt + 0.5 * dt, Z + dt / 2 * k1)
        k3 = fun(t + i * dt + 0.5 * dt, Z + dt / 2 * k2)
        k4 = fun(t + i * dt + dt, Z + dt * k3)

        # Update next value
        Z += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return Z


def oscillator(t, X):

    M = np.array([[0, 1], [-2, -0.5]])

    return M @ X


N = 40  # Ensemble size
D = 2  # State space dims
T = 10  # Asasimilation steps
dt = 0.01  # Time step length

# X   = np.repeat(np.array([np.pi/2, 0.5])[:,None],axis=1,repeats = N)
X = scipy.stats.multivariate_normal.rvs(
    mean=np.array([np.pi / 2, 0.5]), cov=np.eye(2) * 0.1, size=N  # type: ignore
).T

# Save states
X_stored = np.zeros((int(T / dt) + 1, D, N))
X_stored[0, ...] = copy.copy(X)

# Ensemble bias estimates
b_stored = np.zeros((int(T / dt) + 1, D, N))
b_stored[0, ...] = scipy.stats.multivariate_normal.rvs(
    mean=np.zeros(2), cov=np.eye(2) * 0.01, size=N  # type: ignore
).T

# Observation predictions
Y_stored = np.zeros((int(T / dt) + 1, D, N))

# Observation error covariance matrix
R = np.eye(2) * 0.05

# Forecast error matrix
Q_X = np.eye(2) * 0.01  # For states
Q_b = np.eye(2) * 0.0001  # For bias

# Generate the true state
X_true = np.array([np.pi / 2, 0.5])[:, None]
X_true_stored = np.zeros((int(T / dt) + 1, D, 1))
X_true_stored[0, ...] = copy.copy(X_true)
b_true = -np.ones(2) * 0.1

# Run the true dynamics
for idx, t in enumerate(np.arange(0, T, dt)):

    X_true_stored[idx + 1, ...] = rk4(
        Z=copy.copy(X_true_stored[idx, ...]),
        fun=oscillator,
        t=0,
        dt=dt,  # type: ignore
        nt=1,
    )

    X_true_stored[idx + 1, ...] += b_true[:, None]

nT = idx + 2  # Number of timesteps overall

plt.figure()
plt.plot(X_true_stored[:, 0, 0])
plt.plot(X_true_stored[:, 1, 0])

# Generate observations
y_obs = (
    X_true_stored
    + scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(2),
        cov=R,  # type: ignore
        size=nT,
    )[..., None]
)


plt.plot(y_obs[:, 0, 0], zorder=-1)
plt.plot(y_obs[:, 1, 0], zorder=-1)

# %%

# Get the observation operator
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# Start assimilation for loop
for idx, t in enumerate(np.arange(0, T, dt)):

    # Sample observation predictions
    Y_stored[idx, :, :] = (
        copy.copy(X_stored[idx, ...])
        + scipy.stats.multivariate_normal.rvs(
            mean=np.zeros(2), cov=R, size=N  # type: ignore
        ).T
    )

    # Make the filtering update
    X_concat = np.vstack([X_stored[idx, ...], b_stored[idx, ...]])

    # Get the empirical covariance matrix
    X_concat_cov = np.cov(X_concat)

    # Calculate the Kalman gain
    K = X_concat_cov @ H.T @ np.linalg.inv(H @ X_concat_cov @ H.T + R)

    # Apply the Kalman filter update
    X_concat -= K @ (Y_stored[idx, :, :] - y_obs[idx, ...])

    # Separate the state and bias again
    X_stored[idx, ...] = copy.copy(X_concat[:2, :])
    b_stored[idx, ...] = copy.copy(X_concat[2:, :])

    # ===================================================================

    # Make a forecast
    X_stored[idx + 1, ...] = rk4(
        Z=copy.copy(X_stored[idx, ...]),
        fun=oscillator,
        t=0,
        dt=dt,  # type: ignore
        nt=1,
    )

    # Add bias estimates
    X_stored[idx + 1, ...] += b_stored[idx, ...]

    # Add forecast error
    X_stored[idx + 1, ...] += scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(2), cov=Q_X, size=N  # type: ignore
    ).T

    # Make a forecast for the bias
    b_stored[idx + 1, ...] = copy.copy(b_stored[idx, ...])

    # Max: be careful with the auto-regressive model; here, the true bias does
    # not have a mean of zero, so it makes your performance worse. I would keep
    # the noise term to the bias (so uncertainty still increases if you don't
    # assimilate data), but it won't delete your bias estimates like the
    # autoregressive model does

    # # Apply the auto-regression to the bias
    # b_stored[idx + 1, ...] *= 0.95

    # Add forecast error o bias
    b_stored[idx + 1, ...] += scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(2), cov=Q_b, size=N  # type: ignore
    ).T

# raise Exception


# Plot the bias estimates
X_mean = np.mean(X_stored, axis=-1)
X_std = np.std(X_stored, axis=-1)

t_plot = np.array(list(np.arange(0, T, dt)) + [T + dt])

plt.figure()
plt.subplot(2, 1, 1)
plt.title("first state component")
plt.plot(t_plot, X_mean[:, 0], label="ensemble mean")
plt.fill(
    list(t_plot) + list(np.flip(t_plot)),
    list(X_mean[:, 0] + X_std[:, 0]) + list(np.flip(X_mean[:, 0] - X_std[:, 0])),
    zorder=-1,
    alpha=0.25,
)
plt.plot(t_plot, X_true_stored[:, 0, 0])
plt.scatter(t_plot, y_obs[:, 0, 0])
plt.xlabel("time")
plt.ylabel("state")
plt.subplot(2, 1, 2)
plt.title("second state component")
plt.plot(t_plot, X_mean[:, 1], label="ensemble mean")
plt.fill(
    list(t_plot) + list(np.flip(t_plot)),
    list(X_mean[:, 1] + X_std[:, 1]) + list(np.flip(X_mean[:, 1] - X_std[:, 1])),
    zorder=-1,
    alpha=0.25,
)
plt.plot(t_plot, X_true_stored[:, 1, 0])
plt.scatter(t_plot, y_obs[:, 1, 0])
plt.xlabel("time")
plt.ylabel("state")


# Plot the bias estimates
bias_mean = np.mean(b_stored, axis=-1)
bias_std = np.std(b_stored, axis=-1)

t_plot = np.array(list(np.arange(0, T, dt)) + [T + dt])

plt.figure()
plt.subplot(2, 1, 1)
plt.title("first bias component")
plt.plot(t_plot, bias_mean[:, 0], label="ensemble mean")
plt.fill(
    list(t_plot) + list(np.flip(t_plot)),
    list(bias_mean[:, 0] + bias_std[:, 0])
    + list(np.flip(bias_mean[:, 0] - bias_std[:, 0])),
    zorder=-1,
    alpha=0.25,
)
plt.plot(t_plot, np.ones(nT) * b_true[0])
plt.xlabel("time")
plt.ylabel("bias")
plt.subplot(2, 1, 2)
plt.title("second bias component")
plt.plot(t_plot, bias_mean[:, 1], label="ensemble mean")
plt.fill(
    list(t_plot) + list(np.flip(t_plot)),
    list(bias_mean[:, 1] + bias_std[:, 1])
    + list(np.flip(bias_mean[:, 1] - bias_std[:, 1])),
    zorder=-1,
    alpha=0.25,
)
plt.plot(t_plot, np.ones(nT) * b_true[1])
plt.xlabel("time")
plt.ylabel("bias")
plt.show()
