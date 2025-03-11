import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy

# np.random.seed(0)

plt.close("all")


# Fourth-order Runge-Kutta scheme
def rk4(Z, fun, t=0, dt=1, nt=1, **kwargs):  # (x0, y0, x, h):
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
        k1 = fun(t + i * dt, Z, **kwargs)
        k2 = fun(t + i * dt + 0.5 * dt, Z + dt / 2 * k1, **kwargs)
        k3 = fun(t + i * dt + 0.5 * dt, Z + dt / 2 * k2, **kwargs)
        k4 = fun(t + i * dt + dt, Z + dt * k3, **kwargs)

        # Update next value
        Z += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return Z


# These are the dynamics of the Lorenz-96 system
def lorenz_dynamics_96(t, Z, F=8, b=0):

    dZdt = np.zeros(Z.shape)

    if len(Z.shape) == 1:  # Only one particle

        D = len(Z)

        for d in range(D):

            # Calculate the indices
            idc = np.asarray([d - 2, d - 1, d, d + 1])
            idc[np.where(idc < 0)] += D
            idc[np.where(idc >= D)] -= D

            dZdt[d] = (Z[idc[3]] - Z[idc[0]]) * Z[idc[1]] - Z[idc[2]] + F + b

    else:

        D = Z.shape[1]

        for d in range(D):

            # Calculate the indices
            idc = np.asarray([d - 2, d - 1, d, d + 1])
            idc[np.where(idc < 0)] += D
            idc[np.where(idc >= D)] -= D

            dZdt[:, d] = (
                (Z[:, idc[3]] - Z[:, idc[0]]) * Z[:, idc[1]] - Z[:, idc[2]] + F + b
            )

    return dZdt


# Define problem dimensions
O = 20  # Observation space dimensions
D = 40  # State space dimensions

# Set up time
T_spinup = 100  # EnKF spinup period
T = 100  # Assimilation time steps
dt = 0.4  # Time step length

# Ensemble size
N = 1000

# Define a true forcing and a true bias
F_true = 7
b_true = 1

# Observation error
obs_sd = np.sqrt(0.5)
R = np.identity(O) * 0.5
obsindices = np.arange(D)[::2]  # We observe every second state

# Observation operator
H = np.zeros((O, D))
for row, col in enumerate(obsindices):
    H[row, col] = 1

# =============================================================================
# Generate the synthetic truth and the observations
# =============================================================================

# Initialize the synthetic truth
synthetic_truth = np.zeros((T_spinup + T, D))
synthetic_truth[0, :] = scipy.stats.multivariate_normal.rvs(
    mean=np.zeros(D), cov=np.identity(D), size=1
)

# Initialized the synthetic observations
observations = copy.copy(synthetic_truth[:, obsindices])
observations[0, :] += scipy.stats.norm.rvs(loc=0, scale=obs_sd, size=O)

# Go through all timesteps
for t in np.arange(1, T_spinup + T, 1):

    print(f"\r" + "Spinup truth and observations t=" + str(t), end="\r")

    # Make a Lorenz forecast
    synthetic_truth[t, :] = rk4(
        Z=copy.copy(synthetic_truth[t - 1, :][np.newaxis, :]),
        fun=lorenz_dynamics_96,
        t=0,
        dt=0.01,
        nt=int(dt / 0.01),
        F=F_true,
        b=b_true,
    )[0, :]

    # Calculate the synthetic observation
    observations[t, :] = copy.copy(synthetic_truth[t, obsindices])
    observations[t, :] += scipy.stats.multivariate_normal.rvs(
        mean=np.zeros(O), cov=R, size=1
    )

print(f"\r" + "Spinup truth and observations completed.")

# =============================================================================
# Initiate the ensemble with a spin-up period
# =============================================================================

# Initiate particles from a standard Gaussian
X_spinup = np.zeros((T_spinup + 1, N, D))
X_spinup[0, ...] = scipy.stats.norm.rvs(size=(N, D))

RMSE_list_spinup = []

# Go through the spinup period
for t in np.arange(0, T_spinup, 1):

    print(f"\r" + "Spinup initial ensemble t=" + str(t), end="\r")

    # Get the state covariance matrix
    C = np.cov(X_spinup[t, ...].T)  # We need the transpose of X

    # Calculate the Kalman gain
    K = np.linalg.multi_dot(
        (C, H.T, np.linalg.inv(np.linalg.multi_dot((H, C, H.T)) + R))
    )

    # Draw observation error realizations
    v = scipy.stats.multivariate_normal.rvs(mean=np.zeros(R.shape[0]), cov=R, size=N)

    # Perturb the observations
    obs = copy.copy(observations[t, :])[np.newaxis, :] + v

    # Apply the stochastic Kalman update
    X_spinup[t, ...] += np.dot(K, obs.T - np.dot(H, X_spinup[t, ...].T)).T

    # Calculate RMSE
    RMSE = (np.mean(X_spinup[t, :, :D], axis=0) - synthetic_truth[t, :]) ** 2
    RMSE = np.mean(RMSE)
    RMSE = np.sqrt(RMSE)
    RMSE_list_spinup.append(RMSE)

    # After the analysis step, make a forecast to the next timestep
    if t < T_spinup:

        # Make a Lorenz forecast
        X_spinup[t + 1, :, :] = rk4(
            Z=copy.copy(X_spinup[t, :, :]),
            fun=lorenz_dynamics_96,
            t=0,
            dt=dt / 40,
            nt=int(dt / 0.01),
            F=F_true,
            b=b_true,
        )

print(f"\r" + "Spinup initial ensemble completed.")

# Now throw away the spinup period. We no longer need it.
# This is our starting point for the real DA problem.
observations = observations[T_spinup:, :]
synthetic_truth = synthetic_truth[T_spinup:, :]
X_initial = X_spinup[T_spinup, :, :]


# =============================================================================
# Initiate the ensemble with a spin-up period
# =============================================================================

# Pre-allocate some space for the augmented state vector
# Dimensions 1 to D : states
# Dimension D+1     : forcing
# Dimension D+2     : bias
X = np.zeros((T, N, D + 2))
X[0, :, :D] = copy.copy(X_initial)

# Now also define a prior for the forcings and the biases
X[0, :, D] = scipy.stats.norm.rvs(loc=F_true + 2, scale=2, size=N)

X[0, :, D + 1] = scipy.stats.norm.rvs(loc=b_true + 1, scale=2, size=N)

# Save the initial states, forcings, and bias for plotting
X_initial = copy.copy(X[:1, ...])

# Define a new observation operator that takes into account the two extended dimensions
H_new = np.hstack((H, np.zeros((O, 2))))

RMSE_list = []

# Go through the spinup period
for t in np.arange(0, T, 1):

    print(f"\r" + "Running the bias-aware filter t=" + str(t), end="\r")

    # Get the state covariance matrix
    C = np.cov(X[t, ...].T)  # We need the transpose of X

    # Calculate the Kalman gain
    K = np.linalg.multi_dot(
        (C, H_new.T, np.linalg.inv(np.linalg.multi_dot((H_new, C, H_new.T)) + R))
    )

    # Draw observation error realizations
    v = scipy.stats.multivariate_normal.rvs(mean=np.zeros(R.shape[0]), cov=R, size=N)

    # Perturb the observations
    obs = copy.copy(observations[t, :])[np.newaxis, :] + v

    # Apply the stochastic Kalman update
    X[t, ...] += np.dot(K, obs.T - np.dot(H_new, X[t, ...].T)).T

    # Calculate RMSE
    RMSE = (np.mean(X[t, :, :D], axis=0) - synthetic_truth[t, :]) ** 2
    RMSE = np.mean(RMSE)
    RMSE = np.sqrt(RMSE)
    RMSE_list.append(RMSE)

    # After the analysis step, make a forecast to the next timestep
    if t < T - 1:

        # Make a Lorenz forecast
        X[t + 1, :, :D] = rk4(
            Z=copy.copy(X[t, :, :D]),  # The states
            fun=lorenz_dynamics_96,
            t=0,
            dt=dt / 40,
            nt=int(dt / 0.01),
            F=X[t, :, D],  # The forcing
            b=X[t, :, D + 1],
        )  # The bias

        # Forecast for bias and forcing is just identity
        X[t + 1, :, D:] = copy.copy(X[t, :, D:])

print(f"\r" + "Running the bias-aware filter completed.")

# Append the initial (pre-first-assimilation) states for plotting purposes
X = np.concatenate((X_initial, X), axis=0)

# =============================================================================
# Figure: plot the sum of F and b
# =============================================================================

plt.figure()
sum_mean = np.asarray([np.mean(np.sum(X[t, :, D:], axis=-1)) for t in range(T + 1)])
sum_std = np.asarray([np.std(np.sum(X[t, :, D:], axis=-1)) for t in range(T + 1)])

plt.plot(sum_mean, color="r", label="ensemble mean $F + b$")
plt.plot(sum_mean + sum_std, color="k", label="ensemble mean +/- std $F + b$")
plt.plot(sum_mean - sum_std, color="k")
plt.plot(
    [0, T],
    [F_true + b_true, F_true + b_true],
    color="xkcd:grey",
    ls="--",
    label="truth",
)
plt.legend()
plt.ylabel("$F + b$")
plt.xlabel("assimilation steps")

# =============================================================================
# Figure: plot F and b
# =============================================================================

plt.figure()

plt.subplot(2, 1, 1)
F_mean = np.asarray([np.mean(X[t, :, D]) for t in range(T + 1)])
F_std = np.asarray([np.std(X[t, :, D]) for t in range(T + 1)])

plt.plot(F_mean, color="r", label="ensemble mean")
plt.plot(F_mean + F_std, color="k", label="ensemble mean +/- std")
plt.plot(F_mean - F_std, color="k")
plt.plot([0, T], [F_true, F_true], color="xkcd:grey", ls="--", label="truth")
plt.ylabel("forcing $F$")
plt.xlabel("assimilation steps")
plt.legend()

plt.subplot(2, 1, 2)
b_mean = np.asarray([np.mean(X[t, :, D + 1]) for t in range(T + 1)])
b_std = np.asarray([np.std(X[t, :, D + 1]) for t in range(T + 1)])

plt.plot(b_mean, color="r", label="ensemble mean")
plt.plot(b_mean + b_std, color="k", label="ensemble mean +/- std")
plt.plot(b_mean - b_std, color="k")
plt.plot([0, T], [b_true, b_true], color="xkcd:grey", ls="--", label="truth")
plt.ylabel("bias $b$")
plt.xlabel("assimilation steps")
plt.legend()

# =============================================================================
# Figure: scatter plot of F and b
# =============================================================================

plt.figure()
for n in range(N):
    plt.plot(X[:, n, D], X[:, n, D + 1], color="xkcd:grey", lw=0.5, zorder=-1)
    plt.xlabel("forcing $F$")
    plt.ylabel("bias $b$")
plt.scatter(X[0, :, D], X[0, :, D + 1], color="b", zorder=5, label="prior samples")
plt.scatter(
    X[-1, :, D], X[-1, :, D + 1], color="r", zorder=10, label="posterior samples"
)
xlims = plt.gca().get_xlim()
ylims = plt.gca().get_ylim()
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

plt.plot([0, 16], [8, -8], label="$F + b = 8$")

plt.scatter(F_true, b_true, zorder=100, marker="x", color="k", label="truth")

plt.title("forcing and bias estimation - bias-aware EnKF with feedback")

plt.legend()

# =============================================================================
# Figure: RMSEs
# =============================================================================

# Plot a figure of the RMSE
plt.figure()
plt.plot(
    np.arange(-T_spinup, 0), RMSE_list_spinup, label="RMSE during spinup (true values)"
)
plt.plot(np.arange(0, T), RMSE_list, label="RMSE during assimilation (inferred values)")
plt.legend()
plt.xlabel("time steps")
plt.ylabel("ensemble mean RMSE relative to synthetic truth")
