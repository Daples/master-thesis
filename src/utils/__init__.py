# get RMSE
from numpy.random import default_rng
from numpy.typing import NDArray
from numpy.linalg import inv

default_generator = default_rng(123456789)


def kalman_gain(P: NDArray, H: NDArray, R: NDArray) -> NDArray:
    """It computes the Kalman gain from the input matrices.

    Parameters
    ----------
    P: NDArray
        The forecast state covariance matrix.
    H: NDArray
        The observation model.
    R: NDArray
        The observation error covariance matrix.

    Returns
    -------
    NDArray
        The Kalman gain correction matrix.
    """

    return (P @ H.T) @ inv(H @ P @ H.T + R)