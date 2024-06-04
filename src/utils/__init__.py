from numpy.linalg import inv
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
import numpy as np

default_generator = default_rng(123456789)


def get_generator() -> Generator:
    """It returns a default RNG.

    Returns
    -------
    Generator
        The RNG.
    """

    return default_generator


def block_diag(blocks: list[NDArray]) -> NDArray:
    """Create a block diagonal matrix from the input square matrices.

    Parameters
    ----------
    blocks: list[NDArray]
        The collection of square matrices to append.

    Returns
    -------
    NDArray
        The resulting block diagonal matrix.
    """

    def join(upper_block: NDArray, lower_block: NDArray) -> NDArray:
        """Join two square matrices diagonally.

        Parameters
        ----------
        upper_block: NDArray
            The first diagonal block.
        lower_block: NDArray
            The last diagonal block.

        Returns
        -------
        NDArray
            The augmented block diagonal matrix.
        """

        aux = np.zeros((upper_block.shape[0], lower_block.shape[1]))
        if upper_block.size == 0:
            return np.block([aux.T, lower_block])
        if lower_block.size == 0:
            return np.block([upper_block, aux])
        return np.block([[upper_block, aux], [aux.T, lower_block]])

    matrix = blocks[0]
    for block in blocks[1:]:
        matrix = join(matrix, block)
    return matrix


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
