from typing import Callable

from numpy.typing import NDArray

Integrator = Callable[
    [Callable[[float, NDArray], NDArray], NDArray, float, float, float],
    tuple[NDArray, NDArray],
]
DynamicMatrix = Callable[[float], NDArray]
