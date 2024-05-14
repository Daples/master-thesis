from typing import Callable, TypeAlias, TypeVar, ParamSpec

from numpy.typing import NDArray

T = TypeVar("T")
P = ParamSpec("P")
Integrator: TypeAlias = Callable[
    [Callable[[float, NDArray], NDArray], NDArray, float, float, float],
    tuple[NDArray, NDArray],
]
DynamicMatrix: TypeAlias = Callable[[float], NDArray]
