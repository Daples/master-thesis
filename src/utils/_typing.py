from typing import Callable, TypeAlias, TypeVar, ParamSpec

from numpy.typing import NDArray

T = TypeVar("T")
P = ParamSpec("P")

# State types
State: TypeAlias = NDArray
Input: TypeAlias = NDArray
Observation: TypeAlias = NDArray
Time: TypeAlias = float
InputFunction: TypeAlias = Callable[[float, State], Input]
SystemDynamics: TypeAlias = Callable[[float, State], NDArray]

Integrator: TypeAlias = Callable[
    [SystemDynamics, NDArray, float, float, float],
    tuple[NDArray, NDArray],
]
DynamicMatrix: TypeAlias = Callable[[float], NDArray]
