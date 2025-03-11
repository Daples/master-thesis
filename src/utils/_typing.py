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
InputSystem: TypeAlias = Callable[[float, State, Input], NDArray]

Integrator: TypeAlias = Callable[
    [SystemDynamics, State, Time, Time, float, InputFunction],
    tuple[NDArray, NDArray],
]
DynamicMatrix: TypeAlias = Callable[[float], NDArray]
