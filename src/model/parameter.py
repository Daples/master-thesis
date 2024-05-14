from dataclasses import dataclass


@dataclass
class Parameter:
    """A class to represent model parameters that are (potentially) uncertain.

    Attributes
    ----------
    init_value: float
        The parameter initial value.
    uncertainty: float
        The parameter uncertainty (standard deviation of error).
    name: str
        The parameter name/ID. Default: ""
    estimate: bool
        If the parameter should be estimated through DA. Default: False
    """

    init_value: float
    uncertainty: float
    current_value: float = 0
    name: str = ""
    estimate: bool = False

    def __post_init__(self) -> None:
        self.current_value = self.init_value
