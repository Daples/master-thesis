from dataclasses import dataclass
from numpy.random import Generator


@dataclass
class Parameter:
    """A class to represent model parameters that are (potentially) uncertain.

    Attributes
    ----------
    init_value: float
        The parameter initial value.
    current_value: float
        The parameter current value.
    uncertainty: float
        The parameter uncertainty (standard deviation of error).
    name: str
        The parameter name/ID. Default: ""
    estimate: bool
        If the parameter should be estimated through DA. Default: False
    add_noise: bool
        If noise should be added at the forecast step.
    """

    init_value: float
    uncertainty: float
    current_value: float = 0
    name: str = ""
    estimate: bool = False
    stochastic_propagation: bool = False
    stochastic_integration: bool = False

    def __post_init__(self) -> None:
        self.current_value = self.init_value

    def forward(self, generator: Generator | None = None) -> None:
        """It propagates the parameter (optionally stochastically).

        Parameters
        ----------
        generator: Generator | None, optional
            The RNG. Default: None
        """

        if generator is not None:
            self.current_value += generator.normal(loc=0, scale=self.uncertainty)
