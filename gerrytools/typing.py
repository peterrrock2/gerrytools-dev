from typing import Any, Callable, Literal, Sequence, TypeAlias, Union

from numpy.typing import NDArray

Color: TypeAlias = Union[str, tuple[int | float, int | float, int | float]]
MplCompatibleColor: TypeAlias = Union[
    str,
    tuple[str, int | float],
    tuple[int | float, int | float, int | float],
    tuple[tuple[int | float, int | float, int | float], int | float],
    tuple[int | float, int | float, int | float, int | float],
]

# Format takes in original value and currently rendered string
# and returns original value and new rendered string
CellWrapper: TypeAlias = Callable[[Any, str], tuple[Any, str]]

# Type alias for tick types in Matplotlib
TickType = Literal["major", "minor", "both"]

# Type alias for histogram
BinsType = int | Sequence[float] | str | NDArray
HistType = Literal["overlay", "stack", "weave", "outline"]
