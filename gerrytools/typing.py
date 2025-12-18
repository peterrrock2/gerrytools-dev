import re
from typing import Any, Callable, Literal, TypeAlias, Union

Color: TypeAlias = Union[str, tuple[int | float, int | float, int | float]]

# Format takes in original value and currently rendered string
# and returns original value and new rendered string
CellWrapper: TypeAlias = Callable[[Any, str], tuple[Any, str]]

# Type aliases for Matplotlib color types. Str allows for named colors and hex strings.
mplRGBColorType: TypeAlias = tuple[float, float, float] | str
mplRGBAColorType: TypeAlias = (
    str
    | tuple[float, float, float, float]
    | tuple[mplRGBColorType, float]
    | tuple[tuple[float, float, float, float], float]
)

mplColorType: TypeAlias = mplRGBColorType | mplRGBAColorType

# Type alias for tick types in Matplotlib
TickType = Literal["major", "minor", "both"]


def _check_is_hex_color(color: Any) -> bool:
    """Check if a string is a valid hex color.

    Args:
        color (str): A string to check.

    Returns:
        bool: True if the string is a valid hex color, False otherwise.
    """
    if not isinstance(color, str):
        return False

    _HEX_RE = re.compile(r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
    return bool(_HEX_RE.match(color))
