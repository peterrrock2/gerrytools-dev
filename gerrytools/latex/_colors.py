"""Shared color parsing helpers for LaTeX output."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, TypeAlias

from gerrytools.typing import Color

_HEX_COLOR_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")

LatexColorType: TypeAlias = Literal["NAME", "HTML", "rgb", "RGB"]
LatexColorValue: TypeAlias = str | tuple[float, float, float] | tuple[int, int, int]
LatexColorSpec: TypeAlias = tuple[LatexColorType, LatexColorValue]


def is_hex_color(value: str) -> bool:
    """Check whether a string is a 6-digit hex color code.

    Args:
        value (str): Candidate color string.

    Returns:
        bool: True if ``value`` is ``RRGGBB`` or ``#RRGGBB``.
    """
    return _HEX_COLOR_RE.fullmatch(value.strip()) is not None


def normalize_hex_color(value: str) -> str:
    """Normalize a hex color string.

    Args:
        value (str): Candidate color string.

    Returns:
        str: Lowercase hex color text without a leading ``#``.

    Raises:
        ValueError: If ``value`` is not a valid 6-digit hex color string.
    """
    stripped = value.strip()
    if not is_hex_color(stripped):
        raise ValueError("Color string must be a HEX string in the format '#RRGGBB' or 'RRGGBB'.")
    return stripped.lower().lstrip("#")


def to_latex_color_spec(color: Color) -> LatexColorSpec:
    """Classify a ``Color`` value into a LaTeX color mode and normalized value.

    Args:
        color (Color): Color value represented as a name, hex string, or RGB tuple.

    Returns:
        LatexColorSpec: Tuple ``(color_type, color_value)`` used by LaTeX color emitters.

    Raises:
        ValueError: If the color cannot be interpreted as a valid name/hex/RGB value.
    """
    if isinstance(color, str):
        stripped = color.strip()
        if is_hex_color(stripped):
            return ("HTML", normalize_hex_color(stripped))
        return ("NAME", stripped)

    if not isinstance(color, Sequence) or len(color) != 3:
        raise ValueError("Color must be a LaTeX color name, HEX string, or RGB tuple of length 3.")

    c0 = float(color[0])
    c1 = float(color[1])
    c2 = float(color[2])
    if all(0.0 <= c <= 1.0 for c in (c0, c1, c2)):
        return ("rgb", (c0, c1, c2))
    if all(0.0 <= c <= 255.0 for c in (c0, c1, c2)):
        return ("RGB", (int(round(c0)), int(round(c1)), int(round(c2))))

    raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")


def cellcolor_prefix(color: Color) -> str:
    """Build a ``\\cellcolor`` prefix string.

    Args:
        color (Color): Color value represented as a name, hex string, or RGB tuple.

    Returns:
        str: LaTeX snippet like ``\\cellcolor{...}``, ``\\cellcolor[HTML]{...}``, etc.
    """
    color_type, color_value = to_latex_color_spec(color)
    if color_type == "NAME":
        assert isinstance(color_value, str)
        return rf"\cellcolor{{{color_value}}}"
    if color_type == "HTML":
        assert isinstance(color_value, str)
        return rf"\cellcolor[HTML]{{{color_value}}}"
    if color_type == "rgb":
        assert isinstance(color_value, tuple)
        return (
            rf"\cellcolor[rgb]{{{color_value[0]:0.2f},{color_value[1]:0.2f},{color_value[2]:0.2f}}}"
        )
    assert isinstance(color_value, tuple)
    return rf"\cellcolor[RGB]{{{color_value[0]},{color_value[1]},{color_value[2]}}}"
