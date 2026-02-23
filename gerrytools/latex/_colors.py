"""Shared color parsing helpers for LaTeX output."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, TypeAlias

from gerrytools.colors import LATEX_COLOR_DICT, convert_color_to_hexa_or_none
from gerrytools.typing import Color

_HEX_COLOR_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")
_LATEX_COLOR_NAMES_LOWER = {name.strip().lower() for name in LATEX_COLOR_DICT}

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


def is_latex_color_expression(value: str) -> bool:
    """Check whether a string is a valid xcolor expression using known LaTeX names.

    Accepted forms are:
    - ``name``
    - ``name!p!name!p!...`` where ``p`` is a percentage in ``[0, 100]``.

    Args:
        value (str): Candidate xcolor expression.

    Returns:
        bool: True if ``value`` parses as a supported xcolor expression and all
            color-name components appear in ``LATEX_COLOR_DICT``.
    """
    color_expr = value.strip()
    if len(color_expr) == 0:
        return False

    tokens = [token.strip() for token in color_expr.split("!") if len(token.strip()) > 0]
    if len(tokens) == 0:
        return False

    for idx, token in enumerate(tokens):
        if idx % 2 == 0:
            if token.lower() not in _LATEX_COLOR_NAMES_LOWER:
                return False
        else:
            try:
                pct = float(token)
            except ValueError:
                return False
            if not (0.0 <= pct <= 100.0):
                return False

    return True


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


def to_latex_xcolor_or_html_spec(color: Color) -> LatexColorSpec:
    """Classify a color value for xcolor usage with expression preservation.

    Rules:
    - Valid xcolor expressions (``name`` or ``name!p!name!...``) are preserved as ``("NAME", ...)``.
    - Hex strings are normalized as ``("HTML", "rrggbb")``.
    - Other string colors are converted via GerryTools/Matplotlib parsing and emitted as
      ``("HTML", "rrggbb")``.
    - RGB tuples retain ``"rgb"``/``"RGB"`` behavior from :func:`to_latex_color_spec`.

    Args:
        color (Color): Color value represented as a name, xcolor expression, hex string,
            or RGB tuple.

    Returns:
        LatexColorSpec: Tuple ``(color_type, color_value)`` suitable for xcolor emitters.

    Raises:
        ValueError: If ``color`` cannot be parsed as a supported color value.
    """
    if isinstance(color, str):
        stripped = color.strip()
        if is_latex_color_expression(stripped):
            return ("NAME", stripped)
        if is_hex_color(stripped):
            return ("HTML", normalize_hex_color(stripped))

        hex8_or_none = convert_color_to_hexa_or_none(stripped)
        if hex8_or_none.lower() == "none":
            return ("NAME", "none")
        return ("HTML", hex8_or_none.lstrip("#")[:6].lower())

    return to_latex_color_spec(color)


def cellcolor_prefix(color: Color) -> str:
    """Build a ``\\cellcolor`` prefix string.

    Args:
        color (Color): Color value represented as an xcolor expression, hex string,
            RGB tuple, or other parseable color name.

    Returns:
        str: LaTeX snippet like ``\\cellcolor{...}``, ``\\cellcolor[HTML]{...}``, etc.
    """
    color_type, color_value = to_latex_xcolor_or_html_spec(color)
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
