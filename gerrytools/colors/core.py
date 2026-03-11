import logging
import math
from numbers import Real
from typing import Literal, TypeAlias, TypeGuard, cast

import matplotlib.colors as mcolors

from gerrytools.colors._regex import HEX8_OR_NONE_PATTERN
from gerrytools.colors.districtr import DISTRICTR_COLOR_DICT
from gerrytools.colors.latex import get_color_from_latex_string
from gerrytools.colors.latex_full import LATEX_COLOR_DICT
from gerrytools.logging import get_logger
from gerrytools.typing import Color, HexColor, MplBaseColor, MplCompatibleColor, ResolvedColor

gt_logger = get_logger(__name__)
_MplBaseColorWithAlpha: TypeAlias = tuple[MplBaseColor, Real]

DEFAULT_GREY = "#5c676f"
"""
Default grey plotting color; used in histograms, violin plots, and arrows.
"""

CITIZEN_BLUE = "#4693b3"
"""
Citizen ensemble blue color; used in histograms, violin plots, and arrows.
"""

OVERLAYS = ("gainsboro", "silver", "darkgray", "gray", "dimgrey")
"""
Overlay colors for choropleth maps.
"""


ENSEMBLE_COLORS = {
    "ensemble:smc": "#ffca5d",
    "ensemble:forest": "#00cd99",
    "ensemble:rrc": "#0099cd",
    "ensemble:revrecom": "#0099cd",
    "ensemble:recom": "#0099cd",
    "ensemble:recoma": "#99cd00",
    "ensemble:recomb": "#cd0099",
    "ensemble:recomc": "#9900cd",
    "ensemble:recomd": "#8dd3c7",
}
"""
A dictionary mapping ensemble abbreviations to their corresponding standard colors.

These were the colors used in the final version of the RRC paper.
"""

COLOR_CORECTED_BASESET = {
    "cc:applegreen": "#73b900",
    "cc:denim": "#0064bd",
    "cc:cherryblossompink": "#ffb0c5",
    "cc:darktangerine": "#ff9f0f",
    "cc:cadmiumgreen": "#006f3c",
    "cc:purpleheart": "#872f9c",
    "cc:alizarin": "#d91b00",
    "cc:greenishcyan": "#009983",
    "cc:lightblue": "#92dbe6",
    "cc:amber": "#ffb900",
    "cc:muddy": "#9b3200",
    "cc:lostinspace": "#003e64",
    "cc:teagreen": "#d0f0c0",
}
"""
A small set of colors from that are color-corrected for better visibility by color-blind users.
"""


GERRYTOOLS_EXTRA_COLORS_DICT = (
    {
        "default_grey": DEFAULT_GREY,
        "default_gray": DEFAULT_GREY,
        "citizen_blue": CITIZEN_BLUE,
    }
    | {name: mcolors.to_hex(name) for name in OVERLAYS}
    | ENSEMBLE_COLORS
)


def get_all_supported_colors_dict() -> dict[str, Color]:
    """Get a dictionary of all supported color names mapping to their values."""
    result: dict[str, Color] = cast(dict[str, Color], dict(mcolors.get_named_colors_mapping()))
    result.update(DISTRICTR_COLOR_DICT)
    result.update(LATEX_COLOR_DICT)
    result.update(GERRYTOOLS_EXTRA_COLORS_DICT)
    result["green"] = "#00ff00"  # Override matplotlib's dark "green"
    result["none"] = "none"
    return result


def get_named_color(name: str) -> Color:
    """Get a color value from the supported color names.

    Args:
        name (str): The name of the color.

    Returns:
        Color: The corresponding color value.

    Raises:
        KeyError: If the color name is not recognized.
    """
    key = name.lower()

    if key == "green":
        return "#00ff00"

    # try “as-is” and lowercased in each mapping
    for color_map in (GERRYTOOLS_EXTRA_COLORS_DICT, DISTRICTR_COLOR_DICT, LATEX_COLOR_DICT):
        value = color_map.get(name)
        if value is not None:
            return value
        value = color_map.get(key)
        if value is not None:
            return value

    mpl_named_colors = mcolors.get_named_colors_mapping()
    value = mpl_named_colors.get(name)
    if value is not None:
        return mcolors.to_hex(value)
    value = mpl_named_colors.get(key)
    if value is not None:
        return mcolors.to_hex(value)

    raise KeyError(f"Unknown color name: {name!r}")


def _is_real(x: object) -> TypeGuard[Real]:
    return (
        isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(float(x))
    )  # filters NaN too


def _is_rgb_tuple(x: object) -> TypeGuard[tuple[int | float, int | float, int | float]]:
    return isinstance(x, tuple) and len(x) == 3 and all(_is_real(v) for v in x)


def _is_rgba_tuple(
    x: object,
) -> TypeGuard[tuple[int | float, int | float, int | float, int | float]]:
    return isinstance(x, tuple) and len(x) == 4 and all(_is_real(v) for v in x)


def _is_mpl_base_color(x: object) -> TypeGuard[MplBaseColor]:
    return isinstance(x, str) or _is_rgb_tuple(x) or _is_rgba_tuple(x)


def _is_mpl_base_color_with_alpha(x: object) -> TypeGuard[_MplBaseColorWithAlpha]:
    return isinstance(x, tuple) and len(x) == 2 and _is_mpl_base_color(x[0]) and _is_real(x[1])


def convert_color_to_hexa_or_none(
    color: MplCompatibleColor | None,
) -> HexColor | Literal["none"]:
    """Convert a color input to a hex8 string or "none".

    Args:
        color (MplCompatibleColor | None): The color input to convert. This can be a named color,
            a LaTeX color string, an RGB(A) tuple, or other formats supported
            by Matplotlib.
    """
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return "none"

    # ---- strings: named / latex / plain mpl / hex ----
    log_string = ""
    resolved_color: MplBaseColor | None = None
    if isinstance(color, str):
        try:
            resolved_color = get_named_color(color)
        except KeyError as e:
            log_string += f"Color {color!r} is not a known Matplotlib named color string: {e}"

        # LaTeX/xcolor string support
        if resolved_color is None:
            try:
                resolved_color = get_color_from_latex_string(color)
            except Exception as e:
                log_string += f" | Color {color!r} is not parsable as a LaTeX color string: {e}"

        # Generic matplotlib parsing (also covers '#RRGGBB' and '#RRGGBBAA')
        if resolved_color is None:
            try:
                resolved_color = mcolors.to_rgba(color)
            except Exception as e:
                log_string += f" | Color {color!r} not parseable by Matplotlib: {e}"

        if resolved_color is None:
            gt_logger.debug(log_string)

    # ---- (base, alpha) tuples ----
    elif _is_mpl_base_color_with_alpha(color):
        base, a = color  # type narrowed by _is_mpl_base_color_with_alpha
        alpha_value = _validate_alpha(float(a), field="alpha in (base, alpha) color tuple")
        base_hex8 = convert_color_to_hexa_or_none(base)
        if base_hex8.lower() == "none":
            resolved_color = (0.0, 0.0, 0.0, 0.0)
        else:
            resolved_color = mcolors.to_rgba(base_hex8[:7], alpha=alpha_value)

    # ---- numeric RGB(A) tuples (0–1 or 0–255) ----
    elif _is_rgba_tuple(color):
        rgba = color  # type narrowed by _is_rgba_tuple
        vals: list[float] = [float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])]
        r, g, b = vals[:3]

        a_raw = vals[3]
        if a_raw < 0:
            raise ValueError(f"Alpha must be non-negative: {color!r}")

        # if interpreting 0–255, enforce bounds
        if max(r, g, b) > 1.0:
            if any(v < 0.0 for v in (r, g, b)):
                raise ValueError(f"RGB values must be non-negative: {color!r}")
            if max(r, g, b) > 255.0:
                raise ValueError(f"RGB values must be <=255 when using 0-255 scale: {color!r}")
            if max(r, g, b) < 2.0:
                raise ValueError(
                    f"Ambiguous RGB tuple {color!r}: values >1 but <2; "
                    "use 0–1 floats or 0–255 ints."
                )

            r, g, b = r / 255.0, g / 255.0, b / 255.0

        if a_raw > 1.0:
            if a_raw > 255.0:
                raise ValueError(f"Alpha must be <=255 when using 0-255 scale: {color!r}")
            a = a_raw / 255.0
        else:
            a = a_raw

        if not (0.0 <= a <= 1.0):
            raise ValueError(f"Alpha must be in [0,1]: {color!r}")

        resolved_color = (r, g, b, a)
    elif _is_rgb_tuple(color):
        rgb = color  # type narrowed by _is_rgb_tuple
        vals: list[float] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
        r, g, b = vals[:3]
        a = 1.0

        # if interpreting 0–255, enforce bounds
        if max(r, g, b) > 1.0:
            if any(v < 0.0 for v in (r, g, b)):
                raise ValueError(f"RGB values must be non-negative: {color!r}")
            if max(r, g, b) > 255.0:
                raise ValueError(f"RGB values must be <=255 when using 0-255 scale: {color!r}")
            if max(r, g, b) < 2.0:
                raise ValueError(
                    f"Ambiguous RGB tuple {color!r}: values >1 but <2; "
                    "use 0–1 floats or 0–255 ints."
                )

            r, g, b = r / 255.0, g / 255.0, b / 255.0

        resolved_color = (r, g, b, a)

    if resolved_color is None:
        raise ValueError(f"Unknown color value: {color!r}")

    return mcolors.to_hex(mcolors.to_rgba(resolved_color), keep_alpha=True)


def _validate_alpha(alpha: float, *, field: str = "alpha") -> float:
    """Validate that alpha is convertible to a float and between 0.0 and 1.0."""
    a = float(alpha)
    if not math.isfinite(a):
        raise ValueError(f"{field} must be finite")
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"{field} must be between 0.0 and 1.0")
    return a


def resolve_color_and_alpha(
    color: MplCompatibleColor | None,
    alpha: float | None = None,
    *,
    allow_none: bool = True,
    field: str = "color",
    owner: str | None = None,
    logger: logging.Logger | None = None,
) -> ResolvedColor:
    """Normalize a (color, alpha) pair into (hex6_or_none, resolved_alpha).

    Rules:
    - color is converted via convert_color_to_hexa_or_none -> "none" or "#RRGGBBAA"
    - if color resolves to "none":
        - allow_none=True  => ("none", 0.0)  (alpha is forced to 0)
        - allow_none=False => ValueError
    - if color resolves to "#RRGGBBAA":
        - returns ("#RRGGBB", alpha_from_color) if alpha is None
        - returns ("#RRGGBB", alpha) if alpha is provided (validated),
          optionally logs when it overrides embedded alpha.


    Args:
        color (MplCompatibleColor | None): The color input to convert.
        alpha (float | None): An optional explicit alpha value between 0.0 and 1.0.
        allow_none (bool): Whether "none" is an acceptable color. Defaults to True.
        field (str): The name of the field being processed, for error messages.
        owner (str | None): An optional owner name for logging context.
        logger (logging.Logger | None): An optional logger for debug messages.

    Returns:
        ResolvedColor: A tuple of (hex6_or_none, resolved_alpha).
    """
    hex8 = convert_color_to_hexa_or_none(color)
    if HEX8_OR_NONE_PATTERN.match(hex8) is None:
        raise ValueError(f"{field} could not be converted to a valid color: {color!r} -> {hex8!r}")

    if hex8.lower() == "none":
        if not allow_none:
            raise ValueError(f"{field} cannot be 'none'.")
        return "none", 0.0

    hex6 = hex8[:7]
    alpha_from_color = int(hex8[7:], 16) / 255.0

    if alpha is None:
        return hex6, alpha_from_color

    a = _validate_alpha(alpha, field=f"{field} alpha")

    if logger is not None and not math.isclose(a, alpha_from_color, abs_tol=1e-4):
        prefix = f"For {owner}: " if owner else ""
        logger.log(
            level=logging.DEBUG,
            msg=(
                f"{prefix}Ignoring alpha embedded in {field} {hex8!r} "
                f"because explicit alpha {a} was provided."
            ),
        )

    return hex6, a
