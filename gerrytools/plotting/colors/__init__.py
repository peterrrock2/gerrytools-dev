import re
from typing import Any, TypeGuard

import matplotlib.colors as mcolors

from gerrytools.plotting.colors.districtr import DISTRICTR_COLOR_DICT, districtr
from gerrytools.plotting.colors.latex import get_color_from_latex_string
from gerrytools.plotting.colors.latex_full import LATEX_COLOR_DICT
from gerrytools.plotting.colors.seaborn import flare, greens, purples, redbluecmap
from gerrytools.plotting.colors.utils import compare_palettes, preview_palette
from gerrytools.typing import Color, _check_is_hex_color, mplColorType

DEFAULT_GREY = "#5c676f"
"""
Default grey plotting color; used in histograms, violin plots, and arrows.
"""

CITIZEN_BLUE = "#4693b3"
"""
Citizen ensemble blue color; used in histograms, violin plots, and arrows. (Aka
Citizen Kane).
"""

OVERLAYS = ("gainsboro", "silver", "darkgray", "gray", "dimgrey")
"""
Overlay colors for choropleth maps.
"""


ENSEMBLE_COLORS = {
    "ensemble_smc": "#ffca5d",
    "ensemble_forest": "#00cd99",
    "ensemble_rrc": "#0099cd",
    "ensemble_revrecom": "#0099cd",
    "ensemble_recoma": "#99cd00",
    "ensemble_recomb": "#cd0099",
    "ensemble_recomc": "#9900cd",
    "ensemble_recomd": "#8dd3c7",
}
"""
A dictionary mapping ensemble abbreviations to their corresponding standard colors.

These were the colors used in the final version of the RRC paper.
"""

HEX8_PATTERN = re.compile(r"^#[0-9A-Fa-f]{8}$")
HEX8_OR_NONE_PATTERN = re.compile(r"^(#[0-9A-Fa-f]{8}|none)$", re.IGNORECASE)


def _is_mpl_unnamed_color(color_val: Any) -> TypeGuard[mplColorType]:
    if isinstance(color_val, str) and _check_is_hex_color(color_val):
        return True

    if (
        isinstance(color_val, tuple)
        and all(isinstance(c, (float, int)) for c in color_val)
        and len(color_val) in {3, 4}
        and all(0.0 <= float(c) <= 1.0 for c in color_val)
    ):
        return True

    base = "Not a color"
    alpha = -1
    if isinstance(color_val, tuple) and len(color_val) == 2:
        base, alpha = color_val

    if not isinstance(alpha, (float, int)) or not (0.0 <= float(alpha) <= 1.0):
        return False

    if isinstance(base, str) and _check_is_hex_color(base):
        return True

    if (
        isinstance(base, tuple)
        and all(isinstance(c, (float, int)) for c in base)
        and len(base) == 3
        and all(0.0 <= float(c) <= 1.0 for c in base)
    ):
        return True

    return False


def convert_color_to_hexa_or_none(color: Any) -> str:
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return "none"

    color_value = None
    if (
        isinstance(color, str) and color.lower() in mcolors.get_named_colors_mapping()
    ):  # Matplotlib named color
        color_value = mcolors.get_named_colors_mapping()[color.lower()]
        if color.lower() == "green":
            # Matplotlib "green" is very dark; use a brighter green to be compatible with latex
            color_value = "#00ff00"
    elif isinstance(color, str) and color.lower() in DISTRICTR_COLOR_DICT:  # Districtr color
        color_value = DISTRICTR_COLOR_DICT[color.lower()]
    else:
        try:
            color_value = get_color_from_latex_string(color)
        except Exception:
            standardized_color = color
            if (
                isinstance(color, tuple)
                and all(isinstance(c, (int, float)) for c in color)
                and len(color) in {3, 4}
                and all(0 <= float(c) <= 255 for c in color[:3])
            ):
                # Convert 0-255 RGB to 0-1. Preserve alpha if it is already in [0, 1].
                r, g, b = (float(color[0]), float(color[1]), float(color[2]))
                rgb = (r / 255.0, g / 255.0, b / 255.0)
                if len(color) == 3:
                    standardized_color = rgb
                else:
                    a = float(color[3])
                    if 0.0 <= a <= 1.0:
                        standardized_color = (*rgb, a)
                    else:
                        standardized_color = (*rgb, a / 255.0)

            if not _is_mpl_unnamed_color(standardized_color):
                raise ValueError(f"Unknown color value: {color!r}")

            if isinstance(standardized_color, tuple) and len(standardized_color) == 2:
                base, a = standardized_color
                rgba = mcolors.to_rgba(base, alpha=float(a))
                color_value = mcolors.to_hex(rgba, keep_alpha=True)
            else:
                color_value = mcolors.to_hex(standardized_color, keep_alpha=True)

    if color_value is None:
        raise ValueError(f"Could not convert color: {color!r}")
    return mcolors.to_hex(color_value, keep_alpha=True)


def get_all_supported_colors_dict() -> dict[str, Color]:
    return mcolors.get_named_colors_mapping() | DISTRICTR_COLOR_DICT | LATEX_COLOR_DICT


__all__ = [
    "districtr",
    "redbluecmap",
    "flare",
    "purples",
    "greens",
    "convert_color_to_hexa_or_none",
    "get_color_from_latex_string",
    "get_all_supported_colors_dict",
    "DEFAULT_GREY",
    "CITIZEN_BLUE",
    "OVERLAYS",
    "ENSEMBLE_COLORS",
    "compare_palettes",
    "preview_palette",
]
