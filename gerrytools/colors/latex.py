from __future__ import annotations

from warnings import warn

from gerrytools.colors._regex import VALID_COLOR_HEX_RE
from gerrytools.colors._sources import _resolve_named_color


def _norm_hex(s: str) -> str:
    """Normalize a hex color string to 6-digit lowercase form "#rrggbb".

    Args:
        s (str): A hex color string.

    Returns:
        str: A normalized hex color string in the form "#rrggbb"

    Raises:
        ValueError: If the input string is not a valid hex color.
    """
    s = s.strip()
    if VALID_COLOR_HEX_RE.match(s) is None:
        raise ValueError(f"Not a valid hex color: {s!r}")
    if len(s.strip("#")) == 3:
        s = "#" + "".join(c * 2 for c in s.strip("#"))
    elif len(s.strip("#")) == 4:
        warn("Ignoring alpha channel in hex color: " + s)
        s = "#" + "".join(c * 2 for c in s.strip("#")[:3])
    elif len(s.strip("#")) == 8:
        warn("Ignoring alpha channel in hex color: " + s)
        s = "#" + s.strip("#")[:6]
    elif len(s.strip("#")) == 6:
        s = "#" + s.strip("#")
    return s.lower()


def _hex_to_rgb(hex6: str) -> tuple[int, int, int]:
    """Convert a 6-digit hex color string to an RGB tuple.

    Args:
        hex6 (str): A hex color string in the form "#RRGGBB
    Returns:
        tuple[int, int, int]: A tuple of (R, G, B) values.
    """
    hex6 = _norm_hex(hex6)
    return (int(hex6[1:3], 16), int(hex6[3:5], 16), int(hex6[5:7], 16))


def _rgb_to_hex(rgb: tuple[int | float, int | float, int | float]) -> str:
    """Convert an RGB tuple to a 6-digit hex color string.

    Args:
        rgb (tuple[int, int, int]): A tuple of (R, G, B) values.

    Returns:
        str: A hex color string in the form "#RRGGBB"
    """
    r, g, b = rgb
    r = max(0, min(255, int(round(r))))
    g = max(0, min(255, int(round(g))))
    b = max(0, min(255, int(round(b))))
    return f"#{r:02x}{g:02x}{b:02x}"


def _xcolor_mix_hex(hex_colors_list: list[str], percentages_list: list[float | int]) -> str:
    """Allows for mixing of two hex colors according to xcolor semantics.


    See page 44 of the xcolor manual for details:
        https://ctan.math.washington.edu/tex-archive/macros/latex/contrib/xcolor/xcolor.pdf

    Args:
        hex_colors_list (list[str]): A list of hex color strings in the form "#RRGGBB"
        percentages_list (list[float | int]): A list of percentages (0-100) for mixing

    Returns:
        str: The resulting mixed hex color string in the form "#RRGGBB"
    """
    if len(hex_colors_list) != len(percentages_list) + 1:
        raise ValueError(
            "Number of colors must be one more than number of percentages to define the : "
            f"interpolation correctly. Found {len(hex_colors_list)} colors and "
            f"{len(percentages_list)} percentages."
        )

    r, g, b = _hex_to_rgb(hex_colors_list[0])

    for color, percent in zip(hex_colors_list[1:], percentages_list):
        p = percent / 100.0
        r_mix, g_mix, b_mix = _hex_to_rgb(color)
        r = p * r + (1 - p) * r_mix
        g = p * g + (1 - p) * g_mix
        b = p * b + (1 - p) * b_mix
    return _rgb_to_hex((r, g, b))


def get_color_from_latex_string(latex_color_string: str) -> str:
    """Resolve an xcolor-style mix expression into a hex color.

    Supported forms:
      - "name"                   (just a color name)
      - "name!p"                 == "name!p!white"
      - "name!p!other"
      - "name!p!other!q!third"   left-folded: ((name!p!other)!q!third)
      etc.

    Args:
        latex_color_string (str): The xcolor expression (e.g., "amber!10!denim").

    Returns:
        str: The resulting hex color string, in the form "#RRGGBB"
    """

    def resolve_color_name_to_hex(name: str) -> str:
        return _norm_hex(_resolve_named_color(name.strip()))

    tokens = [t.strip() for t in latex_color_string.strip().split("!") if t.strip()]

    if not tokens:
        raise ValueError("Empty color expression.")

    if len(tokens) == 1:
        return resolve_color_name_to_hex(tokens[0])

    raw_color_tokens = tokens[::2]
    color_tokens = list(map(resolve_color_name_to_hex, raw_color_tokens))
    pct_tokens = list(map(float, tokens[1::2]))
    if not all(0.0 <= p <= 100.0 for p in pct_tokens):
        raise ValueError(
            f"Percentages must be in [0,100], interpreted the following perdcentages: "
            f"{pct_tokens} in {latex_color_string!r}"
        )

    if tokens[-1] != raw_color_tokens[-1]:
        color_tokens.append("#ffffff")

    assert len(color_tokens) == len(pct_tokens) + 1

    ret = _xcolor_mix_hex(color_tokens, pct_tokens)
    return ret
