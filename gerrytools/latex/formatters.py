from __future__ import annotations

from typing import TYPE_CHECKING, Any
from numbers import Real
import re

if TYPE_CHECKING:
    from gerrytools.latex.table import CellWrapper
    from gerrytools.latex.document import ColorLike


def wrap_with_tex_command(cmd: str) -> CellWrapper:
    """Wraps cell content with a LaTeX command.

    Indtended for use in the set_col_formatter method of TexTable.

    Args:
        cmd (str): The LaTeX command to wrap around the cell content
            (e.g., "textbf" for bold text).

    Returns:
        CellFormatter: A function that formats cell content with the specified command.
    """

    def _inner(cell_value: Any, rendered_str: str) -> tuple[Any, str]:
        return cell_value, rf"\{cmd}{{{rendered_str}}}"

    return _inner


def compose_formatters(*funcs: CellWrapper) -> CellWrapper:
    """Composes multiple formatter functions into a single formatter.

    Note: Each of the formatting functions is expected to adhere to the
    CellWrapper signature, taking in the original value and the currently
    rendered string, and returning the original value and the newly rendered string.

    Args:
        *funcs (CellWrapper): Formatter functions to compose (an arbitrary number
        of such functions can be provided).

    Returns:
        CellWrapper: A single formatter function that applies all provided
    """

    # compose2(f, g, h)(v, s) == f(g(h(v, s)))
    def run(v: str | Real, s: str) -> tuple[str | Real, str]:
        for f in reversed(funcs):
            v, s = f(v, s)
        return v, s

    return run


def round_decimals(decimal_places: int) -> CellWrapper:
    """Generates a formatter that rounds numerical values to a specified number of decimal places.

    Args:
        decimal_places (int): The number of decimal places to round to.

    Returns:
        CellWrapper: A function that rounds numerical values to the specified.
    """

    def _inner_round_decimals(v: Any, s: str) -> tuple[Any, str]:
        if isinstance(v, Real):
            return v, f"{v:.{decimal_places}f}"
        return v, s

    return _inner_round_decimals


def highlight_gt(thresh: Real, color: ColorLike = "yellow") -> CellWrapper:
    """Generates a formatter that highlights numerical values greater than a threshold.

    Args:
        thresh (float): The threshold value.
        color (ColorLike): The LaTeX color name string or RGB tuple or hex string to use for
            highlighting. Default is "yellow".

    Returns:
        CellWrapper: A function that highlights numerical values greater than the threshold.
    """

    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.upper().lstrip("#")

            def _inner_highlight_gt(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v > thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_gt

        def _inner_highlight_gt(v: Real, s: str) -> tuple[Real, str]:
            if isinstance(v, Real) and v > thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_gt

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_gt(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v > thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_gt
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_gt(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v > thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_gt
        else:
            raise ValueError(
                "RGB color components must be in the range [0.0, 1.0] or [0, 255]."
            )


def highlight_ge(thresh: Real, color: ColorLike = "yellow") -> CellWrapper:
    """Generates a formatter that highlights numerical values greater than or equal to a threshold.

    Args:
        thresh (float): The threshold value.
        color (ColorLike): The LaTeX color name string or RGB tuple or hex string to use for
            highlighting. Default is "yellow".

    Returns:
        CellWrapper: A function that highlights numerical values greater than the threshold.
    """

    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.upper().lstrip("#")

            def _inner_highlight_ge(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v >= thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_ge

        def _inner_highlight_ge(v: Real, s: str) -> tuple[Real, str]:
            if isinstance(v, Real) and v >= thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_ge

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_ge(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v >= thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_ge
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_ge(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v >= thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_ge
        else:
            raise ValueError(
                "RGB color components must be in the range [0.0, 1.0] or [0, 255]."
            )


def highlight_lt(thresh: float, color: ColorLike = "yellow") -> CellWrapper:
    """Generates a formatter that highlights numerical values less than a threshold.

    Args:
        thresh (float): The threshold value.
        color (str): The LaTeX color name string to use for highlighting. Default is "yellow".

    Returns:
        CellWrapper: A function that highlights numerical values less than the threshold.
    """
    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.upper().lstrip("#")

            def _inner_highlight_lt(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v < thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_lt

        def _inner_highlight_lt(v: Real, s: str) -> tuple[Real, str]:
            if isinstance(v, Real) and v < thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_lt

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_lt(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v < thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_lt
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_lt(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v < thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_lt
        else:
            raise ValueError(
                "RGB color components must be in the range [0.0, 1.0] or [0, 255]."
            )


def highlight_le(thresh: float, color: ColorLike = "yellow") -> CellWrapper:
    """Generates a formatter that highlights numerical values less than or equal to a threshold.

    Args:
        thresh (float): The threshold value.
        color (str): The LaTeX color name string to use for highlighting. Default is "yellow".

    Returns:
        CellWrapper: A function that highlights numerical values less than the threshold.
    """
    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.upper().lstrip("#")

            def _inner_highlight_le(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v <= thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_le

        def _inner_highlight_le(v: Real, s: str) -> tuple[Real, str]:
            if isinstance(v, Real) and v <= thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_le

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_le(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v <= thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_le
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_le(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and v <= thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_le
        else:
            raise ValueError(
                "RGB color components must be in the range [0.0, 1.0] or [0, 255]."
            )


def highlight_between(
    lower_bound: Real, upper_bound: Real, color: ColorLike = "yellow"
) -> CellWrapper:
    """Generates a formatter that highlights numerical values between two bounds (inclusive).

    Args:
        lower_bound (Real): The lower bound.
        upper_bound (Real): The upper bound.
        color (str): The LaTeX color name string to use for highlighting. Default is "yellow".

    Returns:
        CellWrapper: A function that highlights numerical values between the bounds.
    """
    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.upper().lstrip("#")

            def _inner_highlight_btwn(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and lower_bound <= v <= upper_bound:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_btwn

        def _inner_highlight_btwn(v: Real, s: str) -> tuple[Real, str]:
            if isinstance(v, Real) and lower_bound <= v <= upper_bound:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_btwn

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_btwn(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and lower_bound <= v <= upper_bound:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_btwn
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_btwn(v: Real, s: str) -> tuple[Real, str]:
                if isinstance(v, Real) and lower_bound <= v <= upper_bound:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_btwn
        else:
            raise ValueError(
                "RGB color components must be in the range [0.0, 1.0] or [0, 255]."
            )


def _consume_balanced(s: str, i: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    if i >= len(s) or s[i] != open_ch:
        raise ValueError(f"Expected '{open_ch}' at position {i}")
    depth = 1
    i += 1
    start = i

    while i < len(s) and depth:
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
        i += 1

    # Failed to find a matching closing character
    if depth != 0:
        raise ValueError(f"Unbalanced {open_ch}{close_ch} in format string")

    return s[start : i - 1], i


def _parse_tabular_preamble(fmt: str):
    i, n = 0, len(fmt)
    colspecs: list[str] = []
    vrules: list[int] = [0]
    extras: list[str] = [""]

    def skip_ws(i: int) -> int:
        while i < n and fmt[i].isspace():
            i += 1
        return i

    SIMPLE_COLS = set("lcr")

    while True:
        i = skip_ws(i)
        if i >= n:
            break

        start_i = i
        ch = fmt[i]

        # reject stray grouping tokens early (prevents hangs and weird acceptance)
        if ch in "{}[]":
            raise ValueError(f"Stray {ch!r} at pos {i} in preamble: {fmt!r}")

        # vertical rules
        if ch == "|":
            vrules[-1] += 1
            i += 1
            continue

        # boundary extras: @{}  !{}  >{}  <{}
        if ch in ("@", "!", ">", "<"):
            if i + 1 >= n or fmt[i + 1] != "{":
                raise ValueError(
                    f"Expected '{{' after {ch} at pos {i} in preamble: {fmt!r}"
                )
            grp, i = _consume_balanced(fmt, i + 1, "{", "}")
            extras[-1] += f"{ch}{{{grp}}}"
            continue

        # p{..}, m{..}, b{..}
        if ch in ("p", "m", "b"):
            if i + 1 >= n or fmt[i + 1] != "{":
                raise ValueError(
                    f"Expected '{{' after {ch} at pos {i} in preamble: {fmt!r}"
                )
            tok, i = _consume_balanced(fmt, i + 1, "{", "}")
            colspecs.append(f"{ch}{{{tok}}}")
            vrules.append(0)
            extras.append("")
            continue

        # S or S[...]
        if ch == "S":
            i += 1
            tok = "S"
            i = skip_ws(i)
            if i < n and fmt[i] == "[":
                grp, i = _consume_balanced(fmt, i, "[", "]")
                tok += f"[{grp}]"
            colspecs.append(tok)
            vrules.append(0)
            extras.append("")
            continue

        # D{in}{out}{places}
        if ch == "D":
            i += 1
            i = skip_ws(i)
            if i >= n or fmt[i] != "{":
                raise ValueError(
                    f"Expected '{{' after D at pos {i} in preamble: {fmt!r}"
                )
            g1, i = _consume_balanced(fmt, i, "{", "}")
            i = skip_ws(i)
            g2, i = _consume_balanced(fmt, i, "{", "}")
            i = skip_ws(i)
            g3, i = _consume_balanced(fmt, i, "{", "}")
            colspecs.append(f"D{{{g1}}}{{{g2}}}{{{g3}}}")
            vrules.append(0)
            extras.append("")
            continue

        # runs of simple one-letter columns, e.g. "ccr"
        if ch in SIMPLE_COLS or ch == "c":
            while True:
                i = skip_ws(i)
                if i >= n:
                    break
                if fmt[i] == "c" or fmt[i] in SIMPLE_COLS:
                    colspecs.append(fmt[i])
                    vrules.append(0)
                    extras.append("")
                    i += 1
                    continue
                break
            continue

        # if we get here, we don't recognize the token
        if i == start_i:
            raise ValueError(
                f"Unsupported token {fmt[i]!r} at pos {i} in preamble: {fmt!r}"
            )

    return colspecs, vrules, extras


def _convert_hex_string_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Converts a hex color string to an RGB tuple.

    Args:
        hex_str (str): The hex color string (e.g., "FF5733" or "#FF5733").

    Returns:
        tuple[float, float, float]: The RGB color components as floats in the range [0.0, 1.0].
    """
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        raise ValueError("Hex color string must be 6 characters long.")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)
