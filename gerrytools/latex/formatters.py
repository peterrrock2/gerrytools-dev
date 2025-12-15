from __future__ import annotations

import math
import re
from typing import Any

from gerrytools.latex.commands import _validate_command_name
from gerrytools.typing import CellWrapper, Color


def wrap_with_tex_command(cmd_str: str) -> CellWrapper:
    """Wrap cell content with a LaTeX command like \\textbf{...}.

    Validates that `cmd` is a legal LaTeX command name (no leading backslash,
    no underscores, braces, spaces, etc.).
    """
    _validate_command_name(cmd_str)

    def _inner(cell_value: Any, rendered_str: str) -> tuple[Any, str]:
        return cell_value, rf"\{cmd_str}{{{rendered_str}}}"

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
    def run(v: str | int | float, s: str) -> tuple[str | int | float, str]:
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
        if isinstance(v, int | float):
            return v, f"{v:.{decimal_places}f}"
        return v, s

    return _inner_round_decimals


def _safe_round(v: Any, round_to: int | None) -> Any:  # pragma: no cover
    """Rounds a numerical value to a specified number of decimal places, handling special cases.

    Args:
        v (Any): The numerical value to round.
        round_to (int): The number of decimal places to round to.

    Returns:
        str: The rounded value as a string, or special strings for NaN and infinity.
    """
    if isinstance(v, int | float) and round_to is not None:
        assert isinstance(round_to, int), "value of `round_to` must be an integer"
        if v != v:  # NaN check
            return v
        if v == float("inf"):
            return v
        if v == float("-inf"):
            return v
        return float(round(v, round_to))
    return v


def highlight_gt(
    thresh: int | float, color: Color = "yellow", *, round_to: int | None = None
) -> CellWrapper:
    """Generates a formatter that highlights numerical values greater than a threshold.

    Args:
        thresh (float): The threshold value.
        color (Color): The LaTeX color name string or RGB tuple or hex string to use for
            highlighting. Default is "yellow".

    Kwargs:
        round_to (int | None): If provided, numerical values will be rounded to this
            number of decimal places before comparison.

    Returns:
        CellWrapper: A function that highlights numerical values greater than the threshold.
    """

    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.lower().lstrip("#")

            def _inner_highlight_gt(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) > thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_gt

        def _inner_highlight_gt(v: int | float, s: str) -> tuple[int | float, str]:
            if isinstance(v, int | float) and _safe_round(v, round_to) > thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_gt

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_gt(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) > thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_gt
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_gt(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) > thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_gt
        else:  # pragma: no cover
            raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")


def highlight_ge(
    thresh: int | float, color: Color = "yellow", *, round_to: int | None = None
) -> CellWrapper:
    """Generates a formatter that highlights numerical values greater than or equal to a threshold.

    Args:
        thresh (float): The threshold value.
        color (Color): The LaTeX color name string or RGB tuple or hex string to use for
            highlighting. Default is "yellow".

    Kwargs:
        round_to (int | None): If provided, numerical values will be rounded to this
            number of decimal places before comparison.

    Returns:
        CellWrapper: A function that highlights numerical values greater than the threshold.
    """
    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.lower().lstrip("#")

            def _inner_highlight_ge(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) >= thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_ge

        def _inner_highlight_ge(v: int | float, s: str) -> tuple[int | float, str]:
            if isinstance(v, int | float) and _safe_round(v, round_to) >= thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_ge

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_ge(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) >= thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_ge
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_ge(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) >= thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_ge
        else:  # pragma: no cover
            raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")


def highlight_lt(
    thresh: float, color: Color = "yellow", *, round_to: int | None = None
) -> CellWrapper:
    """Generates a formatter that highlights numerical values less than a threshold.

    Args:
        thresh (float): The threshold value.
        color (str): The LaTeX color name string to use for highlighting. Default is "yellow".

    Kwargs
        round_to (int | None): If provided, numerical values will be rounded to this
            number of decimal places before comparison.

    Returns:
        CellWrapper: A function that highlights numerical values less than the threshold.
    """
    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.lower().lstrip("#")

            def _inner_highlight_lt(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) < thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_lt

        def _inner_highlight_lt(v: int | float, s: str) -> tuple[int | float, str]:
            if isinstance(v, int | float) and _safe_round(v, round_to) < thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_lt

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_lt(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) < thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_lt
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_lt(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) < thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_lt
        else:  # pragma: no cover
            raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")


def highlight_le(
    thresh: float, color: Color = "yellow", *, round_to: int | None = None
) -> CellWrapper:
    """Generates a formatter that highlights numerical values less than or equal to a threshold.

    Args:
        thresh (float): The threshold value.
        color (str): The LaTeX color name string to use for highlighting. Default is "yellow".

    Kwargs:
        round_to (int | None): If provided, numerical values will be rounded to this
            number of decimal places before comparison.

    Returns:
        CellWrapper: A function that highlights numerical values less than the threshold.
    """
    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.lower().lstrip("#")

            def _inner_highlight_le(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) <= thresh:
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_le

        def _inner_highlight_le(v: int | float, s: str) -> tuple[int | float, str]:
            if isinstance(v, int | float) and _safe_round(v, round_to) <= thresh:
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_le

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_le(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) <= thresh:
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_le
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_le(v: int | float, s: str) -> tuple[int | float, str]:
                if isinstance(v, int | float) and _safe_round(v, round_to) <= thresh:
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_le
        else:  # pragma: no cover
            raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")


def highlight_between(
    lower_bound: int | float,
    upper_bound: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    include_bounds: bool = True,
) -> CellWrapper:
    """Generates a formatter that highlights numerical values between two bounds.

    Args:
        lower_bound (int | float): The lower bound.
        upper_bound (int | float): The upper bound.
        color (str): The LaTeX color name string to use for highlighting. Default is "yellow".

    Kwargs:
        round_to (int | None): If provided, numerical values will be rounded to this
            number of decimal places before comparison.
        include_bounds (bool): If True, values equal to the bounds are included. Default is True.

    Returns:
        CellWrapper: A function that highlights numerical values between the bounds.
    """
    if not include_bounds:
        lower_bound = math.nextafter(
            float(lower_bound), math.inf
        )  # smallest float > original lower_bound
        upper_bound = math.nextafter(
            float(upper_bound), -math.inf
        )  # largest float < original upper_bound

    if isinstance(color, str):
        if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
            # hex string
            color_hex = color.lower().lstrip("#")

            def _inner_highlight_btwn(v: int | float, s: str) -> tuple[int | float, str]:
                if (
                    isinstance(v, int | float)
                    and lower_bound <= _safe_round(v, round_to) <= upper_bound
                ):
                    return v, rf"\cellcolor[HTML]{{{color_hex}}}{s}"
                return v, s

            return _inner_highlight_btwn

        def _inner_highlight_btwn(v: int | float, s: str) -> tuple[int | float, str]:
            if (
                isinstance(v, int | float)
                and lower_bound <= _safe_round(v, round_to) <= upper_bound
            ):
                return v, rf"\cellcolor{{{color}}}{s}"
            return v, s

        return _inner_highlight_btwn

    if isinstance(color, tuple) and len(color) == 3:
        if all(0.0 <= c <= 1.0 for c in color):  # type: ignore

            def _inner_highlight_btwn(v: int | float, s: str) -> tuple[int | float, str]:
                if (
                    isinstance(v, int | float)
                    and lower_bound <= _safe_round(v, round_to) <= upper_bound
                ):
                    return (
                        v,
                        rf"\cellcolor[rgb]{{{color[0]:0.2f},{color[1]:0.2f},{color[2]:0.2f}}}{s}",
                    )
                return v, s

            return _inner_highlight_btwn
        elif all(0 <= c <= 255 for c in color):  # type: ignore

            def _inner_highlight_btwn(v: int | float, s: str) -> tuple[int | float, str]:
                if (
                    isinstance(v, int | float)
                    and lower_bound <= _safe_round(v, round_to) <= upper_bound
                ):
                    return (
                        v,
                        rf"\cellcolor[RGB]{{{int(color[0])},{int(color[1])},{int(color[2])}}}{s}",
                    )
                return v, s

            return _inner_highlight_btwn
        else:  # pragma: no cover
            raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")
