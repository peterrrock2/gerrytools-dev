from __future__ import annotations

import math
from numbers import Real
from typing import Callable

from gerrytools.latex._colors import cellcolor_prefix
from gerrytools.latex.commands import validate_command_name
from gerrytools.typing import CellWrapper, Color, TableCellValue


def boxed_center(width: int, height: int | None = None, unit: str = "mm") -> CellWrapper:
    """Create a formatter that centers content in a fixed-size LaTeX ``\\parbox``.

    Args:
        width (int): Box width value.
        height (int | None, optional): Box height value. If None, uses ``width``.
            Defaults to None.
        unit (str, optional): LaTeX unit suffix for ``width``/``height`` (for example ``"mm"``).
            Defaults to ``"mm"``.

    Returns:
        CellWrapper: Formatter that wraps rendered text in a centered parbox.
    """
    if height is None:
        height = width

    def _inner_boxed_center(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Format one cell value into a centered fixed-size ``\\parbox``.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and wrapped LaTeX string.
        """
        return v, rf"\parbox[c][{height}{unit}][c]{{{width}}}{{\centering\strut {s}}}"

    return _inner_boxed_center


def wrap_with_tex_command(cmd_str: str) -> CellWrapper:
    """Wrap rendered cell text in a LaTeX command.

    Args:
        cmd_str (str): LaTeX command name without a leading backslash.

    Returns:
        CellWrapper: Formatter that renders output as ``\\<cmd_str>{...}``.
    """
    validate_command_name(cmd_str)

    def _inner(cell_value: TableCellValue, rendered_str: str) -> tuple[TableCellValue, str]:
        """Wrap one rendered string in the configured TeX command.

        Args:
            cell_value (TableCellValue): Original unformatted cell value.
            rendered_str (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and wrapped LaTeX string.
        """
        return cell_value, rf"\{cmd_str}{{{rendered_str}}}"

    return _inner


def compose_formatters(*funcs: CellWrapper) -> CellWrapper:
    """Compose multiple ``CellWrapper`` formatters into one formatter.

    Args:
        *funcs (CellWrapper): One or more formatter callables.

    Returns:
        CellWrapper: A formatter that applies all provided formatters from right to left.
    """

    # compose(f, g, h)(v, s) == f(g(h(v, s)))
    def run(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Run composed formatters over a single value/string pair.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Updated value/string pair after all formatters.
        """
        for formatter in reversed(funcs):
            v, s = formatter(v, s)
        return v, s

    return run


def round_decimals(decimal_places: int) -> CellWrapper:
    """Create a formatter that renders numeric values with fixed decimal places.

    Args:
        decimal_places (int): Number of decimal places to render.

    Returns:
        CellWrapper: Formatter that applies fixed-point rendering to numeric values.
    """

    def _inner_round_decimals(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Render one numeric value with the configured decimal precision.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and precision-formatted output string.
        """
        if isinstance(v, Real):
            return v, f"{v:.{decimal_places}f}"
        return v, s

    return _inner_round_decimals


def _safe_round(v: TableCellValue, round_to: int | None) -> TableCellValue:  # pragma: no cover
    """Round numeric values while preserving ``NaN`` and infinite values.

    Args:
        v (TableCellValue): Candidate value.
        round_to (int | None): Decimal places to round to, if provided.

    Returns:
        TableCellValue: Rounded numeric value or the original input.
    """
    if isinstance(v, Real) and round_to is not None:
        if v != v:  # NaN check
            return v
        if v == float("inf"):
            return v
        if v == float("-inf"):
            return v
        return float(round(v, round_to))
    return v


def _make_numeric_highlighter(
    predicate: Callable[[float], bool],
    color: Color,
    *,
    round_to: int | None,
) -> CellWrapper:
    """Build a numeric highlighter formatter from a predicate.

    Args:
        predicate (Callable[[float], bool]): Comparison predicate.
        color (Color): Highlight color.
        round_to (int | None): Decimal places to round values before comparison.

    Returns:
        CellWrapper: Formatter that prepends a LaTeX ``\\cellcolor`` command when matched.
    """
    prefix = cellcolor_prefix(color)

    def _inner(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Apply conditional cell highlighting to one value/string pair.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and highlighted (or unchanged) output string.
        """
        if isinstance(v, Real):
            rounded_value = _safe_round(v, round_to)
            if isinstance(rounded_value, Real) and predicate(float(rounded_value)):
                return v, f"{prefix}{s}"
        return v, s

    return _inner


def highlight_gt(
    thresh: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Highlight values strictly greater than a threshold.

    Args:
        thresh (int | float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(lambda x: x > float(thresh), color, round_to=round_to)


def highlight_ge(
    thresh: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Highlight values greater than or equal to a threshold.

    Args:
        thresh (int | float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(lambda x: x >= float(thresh), color, round_to=round_to)


def highlight_lt(
    thresh: float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Highlight values strictly less than a threshold.

    Args:
        thresh (float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(lambda x: x < float(thresh), color, round_to=round_to)


def highlight_le(
    thresh: float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Highlight values less than or equal to a threshold.

    Args:
        thresh (float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(lambda x: x <= float(thresh), color, round_to=round_to)


def highlight_between(
    lower_bound: int | float,
    upper_bound: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    include_lower: bool = True,
    include_upper: bool = True,
) -> CellWrapper:
    """Highlight values between lower and upper bounds.

    Args:
        lower_bound (int | float): Lower bound.
        upper_bound (int | float): Upper bound.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        include_lower (bool, optional): Whether the lower bound is inclusive.
            Defaults to True.
        include_upper (bool, optional): Whether the upper bound is inclusive.
            Defaults to True.

    Returns:
        CellWrapper: Highlight formatter.
    """
    if not include_lower:
        lower_bound = math.nextafter(float(lower_bound), math.inf)  # smallest float > lower_bound
    if not include_upper:
        upper_bound = math.nextafter(float(upper_bound), -math.inf)  # largest float < upper_bound

    low = float(lower_bound)
    high = float(upper_bound)
    return _make_numeric_highlighter(lambda x: low <= x <= high, color, round_to=round_to)
