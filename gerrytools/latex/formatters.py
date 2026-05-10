from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable

from gerrytools.colors import convert_color_to_hexa_or_none
from gerrytools.latex._colors import cellcolor_prefix
from gerrytools.latex.commands import (
    tex_cell_highlight_command,
    tex_diverging_gradient_command,
    validate_command_name,
)
from gerrytools.typing import CellWrapper, Color, TableCellValue

_LATEX_COMMANDS_ATTR = "_gerrytools_latex_commands"
_LATEX_COMMAND_SPECS_ATTR = "_gerrytools_latex_command_specs"


@dataclass
class LatexCommandSpec:
    """LaTeX command required by a formatter.

    ``selected_name`` is intentionally mutable: ``TexTable`` may rename a
    formatter's command during registration to avoid command-name collisions.
    """

    base_name: str
    selected_name: str
    command_factory: Callable[[str], str]
    suffix_sequence: bool = False

    def command(self) -> str:
        """Generate the LaTeX command for the currently selected command name."""
        return self.command_factory(self.selected_name)


def latex_commands_for(formatter: Callable) -> tuple[str, ...]:
    """Return LaTeX preamble commands required by a formatter.

    Args:
        formatter (Callable): Formatter callable to inspect.

    Returns:
        tuple[str, ...]: Required LaTeX preamble command definitions.
    """
    commands = list(getattr(formatter, _LATEX_COMMANDS_ATTR, ()))
    commands.extend(spec.command() for spec in latex_command_specs_for(formatter))
    return tuple(commands)


def static_latex_commands_for(formatter: Callable) -> tuple[str, ...]:
    """Return static LaTeX commands required by a formatter.

    Unlike :func:`latex_commands_for`, this excludes mutable command specs that
    may need table-local renaming.
    """
    return tuple(getattr(formatter, _LATEX_COMMANDS_ATTR, ()))


def latex_command_specs_for(formatter: Callable) -> tuple[LatexCommandSpec, ...]:
    """Return mutable LaTeX command specs required by a formatter.

    Args:
        formatter (Callable): Formatter callable to inspect.

    Returns:
        tuple[LatexCommandSpec, ...]: Required command specs.
    """
    return tuple(getattr(formatter, _LATEX_COMMAND_SPECS_ATTR, ()))


def _with_latex_commands(formatter: CellWrapper, commands: tuple[str, ...]) -> CellWrapper:
    """Attach required LaTeX preamble commands to a formatter.

    Args:
        formatter (CellWrapper): Formatter to annotate.
        commands (tuple[str, ...]): LaTeX command definitions required by ``formatter``.

    Returns:
        CellWrapper: The same formatter, annotated with command metadata.
    """
    if commands:
        setattr(formatter, _LATEX_COMMANDS_ATTR, commands)
    return formatter


def _with_latex_command_specs(
    formatter: CellWrapper,
    specs: tuple[LatexCommandSpec, ...],
) -> CellWrapper:
    """Attach mutable LaTeX command specs to a formatter.

    Args:
        formatter (CellWrapper): Formatter to annotate.
        specs (tuple[LatexCommandSpec, ...]): Required LaTeX command specs.

    Returns:
        CellWrapper: The same formatter, annotated with command metadata.
    """
    if specs:
        setattr(formatter, _LATEX_COMMAND_SPECS_ATTR, specs)
    return formatter


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

    commands: list[str] = []
    specs: list[LatexCommandSpec] = []
    for formatter in funcs:
        commands.extend(getattr(formatter, _LATEX_COMMANDS_ATTR, ()))
        specs.extend(latex_command_specs_for(formatter))

    run = _with_latex_commands(run, tuple(commands))
    return _with_latex_command_specs(run, tuple(specs))


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
    command_prefix: str | None,
) -> CellWrapper:
    """Build a numeric highlighter formatter from a predicate.

    Args:
        predicate (Callable[[float], bool]): Comparison predicate.
        color (Color): Highlight color.
        round_to (int | None): Decimal places to round values before comparison.
        command_prefix (str | None): Command-name prefix for compact command
            output. Pass ``None`` to emit literal ``\\cellcolor`` prefixes.

    Returns:
        CellWrapper: Formatter that prepends a LaTeX ``\\cellcolor`` command when matched.
    """
    if command_prefix is not None:
        validate_command_name(command_prefix)
        spec = LatexCommandSpec(
            base_name=command_prefix,
            selected_name=f"{command_prefix}a",
            command_factory=lambda selected_name: tex_cell_highlight_command(selected_name, color),
            suffix_sequence=True,
        )

        def _inner_command(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
            if isinstance(v, Real):
                rounded_value = _safe_round(v, round_to)
                if isinstance(rounded_value, Real) and predicate(float(rounded_value)):
                    return v, rf"\{spec.selected_name}{{{s}}}"
            return v, s

        return _with_latex_command_specs(_inner_command, (spec,))

    prefix = cellcolor_prefix(color)

    def _inner_literal(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
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

    return _inner_literal


def _make_numeric_wrapper(
    predicate: Callable[[float], bool],
    wrap_cmd: str,
    *,
    round_to: int | None,
) -> CellWrapper:
    """Build a numeric wrapper formatter from a predicate.

    Args:
        predicate (Callable[[float], bool]): Comparison predicate.
        wrap_cmd (str): LaTeX command name without a leading backslash.
        round_to (int | None): Decimal places to round values before comparison.

    Returns:
        CellWrapper: Formatter that wraps matching values in the specified LaTeX command.
    """
    validate_command_name(wrap_cmd)
    prefix = rf"\{wrap_cmd}{{"

    def _inner(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Apply conditional wrapping to one value/string pair.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and wrapped (or unchanged) output string.
        """
        if isinstance(v, Real):
            rounded_value = _safe_round(v, round_to)
            if isinstance(rounded_value, Real) and predicate(float(rounded_value)):
                return v, f"{prefix}{s}}}"
        return v, s

    return _inner


def highlight_gt(
    thresh: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = "gt",
) -> CellWrapper:
    """Highlight values strictly greater than a threshold.

    Args:
        thresh (int | float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to ``"gt"``.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x > float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
    )


def wrap_gt(
    thresh: int | float,
    wrap_cmd: str,
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Wrap values strictly greater than a threshold in a LaTeX command.

    Args:
        thresh (int | float): Threshold value.
        wrap_cmd (str): LaTeX command name without a leading backslash.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Formatter that wraps matching values in the specified LaTeX command.
    """
    return _make_numeric_wrapper(lambda x: x > float(thresh), wrap_cmd, round_to=round_to)


def highlight_ge(
    thresh: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = "ge",
) -> CellWrapper:
    """Highlight values greater than or equal to a threshold.

    Args:
        thresh (int | float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to ``"ge"``.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x >= float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
    )


def wrap_ge(
    thresh: int | float,
    wrap_cmd: str,
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Wrap values greater than or equal to a threshold in a LaTeX command.

    Args:
        thresh (int | float): Threshold value.
        wrap_cmd (str): LaTeX command name without a leading backslash.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Formatter that wraps matching values in the specified LaTeX command.
    """
    return _make_numeric_wrapper(lambda x: x >= float(thresh), wrap_cmd, round_to=round_to)


def highlight_lt(
    thresh: float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = "lt",
) -> CellWrapper:
    """Highlight values strictly less than a threshold.

    Args:
        thresh (float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to ``"lt"``.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x < float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
    )


def wrap_lt(
    thresh: float,
    wrap_cmd: str,
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Wrap values strictly less than a threshold in a LaTeX command.

    Args:
        thresh (float): Threshold value.
        wrap_cmd (str): LaTeX command name without a leading backslash.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Formatter that wraps matching values in the specified LaTeX command.
    """
    return _make_numeric_wrapper(lambda x: x < float(thresh), wrap_cmd, round_to=round_to)


def highlight_le(
    thresh: float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = "le",
) -> CellWrapper:
    """Highlight values less than or equal to a threshold.

    Args:
        thresh (float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to ``"le"``.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x <= float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
    )


def wrap_le(
    thresh: float,
    wrap_cmd: str,
    *,
    round_to: int | None = None,
) -> CellWrapper:
    """Wrap values less than or equal to a threshold in a LaTeX command.

    Args:
        thresh (float): Threshold value.
        wrap_cmd (str): LaTeX command name without a leading backslash.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.

    Returns:
        CellWrapper: Formatter that wraps matching values in the specified LaTeX command.
    """
    return _make_numeric_wrapper(lambda x: x <= float(thresh), wrap_cmd, round_to=round_to)


def highlight_between(
    lower_bound: int | float,
    upper_bound: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    include_lower: bool = True,
    include_upper: bool = True,
    command_prefix: str | None = "btw",
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
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to ``"btw"``.

    Returns:
        CellWrapper: Highlight formatter.
    """
    if not include_lower:
        lower_bound = math.nextafter(float(lower_bound), math.inf)  # smallest float > lower_bound
    if not include_upper:
        upper_bound = math.nextafter(float(upper_bound), -math.inf)  # largest float < upper_bound

    low = float(lower_bound)
    high = float(upper_bound)
    return _make_numeric_highlighter(
        lambda x: low <= x <= high,
        color,
        round_to=round_to,
        command_prefix=command_prefix,
    )


def wrap_between(
    lower_bound: int | float,
    upper_bound: int | float,
    wrap_cmd: str,
    *,
    round_to: int | None = None,
    include_lower: bool = True,
    include_upper: bool = True,
) -> CellWrapper:
    """Wrap values between lower and upper bounds in a LaTeX command.

    Args:
        lower_bound (int | float): Lower bound.
        upper_bound (int | float): Upper bound.
        wrap_cmd (str): LaTeX command name without a leading backslash.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        include_lower (bool, optional): Whether the lower bound is inclusive.
            Defaults to True.
        include_upper (bool, optional): Whether the upper bound is inclusive.
            Defaults to True.

    Returns:
        CellWrapper: Formatter that wraps matching values in the specified LaTeX command.
    """
    if not include_lower:
        lower_bound = math.nextafter(float(lower_bound), math.inf)  # smallest float > lower_bound
    if not include_upper:
        upper_bound = math.nextafter(float(upper_bound), -math.inf)  # largest float < upper_bound

    low = float(lower_bound)
    high = float(upper_bound)
    return _make_numeric_wrapper(lambda x: low <= x <= high, wrap_cmd, round_to=round_to)


def diverging_gradient_formatter(
    lo: float = 0.0,
    mid: float = 0.5,
    hi: float = 1.0,
    color_lo: Color = "darkpastelgreen",
    color_hi: Color = "richlavender",
    color_mid: Color = "white",
    *,
    command_name: str | None = "divgrad",
    precision: int = 4,
) -> CellWrapper:
    """Formatter that applies a diverging gradient cell background.

    By default, renders numeric cells as compact LaTeX command calls like
    ``\\divgrad{0.774}`` and exposes the matching preamble command so
    ``TexTable`` can add it automatically when the formatter is set.

    Pass ``command_name=None`` to use the literal-color path instead.  That
    computes the interpolated background color in Python and prepends a
    ``\\cellcolor[HTML]{RRGGBB}`` to the rendered string.  This is the preferred
    path for ``TikzTable``, where literal ``\\cellcolor`` prefixes are converted
    to post-matrix TikZ ``\\fill`` commands with correct column-width extent.

    Args:
        lo (float): Lower bound of the gradient range. Defaults to ``0.0``.
        mid (float): Midpoint of the gradient range. Defaults to ``0.5``.
        hi (float): Upper bound of the gradient range. Defaults to ``1.0``.
        color_lo (Color): Color at the lower bound. Defaults to ``"darkpastelgreen"``.
        color_hi (Color): Color at the upper bound. Defaults to ``"richlavender"``.
        color_mid (Color): Color at the midpoint. Defaults to ``"white"``.
        command_name (str | None): LaTeX command name for compact command-based
            output. Pass ``None`` for literal ``\\cellcolor[HTML]{...}``
            prefixes. Defaults to ``"divgrad"``.
        precision (int): ``siunitx`` round precision used by the generated
            command when ``command_name`` is provided. Defaults to ``4``.

    Returns:
        CellWrapper: Formatter that applies gradient coloring to numeric cells.
    """
    if command_name is not None:
        if not all(isinstance(c, str) for c in (color_lo, color_mid, color_hi)):
            raise ValueError("command-based gradients require LaTeX color names/expressions")

        spec = LatexCommandSpec(
            base_name=command_name,
            selected_name=command_name,
            command_factory=lambda selected_name: tex_diverging_gradient_command(
                selected_name,
                lo=lo,
                mid=mid,
                hi=hi,
                color_lo=color_lo,
                color_mid=color_mid,
                color_hi=color_hi,
                precision=precision,
            ),
        )

        def _inner_command(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
            if not isinstance(v, Real):
                return v, s
            return v, rf"\{spec.selected_name}{{{float(v):.{precision}f}}}"

        return _with_latex_command_specs(_inner_command, (spec,))

    def _resolve(c: Color) -> tuple[int, int, int]:
        hex_str = convert_color_to_hexa_or_none(c)
        h = hex_str.lstrip("#")[:6]
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    rgb_lo = _resolve(color_lo)
    rgb_mid = _resolve(color_mid)
    rgb_hi = _resolve(color_hi)
    lo_f, mid_f, hi_f = float(lo), float(mid), float(hi)
    left_w = max(mid_f - lo_f, 1e-12)
    right_w = max(hi_f - mid_f, 1e-12)

    def _lerp(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (
            round(c1[0] + (c2[0] - c1[0]) * t),
            round(c1[1] + (c2[1] - c1[1]) * t),
            round(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _inner(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        if not isinstance(v, Real):
            return v, s
        x = float(v)
        x = max(lo_f, min(hi_f, x))
        if x < mid_f:
            rgb = _lerp(rgb_lo, rgb_mid, (x - lo_f) / left_w)
        else:
            rgb = _lerp(rgb_mid, rgb_hi, (x - mid_f) / right_w)
        hex_color = f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        return v, rf"\cellcolor[HTML]{{{hex_color}}}{s}"

    return _inner
