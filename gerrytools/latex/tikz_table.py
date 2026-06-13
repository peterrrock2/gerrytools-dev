"""TikZ-augmented table renderer from a pandas DataFrame.

Mirrors the :class:`TexTable` interface but generates a ``nicematrix``
``NiceTabular`` environment instead of a plain ``tabular``.  LaTeX's own
tabular engine does the layout — so rules, spacing, and double rules are
pixel-identical to :class:`TexTable` by construction — while nicematrix
exposes every cell as a PGF/TikZ node, so you can still:

* Add arbitrary ``\\draw`` commands referencing cell nodes such as
  ``(table-2-3.north west)`` or the boundary lattice ``(row-i)``/``(col-j)``.
* Control individual cell borders (top/right/bottom/left) per cell.
* Fill rows and cells with background colours.

The generated LaTeX requires two compile passes (nicematrix stores node
positions in the aux file); :attr:`document` configures that automatically.
"""

from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable, Iterable, Optional, cast

import pandas as pd

from gerrytools.latex._colors import to_latex_xcolor_or_html_spec
from gerrytools.latex._table_preamble import (
    _parse_tabular_preamble,
)
from gerrytools.latex._text import latex_escape
from gerrytools.latex.document import TexDocument
from gerrytools.latex.formatters import round_decimals
from gerrytools.logging import get_logger
from gerrytools.typing import (
    CellWrapper,
    Color,
    IndexCellWrapper,
    TableCellValue,
    TableIndexValue,
)

logger = get_logger(__name__)


def _latex_escape_wrapper(value: TableCellValue, prev: str) -> tuple[TableCellValue, str]:
    return value, latex_escape(prev)


def _latex_foreach_list(indices: Iterable[int]) -> str:
    """Return a compact TikZ ``\\foreach`` list for sorted integer indices."""
    values = sorted(set(indices))
    if not values:
        return ""

    parts: list[str] = []
    start = prev = values[0]
    for cur in values[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        run_len = prev - start + 1
        if run_len >= 3:
            parts.append(f"{start},...,{prev}")
        else:
            parts.extend(str(v) for v in range(start, prev + 1))
        start = prev = cur

    run_len = prev - start + 1
    if run_len >= 3:
        parts.append(f"{start},...,{prev}")
    else:
        parts.extend(str(v) for v in range(start, prev + 1))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------


@dataclass
class TikzTableOptions:
    r"""Options for a TikZ-based LaTeX table.

    Most attributes mirror :class:`TableOptions` exactly.  TikZ-specific
    attributes are grouped at the bottom.
    """

    # --- rules (same semantics as TableOptions) ---
    toprule_cmd: str | None = None
    bottomrule_cmd: str | None = None
    hrule_cmd: str = r"\hline"

    # --- header visibility ---
    include_column_headers: bool = True
    include_group_headers: bool = True

    # --- index ---
    include_index: bool = False
    index_name: Optional[str] = None
    index_alignment: Optional[str] = None

    nan_string: str = "NaN"

    # --- rule counts (same semantics as TableOptions) ---
    hrule_counts: list[int] = field(default_factory=list)
    vrule_counts: list[int] = field(default_factory=list)
    tabular_alignments: list[str] = field(default_factory=list)
    boundary_extras: list[str] = field(default_factory=list)

    # API-parity fields
    group_vrule_counts: list[int] | None = None
    group_tabular_alignments: list[str] | None = None
    group_boundary_extras: list[str] | None = None

    # --- highlighting ---
    row_highlight_colors: list = field(default_factory=list)

    # --- formatters ---
    number_fmt_fn: Optional[CellWrapper] = None
    str_fmt_fn: Optional[CellWrapper] = _latex_escape_wrapper
    col_formatters: dict[str, CellWrapper] = field(default_factory=dict)
    row_formatters: dict[int, CellWrapper] = field(default_factory=dict)
    index_fmt_fn: Optional[IndexCellWrapper] = None

    # --- grouping ---
    groups_to_cols: dict[str, list[str]] = field(default_factory=dict)

    # --- text styling ---
    bold_group_headers: bool = True
    italic_group_headers: bool = False
    bold_column_headers: bool = False
    italic_column_headers: bool = False

    # --- NiceTabular-specific ---
    cell_space_limits: str = "1pt"
    extra_draws: list[str] = field(default_factory=list)

    # Per-cell border control: maps (tikz_row_1based, tikz_col_1based) to a
    # set of sides.  Valid sides: "top", "bottom", "left", "right".
    cell_borders: dict[tuple[int, int], set[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TikzTable:
    """Generate a TikZ ``matrix of nodes`` table from a pandas DataFrame.

    The public API is identical to :class:`TexTable`.  The default output
    matches the visual appearance of ``TexTable``: no cell borders, horizontal
    rules where ``\\hline`` would appear, vertical rules where ``|`` would
    appear in the tabular preamble.

    TikZ-specific extras let you control cell geometry, inject raw
    ``\\draw`` commands, and set per-cell borders.

    The generated LaTeX requires::

        \\usepackage{tikz}
        \\usetikzlibrary{matrix,fit,calc}

    (The :attr:`document` property adds them automatically.)

    Args:
        df (pd.DataFrame): Source data.
        use_defaults (bool): When ``True`` bold column-headers, 4 decimal
            places, and a double rule above the first data row are applied
            — the same defaults as :class:`TexTable`.
    """

    def __init__(self, df: pd.DataFrame, *, use_defaults: bool = True) -> None:
        self.df = df.copy()
        self._document = TexDocument()
        self._document.add_packages(["nicematrix", "tikz"])
        self._document.add_command(r"\usetikzlibrary{calc}")
        # nicematrix stores cell-node positions in the aux file, so the
        # document needs a second compile pass for CodeBefore/CodeAfter
        # material to land in the right place.
        self._document.compile_passes = 2

        if use_defaults:
            self._options = TikzTableOptions(
                groups_to_cols={"": list(df.columns)},
                tabular_alignments=["c"] * df.shape[1],
                vrule_counts=[0] * (df.shape[1] + 1),
                row_highlight_colors=[("NONE", "")] * df.shape[0],
                bold_group_headers=True,
                bold_column_headers=True,
            )
            self.add_hrule_above(0, 2)
            self.set_decimal_count(4)
        else:
            self._options = TikzTableOptions(
                groups_to_cols={"": list(df.columns)},
                tabular_alignments=["c"] * df.shape[1],
                row_highlight_colors=[("NONE", "")] * df.shape[0],
                vrule_counts=[0] * (df.shape[1] + 1),
            )

    def __repr__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    def __str__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    @property
    def document(self) -> TexDocument:
        """TexDocument: The LaTeX document associated with this table."""
        self._document.body_string = self._generate_latex()
        return self._document

    def clear_options(self) -> None:
        """Reset all table options to plain defaults."""
        self._options = TikzTableOptions(
            groups_to_cols={"": list(self.df.columns)},
            tabular_alignments=["c"] * self.df.shape[1],
            row_highlight_colors=[("NONE", "")] * self.df.shape[0],
            vrule_counts=[0] * (self.df.shape[1] + 1),
        )

    def preview(self) -> None:  # pragma: no cover
        """Render and preview the table through its ``TexDocument``."""
        self.document.preview()

    # ==================================================================
    #   FEATURE ADDITION — same API as TexTable
    # ==================================================================

    # --- index ---------------------------------------------------------

    def include_index(
        self, name: Optional[str] = None, alignment: str = "c", include: bool = True
    ) -> None:
        """Add or remove the DataFrame index column.

        Args:
            name (Optional[str]): Override label for the index header.
            alignment (str): Alignment token for the index column.
            include (bool): Whether to include the index.
        """
        self._options.index_alignment = alignment
        self._options.index_name = latex_escape(str(name)) if name is not None else ""

        n_data_cols = self.df.shape[1]

        def _ensure_extras(ncols: int) -> None:
            if not self._options.boundary_extras:
                self._options.boundary_extras = [""] * (ncols + 1)
            if len(self._options.boundary_extras) != ncols + 1:
                self._options.boundary_extras = [""] * (ncols + 1)

        if include and not self._options.include_index:
            self._options.include_index = True
            if len(self._options.tabular_alignments) != n_data_cols:
                raise ValueError(
                    "Current tabular format does not match DataFrame columns. Got "
                    f"{len(self._options.tabular_alignments)} colspecs but expected {n_data_cols}."
                )
            # Parse preamble-style alignment (e.g. ">{\bfseries}c|") to
            # extract colspec, vrules, and boundary extras.
            idx_specs, idx_vrules, idx_extras = _parse_tabular_preamble(alignment)
            if len(idx_specs) != 1:
                raise ValueError(
                    f"Index alignment must specify exactly 1 column, got {len(idx_specs)}."
                )
            self._options.tabular_alignments.insert(0, idx_specs[0])
            _ensure_extras(n_data_cols)
            # Merge left-boundary vrule of the index into position 0,
            # right-boundary vrule into position 1.
            self._options.vrule_counts.insert(0, idx_vrules[0])
            self._options.vrule_counts[1] += idx_vrules[1]
            # Merge boundary extras: left extra at position 0, right at 1.
            self._options.boundary_extras.insert(0, idx_extras[0])
            if idx_extras[1]:
                self._options.boundary_extras[1] = idx_extras[1] + self._options.boundary_extras[1]

        if (not include) and self._options.include_index:
            self._options.include_index = False
            if len(self._options.tabular_alignments) != n_data_cols + 1:
                raise ValueError(
                    "Current tabular format does not match DataFrame columns+index. Got "
                    f"{len(self._options.tabular_alignments)} colspecs but expected "
                    f"{n_data_cols + 1}."
                )
            self._options.tabular_alignments.pop(0)
            _ensure_extras(n_data_cols + 1)
            self._options.vrule_counts.pop(0)
            self._options.boundary_extras.pop(0)

    def remove_index(self) -> None:
        """Remove the index column from the table."""
        self.include_index(include=False)

    # --- horizontal rules ----------------------------------------------

    def add_hrule_above(self, index: int | list[int], count: int = 1) -> None:
        """Add horizontal rules above the specified data row(s).

        Row index 0 means *above the first data row* (below the header).

        Args:
            index (int | list[int]): Data-row index or list of indices.
            count (int): Number of rules to add at each position.
        """
        indices = [index] if isinstance(index, int) else list(index)
        if not self._options.hrule_counts:
            self._options.hrule_counts = [0] * len(self.df)
        if len(self._options.hrule_counts) < len(self.df):
            self._options.hrule_counts.extend(
                [0] * (len(self.df) - len(self._options.hrule_counts))
            )
        for idx in indices:
            if idx < 0 or idx > len(self.df):
                raise ValueError(
                    f"Row index {idx} is out of bounds for DataFrame with {len(self.df)} rows."
                )
            self._options.hrule_counts[idx] += count

    def add_hrule_above_all(self, count: int = 1) -> None:
        """Add horizontal rules above every data row."""
        if not self._options.hrule_counts:
            self._options.hrule_counts = [0] * len(self.df)
        for idx in range(len(self.df)):
            self._options.hrule_counts[idx] += count

    def clear_all_hrule(self) -> None:
        """Remove all interior horizontal rules."""
        self._options.hrule_counts = []

    def set_all_hrule(self, count: int) -> None:
        """Set a uniform horizontal-rule count for every data row boundary.

        This draws *count* lines above every data row **and** below the last
        data row, producing a complete grid when combined with
        :meth:`add_vrule_all`.
        """
        self._options.hrule_counts = [count] * (len(self.df) + 1)

    def add_toprule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule at the very top of the table.

        Args:
            cmd (Optional[str]): LaTeX rule command (e.g. ``r"\\toprule"``).
                Defaults to the current ``hrule_cmd`` (``r"\\hline"``).
        """
        self._options.toprule_cmd = cmd if cmd is not None else self._options.hrule_cmd

    def remove_toprule(self) -> None:
        self._options.toprule_cmd = None

    def add_bottomrule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule at the very bottom of the table.

        Args:
            cmd (Optional[str]): LaTeX rule command (e.g. ``r"\\bottomrule"``).
                Defaults to the current ``hrule_cmd`` (``r"\\hline"``).
        """
        self._options.bottomrule_cmd = cmd if cmd is not None else self._options.hrule_cmd

    def remove_bottomrule(self) -> None:
        self._options.bottomrule_cmd = None

    def set_hrule_command(self, cmd: str) -> None:
        r"""Set the LaTeX command used for interior horizontal rules.

        Args:
            cmd (str): Rule command (e.g. ``r"\hline"``).
        """
        self._options.hrule_cmd = cmd

    def set_toprule_command(self, cmd: str | None = None) -> None:
        self._options.toprule_cmd = cmd if cmd is not None else self._options.hrule_cmd

    def set_bottomrule_command(self, cmd: str | None = None) -> None:
        self._options.bottomrule_cmd = cmd if cmd is not None else self._options.hrule_cmd

    # --- vertical rules ------------------------------------------------

    def add_vrule_left_of(self, col_idx: int | list[int], count: int = 1) -> None:
        cols = [col_idx] if isinstance(col_idx, int) else list(col_idx)
        ioff = 1 if self._options.include_index else 0
        if not self._options.vrule_counts:
            self._options.vrule_counts = [0] * (len(self.df.columns) + 1 + ioff)
        for cidx in cols:
            if cidx < 0 or cidx > len(self.df.columns) + ioff:
                raise ValueError(f"Column index {cidx} is out of bounds.")
            self._options.vrule_counts[cidx] += count

    def add_vrule_right_of(self, col_idx: int | list[int], count: int = 1) -> None:
        cols = [col_idx] if isinstance(col_idx, int) else list(col_idx)
        ioff = 1 if self._options.include_index else 0
        if not self._options.vrule_counts:
            self._options.vrule_counts = [0] * (len(self.df.columns) + 1 + ioff)
        for cidx in cols:
            if cidx < -1 or cidx > len(self.df.columns) + ioff - 1:
                raise ValueError(f"Column index {cidx} is out of bounds.")
            self._options.vrule_counts[cidx + 1] += count

    def add_vrule_all(self, count: int = 1) -> None:
        total = len(self.df.columns) + int(self._options.include_index)
        if not self._options.vrule_counts:
            self._options.vrule_counts = [0] * (total + 1)
        for idx in range(total + 1):
            self._options.vrule_counts[idx] += count

    def clear_all_vrule(self) -> None:
        ioff = int(self._options.include_index)
        self._options.vrule_counts = [0] * (len(self.df.columns) + 1 + ioff)

    # --- highlighting --------------------------------------------------

    def highlight_rows(self, rows: int | Iterable[int], color: Color = "yellow") -> None:
        """Fill data-row cell backgrounds with a colour.

        Args:
            rows (int | Iterable[int]): Row index or indices to highlight.
            color (Color): xcolor name, hex string, or RGB tuple.
        """
        row_indices = [rows] if isinstance(rows, int) else list(rows)
        try:
            color_type, color_value = to_latex_xcolor_or_html_spec(color)
        except ValueError as exc:
            if isinstance(color, tuple) and len(color) == 3:
                raise
            raise ValueError("Invalid color specification for row highlighting") from exc
        if not self._options.row_highlight_colors:
            self._options.row_highlight_colors = [("NONE", "")] * len(self.df)
        for ridx in row_indices:
            self._options.row_highlight_colors[ridx] = (color_type, color_value)  # type: ignore[invalid-assignment]

    # --- header visibility ---------------------------------------------

    def remove_column_headers(self) -> None:
        self._options.include_column_headers = False

    def remove_group_headers(self) -> None:
        self._options.include_group_headers = False

    def remove_all_headers(self) -> None:
        self.remove_column_headers()
        self.remove_group_headers()

    def include_column_headers(self) -> None:
        self._options.include_column_headers = True

    def include_group_headers(self) -> None:
        self._options.include_group_headers = True

    def include_all_headers(self) -> None:
        self.include_column_headers()
        self.include_group_headers()

    # --- text styling --------------------------------------------------

    def set_column_headers_text_format(self, bold: bool = True, italic: bool = False) -> None:
        self._options.bold_column_headers = bold
        self._options.italic_column_headers = italic

    def set_group_headers_text_format(self, bold: bool = True, italic: bool = False) -> None:
        self._options.bold_group_headers = bold
        self._options.italic_group_headers = italic

    # ==================================================================
    #   OPTION SETTERS — same API as TexTable
    # ==================================================================

    def set_decimal_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("Decimal count must be non-negative")
        self._options.number_fmt_fn = round_decimals(count)

    def set_nan_string(self, nan_str: str) -> None:
        self._options.nan_string = nan_str

    def set_number_formatter(self, fmt_fn: CellWrapper | Callable[[Real], str]) -> None:
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[Real], str], fmt_fn)

            def _w(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                return (v, s) if not isinstance(v, Real) else (v, one_arg(v))

            self._options.number_fmt_fn = _w
        else:
            self._options.number_fmt_fn = cast(CellWrapper, fmt_fn)

    def set_string_formatter(self, fmt_fn: CellWrapper | Callable[[str], str]) -> None:
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[str], str], fmt_fn)

            def _w(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                return (v, s) if not isinstance(v, str) else (v, one_arg(v))

            self._options.str_fmt_fn = _w
        else:
            self._options.str_fmt_fn = cast(CellWrapper, fmt_fn)

    def set_column_formatter(
        self, col: str | list[str], fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        for c in [col] if isinstance(col, str) else col:
            self._set_single_col_formatter(c, fmt_fn)

    def _set_single_col_formatter(
        self, col: str, fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame.")
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[TableCellValue], str], fmt_fn)

            def _w(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                return v, one_arg(v)

            self._options.col_formatters[col] = _w
        else:
            self._options.col_formatters[col] = cast(CellWrapper, fmt_fn)

    def set_row_formatter(
        self, row_idx: int | list[int], fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        for ridx in [row_idx] if isinstance(row_idx, int) else list(row_idx):
            self._set_single_row_formatter(ridx, fmt_fn)

    def _set_single_row_formatter(
        self, row_idx: int, fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        if row_idx < 0 or row_idx >= len(self.df):
            raise ValueError(
                f"Row index {row_idx} is out of bounds for DataFrame with {len(self.df)} rows."
            )
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[TableCellValue], str], fmt_fn)

            def _w(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                return v, one_arg(v)

            self._options.row_formatters[row_idx] = _w
        else:
            self._options.row_formatters[row_idx] = cast(CellWrapper, fmt_fn)

    def set_index_formatter(
        self, fmt_fn: IndexCellWrapper | Callable[[TableIndexValue], str]
    ) -> None:
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[TableIndexValue], str], fmt_fn)

            def _w(v: TableIndexValue, s: str) -> tuple[TableIndexValue, str]:
                return v, one_arg(v)

            self._options.index_fmt_fn = _w
        else:
            self._options.index_fmt_fn = cast(IndexCellWrapper, fmt_fn)

    # --- tabular / group format ----------------------------------------

    def set_tabular_format(self, fmt: str) -> None:
        """Parse a LaTeX tabular preamble and apply column alignments/widths.

        ``p{width}`` specs set a fixed column width.  Vertical-rule counts
        from ``|`` characters become post-matrix ``\\draw`` lines.  ``>{}``
        and ``<{}`` decorators wrap cell content.

        Args:
            fmt (str): Tabular preamble string (e.g. ``"|l|p{2cm}||c|"``).
        """
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)
        expected = self.df.shape[1] + (1 if self._options.include_index else 0)
        if len(colspecs) != expected:
            raise ValueError(
                f"Format implies {len(colspecs)} columns but expected {expected} "
                f"({'with' if self._options.include_index else 'without'} index)."
            )
        self._options.tabular_alignments = colspecs
        self._options.vrule_counts = vrules
        self._options.boundary_extras = extras

    def set_group_tabular_format(self, fmt: str) -> None:
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)
        group_cells = len(self._options.groups_to_cols) + (1 if self._options.include_index else 0)
        if len(colspecs) == group_cells - 1:
            colspecs, vrules, extras = _parse_tabular_preamble(fmt + "c")
        if len(colspecs) != group_cells:
            raise ValueError(
                f"Group-header format implies {len(colspecs)} cells but expected {group_cells}."
            )
        self._options.group_tabular_alignments = colspecs
        self._options.group_vrule_counts = vrules
        self._options.group_boundary_extras = extras

    def clear_header_groups(self) -> None:
        self._options.groups_to_cols = {"": list(self.df.columns)}
        self._options.group_tabular_alignments = None
        self._options.group_vrule_counts = None
        self._options.group_boundary_extras = None

    def set_header_groups(self, groups_to_columns: dict[str, Iterable[str]]) -> None:
        df_cols = list(map(str, self.df.columns))
        retyped: dict[str, list[str]] = {
            str(g): [str(c) for c in cols] for g, cols in groups_to_columns.items()
        }
        all_listed = [c for cols in retyped.values() for c in cols]
        unknown = set(all_listed) - set(df_cols)
        if unknown:
            raise ValueError(f"Unknown columns in groups_to_columns: {sorted(unknown)}")
        missing = [c for c in df_cols if c not in set(all_listed)]
        if "" in retyped:
            retyped[""].extend(missing)
        elif missing:
            retyped[""] = missing
        self._options.groups_to_cols = retyped
        self._options.group_tabular_alignments = None
        self._options.group_vrule_counts = None
        self._options.group_boundary_extras = None

    # ==================================================================
    #   NiceTabular-specific setters
    # ==================================================================

    def set_cell_space_limits(self, limit: str) -> None:
        """Set nicematrix's minimal vertical space around cell content.

        Args:
            limit (str): LaTeX dimension (e.g. ``"2pt"``).
        """
        self._options.cell_space_limits = limit

    def add_draw(self, draw_cmd: str) -> None:
        r"""Append a raw TikZ command drawn over the finished table.

        The environment is named ``table``; the content node of cell
        *(i, j)* (1-indexed, header rows count) is ``(table-i-j)``.  The
        boundary lattice is also available: ``(row-i)`` and ``(col-j)``
        coordinates combine as ``(row-2-|col-1)``.
        """
        self._options.extra_draws.append(draw_cmd)

    def clear_extra_draws(self) -> None:
        self._options.extra_draws = []

    def set_cell_border(
        self,
        row: int | list[int],
        col: int | list[int],
        sides: str | Iterable[str],
    ) -> None:
        """Specify which borders to draw on individual cells.

        The (row, col) indices are 1-based TikZ matrix coordinates — row 1
        is the first rendered row (group-header or column-header), and
        column 1 is the leftmost column.

        Args:
            row (int | list[int]): TikZ row index (or list of indices).
            col (int | list[int]): TikZ column index (or list of indices).
            sides (str | Iterable[str]): ``"top"``, ``"bottom"``, ``"left"``,
                ``"right"`` — or an iterable of them, or ``"all"``.
        """
        rows = [row] if isinstance(row, int) else list(row)
        cols = [col] if isinstance(col, int) else list(col)
        if isinstance(sides, str):
            side_set = {"top", "bottom", "left", "right"} if sides == "all" else {sides}
        else:
            side_set = set(sides)
        for r in rows:
            for c in cols:
                self._options.cell_borders.setdefault((r, c), set()).update(side_set)

    def clear_cell_borders(self) -> None:
        """Remove all per-cell border specifications."""
        self._options.cell_borders = {}

    # ==================================================================
    #   Internal helpers
    # ==================================================================

    def _has_groups(self) -> bool:
        return set(self._options.groups_to_cols.keys()) != {""}

    def _get_ncols(self) -> int:
        return len(self.df.columns) + (1 if self._options.include_index else 0)

    def _get_column_ordering(self) -> list[str]:
        if self._has_groups():
            return [c for cols in self._options.groups_to_cols.values() for c in cols]
        return list(self.df.columns)

    # Patterns for extracting >{ } and <{ } array decorators.
    _GT_RE = re.compile(r">\{((?:[^{}]|\{[^{}]*\})*)\}")
    _LT_RE = re.compile(r"<\{((?:[^{}]|\{[^{}]*\})*)\}")

    # Pattern for \cellcolor{name}, \cellcolor[model]{value} at start of text.
    _CELLCOLOR_RE = re.compile(
        r"^\\cellcolor"
        r"(?:\[([^\]]*)\])?"  # optional [model]
        r"\{([^}]*)\}"  # {value}
    )

    def _strip_cellcolor_spec(self, text: str) -> tuple[str | None, str]:
        r"""Strip a leading ``\cellcolor`` prefix, returning ``(spec, clean_text)``.

        ``spec`` is an inline xcolor argument ready to follow ``\cellcolor`` or
        ``\rowcolor`` in nicematrix's ``\CodeBefore`` — e.g. ``[HTML]{B481D6}``
        or ``{teal}`` — or ``None`` when no ``\cellcolor`` prefix is present.

        Emitting the colour inline (the same approach TexTable uses) rather
        than registering a body-level ``\definecolor`` avoids the spurious
        horizontal space nicematrix reserves for every ``\definecolor`` that
        appears in the document body.
        """
        m = self._CELLCOLOR_RE.match(text)
        if not m:
            return None, text
        model, value = m.group(1), m.group(2)
        rest = text[m.end() :]
        spec = f"[{model}]{{{value}}}" if model else f"{{{value}}}"
        return spec, rest

    def _format_cell_value(self, row_idx: int, col: str, cell_value: object) -> str:
        if pd.isna(cell_value):
            return self._options.nan_string
        if col in self._options.col_formatters:
            return self._options.col_formatters[col](cell_value, str(cell_value))[1]
        if row_idx in self._options.row_formatters:
            return self._options.row_formatters[row_idx](cell_value, str(cell_value))[1]
        if isinstance(cell_value, Real) and self._options.number_fmt_fn is not None:
            return self._options.number_fmt_fn(cell_value, str(cell_value))[1]
        if isinstance(cell_value, str) and self._options.str_fmt_fn is not None:
            return self._options.str_fmt_fn(cell_value, cell_value)[1]
        raw = latex_escape(str(cell_value))
        if self._options.str_fmt_fn is not None:
            return self._options.str_fmt_fn(raw, raw)[1]
        return raw

    def _row_fill_spec(self, row_idx: int) -> str:
        r"""Return the inline ``\rowcolor`` colour spec for a data row, or ``""``.

        The spec is an xcolor argument such as ``[HTML]{F6E8C3}`` or ``{teal}``
        emitted inline in ``\CodeBefore`` (no body-level ``\definecolor`` — see
        :meth:`_strip_cellcolor_spec`).
        """
        if not self._options.row_highlight_colors:
            return ""
        color_type, color_value = self._options.row_highlight_colors[row_idx]
        match color_type:
            case "NONE":
                return ""
            case "NAME":
                return f"{{{color_value}}}"
            case "HTML":
                return f"[HTML]{{{str(color_value).lstrip('#').upper()}}}"
            case "RGB":
                r, g, b = color_value
                return f"[RGB]{{{r},{g},{b}}}"
            case "rgb":
                r, g, b = color_value
                return f"[rgb]{{{r:.3f},{g:.3f},{b:.3f}}}"
            case _:
                warnings.warn(
                    f"Unsupported color type '{color_type}' for row highlighting; skipping.",
                    stacklevel=2,
                )
                return ""

    # ==================================================================
    #   Rendering
    # ==================================================================

    def _boundary_spec(self, boundary: int) -> str:
        """Build the colspec tokens for one column boundary.

        ``<{...}`` decorators (which close the previous column) come first,
        then vertical rules, then ``>{...}`` decorators (which open the next
        column) — reconstructing a valid tabular preamble ordering.
        """
        extras = self._options.boundary_extras
        extra = extras[boundary] if boundary < len(extras) else ""
        vrules = self._options.vrule_counts
        vrule = "|" * (vrules[boundary] if boundary < len(vrules) else 0)
        closing = "".join(f"<{{{m.group(1)}}}" for m in self._LT_RE.finditer(extra))
        opening = "".join(f">{{{m.group(1)}}}" for m in self._GT_RE.finditer(extra))
        return closing + vrule + opening

    def _column_format(self) -> str:
        """Build the full NiceTabular column specification."""
        ncols = self._get_ncols()
        alignments = self._options.tabular_alignments
        parts: list[str] = []
        for col_i in range(ncols):
            parts.append(self._boundary_spec(col_i))
            parts.append(alignments[col_i] if col_i < len(alignments) else "c")
        parts.append(self._boundary_spec(ncols))
        return "".join(parts)

    def _styled_header_text(self, text: str, *, bold: bool, italic: bool) -> str:
        if bold:
            text = rf"\textbf{{{text}}}"
        if italic:
            text = rf"\textit{{{text}}}"
        return text

    def _header_rule_metrics(self) -> tuple[str, str]:
        """Line width and inter-rule step for the pushed-up header rule stack.

        The values are LaTeX length expressions matching the active ``hrule_cmd``
        so the drawn rules look like real stacked rules: booktabs ``\\midrule``
        uses ``\\lightrulewidth``, ``\\toprule``/``\\bottomrule`` use
        ``\\heavyrulewidth``, and ``\\hline`` (or anything else) uses
        ``\\arrayrulewidth``. The step is ``\\doublerulesep`` throughout, which
        reproduces a tabular double rule and reads as a clean multi-rule.
        """
        cmd = self._options.hrule_cmd
        if cmd == r"\midrule":
            return r"\lightrulewidth", r"\doublerulesep"
        if cmd in (r"\toprule", r"\bottomrule"):
            return r"\heavyrulewidth", r"\doublerulesep"
        return r"\arrayrulewidth", r"\doublerulesep"

    def _header_rule_strut(self) -> str:
        """Zero-width depth strut giving the last header row room for the stack.

        Depth ``(count - 1) * step + width`` reaches just past the topmost drawn
        rule so it sits in page-coloured space inside the header rather than
        bleeding into the first data row's colour band.
        """
        width, step = self._header_rule_metrics()
        extra = self._options.hrule_counts[0] - 1
        return rf"\rule[-\dimexpr{extra}{step}+{width}\relax]{{0pt}}{{0pt}}"

    def _generate_latex(self) -> str:
        """Build the complete ``NiceTabular`` string."""
        column_ordering = self._get_column_ordering()
        has_groups = self._has_groups()
        group_header_present = has_groups and self._options.include_group_headers
        column_headers_present = self._options.include_column_headers
        data_start = 1 + int(group_header_present) + int(column_headers_present)
        ndata = len(self.df)

        # A stack of two or more rules directly under the header (e.g. a double
        # \hline, or \midrule×3 once use_defaults' header rule and an explicit
        # add_hrule_above pile up) absorbs the rules' vertical gaps into the
        # first data row's colour band under nicematrix, so that one shaded row
        # is visibly taller than the rest. Instead, emit a single rule at the
        # boundary (so all data rows stay the same height) and draw the extra
        # rules up inside the header via \CodeAfter, with a depth strut on the
        # last header row to give it room. The gap between rules is the page
        # colour, every shaded row is identical, and the multi-rule still reads
        # the way the same calls render under colortbl (TexTable).
        push_header_double = (
            bool(self._options.hrule_counts)
            and self._options.hrule_counts[0] >= 2
            and (column_headers_present or group_header_present)
        )

        # ---- body rows + per-cell fills (stripped \cellcolor prefixes) ----
        # (nicematrix_row, col_1based, color_spec) where color_spec is an
        # inline xcolor argument such as ``[HTML]{B481D6}`` or ``{teal}``.
        # The colour is emitted inline in \CodeBefore rather than via a
        # body-level \definecolor: nicematrix adds spurious horizontal space
        # for every \definecolor that appears in the document body, so the
        # table drifts right by an amount that grows with the colour count.
        cell_fills: list[tuple[int, int, str]] = []

        body_rows: list[str] = []
        for row_idx, (df_row_idx, row) in enumerate(self.df.iterrows()):
            nicerow = data_start + row_idx
            cells: list[str] = []
            col_1based = 1
            if self._options.include_index:
                escaped = latex_escape(str(df_row_idx))
                text = (
                    self._options.index_fmt_fn(cast(TableIndexValue, df_row_idx), escaped)[1]
                    if self._options.index_fmt_fn is not None
                    else escaped
                )
                fill_spec, text = self._strip_cellcolor_spec(text)
                if fill_spec is not None:
                    cell_fills.append((nicerow, col_1based, fill_spec))
                cells.append(text)
                col_1based += 1
            for col in column_ordering:
                text = self._format_cell_value(row_idx, col, row[col])
                fill_spec, text = self._strip_cellcolor_spec(text)
                if fill_spec is not None:
                    cell_fills.append((nicerow, col_1based, fill_spec))
                cells.append(text)
                col_1based += 1
            body_rows.append(" & ".join(cells) + r" \\")

        # ---- assemble ----
        lines: list[str] = []
        env_options = f"[name=table, cell-space-limits={self._options.cell_space_limits}]"
        lines.append(f"\\begin{{NiceTabular}}{{{self._column_format()}}}{env_options}")

        # ---- CodeBefore: row + cell background fills ----
        code_before: list[str] = []
        for row_idx in range(ndata):
            spec = self._row_fill_spec(row_idx)
            if spec:
                code_before.append(f"\\rowcolor{spec}{{{data_start + row_idx}}}")
        for nicerow, col_1based, spec in cell_fills:
            code_before.append(f"\\cellcolor{spec}{{{nicerow}-{col_1based}}}")
        if code_before:
            lines.append(r"\CodeBefore")
            lines.extend(f"  {entry}" for entry in code_before)
            lines.append(r"\Body")

        if self._options.toprule_cmd is not None:
            lines.append(self._options.toprule_cmd)

        # ---- group-header row ----
        if group_header_present:
            cells = [""] if self._options.include_index else []
            for group_name, group_cols in self._options.groups_to_cols.items():
                span = len(group_cols)
                if group_name == "" or span == 0:
                    cells.extend([""] * span)
                    continue
                text = self._styled_header_text(
                    latex_escape(str(group_name)),
                    bold=self._options.bold_group_headers,
                    italic=self._options.italic_group_headers,
                )
                cells.append(rf"\Block[c]{{1-{span}}}{{{text}}}")
                cells.extend([""] * (span - 1))
            # When the group row is the last header row, it carries the depth
            # strut that makes white room for the pushed-up rule stack.
            strut = (
                self._header_rule_strut()
                if (push_header_double and not column_headers_present)
                else ""
            )
            lines.append(" & ".join(cells) + strut + r" \\")

        # ---- column-header row ----
        if column_headers_present:
            header_cells: list[str] = []
            if self._options.include_index:
                idx_name = (
                    self._options.index_name
                    if self._options.index_name is not None
                    else (self.df.index.name if self.df.index.name is not None else "")
                )
                header_cells.append(
                    self._styled_header_text(
                        latex_escape(str(idx_name)),
                        bold=self._options.bold_column_headers,
                        italic=self._options.italic_column_headers,
                    )
                )
            for col in column_ordering:
                header_cells.append(
                    self._styled_header_text(
                        latex_escape(str(col)),
                        bold=self._options.bold_column_headers,
                        italic=self._options.italic_column_headers,
                    )
                )
            strut = self._header_rule_strut() if push_header_double else ""
            lines.append(" & ".join(header_cells) + strut + r" \\")

        # ---- data rows with interior horizontal rules ----
        # Rules are emitted verbatim so row geometry and the (row-i) boundary
        # lattice match a real tabular — except a header rule stack, which keeps
        # a single rule here and draws the rest in \CodeAfter (see
        # push_header_double) to keep shaded rows uniform.
        hrule_counts = self._options.hrule_counts
        for row_idx, row_text in enumerate(body_rows):
            count = hrule_counts[row_idx] if row_idx < len(hrule_counts) else 0
            if row_idx == 0 and push_header_double:
                count = 1
            lines.extend([self._options.hrule_cmd] * count)
            lines.append(row_text)
        trailing = hrule_counts[ndata] if len(hrule_counts) > ndata else 0
        lines.extend([self._options.hrule_cmd] * trailing)

        if self._options.bottomrule_cmd is not None:
            lines.append(self._options.bottomrule_cmd)

        # ---- CodeAfter: group-header vrules + per-cell borders + user draws ----
        code_after: list[str] = []

        # Extra rules of the pushed-up header stack, drawn inside the header
        # (above the single rule at the header/data boundary) so the gaps
        # between rules are the page colour and the first data row's colour band
        # matches the others. One rule sits at the boundary in the body; the
        # remaining (count - 1) are drawn here at multiples of the step above it.
        # See push_header_double.
        if push_header_double:
            ncols = self._get_ncols()
            width, step = self._header_rule_metrics()
            extra = self._options.hrule_counts[0] - 1
            for k in range(1, extra + 1):
                code_after.append(
                    rf"\draw[line width={width}] "
                    f"([yshift={k}{step}]row-{data_start}-|col-1) -- "
                    f"([yshift={k}{step}]row-{data_start}-|col-{ncols + 1});"
                )

        # Group-header vertical rules (from set_group_tabular_format) span only
        # the group-header row, matching \multicolumn{n}{|c|}{} in TexTable.
        # They are drawn on nicematrix's boundary lattice at the column
        # position of each group boundary.
        if group_header_present and self._options.group_vrule_counts:
            group_spans: list[int] = []
            if self._options.include_index:
                group_spans.append(1)
            group_spans.extend(len(cols) for cols in self._options.groups_to_cols.values())
            boundary_cols = [0]
            for span in group_spans:
                boundary_cols.append(boundary_cols[-1] + span)
            for boundary, count in enumerate(self._options.group_vrule_counts):
                if count <= 0 or boundary >= len(boundary_cols):
                    continue
                lattice_col = boundary_cols[boundary] + 1  # nicematrix col-j is left edge of col j
                for k in range(count):
                    shift = f"[xshift={k * 2}pt]" if k > 0 else ""
                    code_after.append(
                        f"\\draw{shift} (row-1-|col-{lattice_col}) -- (row-2-|col-{lattice_col});"
                    )

        # Canonicalize (row, col, side) requests into boundary segments so
        # shared edges between adjacent cells are drawn exactly once.
        horiz_edges: set[tuple[int, int]] = set()
        vert_edges: set[tuple[int, int]] = set()
        for (r, c), sides in self._options.cell_borders.items():
            for side in sides:
                if side == "top":
                    horiz_edges.add((c, r))
                elif side == "bottom":
                    horiz_edges.add((c, r + 1))
                elif side == "left":
                    vert_edges.add((r, c))
                elif side == "right":
                    vert_edges.add((r, c + 1))
        for c, row_boundary in sorted(horiz_edges):
            code_after.append(
                f"\\draw (row-{row_boundary}-|col-{c}) -- (row-{row_boundary}-|col-{c + 1});"
            )
        for r, col_boundary in sorted(vert_edges):
            code_after.append(
                f"\\draw (row-{r}-|col-{col_boundary}) -- (row-{r + 1}-|col-{col_boundary});"
            )
        code_after.extend(self._options.extra_draws)
        if code_after:
            lines.append(r"\CodeAfter")
            lines.append(r"\begin{tikzpicture}")
            lines.extend(f"  {entry}" for entry in code_after)
            lines.append(r"\end{tikzpicture}")

        lines.append(r"\end{NiceTabular}")
        return "\n".join(lines)
