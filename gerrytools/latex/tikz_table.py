"""TikZ-based table renderer from a pandas DataFrame.

Mirrors the :class:`TexTable` interface but generates a ``tikzpicture``
``matrix of nodes`` instead of a ``tabular`` environment.  Because every
cell is a TikZ node you can:

* Change the node shape (``rectangle``, ``rounded rectangle``, …).
* Add arbitrary ``\\draw`` commands referencing matrix-node anchors such as
  ``(table-2-3.north west)``.
* Control individual cell borders (top/right/bottom/left) per cell.
* Override per-column widths, per-row heights, and per-cell styles.

The generated code follows the same visual conventions as a ``tabular``:
no cell borders are drawn by default; horizontal and vertical rules are
placed exactly where :class:`TexTable` would put ``\\hline`` and ``|``.
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

    # --- rules ---
    toprule_style: str | None = None
    bottomrule_style: str | None = None
    hrule_style: str = ""
    vrule_style: str = ""

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

    # --- TikZ-specific ---
    cell_width: str = ""
    cell_height: str = "0.7cm"
    exact_cell_size: bool = False
    normalize_cell_text_metrics: bool = False
    row_sep: str = "0pt"
    column_sep: str = "0pt"
    inner_sep: str = "3pt"
    node_shape: str = "rectangle"
    extra_node_style: str = ""
    col_widths: dict[int, str] = field(default_factory=dict)
    row_heights: dict[int, str] = field(default_factory=dict)
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
        self._document.add_packages("tikz")
        self._document.add_command(r"\usetikzlibrary{matrix,fit,calc,backgrounds}")

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
            cmd (Optional[str]): TikZ draw style.  Defaults to ``hrule_style``.
        """
        self._options.toprule_style = cmd if cmd is not None else self._options.hrule_style

    def remove_toprule(self) -> None:
        self._options.toprule_style = None

    def add_bottomrule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule at the very bottom of the table.

        Args:
            cmd (Optional[str]): TikZ draw style.  Defaults to ``hrule_style``.
        """
        self._options.bottomrule_style = cmd if cmd is not None else self._options.hrule_style

    def remove_bottomrule(self) -> None:
        self._options.bottomrule_style = None

    def set_hrule_command(self, cmd: str) -> None:
        r"""Set the TikZ draw style for interior horizontal rules.

        Args:
            cmd (str): TikZ draw-option string (e.g. ``"line width=1pt"``).
        """
        self._options.hrule_style = cmd

    def set_toprule_command(self, cmd: str | None = None) -> None:
        self._options.toprule_style = cmd if cmd is not None else self._options.hrule_style

    def set_bottomrule_command(self, cmd: str | None = None) -> None:
        self._options.bottomrule_style = cmd if cmd is not None else self._options.hrule_style

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
    #   TikZ-specific setters
    # ==================================================================

    def set_cell_size(self, width: str, height: str) -> None:
        """Set the default minimum width and height for all cells.

        Pass ``""`` for *width* to let columns auto-size to their widest
        content (the default).

        Args:
            width (str): TikZ dimension (e.g. ``"3cm"``), or ``""``.
            height (str): TikZ dimension (e.g. ``"0.8cm"``).
        """
        self._options.cell_width = width
        self._options.cell_height = height
        self._options.exact_cell_size = False
        self._options.normalize_cell_text_metrics = True

    def set_exact_cell_size(self, width: str, height: str) -> None:
        """Force all cells to render at exactly ``width × height`` when possible.

        Unlike :meth:`set_cell_size`, this wraps content in fixed-size TeX boxes
        so glyph ascenders/descenders do not change the rendered node size.
        Content that is too large may still overflow visually.

        Args:
            width (str): TikZ dimension (e.g. ``"1cm"``), or ``""`` to leave width auto.
            height (str): TikZ dimension (e.g. ``"1cm"``).
        """
        self._options.cell_width = width
        self._options.cell_height = height
        self._options.exact_cell_size = True
        self._options.normalize_cell_text_metrics = True

    def set_col_width(self, col_idx: int | list[int], width: str) -> None:
        for i in [col_idx] if isinstance(col_idx, int) else list(col_idx):
            self._options.col_widths[i] = width

    def set_row_height(self, row_idx: int | list[int], height: str) -> None:
        for i in [row_idx] if isinstance(row_idx, int) else list(row_idx):
            self._options.row_heights[i] = height

    def set_node_shape(self, shape: str) -> None:
        self._options.node_shape = shape

    def set_row_sep(self, sep: str) -> None:
        """Set the TikZ matrix row separation."""
        self._options.row_sep = sep

    def set_column_sep(self, sep: str) -> None:
        """Set the TikZ matrix column separation."""
        self._options.column_sep = sep

    def set_inner_sep(self, sep: str) -> None:
        self._options.inner_sep = sep

    def set_extra_node_style(self, style: str) -> None:
        self._options.extra_node_style = style

    def set_vrule_style(self, style: str) -> None:
        self._options.vrule_style = style

    def add_draw(self, draw_cmd: str) -> None:
        r"""Append a raw TikZ line after the matrix.

        The matrix is named ``table``; cell *(i, j)* (1-indexed) is
        ``(table-i-j)``.
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

    def _col_spec_to_tikz_align(self, spec: str) -> str:
        if spec.startswith("l"):
            return "left"
        if spec.startswith("r"):
            return "right"
        return "center"

    def _col_spec_fixed_width(self, spec: str) -> Optional[str]:
        m = re.match(r"[pmb]\{(.+)\}$", spec)
        return m.group(1) if m else None

    # Patterns for extracting >{ } and <{ } array decorators.
    _GT_RE = re.compile(r">\{((?:[^{}]|\{[^{}]*\})*)\}")
    _LT_RE = re.compile(r"<\{((?:[^{}]|\{[^{}]*\})*)\}")

    # Pattern for \cellcolor{name}, \cellcolor[model]{value} at start of text.
    _CELLCOLOR_RE = re.compile(
        r"^\\cellcolor"
        r"(?:\[([^\]]*)\])?"  # optional [model]
        r"\{([^}]*)\}"  # {value}
    )

    def _col_decorators(self, col_i: int) -> tuple[str, str]:
        """Return ``(pre, post)`` TeX content from ``>{}`` / ``<{}`` specs."""
        extras = self._options.boundary_extras
        if not extras:
            return "", ""
        left = extras[col_i] if col_i < len(extras) else ""
        right = extras[col_i + 1] if col_i + 1 < len(extras) else ""
        pre = "".join(m.group(1) for m in self._GT_RE.finditer(left))
        post = "".join(m.group(1) for m in self._LT_RE.finditer(right))
        return pre, post

    @staticmethod
    def _apply_decorators(text: str, pre: str, post: str) -> str:
        if not pre and not post:
            return text
        sep = " " if pre and pre[-1].isalpha() else ""
        return f"{pre}{sep}{text}{post}"

    @staticmethod
    def _value_to_color_basename(value: object) -> str:
        r"""Build a descriptive xcolor base name for a cell's source value.

        Names look like ``gradc_55`` (for value 0.549, percent-rounded)
        so the user can scan the colour list and match each colour back
        to the cell value that produced it.

        * ``0 <= value <= 1``  → ``gradc_<NN>`` where NN = round(v*100).
        * Other numerics       → ``gradc_<sanitised>`` with ``.``→``p``,
          ``-``→``n`` (LaTeX color names disallow ``.`` and ``-``).
        * Non-numeric / NaN    → ``gradcx`` (collisions get suffixes).
        """
        if isinstance(value, Real) and value == value:  # not NaN
            f = float(value)
            if 0.0 <= f <= 1.0:
                return f"gradc_{int(round(f * 100)):02d}"
            s = f"{f:.4g}".replace(".", "p").replace("-", "n").replace("+", "")
            return f"gradc_{s}"
        return "gradcx"

    @staticmethod
    def _cellcolor_to_hex_or_name(
        model: str | None, value: str
    ) -> tuple[str, str] | tuple[None, None]:
        r"""Normalise a ``\cellcolor`` model+value into ``(kind, value)``.

        Returns:
            ``("hex", "RRGGBB")`` for model-based colours that map to a
                fixed RGB triple (HTML, RGB, rgb).  These can be emitted
                with the row-grouped ``\gtcellrowfills`` macro.
            ``("name", "<name>")`` for unmodelled named colours
                (e.g. ``teal``, ``gray!50``, or a deduped fallback name).
            ``(None, None)`` for unrecognised models — caller falls
                back to a deduped ``\definecolor``.
        """
        if model is None:
            return "name", value
        m = model.upper()
        if m == "HTML":
            h = value.lstrip("#")[:6].upper()
            try:
                int(h, 16)
            except ValueError:
                return None, None
            return "hex", h
        if m == "RGB":
            parts = [p.strip() for p in value.split(",")]
            if len(parts) != 3:
                return None, None
            try:
                ints = [int(p) for p in parts]
            except ValueError:
                return None, None
            return "hex", "".join(f"{i:02X}" for i in ints)
        if model == "rgb":
            parts = [p.strip() for p in value.split(",")]
            if len(parts) != 3:
                return None, None
            try:
                floats = [float(p) for p in parts]
            except ValueError:
                return None, None
            ints = [max(0, min(255, round(f * 255))) for f in floats]
            return "hex", "".join(f"{i:02X}" for i in ints)
        return None, None

    def _strip_cellcolor(
        self, text: str, fallback_color_map: dict[tuple[str, str], str]
    ) -> tuple[str, str, str]:
        r"""Strip a ``\cellcolor`` prefix and return ``(kind, value, clean_text)``.

        ``kind`` is one of:

        * ``"hex"`` — *value* is a 6-char uppercase hex string; the
          caller groups these by row into ``\gtcellrowfills`` calls.
        * ``"name"`` — *value* is an xcolor name; the caller emits one
          ``\gtcellfill{<name>}{row}{col}`` per cell.
        * ``""`` — no ``\cellcolor`` was present.

        For unrecognised models, *fallback_color_map* dedups by
        (model, value) and returns the assigned ``tikzcc<N>`` as a name.
        """
        m = self._CELLCOLOR_RE.match(text)
        if not m:
            return "", "", text
        model, value = m.group(1), m.group(2)
        rest = text[m.end() :]
        kind, normalised = self._cellcolor_to_hex_or_name(model, value)
        if kind is not None:
            normalised = "" if normalised is None else normalised
            return kind, normalised, rest
        # Unknown model — fall back to dedup'd \definecolor.
        key = (model, value)
        existing = fallback_color_map.get(key)
        if existing is not None:
            return "name", existing, rest
        cname = f"tikzcc{len(fallback_color_map)}"
        fallback_color_map[key] = cname
        return "name", cname, rest

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

    def _build_color_defs(self) -> list[str]:
        lines: list[str] = []
        for row_idx, (color_type, color_value) in enumerate(self._options.row_highlight_colors):
            name = f"tikztblc{row_idx}"
            match color_type:
                case "NONE" | "NAME":
                    pass
                case "HTML":
                    hex_val = str(color_value).lstrip("#").upper()
                    lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{hex_val}}}")
                case "RGB":
                    r, g, b = color_value
                    lines.append(f"\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}")
                case "rgb":
                    r, g, b = color_value
                    lines.append(f"\\definecolor{{{name}}}{{rgb}}{{{r:.3f},{g:.3f},{b:.3f}}}")
                case _:
                    warnings.warn(
                        f"Unsupported color type '{color_type}' for row highlighting; skipping.",
                        stacklevel=2,
                    )
        return lines

    def _row_fill_color(self, row_idx: int) -> str:
        """Return a color name/spec for row highlighting, or ``""``."""
        if not self._options.row_highlight_colors:
            return ""
        color_type, color_value = self._options.row_highlight_colors[row_idx]
        if color_type == "NONE":
            return ""
        if color_type == "NAME":
            return str(color_value)
        return f"tikztblc{row_idx}"

    # ==================================================================
    #   Rendering
    # ==================================================================

    def _generate_latex(self) -> str:
        """Build the complete ``tikzpicture`` string."""
        ncols = self._get_ncols()
        column_ordering = self._get_column_ordering()
        has_groups = self._has_groups()
        alignments = self._options.tabular_alignments

        lines: list[str] = []

        # --- inline color definitions ---
        # Row-highlight colour defs; per-cell cellcolor defs are added
        # after the data rows are processed (see cellcolor_map below).
        color_def_insert_idx = len(lines)
        row_highlight_color_defs = self._build_color_defs()
        lines.extend(row_highlight_color_defs)

        # --- base node style ---
        # Cells have NO border by default (matching tabular).  Borders are
        # drawn explicitly via \draw commands after the matrix, exactly where
        # \hline and | would appear.
        # anchor=center is intentionally NOT set here.  In TikZ, the
        # nodes={} style is applied AFTER column/.style options, so placing
        # anchor=center here would silently override any per-column
        # anchor=west / anchor=east set for l / r aligned columns.
        # TikZ's built-in default anchor is already "center", so omitting
        # it from nodes={} leaves column styles free to override it.
        node_parts: list[str] = [
            self._options.node_shape,
            f"minimum height={self._options.cell_height}",
            f"inner sep={self._options.inner_sep}",
            "outer sep=0pt",
            "align=center",
        ]
        if self._options.cell_width:
            node_parts.insert(1, f"minimum width={self._options.cell_width}")
        if self._options.extra_node_style:
            node_parts.append(self._options.extra_node_style)
        node_style = ", ".join(node_parts)

        # --- per-column styles ---
        col_styles: list[str] = []
        # col_align_chars[i] = \makebox alignment character for TikZ column i+1.
        col_align_chars: list[str] = []
        for col_i in range(ncols):
            parts: list[str] = []
            align = "center"
            fixed_w: Optional[str] = None
            if alignments and col_i < len(alignments):
                spec = alignments[col_i]
                fixed_w = self._col_spec_fixed_width(spec)
                align = self._col_spec_to_tikz_align(spec)
            explicit_w = self._options.col_widths.get(col_i)
            effective_w = explicit_w or fixed_w
            if effective_w:
                parts.append(f"minimum width={effective_w}")
                if fixed_w and not explicit_w:
                    parts.append(f"text width={fixed_w}")
            if align == "left":
                parts.append("align=left")
                parts.append("anchor=west")
            elif align == "right":
                parts.append("align=right")
                parts.append("anchor=east")
            if parts:
                col_styles.append(f"  column {col_i + 1}/.style={{{', '.join(parts)}}}")
            col_align_chars.append("l" if align == "left" else "r" if align == "right" else "c")

        # When cell_width / normalize_cell_text_metrics are active, wrap
        # cell text via short helper macros (\gtcell / \gtcellbox / \gtcellraise)
        # defined at the top of the picture.  The macros expand to
        # \makebox / \raisebox / \smash combinations that
        #   * force the node to cell_width regardless of content
        #     (subtracting 2*inner_sep so total width = cell_width); and
        #   * normalize ascender/descender so anchors line up.
        # The dimension expressions live in single \def's at the top of
        # the picture, so the matrix body stays readable.
        _has_makebox = bool(self._options.cell_width)
        _has_raisebox = bool(
            self._options.normalize_cell_text_metrics and self._options.cell_height
        )

        def _wrap_cell_text(text: str, align_char: str) -> str:
            if _has_makebox and _has_raisebox:
                return f"\\gtcell{{{align_char}}}{{{text}}}"
            if _has_makebox:
                return f"\\gtcellbox{{{align_char}}}{{{text}}}"
            if _has_raisebox:
                return f"\\gtcellraise{{{text}}}"
            return text

        # --- matrix preamble ---
        _half_col_sep = r"\gtcolsephalf"

        lines.append("")
        lines.append("% --- matrix (cell content) ---")
        lines.append("\\matrix (table) [")
        lines.append("  matrix of nodes,")
        lines.append("  nodes in empty cells,")
        lines.append(f"  row sep={self._options.row_sep},")
        lines.append(f"  column sep={self._options.column_sep},")
        lines.append(f"  nodes={{{node_style}}},")
        for cs in col_styles:
            lines.append(f"  {cs},")
        lines.append("] {")

        tikz_row = 1

        # ---- build column-header texts (needed by both header rows) ----
        col_header_texts: list[str] = []
        if self._options.include_column_headers or (
            has_groups and self._options.include_group_headers
        ):
            if self._options.include_index:
                idx_name = (
                    self._options.index_name
                    if self._options.index_name is not None
                    else (self.df.index.name if self.df.index.name is not None else "")
                )
                text = latex_escape(str(idx_name))
                if self._options.bold_column_headers:
                    text = rf"\textbf{{{text}}}"
                if self._options.italic_column_headers:
                    text = rf"\textit{{{text}}}"
                col_header_texts.append(text)
            for _, col_list in self._options.groups_to_cols.items():
                for col in col_list:
                    text = latex_escape(str(col))
                    if self._options.bold_column_headers:
                        text = rf"\textbf{{{text}}}"
                    if self._options.italic_column_headers:
                        text = rf"\textit{{{text}}}"
                    col_header_texts.append(text)

        # When forcing cell width or normalizing text metrics, wrap
        # header texts in the same helper macros used for data cells.
        if (_has_makebox or _has_raisebox) and col_header_texts:
            col_header_texts = [
                _wrap_cell_text(
                    t,
                    col_align_chars[i] if i < len(col_align_chars) else "c",
                )
                for i, t in enumerate(col_header_texts)
            ]

        # ---- group-header row ----
        # Use \phantom of column header text in each cell so the nodes are
        # at least as wide as the column headers.  This ensures node anchors
        # (.north east etc.) align vertically with data-row anchors.
        if has_groups and self._options.include_group_headers:
            phantom_cells: list[str] = []
            for ht in col_header_texts:
                phantom_cells.append(rf"\phantom{{{ht}}}")
            lines.append("  " + " & ".join(phantom_cells) + r" \\")
            tikz_row += 1

        # ---- column-header row ----
        if self._options.include_column_headers:
            lines.append("  " + " & ".join(col_header_texts) + r" \\")
            tikz_row += 1

        # ---- data rows ----
        # Row highlighting is drawn on the background layer after the
        # matrix.  Per-cell \cellcolor{} from formatters is converted to
        # TikZ |[fill=...]| per-cell overrides.  cellcolor_map dedups
        # by (model, value): the diverging gradient formatter often
        # produces identical hex codes for nearby cells, and we want a
        # single \definecolor per unique colour.
        cellcolor_map: dict[tuple[str, str], str] = {}
        # Per-cell fills collected as (tikz_row, col_1based, kind, value, raw_value):
        #   kind="hex"  → value is a 6-char HEX, raw_value is the source cell
        #                 value (used to derive a meaningful colour name).
        #   kind="name" → value is an xcolor name; raw_value is unused.
        # Hex fills end up grouped by row into a single \gtcellrowfills
        # call referring to named colours; name fills emit one
        # \gtcellfill per cell.
        cell_fills: list[tuple[int, int, str, str, object]] = []
        data_start_tikz = tikz_row
        for row_idx, (df_row_idx, row) in enumerate(self.df.iterrows()):
            cells: list[str] = []
            data_col_i = 0

            if self._options.include_index:
                raw = str(df_row_idx)
                esc = latex_escape(raw)
                text = (
                    self._options.index_fmt_fn(cast(TableIndexValue, df_row_idx), esc)[1]
                    if self._options.index_fmt_fn is not None
                    else esc
                )
                pre, post = self._col_decorators(data_col_i)
                text = self._apply_decorators(text, pre, post)
                fill_kind, fill_value, text = self._strip_cellcolor(text, cellcolor_map)
                if fill_kind:
                    cell_fills.append((tikz_row, data_col_i + 1, fill_kind, fill_value, df_row_idx))
                if _has_makebox or _has_raisebox:
                    ma = col_align_chars[data_col_i] if data_col_i < len(col_align_chars) else "c"
                    text = _wrap_cell_text(text, ma)
                cells.append(text)
                data_col_i += 1

            for col in column_ordering:
                raw_value = row[col]
                text = self._format_cell_value(row_idx, col, raw_value)
                pre, post = self._col_decorators(data_col_i)
                text = self._apply_decorators(text, pre, post)
                fill_kind, fill_value, text = self._strip_cellcolor(text, cellcolor_map)
                if fill_kind:
                    cell_fills.append((tikz_row, data_col_i + 1, fill_kind, fill_value, raw_value))
                if _has_makebox or _has_raisebox:
                    ma = col_align_chars[data_col_i] if data_col_i < len(col_align_chars) else "c"
                    text = _wrap_cell_text(text, ma)
                cells.append(text)
                data_col_i += 1

            lines.append("  " + " & ".join(cells) + r" \\")
            tikz_row += 1

        lines.append("};")

        nrows = tikz_row - 1  # total TikZ rows emitted

        # ==============================================================
        # Post-matrix draws — rules, overlays, and fills
        # ==============================================================

        # Build fit nodes for columns/rows that need canonical boundaries.
        # In a TikZ matrix of nodes, each node sizes to its own content so
        # anchors can vary within the same visual row/column. Full-row/full-
        # column fit nodes give stable geometry for fills and rules.
        _colfit_needed: set[int] = set()
        _rowfit_needed: set[int] = set()

        def _ensure_colfit(col_1based: int) -> str:
            """Mark column ``col_1based`` for a canonical fit node and return its name."""
            _colfit_needed.add(col_1based)
            return f"colfitV{col_1based}"

        def _ensure_rowfit(row_1based: int) -> str:
            """Mark row ``row_1based`` for a canonical fit node and return its name."""
            _rowfit_needed.add(row_1based)
            return f"rowfitH{row_1based}"

        # Pre-create colfit nodes for leftmost and rightmost columns so
        # that row highlights and hlines span the true column boundaries
        # rather than the narrow data-cell boundaries.
        colfit_left = _ensure_colfit(1)
        colfit_right = _ensure_colfit(ncols)
        post_insert_idx = len(lines)

        # Macro-need flags — populated as we emit and consumed when we
        # build the picture-local macro preamble.
        needs_gtrowfill = False
        needs_gtcellfill = False
        needs_gtcellrowfills = False
        needs_gthline = False
        needs_gthlinestyled = False

        # ---- row highlighting (background layer) ----
        # Use \fill on the background layer so the colour sits behind the
        # text and spans the full row width (edge-to-edge).
        bg_fill_groups: dict[str, list[int]] = {}
        for row_idx in range(len(self.df)):
            color = self._row_fill_color(row_idx)
            if not color:
                continue
            tgt = data_start_tikz + row_idx
            _ensure_rowfit(tgt)
            bg_fill_groups.setdefault(color, []).append(tgt)
        if bg_fill_groups:
            needs_gtrowfill = True

        # ---- per-cell highlights (background layer, drawn over row fills) ----
        # Hex-typed fills get a stable, descriptive xcolor name derived
        # from the source cell value (e.g. value 0.549 → ``gradc_55``).
        # Each row's hex fills are then collapsed into a single
        # ``\gtcellrowfills{row}{col1/name1, col2/name2, ...}`` call.
        # Multiple distinct hexes that fall in the same value bucket
        # get suffixes (gradc_55, gradc_55_b, gradc_55_c, ...).
        # Named-colour fills (teal, gray!50, fallback tikzcc<N>) emit
        # one \gtcellfill{name}{row}{col} per cell.
        hex_fills_by_row: dict[int, list[tuple[int, str]]] = {}
        named_fills: list[tuple[int, int, str]] = []  # (row, col, name)
        # value-bucket -> {hex -> assigned color name}
        gradient_buckets: dict[str, dict[str, str]] = {}
        # ordered list of (color name, hex) for emission as \definecolor
        gradient_color_defs: list[tuple[str, str]] = []
        for tgt_row, col_1based, kind, value, raw_value in cell_fills:
            _ensure_colfit(col_1based)
            _ensure_rowfit(tgt_row)
            if kind == "hex":
                base = self._value_to_color_basename(raw_value)
                bucket = gradient_buckets.setdefault(base, {})
                if value in bucket:
                    cname = bucket[value]
                elif not bucket:
                    cname = base
                    bucket[value] = cname
                    gradient_color_defs.append((cname, value))
                else:
                    suffix = chr(ord("b") + len(bucket) - 1)  # b, c, d, ...
                    cname = f"{base}_{suffix}"
                    bucket[value] = cname
                    gradient_color_defs.append((cname, value))
                hex_fills_by_row.setdefault(tgt_row, []).append((col_1based, cname))
            else:
                named_fills.append((tgt_row, col_1based, value))
        if hex_fills_by_row:
            needs_gtcellrowfills = True
        if named_fills:
            needs_gtcellfill = True

        # ---- group-header overlay nodes ----
        if has_groups and self._options.include_group_headers:
            lines.append("")
            lines.append("% --- group-header overlay nodes ---")
            col_tikz = 2 if self._options.include_index else 1
            g_aligns = self._options.group_tabular_alignments
            g_idx = 0
            for group, cols in self._options.groups_to_cols.items():
                span = len(cols)
                if span == 0:
                    continue
                left_col = col_tikz
                right_col = col_tikz + span - 1
                name = latex_escape(str(group))
                if self._options.bold_group_headers and name:
                    name = rf"\textbf{{{name}}}"
                if self._options.italic_group_headers and name:
                    name = rf"\textit{{{name}}}"
                # Determine alignment from group tabular format.
                anchor = "center"
                if g_aligns and g_idx < len(g_aligns):
                    ga = self._col_spec_to_tikz_align(g_aligns[g_idx])
                    if ga == "left":
                        anchor = "west"
                    elif ga == "right":
                        anchor = "east"
                node_opts = f"inner sep=0pt, fit=(table-1-{left_col})(table-1-{right_col})"
                if anchor != "center":
                    # Use an overlay: fit the area, then place text at the
                    # appropriate anchor.
                    lines.append(f"\\node ({group.replace(' ', '')}_fit) [{node_opts}] {{}};")
                    lines.append(
                        f"\\node[anchor={anchor}]"
                        f" at ({group.replace(' ', '')}_fit.{anchor})"
                        f" {{{name}}};"
                    )
                else:
                    lines.append(f"\\node[{node_opts}] {{{name}}};")
                col_tikz += span
                g_idx += 1

        # ---- horizontal rules ----
        # hrule_counts[i] == number of lines above data row i.
        # hrule_counts[ndata] (optional extra entry) == lines below the last row.
        # This is the TikZ equivalent of \hline in the TexTable body.
        # The common north-edge case is emitted via \gthline{row} / \gthlinestyled{style}{row}
        # macros for readability; the rare south-edge case is emitted inline.
        if self._options.hrule_counts and any(c > 0 for c in self._options.hrule_counts):
            lines.append("")
            lines.append("% --- horizontal rules ---")
            ndata_rows = len(self.df)
            for row_idx, count in enumerate(self._options.hrule_counts):
                if count == 0:
                    continue
                if row_idx < ndata_rows:
                    tgt = data_start_tikz + row_idx
                    edge = "north"
                else:
                    tgt = data_start_tikz + ndata_rows - 1
                    edge = "south"
                _ensure_rowfit(tgt)
                for k in range(count):
                    yshift = f"yshift={k * 0.4}pt" if k > 0 else ""
                    opts = ", ".join(filter(None, [self._options.hrule_style, yshift]))
                    if edge == "north" and not opts:
                        lines.append(f"\\gthline{{{tgt}}}")
                        needs_gthline = True
                    elif edge == "north":
                        lines.append(f"\\gthlinestyled{{{opts}}}{{{tgt}}}")
                        needs_gthlinestyled = True
                    else:
                        style_str = f"[{opts}]" if opts else ""
                        lines.append(
                            f"\\draw{style_str}"
                            f" ({colfit_left}.west |- rowfitH{tgt}.{edge}) --"
                            f" ({colfit_right}.east |- rowfitH{tgt}.{edge});"
                        )

        # ---- top / bottom rules ----
        if self._options.toprule_style is not None or self._options.bottomrule_style is not None:
            lines.append("")
            lines.append("% --- top / bottom rules ---")
        if self._options.toprule_style is not None:
            s = f"[{self._options.toprule_style}]" if self._options.toprule_style else ""
            lines.append(f"\\draw{s} (table.north west) -- (table.north east);")
        if self._options.bottomrule_style is not None:
            s = f"[{self._options.bottomrule_style}]" if self._options.bottomrule_style else ""
            lines.append(f"\\draw{s} (table.south west) -- (table.south east);")

        # ---- vertical rules ----
        # vrule_counts[j] == number of vertical lines at boundary j.
        # Boundary 0 = left edge, boundary ncols = right edge.
        #
        # In a TikZ matrix of nodes, each node sizes to its own content
        # so node anchors (.north east) vary per row even within the
        # same column.  To get the true column boundary x-position we
        # create a \node[fit=...] spanning all rows for each column
        # that needs a vrule.  The fit node's edge is at the widest
        # node in the column — the correct column boundary.
        #
        # When a group-header row is present, vrules extend up through
        # it ONLY at group boundaries (matching \multicolumn in tabular).
        group_header_present = has_groups and self._options.include_group_headers
        vrule_top_row = 2 if group_header_present else 1

        # Compute group-boundary set.
        group_boundary_set: set[int] = set()
        if group_header_present:
            gb = 1 if self._options.include_index else 0
            group_boundary_set.add(gb)
            for cols in self._options.groups_to_cols.values():
                gb += len(cols)
                group_boundary_set.add(gb)
            group_boundary_set.add(0)
            group_boundary_set.add(ncols)

        if self._options.vrule_counts and any(c > 0 for c in self._options.vrule_counts):
            lines.append("")
            lines.append("% --- vertical rules ---")
            for bdry, count in enumerate(self._options.vrule_counts):
                if count == 0:
                    continue
                top = (
                    1 if (not group_header_present or bdry in group_boundary_set) else vrule_top_row
                )
                for k in range(count):
                    xshift = f"xshift={k * 0.4}pt" if k > 0 else ""
                    opts = ", ".join(filter(None, [self._options.vrule_style, xshift]))
                    style_str = f"[{opts}]" if opts else ""
                    if bdry == 0:
                        fit = _ensure_colfit(1)
                        lines.append(
                            f"\\draw{style_str}"
                            f" ({fit}.north west |- table-{top}-1.north) --"
                            f" ({fit}.south west);"
                        )
                    elif bdry == ncols:
                        fit = _ensure_colfit(ncols)
                        lines.append(
                            f"\\draw{style_str}"
                            f" ({fit}.north east |- table-{top}-1.north) --"
                            f" ({fit}.south east);"
                        )
                    else:
                        fit = _ensure_colfit(bdry)
                        lines.append(
                            f"\\draw{style_str}"
                            f" ({fit}.north east |- table-{top}-1.north) --"
                            f" ({fit}.south east);"
                        )

        # ---- per-cell borders ----
        # Convert (r, c, side) requests into canonical boundary segments so
        # that shared edges between adjacent cells are drawn exactly once:
        #   horiz_edges: (c, r_bnd) — horizontal line in column c
        #     r_bnd == 0  → north of TikZ row 1
        #     r_bnd >= 1  → south of TikZ row r_bnd  (= north of row r_bnd+1)
        #   vert_edges:  (r, c_bnd) — vertical line in TikZ row r
        #     c_bnd == 0      → west of column 1
        #     c_bnd == ncols  → east of column ncols
        #     else            → midpoint of gap between column c_bnd and c_bnd+1
        #                       drawn at [xshift=3pt]colfitV{c_bnd}.east
        horiz_edges: set[tuple[int, int]] = set()
        vert_edges: set[tuple[int, int]] = set()
        for (r, c), sides in self._options.cell_borders.items():
            for side in sides:
                if side == "top":
                    horiz_edges.add((c, r - 1))
                elif side == "bottom":
                    horiz_edges.add((c, r))
                elif side == "left":
                    vert_edges.add((r, c - 1))
                elif side == "right":
                    vert_edges.add((r, c))

        if horiz_edges or vert_edges:
            lines.append("")
            lines.append("% --- per-cell borders ---")

        for c, r_bnd in sorted(horiz_edges):
            _ensure_colfit(c)
            colfit = f"colfitV{c}"
            lmod = "" if c == 1 else f"[xshift=-{_half_col_sep}]"
            rmod = "" if c == ncols else f"[xshift={_half_col_sep}]"
            row_ref, edge = (1, "north") if r_bnd == 0 else (r_bnd, "south")
            rowfit = _ensure_rowfit(row_ref)
            lines.append(
                f"\\draw"
                f" ({lmod}{colfit}.west |- {rowfit}.{edge}) --"
                f" ({rmod}{colfit}.east |- {rowfit}.{edge});"
            )

        for r, c_bnd in sorted(vert_edges):
            rowfit = _ensure_rowfit(r)
            if c_bnd == 0:
                _ensure_colfit(1)
                x = "colfitV1.west"
            elif c_bnd == ncols:
                _ensure_colfit(ncols)
                x = f"colfitV{ncols}.east"
            else:
                _ensure_colfit(c_bnd)
                x = f"[xshift={_half_col_sep}]colfitV{c_bnd}.east"
            lines.append(f"\\draw ({x} |- {rowfit}.north) -- ({x} |- {rowfit}.south);")

        post_prefix: list[str] = []
        if _colfit_needed or _rowfit_needed:
            post_prefix.append("")
            post_prefix.append("% --- canonical row/column fit nodes ---")
        if _colfit_needed:
            col_targets = "".join(f"(table-{row_1based}-\\c)" for row_1based in range(1, nrows + 1))
            post_prefix.append(f"\\foreach \\c in {{{_latex_foreach_list(_colfit_needed)}}} {{")
            post_prefix.append(f"  \\node[fit={col_targets}, inner sep=0pt] (colfitV\\c) {{}};")
            post_prefix.append("}")
        if _rowfit_needed:
            row_targets = "".join(f"(table-\\r-{col_1based})" for col_1based in range(1, ncols + 1))
            post_prefix.append(f"\\foreach \\r in {{{_latex_foreach_list(_rowfit_needed)}}} {{")
            post_prefix.append(f"  \\node[fit={row_targets}, inner sep=0pt] (rowfitH\\r) {{}};")
            post_prefix.append("}")
        if bg_fill_groups or hex_fills_by_row or named_fills:
            post_prefix.append("")
            post_prefix.append("% --- background fills (row highlights + per-cell colours) ---")
            post_prefix.append("\\begin{scope}[on background layer]")
            for color, rows in bg_fill_groups.items():
                post_prefix.append(f"  \\foreach \\r in {{{_latex_foreach_list(rows)}}} {{")
                post_prefix.append(f"    \\gtrowfill{{{color}}}{{\\r}}")
                post_prefix.append("  }")
            # One \gtcellrowfills call per row carrying every hex-coloured cell.
            for row in sorted(hex_fills_by_row):
                pairs = sorted(hex_fills_by_row[row])
                body = ", ".join(f"{c}/{h}" for c, h in pairs)
                post_prefix.append(f"  \\gtcellrowfills{{{row}}}{{{body}}}")
            # Named-colour cells emit one fill apiece (rare).
            for tgt_row, col_1based, name in named_fills:
                post_prefix.append(f"  \\gtcellfill{{{name}}}{{{tgt_row}}}{{{col_1based}}}")
            post_prefix.append("\\end{scope}")
        lines[post_insert_idx:post_insert_idx] = post_prefix

        # ---- user extra draws ----
        if self._options.extra_draws:
            lines.append("")
            lines.append("% --- user extra draws ---")
            lines.extend(self._options.extra_draws)

        # ---- picture-local macro preamble ----
        # Definitions go at the very top of the picture so the body is
        # readable.  Each macro is only emitted if something below uses
        # it; \gtcolsephalf is always defined since cell fills, cell
        # borders, and per-cell highlights all reference it.
        macro_lines: list[str] = ["% --- gerrytools tikz table macros (picture-local) ---"]
        macro_lines.append("% \\gtcolsephalf: half of the matrix column separation, used to extend")
        macro_lines.append("%   per-cell borders into adjacent inter-column gaps.")
        macro_lines.append(f"\\def\\gtcolsephalf{{\\dimexpr {self._options.column_sep}/2\\relax}}")
        if _has_makebox:
            macro_lines.append(
                "% \\gtboxwidth: width of the inner makebox so a cell's total width"
                " (content + 2*inner_sep) equals the requested cell_width."
            )
            macro_lines.append(
                f"\\def\\gtboxwidth{{\\dimexpr {self._options.cell_width}"
                f"-{self._options.inner_sep}*2\\relax}}"
            )
        if _has_raisebox:
            macro_lines.append(
                "% \\gtraiseheight / \\gtraiseshift: half-height and vertical shift used by"
                " the \\raisebox wrapper to normalise ascender/descender so all rows"
                " share the same baseline anchor."
            )
            macro_lines.append(
                f"\\def\\gtraiseheight{{\\dimexpr {self._options.cell_height}"
                f"/2-{self._options.inner_sep}\\relax}}"
            )
            macro_lines.append(r"\def\gtraiseshift{\dimexpr (\dp\strutbox-\ht\strutbox)/2\relax}")
        if _has_makebox and _has_raisebox:
            macro_lines.append(
                "% \\gtcell{align}{text}: standard cell wrapper used in the matrix body."
                "  Forces the node to the configured cell_width and centres the"
                " baseline so anchors line up across rows."
            )
            macro_lines.append(
                r"\def\gtcell#1#2{\raisebox{\gtraiseshift}"
                r"[\gtraiseheight][\gtraiseheight]"
                r"{\smash{\strut \makebox[\gtboxwidth][#1]{#2}}}}"
            )
        elif _has_makebox:
            macro_lines.append(
                "% \\gtcellbox{align}{text}: cell wrapper that only forces width"
                " (no baseline normalisation)."
            )
            macro_lines.append(r"\def\gtcellbox#1#2{\makebox[\gtboxwidth][#1]{#2}}")
        elif _has_raisebox:
            macro_lines.append(
                "% \\gtcellraise{text}: cell wrapper that only normalises baseline"
                " (no fixed width)."
            )
            macro_lines.append(
                r"\def\gtcellraise#1{\raisebox{\gtraiseshift}"
                r"[\gtraiseheight][\gtraiseheight]{\smash{\strut #1}}}"
            )
        if needs_gthline:
            macro_lines.append(
                "% \\gthline{row}: horizontal rule across the table at the top of"
                " the given TikZ row (1-indexed, header rows count)."
            )
            macro_lines.append(
                f"\\def\\gthline#1{{\\draw (colfitV1.west |- rowfitH#1.north)"
                f" -- (colfitV{ncols}.east |- rowfitH#1.north);}}"
            )
        if needs_gthlinestyled:
            macro_lines.append(
                "% \\gthlinestyled{tikz options}{row}: same as \\gthline with extra"
                " TikZ draw options (e.g. yshift=0.4pt for double rules)."
            )
            macro_lines.append(
                f"\\def\\gthlinestyled#1#2{{\\draw[#1] (colfitV1.west |- rowfitH#2.north)"
                f" -- (colfitV{ncols}.east |- rowfitH#2.north);}}"
            )
        if needs_gtrowfill:
            macro_lines.append(
                "% \\gtrowfill{color}{row}: fill the entire row width with <color>"
                " (an xcolor name or expression like gray!50)."
            )
            macro_lines.append(
                f"\\def\\gtrowfill#1#2{{\\fill[fill={{#1}}]"
                f" (colfitV1.west |- rowfitH#2.north)"
                f" rectangle (colfitV{ncols}.east |- rowfitH#2.south);}}"
            )
        if needs_gtcellfill:
            macro_lines.append(
                "% \\gtcellfill{color}{row}{col}: fill a single cell with the named xcolor <color>."
            )
            macro_lines.append(
                r"\def\gtcellfill#1#2#3{\fill[fill={#1}]"
                r" (colfitV#3.west |- rowfitH#2.north) rectangle"
                r" (colfitV#3.east |- rowfitH#2.south);}"
            )
        if needs_gtcellrowfills:
            macro_lines.append(
                "% \\gtcellrowfills{row}{col1/name1, col2/name2, ...}: row-grouped"
                " cell fills.  Names refer to \\definecolor entries above; their"
                " suffix encodes the source value (e.g. gradc_55 ~ value 0.55)."
            )
            macro_lines.append(
                r"\def\gtcellrowfills#1#2{\foreach \gtc/\gtn in {#2} {"
                r"\fill[fill={\gtn}]"
                r" (colfitV\gtc.west |- rowfitH#1.north) rectangle"
                r" (colfitV\gtc.east |- rowfitH#1.south);}}"
            )
        # Build the consolidated \definecolor block from every source:
        # gradient cells (named by source value), unknown-model fallback
        # (cellcolor_map), and row-highlight colours (row_highlight_color_defs
        # was already extended into ``lines`` at color_def_insert_idx).
        color_def_lines: list[str] = []
        if gradient_color_defs:
            color_def_lines.extend(
                f"\\definecolor{{{name}}}{{HTML}}{{{hex_value}}}"
                for name, hex_value in gradient_color_defs
            )
        if cellcolor_map:
            color_def_lines.extend(
                f"\\definecolor{{{name}}}{{{model}}}{{{value}}}"
                for (model, value), name in cellcolor_map.items()
            )
        if color_def_lines or row_highlight_color_defs:
            header = ["", "% --- color definitions ---"]
            for i, cd in enumerate(header + color_def_lines):
                lines.insert(color_def_insert_idx + i, cd)
        lines[0:0] = macro_lines

        body = "\n".join(lines)
        return f"\\begin{{tikzpicture}}\n{body}\n\\end{{tikzpicture}}"
