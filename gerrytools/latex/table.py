import inspect
import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, cast

import pandas as pd

from gerrytools.latex._colors import to_latex_xcolor_or_html_spec
from gerrytools.latex._table_preamble import (
    _infer_group_cell_align_from_data,
    _parse_tabular_preamble,
)
from gerrytools.latex._text import latex_escape
from gerrytools.latex.document import TexDocument
from gerrytools.latex.formatters import round_decimals
from gerrytools.logging import get_logger
from gerrytools.typing import CellWrapper, Color, TableCellValue

logger = get_logger(__name__)


def _latex_escape_wrapper(value: TableCellValue, prev: str) -> tuple[TableCellValue, str]:
    """Wrapper function to escape LaTeX special characters in strings.

    Args:
        value (TableCellValue): The original unformatted value.
        prev (str): The currently rendered string.

    Returns:
        tuple[TableCellValue, str]: The original value and the newly escaped version of the
            previously rendered string.
    """
    return value, latex_escape(prev)


@dataclass
class TableOptions:
    r"""Dataclass intended to help track all possible options for a latex table.

    Attributes:
        toprule_cmd (str | None): LaTeX command for the top rule of the table. Default is None.
        bottomrule_cmd (str | None): LaTeX command for the bottom rule of the table.
            Default is None.
        hrule_cmd (str): LaTeX command for horizontal rules in the table. Default is r"\hline".
        include_index (bool): Whether to include the DataFrame index in the table.
            Default is False.
        index_name (Optional[str]): Name to use for the index column header.
            If None and include_index is True, uses df.index.name or "". Default is None.
        index_alignment (Optional[str]): Alignment for the index column. Default is None.
        nan_string (str): String to represent NaN values in the table. Default is "NaN".
        hrule_counts (list[int]): List of counts of horizontal rules at each boundary.
            Boundary 0 is after header, k is after row k-1. Default is empty list.
        vrule_counts (list[int]): List of counts of vertical rules at each boundary.
            Boundary 0 is before the left most column. Default is empty list.
        tabular_alignments (list[str]): List of column alignments for the tabular environment.
            Default is empty list.
        boundary_extras (list[str]): List of extra LaTeX code to insert at each boundary.
            Default is empty list.
        group_vrule_counts (list[int] | None): List of counts of vertical rules for group headers.
            Default is None.
        group_tabular_alignments (list[str] | None): List of column alignments for group headers.
            Default is None.
        group_boundary_extras (list[str] | None): List of extra LaTeX code for group header
            boundaries. Default is None.
        row_highlight_colors (list[tuple[str, Color]]): List of tuples specifying
            row highlight colors. Default is empty list.
        number_fmt_fn (Optional[Wrapper]): Formatter function for numerical values. Default is None.
        str_fmt_fn (Optional[Wrapper]): Formatter function for string values. Default is
            latex_escape_wrapper.
        col_formatters (dict[str, Wrapper]): Dictionary mapping column names to formatter functions.
            Default is empty dict.
        row_formatters (dict[int, Wrapper]): Dictionary mapping row indices to formatter functions.
            Default is empty dict.
        groups_to_cols (dict[str, list[str]]): Dictionary mapping group names to lists of column
            names. Default is empty dict.
        bold_group_headers (bool): Whether to bold group headers. Default is True.
        italic_group_headers (bool): Whether to italicize group headers. Default is False.
        bold_column_headers (bool): Whether to bold column headers. Default is False.
        italic_column_headers (bool): Whether to italicize column headers. Default is False.
    """

    toprule_cmd: str | None = None
    bottomrule_cmd: str | None = None
    hrule_cmd: str = r"\hline"

    include_column_headers: bool = True
    include_group_headers: bool = True

    include_index: bool = False
    index_name: Optional[str] = None  # if None and include_index True, uses df.index.name or ""
    index_alignment: Optional[str] = None

    nan_string: str = "NaN"

    # boundary -> count, where boundary 0 is after header, k is after row k-1
    hrule_counts: list[int] = field(default_factory=list)

    # boundary -> count, where boundary 0 is before the left most column
    vrule_counts: list[int] = field(default_factory=list)
    tabular_alignments: list[str] = field(default_factory=list)
    boundary_extras: list[str] = field(default_factory=list)

    group_vrule_counts: list[int] | None = None
    group_tabular_alignments: list[str] | None = None
    group_boundary_extras: list[str] | None = None

    row_highlight_colors: list[tuple[str, Color]] = field(default_factory=list)

    number_fmt_fn: Optional[CellWrapper] = None
    str_fmt_fn: Optional[CellWrapper] = _latex_escape_wrapper
    col_formatters: dict[str, CellWrapper] = field(default_factory=dict)
    row_formatters: dict[int, CellWrapper] = field(default_factory=dict)
    index_fmt_fn: Optional[CellWrapper] = None

    groups_to_cols: dict[str, list[str]] = field(default_factory=dict)

    bold_group_headers: bool = True
    italic_group_headers: bool = False
    bold_column_headers: bool = False
    italic_column_headers: bool = False

    @property
    def column_format(self) -> str:
        """Generate the tabular preamble (e.g. '|c|p{2cm}||S[...]|')."""
        cols = self.tabular_alignments
        n = len(cols)

        # default extras to empty strings
        extras = self.boundary_extras or ([""] * (n + 1))
        if len(extras) != n + 1:
            raise ValueError("boundary_extras must have length ncols+1")

        fmt = ""
        for i, colspec in enumerate(cols):
            fmt += ("|" * self.vrule_counts[i]) + extras[i] + colspec
        fmt += ("|" * self.vrule_counts[n]) + extras[n]
        return fmt

    @property
    def multicolumn_format(self) -> str:
        r"""Generate the multicolumn header row (group headers).

        Divider rule: boundary dividers belong to the *preceding* cell.
        Implementation: only the first emitted cell includes its left boundary.
        Every cell includes its right boundary.
        """
        group_items = [(g, len(cols)) for g, cols in self.groups_to_cols.items()]
        group_cell_count = len(group_items) + (1 if self.include_index else 0)

        def _normalize_preamble(
            cols: list[str], vr: list[int], ex: list[str]
        ) -> tuple[list[str], list[int], list[str]]:
            """Validate preamble vector lengths for one header row.

            Args:
                cols (list[str]): Column alignment specifications.
                vr (list[int]): Vertical-rule counts with one boundary slot per column edge.
                ex (list[str]): Boundary-extra token strings for each column edge.

            Returns:
                tuple[list[str], list[int], list[str]]: The original inputs when lengths match.

            Raises:
                ValueError: If ``vr`` or ``ex`` lengths do not equal ``len(cols) + 1``.
            """
            n = len(cols)
            if len(vr) != n + 1:
                raise ValueError("vrule_counts must have length ncols+1")
            if len(ex) != n + 1:
                raise ValueError("boundary_extras must have length ncols+1")
            return cols, vr, ex

        def _mc_colspec(
            vr: list[int],
            ex: list[str],
            left_b: int,
            align: str,
            right_b: int,
            *,
            include_left: bool,
        ) -> str:
            """Build one ``\\multicolumn`` colspec from parsed boundary tokens.

            Args:
                vr (list[int]): Vertical-rule counts at boundaries.
                ex (list[str]): Boundary-extra token strings at boundaries.
                left_b (int): Left boundary index.
                align (str): Cell alignment token.
                right_b (int): Right boundary index.
                include_left (bool): Whether to include the left boundary tokens.

            Returns:
                str: A ``\\multicolumn`` colspec string.
            """
            left = (("|" * vr[left_b]) + ex[left_b]) if include_left else ""
            right = ("|" * vr[right_b]) + ex[right_b]
            return left + align + right

        parts: list[str] = []
        first_cell = True

        # ---------- Mode A: group-header preamble drives alignment + boundaries ----------
        if self.group_tabular_alignments is not None:
            if self.group_vrule_counts is None or self.group_boundary_extras is None:
                raise ValueError(
                    "group_* preamble is partially set; set alignments, vrules, and extras together."
                )

            gcols, gvr, gex = _normalize_preamble(
                self.group_tabular_alignments,
                self.group_vrule_counts,
                self.group_boundary_extras,
            )
            if self.include_index:
                index_colspecs, index_vrules, index_extras = _parse_tabular_preamble(
                    self.index_alignment or "c"
                )
                gcols = [index_colspecs[0]] + gcols
                gvr = [index_vrules[0]] + gvr
                gex = [index_extras[0]] + gex
            if len(gcols) != group_cell_count:
                raise ValueError(
                    f"Group-header preamble has {len(gcols)} cells but expected {group_cell_count}."
                )

            cell_i = 0

            if self.include_index:
                idx_align = self.index_alignment or "c"
                col_spec = _mc_colspec(
                    gvr, gex, cell_i, idx_align, cell_i + 1, include_left=first_cell
                )
                parts.append(rf"\multicolumn{{1}}{{{col_spec}}}{{}}")
                first_cell = False
                cell_i += 1

            for group, span in group_items:
                if span == 0:
                    continue

                align = gcols[cell_i]
                name = latex_escape(str(group))
                if self.bold_group_headers and name:
                    name = rf"\textbf{{{name}}}"
                if self.italic_group_headers and name:
                    name = rf"\textit{{{name}}}"

                col_spec = _mc_colspec(gvr, gex, cell_i, align, cell_i + 1, include_left=first_cell)
                parts.append(rf"\multicolumn{{{span}}}{{{col_spec}}}{{{name}}}")
                first_cell = False
                cell_i += 1

            return " & ".join(parts) + r" \\"

        # ---------- Mode B: infer group boundaries from column preamble ----------
        dcols = self.tabular_alignments
        ncols_total = len(dcols)
        dex = [""] * (ncols_total + 1)
        dvr = self.vrule_counts
        _normalize_preamble(dcols, dvr, dex)

        if self.include_index:
            idx_align = self.index_alignment or "c"
            col_spec = _mc_colspec(dvr, dex, 0, idx_align, 1, include_left=first_cell)
            parts.append(rf"\multicolumn{{1}}{{{col_spec}}}{{}}")
            first_cell = False

        data_start = 1 if self.include_index else 0
        offset = 0
        for group, span in group_items:
            if span == 0:
                continue

            left_b = data_start + offset
            right_b = left_b + span
            offset += span

            # inferred alignment for this group cell
            align = _infer_group_cell_align_from_data(dcols, left_b, right_b)

            name = latex_escape(str(group))
            if self.bold_group_headers and name:
                name = rf"\textbf{{{name}}}"
            if self.italic_group_headers and name:
                name = rf"\textit{{{name}}}"

            col_spec = _mc_colspec(dvr, dex, left_b, align, right_b, include_left=first_cell)
            parts.append(rf"\multicolumn{{{span}}}{{{col_spec}}}{{{name}}}")
            first_cell = False

        return " & ".join(parts) + r" \\"


class TexTable:
    """Class for generating LaTeX table code from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be converted to a LaTeX table
        use_defaults (bool, optional): Whether to initialize with default table options
            (bold headers, 4 decimal places, etc.). Defaults to True.

    Attributes:
        df (pd.DataFrame): The DataFrame to be converted to a LaTeX table
    """

    def __init__(self, df: pd.DataFrame, *, use_defaults: bool = True) -> None:
        self.df = df.copy()
        self._document = TexDocument()
        self._document.add_packages("colortbl")
        self.df.index = self.df.index.map(str)
        if use_defaults:
            self._options = TableOptions(
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
            self._options = TableOptions(
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
        """Resets all table options to their default values."""
        self._options = TableOptions(
            groups_to_cols={"": list(self.df.columns)},
            tabular_alignments=["c"] * self.df.shape[1],
            row_highlight_colors=[("NONE", "")] * self.df.shape[0],
            vrule_counts=[0] * (self.df.shape[1] + 1),
        )

    def preview(self) -> None:  # pragma: no cover
        """Render and preview the table through its ``TexDocument``.

        Returns:
            None
        """
        self.document.preview()

    # ====================
    #   FEATURE ADDITION
    # ====================

    def include_index(
        self, name: Optional[str] = None, alignment: str = "c", include: bool = True
    ) -> None:
        """Adds or removes the DataFrame index column to/from the LaTeX table.

        Args:
            name (Optional[str], optional): Name to give the index in the table. Defaults to None.
            alignment (str, optional): Alignment of the index column. Defaults to "c".
            include (bool, optional): Whether to include the index column. Defaults to True.


        Raises:
            ValueError: If the number of tabular alignments in options is inconsistent
                with the DataFrame shape when including or removing the index.
        """

        self._options.index_alignment = alignment
        self._options.index_name = latex_escape(str(name)) if name is not None else ""

        n_data_cols = self.df.shape[1]

        # Ensure boundary_extras exists and has correct length when in parsed-mode
        def _ensure_extras(ncols: int) -> None:
            """Ensure ``boundary_extras`` exists with ``ncols + 1`` boundary slots.

            Args:
                ncols (int): Number of columns currently represented in tabular alignments.

            Returns:
                None
            """
            if not getattr(self._options, "boundary_extras", None):
                self._options.boundary_extras = [""] * (ncols + 1)
            if len(self._options.boundary_extras) != ncols + 1:
                self._options.boundary_extras = [""] * (ncols + 1)

        # ADD index
        if include and not self._options.include_index:
            self._options.include_index = True

            # expected current cols = data cols
            if len(self._options.tabular_alignments) != n_data_cols:
                raise ValueError(
                    "Current tabular format does not match DataFrame columns. Got "
                    f"{len(self._options.tabular_alignments)} colspecs but expected {n_data_cols}."
                )

            # Insert the new index column spec
            self._options.tabular_alignments.insert(0, alignment)

            # vrule_counts/extras: old length was n_data_cols+1, new length should be
            # (n_data_cols+1)+1
            old_ncols = n_data_cols
            _ensure_extras(old_ncols)

            # Insert a NEW boundary at position 1 (between new index col and old first col)
            # Keep boundary 0 (left edge) as-is.
            self._options.vrule_counts.insert(1, 0)
            self._options.boundary_extras.insert(1, "")

        # REMOVE index
        if (not include) and self._options.include_index:
            self._options.include_index = False

            # expected current cols = data cols + 1
            if len(self._options.tabular_alignments) != n_data_cols + 1:
                raise ValueError(
                    "Current tabular format does not match DataFrame columns+index. Got "
                    f"{len(self._options.tabular_alignments)} colspecs but expected "
                    f"{n_data_cols + 1}."
                )

            # Remove the index column spec
            self._options.tabular_alignments.pop(0)

            # Remove boundary 1 (between old index col and first data col)
            new_ncols = n_data_cols + 1
            _ensure_extras(new_ncols)

            self._options.vrule_counts.pop(1)
            self._options.boundary_extras.pop(1)

    def remove_index(self) -> None:
        """Remove the index from the generated latex table (if it exists)"""
        self.include_index(include=False)

    def add_hrule_above(self, index: int | list[int], count: int = 1) -> None:
        """Add horizontal rules above specified rows in the LaTeX table.

        Note: Row index 0 corresponds to the first row of data in the dataframe, so it
        will appear below the header row in the LaTeX table.

        Args:
            index (int | list[int]): Row index or list of row indices above which to add horizontal
                rules.
            count (int, optional): Number of horizontal rules to add. Defaults to 1.
        """
        if isinstance(index, int):
            indices = [index]
        else:
            indices = index

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

    def add_toprule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule to the top of the LaTeX table.

        Args:
            cmd (Optional[str], optional): LaTeX command for the top rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            cmd = self._options.hrule_cmd
        self.set_toprule_command(cmd)

    def remove_toprule(self) -> None:
        """Remove the rule at the top of the LaTeX table."""
        self._options.toprule_cmd = None

    def add_bottomrule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule to the bottom of the LaTeX table.

        Args:
            cmd (Optional[str], optional): LaTeX command for the bottom rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            cmd = self._options.hrule_cmd
        self.set_bottomrule_command(cmd)

    def remove_bottomrule(self) -> None:
        """Remove the rule at the bottom of the LaTeX table."""
        self._options.bottomrule_cmd = None

    def add_hrule_above_all(self, count: int = 1) -> None:
        """Add horizontal rules above all rows in the LaTeX table.

        Args:
            count (int, optional): Number of horizontal rules to add above each row. Defaults to 1.
        """
        if self._options.hrule_counts is None or len(self._options.hrule_counts) == 0:
            self._options.hrule_counts = [0] * len(self.df)

        for idx in range(len(self.df)):
            self._options.hrule_counts[idx] += count

    def clear_all_hrule(self) -> None:
        """Remove all horizontal rules from the LaTeX table."""
        self._options.hrule_counts = []

    def add_vrule_left_of(self, col_idx: int | list[int], count: int = 1) -> None:
        """Add vertical rules to the left of specified columns in the LaTeX table.

        Note: Column index 0 corresponds to the first column in the dataframe which will include
        the index column if include_index is set to True in options.

        Args:
            col_idx (int | list[int]): Column index or list of column indices to the left of which
                to add vertical rules.
            count (int, optional): Number of vertical rules to add. Defaults to 1.
        """
        if isinstance(col_idx, int):
            cols = [col_idx]
        else:
            cols = col_idx

        include_index_offset = 1 if self._options.include_index else 0

        if not self._options.vrule_counts:
            self._options.vrule_counts = [0] * (len(self.df.columns) + 1 + include_index_offset)

        for cidx in cols:
            if cidx < 0 or cidx > len(self.df.columns) + include_index_offset:
                raise ValueError(
                    f"Column index {cidx} is out of bounds for DataFrame with "
                    f"{len(self.df.columns)} columns."
                    f"You may add vrules to the left of index 0 up to index "
                    f"{len(self.df.columns) + include_index_offset} for this dataframe."
                )
            self._options.vrule_counts[cidx] += count

    def clear_all_vrule(self) -> None:
        """Remove all vertical rules from the LaTeX table."""
        self._options.vrule_counts = [0] * (
            len(self.df.columns) + 1 + int(self._options.include_index)
        )

    def add_vrule_right_of(self, col_idx: int | list[int], count: int = 1) -> None:
        """Add vertical rules to the right of specified columns in the LaTeX table.

        Note: Column index 0 corresponds to the first column in the dataframe which will include
        the index column if include_index is set to True in options.

        Args:
            col_idx (int | list[int]): Column index or list of column indices to the right of which
                to add vertical rules.
            count (int, optional): Number of vertical rules to add. Defaults to 1.
        """
        if isinstance(col_idx, int):
            cols = [col_idx]
        else:
            cols = col_idx

        include_index_offset = 1 if self._options.include_index else 0

        if not self._options.vrule_counts:
            self._options.vrule_counts = [0] * (len(self.df.columns) + 1 + include_index_offset)

        for cidx in cols:
            if cidx < -1 or cidx > len(self.df.columns) + include_index_offset - 1:
                raise ValueError(
                    f"Column index {cidx} is out of bounds for DataFrame with "
                    f"{len(self.df.columns)} columns."
                    f"You may add vrules to the right of index -1 up to index "
                    f"{len(self.df.columns) + include_index_offset - 1} for this dataframe."
                )
            self._options.vrule_counts[cidx + 1] += count

    def add_vrule_all(self, count: int = 1) -> None:
        """Add vertical rules around all columns in the LaTeX table.

        Args:
            count (int, optional): Number of vertical rules to add between each column.
                Defaults to 1.
        """
        total_cols = len(self.df.columns) + int(self._options.include_index)
        if not self._options.vrule_counts:
            self._options.vrule_counts = [0] * (total_cols + 1)

        for idx in range(total_cols + 1):
            self._options.vrule_counts[idx] += count

    def highlight_rows(self, rows: int | Iterable[int], color: Color = "yellow") -> None:
        """Highlight specified rows in the LaTeX table.

        Args:
            rows (int | Iterable[int]): Row index or iterable of row indices to highlight.
            color (Color, optional): Color to use for highlighting. Supports valid xcolor
                expressions (for example ``"amber!80!gray"``), hex color strings, RGB tuples,
                and other names parseable by GerryTools/Matplotlib (which are converted to
                HTML color form). Defaults to ``"yellow"``.
        """
        row_indices: list[int]

        if isinstance(rows, int):
            row_indices = [rows]
        else:
            row_indices = list(rows)

        try:
            color_type, color_value = to_latex_xcolor_or_html_spec(color)
        except ValueError as exc:
            if isinstance(color, tuple) and len(color) == 3:
                raise
            raise ValueError("Invalid color specification for row highlighting") from exc
        color_tup = (color_type, color_value)

        if (
            self._options.row_highlight_colors is None
            or len(self._options.row_highlight_colors) == 0
        ):
            self._options.row_highlight_colors = [("NONE", "")] * len(self.df)

        for ridx in row_indices:
            self._options.row_highlight_colors[ridx] = color_tup

    def remove_column_headers(self) -> None:
        """Remove the column headers from the LaTeX table."""
        self._options.include_column_headers = False

    def remove_group_headers(self) -> None:
        """Remove the group headers from the LaTeX table."""
        self._options.include_group_headers = False

    def remove_all_headers(self) -> None:
        """Remove all headers (both column and group) from the LaTeX table."""
        self.remove_column_headers()
        self.remove_group_headers()

    def include_column_headers(self) -> None:
        """Include the column headers in the LaTeX table."""
        self._options.include_column_headers = True

    def include_group_headers(self) -> None:
        """Include the group headers in the LaTeX table."""
        self._options.include_group_headers = True

    def include_all_headers(self) -> None:
        """Include all headers (both column and group) in the LaTeX table."""
        self.include_column_headers()
        self.include_group_headers()

    # ==================
    #   OPTION SETTERS
    # ==================

    def set_column_headers_text_format(self, bold: bool = True, italic: bool = False) -> None:
        """Set column-header emphasis styling.

        Args:
            bold (bool, optional): Whether to apply bold styling. Defaults to True.
            italic (bool, optional): Whether to apply italic styling. Defaults to False.

        Returns:
            None
        """
        self._options.bold_column_headers = bold
        self._options.italic_column_headers = italic

    def set_group_headers_text_format(self, bold: bool = True, italic: bool = False) -> None:
        """Set group-header emphasis styling.

        Args:
            bold (bool, optional): Whether to apply bold styling. Defaults to True.
            italic (bool, optional): Whether to apply italic styling. Defaults to False.

        Returns:
            None
        """
        self._options.bold_group_headers = bold
        self._options.italic_group_headers = italic

    def set_decimal_count(self, count: int) -> None:
        """Set the number of decimal places to round float values to in the LaTeX table.

        Args:
            count (int): Number of decimal places to round to.
        """
        if count < 0:
            raise ValueError("Decimal count must be non-negative")

        self._options.number_fmt_fn = round_decimals(count)

    def set_hrule_command(self, cmd: str) -> None:
        r"""Set the LaTeX command for horizontal rules in the table.

        Args:
            cmd (str): LaTeX command for horizontal rules (e.g., r"\hline").
        """
        self._options.hrule_cmd = cmd

    def set_toprule_command(self, cmd: str | None = None) -> None:
        """Set the LaTeX command for the top rule in the table.

        Args:
            cmd (str | None, optional): LaTeX command for the top rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            self._options.toprule_cmd = self._options.hrule_cmd
        else:
            self._options.toprule_cmd = cmd

    def set_bottomrule_command(self, cmd: str | None = None) -> None:
        """Set the LaTeX command for the bottom rule in the table.

        Args:
            cmd (str | None, optional): LaTeX command for the bottom rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            self._options.bottomrule_cmd = self._options.hrule_cmd
        else:
            self._options.bottomrule_cmd = cmd

    def set_all_hrule(self, count: int) -> None:
        """Set the number of horizontal rules above all rows in the LaTeX table.

        Args:
            count (int): Number of horizontal rules to add above each row.
        """
        self._options.hrule_counts = [count] * len(self.df)

    def set_nan_string(self, nan_str: str) -> None:
        """Set the string to represent NaN values in the LaTeX table.

        Args:
            nan_str (str): String to represent NaN values.
        """
        self._options.nan_string = nan_str

    def set_tabular_format(self, fmt: str) -> None:
        """Set the table-row tabular preamble.

        Supports simple and rich specifications (for example
        ``"|l|p{2cm}||S[table-format=1.3]|>{...}p{..}<{...}|"``) and preserves
        top-level boundary tokens for multicolumn rendering.

        Args:
            fmt (str): Raw LaTeX tabular preamble string.

        Returns:
            None

        Raises:
            ValueError: If the parsed column count does not match the table width.
        """
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        expected_cols = self.df.shape[1] + (1 if self._options.include_index else 0)
        if len(colspecs) != expected_cols:
            raise ValueError(
                f"Format implies {len(colspecs)} columns but expected {expected_cols} "
                f"({'with' if self._options.include_index else 'without'} index)."
            )

        self._options.tabular_alignments = colspecs
        self._options.vrule_counts = vrules
        self._options.boundary_extras = extras

    def set_group_tabular_format(self, fmt: str) -> None:
        """Set the group-header-row tabular preamble.

        Args:
            fmt (str): Raw LaTeX tabular preamble for group-header cells.

        Returns:
            None

        Raises:
            ValueError: If parsed group-header cell count is incompatible with current grouping.
        """
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        # group header has one cell per group-header block (+ index cell if include_index)
        group_cells = len(self._options.groups_to_cols) + (1 if self._options.include_index else 0)

        if len(colspecs) == group_cells - 1:
            colspecs, vrules, extras = _parse_tabular_preamble(fmt + "c")

        if len(colspecs) != group_cells:
            raise ValueError(
                f"Group-header format implies {len(colspecs)} cells but expected {group_cells}. "
                f"({('with' if self._options.include_index else 'without')} index)."
            )

        self._options.group_tabular_alignments = colspecs
        self._options.group_vrule_counts = vrules
        self._options.group_boundary_extras = extras

    def clear_header_groups(self) -> None:
        """Clear any header groups set for the LaTeX table."""
        self._options.groups_to_cols = {"": list(self.df.columns)}

        self._options.group_tabular_alignments = None
        self._options.group_vrule_counts = None
        self._options.group_boundary_extras = None

    def set_header_groups(
        self,
        groups_to_columns: dict[str, Iterable[str]],
    ):
        """Set header groups for the LaTeX table.

        Example:
            If the table has columns ["Col1", "Col2", "Col3"], then calling
            set_header_groups({"GroupA": ["Col1", "Col2"], "GroupB": ["Col3"]},
            "GroupA" spanning "Col1" and "Col2" with center alignment, and "GroupB"
            spanning "Col3" with left alignment.

        Args:
            groups_to_columns (dict[str, Iterable[str]]): Mapping of group names to column names.

        Raises:
            ValueError: If group_align is invalid or if columns in groups_to_columns
                do not exist in the DataFrame.
        """
        df_cols = list(map(str, self.df.columns))

        retyped_group_to_cols: dict[str, list[str]] = {
            str(grp): [str(col) for col in cols] for grp, cols in groups_to_columns.items()
        }

        all_listed: list[str] = []
        for cols in retyped_group_to_cols.values():
            all_listed.extend(cols)

        unknown = set(all_listed) - set(df_cols)
        if unknown:
            raise ValueError(f"Unknown columns in groups_to_columns: {sorted(unknown)}")

        # preserve DF column order: any unlisted columns go into the "" group (at the end)
        missing_cols = [c for c in df_cols if c not in set(all_listed)]

        groups_to_cols = {k: v for k, v in retyped_group_to_cols.items()}
        if "" in groups_to_cols:
            groups_to_cols[""].extend(missing_cols)
        elif len(missing_cols) > 0:
            groups_to_cols[""] = missing_cols

        self._options.groups_to_cols = groups_to_cols

        self._options.group_tabular_alignments = None
        self._options.group_vrule_counts = None
        self._options.group_boundary_extras = None

    def set_index_formatter(self, fmt_fn: CellWrapper | Callable[[TableCellValue], str]) -> None:
        """Set a formatter function for the index column.

        Args:
            fmt_fn (CellWrapper | Callable[[TableCellValue], str]): Either a full two-argument
                ``CellWrapper`` or a one-argument formatter over the raw value.

        Returns:
            None
        """
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[TableCellValue], str], fmt_fn)

            def _wrapped(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                """Adapt a one-argument value formatter to ``CellWrapper`` form.

                Args:
                    v (TableCellValue): Raw cell value.
                    s (str): Existing rendered string (ignored).

                Returns:
                    tuple[TableCellValue, str]: Original value and newly formatted string.
                """
                return v, one_arg(v)

            self._options.index_fmt_fn = _wrapped
        else:
            self._options.index_fmt_fn = cast(CellWrapper, fmt_fn)

    def set_number_formatter(self, fmt_fn: CellWrapper | Callable[[float], str]) -> None:
        """Set the number formatter function for the LaTeX table.

        Used as the default formatter for all float values in the table.

        Args:
            fmt_fn (Wrapper): Formatter function for float values.

        Raises:
            ValueError: If the provided formatter is not callable.
        """
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[float], str], fmt_fn)

            def _wrapped(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                """Adapt a one-argument numeric formatter to ``CellWrapper`` form.

                Args:
                    v (TableCellValue): Raw numeric cell value.
                    s (str): Existing rendered string (ignored).

                Returns:
                    tuple[TableCellValue, str]: Original value and newly formatted string.
                """
                if not isinstance(v, (int, float)):
                    return v, s
                return v, one_arg(float(v))

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self._options.number_fmt_fn = new_fn

    def set_string_formatter(self, fmt_fn: CellWrapper | Callable[[str], str]) -> None:
        """Set the string formatter function for the LaTeX table.

        Used as the default formatter for all string values in the table.
        Args:
            fmt_fn (Wrapper): Formatter function for string values.

        Raises:
            ValueError: If the provided formatter is not callable.
        """
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[str], str], fmt_fn)

            def _wrapped(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                """Adapt a one-argument string formatter to ``CellWrapper`` form.

                Args:
                    v (TableCellValue): Raw cell value passed through formatter.
                    s (str): Existing rendered string (ignored).

                Returns:
                    tuple[TableCellValue, str]: Original value and newly formatted string.
                """
                if not isinstance(v, str):
                    return v, s
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self._options.str_fmt_fn = new_fn

    def __set_single_col_formatter(
        self, col: str, fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        """Set a specific column formatter function for the LaTeX table.

        Args:
            col (str): Column name to set the formatter for.
            fmt_fn (Wrapper): Formatter function for the specified column.

        Raises:
            ValueError: If the specified column does not exist in the DataFrame,
                or if the provided formatter is not callable.
        """

        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame.")

        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[TableCellValue], str], fmt_fn)

            def _wrapped(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                """Adapt a one-argument column formatter to ``CellWrapper`` form.

                Args:
                    v (TableCellValue): Raw cell value.
                    s (str): Existing rendered string (ignored).

                Returns:
                    tuple[TableCellValue, str]: Original value and newly formatted string.
                """
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self._options.col_formatters[col] = new_fn

    def set_column_formatter(
        self, col: str | list[str], fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        """Set a specific column formatter function for the LaTeX table.

        Args:
            col (str | list[str]): Column name or list of column names to set the formatter for.
            fmt_fn (Wrapper): Formatter function for the specified column(s).

        Raises:
            ValueError: If any of the specified columns do not exist in the DataFrame,
                or if the provided formatter is not callable.
        """

        if isinstance(col, list):
            for c in col:
                self.__set_single_col_formatter(c, fmt_fn)
            return

        self.__set_single_col_formatter(col, fmt_fn)

    def __set_single_row_formatter(
        self, row_idx: int, fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        """Set a specific row formatter function for the LaTeX table.

        Args:
            row_idx (int): Row index to set the formatter for.
            fmt_fn (Wrapper): Formatter function for the specified row.

        Raises:
            ValueError: If the specified row index is out of bounds,
                or if the provided formatter is not callable.
        """

        if row_idx < 0 or row_idx >= len(self.df):
            raise ValueError(
                f"Row index {row_idx} is out of bounds for DataFrame with {len(self.df)} rows."
            )

        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[TableCellValue], str], fmt_fn)

            def _wrapped(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
                """Adapt a one-argument row formatter to ``CellWrapper`` form.

                Args:
                    v (TableCellValue): Raw cell value.
                    s (str): Existing rendered string (ignored).

                Returns:
                    tuple[TableCellValue, str]: Original value and newly formatted string.
                """
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self._options.row_formatters[row_idx] = new_fn

    def set_row_formatter(
        self, row_idx: int | list[int], fmt_fn: CellWrapper | Callable[[TableCellValue], str]
    ) -> None:
        """Set a specific row formatter function for the LaTeX table.

        Args:
            row_idx (int | list[int]): Row index or list of row indices to set formatters for.
            fmt_fn (Wrapper): Formatter function for the specified row.

        Raises:
            ValueError: If the specified row index is out of bounds,
                or if the provided formatter is not callable.
        """

        if isinstance(row_idx, list):
            for ridx in row_idx:
                self.__set_single_row_formatter(ridx, fmt_fn)
            return

        self.__set_single_row_formatter(row_idx, fmt_fn)

    # =====================
    #   STRING GENERATORS
    # =====================

    def _generate_header(self) -> str:
        r"""Generate the LaTeX table header string.

        Generally includes the \multicolumn row (if applicable) and the column titles.

        Returns:
            str: LaTeX table header string.
        """
        header_string = f"{{{self._options.column_format}}}"
        if self._options.toprule_cmd is not None:
            header_string += "\n" + self._options.toprule_cmd + "\n"

        column_titles = []
        if (
            set(self._options.groups_to_cols.keys()) != set({""})
            and self._options.include_group_headers
        ):
            header_string += f"\n{self._options.multicolumn_format}"

        if self._options.include_index:
            index_name = (
                self._options.index_name
                if self._options.index_name is not None
                else (self.df.index.name if self.df.index.name is not None else "")
            )
            index_str = latex_escape(str(index_name))
            if self._options.bold_column_headers:
                index_str = rf"\textbf{{{index_str}}}"
            if self._options.italic_column_headers:
                index_str = rf"\textit{{{index_str}}}"
            column_titles.append(index_str)

        for _, col_list in self._options.groups_to_cols.items():
            for col in col_list:
                col_title = latex_escape(str(col))
                if self._options.bold_column_headers:
                    col_title = rf"\textbf{{{col_title}}}"
                if self._options.italic_column_headers:
                    col_title = rf"\textit{{{col_title}}}"

                column_titles.append(col_title)

        if self._options.include_column_headers:
            header_string += "\n" + " & ".join(column_titles) + r" \\"

        header_string = rf"\begin{{tabular}}{header_string}" + "\n"
        logger.log(
            logging.DEBUG,
            "Generated LaTeX table header:\n%s",
            header_string,
            stacklevel=2,
        )
        return header_string

    def _generate_body(self) -> str:
        """Generate the LaTeX table body string.

        Will format each cell according to the specified formatters in options and only
        touches the data rows of the internal dataframe.

        Returns:
            str: LaTeX table body string.
        """
        column_ordering = list(self.df.columns)

        if set(self._options.groups_to_cols.keys()) != {""}:
            column_ordering = []
            for cols in self._options.groups_to_cols.values():
                column_ordering.extend(cols)

        body_string = ""
        for row_idx, (df_row_idx, row) in enumerate(self.df.iterrows()):
            if len(self._options.hrule_counts) > 0 and self._options.hrule_counts[row_idx] > 0:
                body_string += self._options.hrule_cmd * self._options.hrule_counts[row_idx] + "\n"

            color_type, color_value = self._options.row_highlight_colors[row_idx]
            match color_type:
                case "NAME":
                    if not isinstance(color_value, str):
                        raise ValueError(f"Found invalid color value '{color_value}'.")
                    body_string += r"\rowcolor{" + str(color_value) + "}\n"
                case "HTML":
                    if not isinstance(color_value, str):
                        raise ValueError(f"Found invalid hex color value '{color_value}'.")

                    if not re.match(r"^#?[0-9A-Fa-f]{6}$", color_value):
                        raise ValueError(
                            f"Invalid hex color value '{color_value}'. "
                            "Must be 6 hexadecimal digits."
                        )
                    color_value = color_value.lstrip("#").lower()
                    body_string += r"\rowcolor[HTML]{" + color_value + "}\n"

                case "RGB":
                    r_val, g_val, b_val = color_value
                    body_string += r"\rowcolor[RGB]{" f"{r_val},{g_val},{b_val}" + "}\n"

                case "rgb":
                    r_val, g_val, b_val = color_value
                    body_string += r"\rowcolor[rgb]{" f"{r_val:.3f},{g_val:.3f},{b_val:.3f}" + "}\n"

                case "NONE":
                    pass

                case _:
                    warnings.warn(
                        f"Unsupported color type '{color_type}' for row highlighting. "
                        f"Skipping row highlighting for row '{row_idx}'.",
                        stacklevel=2,
                    )

            row_items = []
            if self._options.include_index:
                raw = str(df_row_idx)
                esc = latex_escape(raw)
                if self._options.index_fmt_fn is not None:
                    row_items.append(self._options.index_fmt_fn(df_row_idx, esc)[1])
                else:
                    row_items.append(esc)

            for col in column_ordering:
                cell_value = row[col]
                if pd.isna(cell_value):
                    cell_str = self._options.nan_string
                else:
                    if col in self._options.col_formatters:
                        cell_str = self._options.col_formatters[col](cell_value, str(cell_value))[1]
                    elif row_idx in self._options.row_formatters:
                        cell_str = self._options.row_formatters[row_idx](
                            cell_value, str(cell_value)
                        )[1]
                    elif isinstance(cell_value, float) and self._options.number_fmt_fn is not None:
                        cell_str = self._options.number_fmt_fn(cell_value, str(cell_value))[1]
                    elif isinstance(cell_value, str) and self._options.str_fmt_fn is not None:
                        cell_str = self._options.str_fmt_fn(cell_value, cell_value)[1]
                    else:
                        cell_str_raw = latex_escape(str(cell_value))
                        cell_str = (
                            self._options.str_fmt_fn(cell_str_raw, cell_str_raw)[1]
                            if self._options.str_fmt_fn is not None
                            else cell_str_raw
                        )

                row_items.append(cell_str)
            body_string += " & ".join(row_items) + r" \\" + "\n"

        logger.log(logging.DEBUG, "Generated LaTeX table body:\n%s", body_string, stacklevel=2)
        return body_string

    def _generate_footer(self) -> str:
        """Generate the LaTeX table footer string."""
        footer_str = ""
        if self._options.bottomrule_cmd is not None:
            footer_str += "\n" + self._options.bottomrule_cmd + "\n"
        footer_str += r"\end{tabular}"
        logger.log(logging.DEBUG, "Generated LaTeX table footer:\n%s", footer_str, stacklevel=2)
        return footer_str

    def _generate_latex(self) -> str:
        """Generate the complete LaTeX table string."""
        tex_string = ""
        tex_string += self._generate_header()
        tex_string += self._generate_body()
        tex_string += self._generate_footer()
        return tex_string
