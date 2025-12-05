import warnings
from typing import Callable, Optional, Any, Iterable, TypeAlias, cast
from dataclasses import dataclass, field
import pandas as pd
from numbers import Real
import re
import inspect

from gerrytools.latex.document import TexDocument, ColorLike
from gerrytools.latex.formatters import round_decimals


def latex_escape(s: str) -> str:
    """Escapes special LaTeX characters in a string."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _latex_escape_wrapper(s: str, prev: str) -> tuple[str, str]:
    """Wrapper function to escape LaTeX special characters in strings.

    Args:
        s (str): The original string value.
        prev (str): The currently rendered string.

    Returns:
        tuple[str, str]: The original string and the newly escaped version of the previously
            rendered string.
    """
    return s, latex_escape(prev)


def _infer_group_cell_align_from_data(colspecs: list[str], start: int, end: int) -> str:
    """
    Infer group header align from the underlying spanned *data* colspecs.
    Heuristic:
      - if all are 'l' -> 'l'
      - if all are 'r' -> 'r'
      - if all are 'c' -> 'c'
      - else -> 'c'
    Treat complex specs as 'c' by default unless you want to get fancier.
    """

    def base_align(spec: str) -> str:
        spec = spec.strip()
        if spec in ("l", "c", "r"):
            return spec
        # you can choose to map S/D columns to 'r' if you like:
        # if spec.startswith(("S", "D")): return "r"
        return "c"

    aligns = {base_align(s) for s in colspecs[start:end]}
    return aligns.pop() if len(aligns) == 1 else "c"


# Format takes in original value and currently rendered string
# and returns original value and new rendered string
CellWrapper: TypeAlias = Callable[[Any, str], tuple[Any, str]]


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
        row_highlight_colors (list[tuple[str, ColorLike]]): List of tuples specifying
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

    include_index: bool = False
    index_name: Optional[str] = (
        None  # if None and include_index True, uses df.index.name or ""
    )
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

    row_highlight_colors: list[tuple[str, ColorLike]] = field(default_factory=list)

    number_fmt_fn: Optional[CellWrapper] = None
    str_fmt_fn: Optional[CellWrapper] = _latex_escape_wrapper
    col_formatters: dict[str, CellWrapper] = field(default_factory=dict)
    row_formatters: dict[int, CellWrapper] = field(default_factory=dict)

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
            if len(gcols) != group_cell_count:
                raise ValueError(
                    f"Group-header preamble has {len(gcols)} cells but expected {group_cell_count}."
                )

            cell_i = 0

            if self.include_index:
                idx_align = self.index_alignment or "c"
                parts.append(
                    rf"\multicolumn{{1}}{{{_mc_colspec(gvr, gex, cell_i, idx_align, cell_i + 1, include_left=first_cell)}}}{{}}"
                )
                first_cell = False
                cell_i += 1

            for group, span in group_items:
                if span == 0:
                    continue

                align = gcols[cell_i]  # <-- alignment comes from group preamble
                name = latex_escape(group)
                if self.bold_group_headers and name:
                    name = rf"\textbf{{{name}}}"
                if self.italic_group_headers and name:
                    name = rf"\textit{{{name}}}"

                parts.append(
                    rf"\multicolumn{{{span}}}{{{_mc_colspec(gvr, gex, cell_i, align, cell_i + 1, include_left=first_cell)}}}{{{name}}}"
                )
                first_cell = False
                cell_i += 1

            return " & ".join(parts) + r" \\"

        # ---------- Mode B: infer group boundaries from DATA preamble; infer group alignments ----------
        dcols = self.tabular_alignments
        ncols_total = len(dcols)
        dex = self.boundary_extras or [""] * (ncols_total + 1)
        dvr = self.vrule_counts
        _normalize_preamble(dcols, dvr, dex)

        if self.include_index:
            idx_align = self.index_alignment or "c"
            parts.append(
                rf"\multicolumn{{1}}{{{_mc_colspec(dvr, dex, 0, idx_align, 1, include_left=first_cell)}}}{{}}"
            )
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

            name = latex_escape(group)
            if self.bold_group_headers and name:
                name = rf"\textbf{{{name}}}"
            if self.italic_group_headers and name:
                name = rf"\textit{{{name}}}"

            parts.append(
                rf"\multicolumn{{{span}}}{{{_mc_colspec(dvr, dex, left_b, align, right_b, include_left=first_cell)}}}{{{name}}}"
            )
            first_cell = False

        return " & ".join(parts) + r" \\"


def _consume_balanced(s: str, i: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Consume a balanced group from a string starting at index i."""
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
    """Parse a LaTeX tabular preamble into column specs, vertical rules, and boundary extras."""
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
        self.document = TexDocument()
        self.df.index = self.df.index.map(str)
        if use_defaults:
            self.__options = TableOptions(
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
            self.__options = TableOptions(
                groups_to_cols={"": list(df.columns)},
                tabular_alignments=["c"] * df.shape[1],
                row_highlight_colors=[("NONE", "")] * df.shape[0],
                vrule_counts=[0] * (df.shape[1] + 1),
            )

    def __repr__(self) -> str:
        return self._generate_latex()

    def __str__(self) -> str:
        return self._generate_latex()

    def clear_options(self) -> None:
        """Resets all table options to their default values."""
        self.__options = TableOptions(
            groups_to_cols={"": list(self.df.columns)},
            tabular_alignments=["c"] * self.df.shape[1],
            row_highlight_colors=[("NONE", "")] * self.df.shape[0],
            vrule_counts=[0] * (self.df.shape[1] + 1),
        )

    def preview(self) -> None:
        self.document.body_string = self._generate_latex()
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

        self.__options.index_alignment = alignment
        self.__options.index_name = latex_escape(name) if name is not None else ""

        n_data_cols = self.df.shape[1]

        # Ensure boundary_extras exists and has correct length when in parsed-mode
        def _ensure_extras(ncols: int) -> None:
            if not getattr(self.__options, "boundary_extras", None):
                self.__options.boundary_extras = [""] * (ncols + 1)
            if len(self.__options.boundary_extras) != ncols + 1:
                # best-effort reset; you could be stricter and raise
                self.__options.boundary_extras = [""] * (ncols + 1)

        # ADD index
        if include and not self.__options.include_index:
            self.__options.include_index = True

            # expected current cols = data cols
            if len(self.__options.tabular_alignments) != n_data_cols:
                raise ValueError(
                    "Current tabular format does not match DataFrame columns. "
                    f"Got {len(self.__options.tabular_alignments)} colspecs but expected {n_data_cols}."
                )

            # Insert the new index column spec
            self.__options.tabular_alignments.insert(0, alignment)

            # vrule_counts/extras: old length was n_data_cols+1, new length should be (n_data_cols+1)+1
            old_ncols = n_data_cols
            _ensure_extras(old_ncols)

            # Insert a NEW boundary at position 1 (between new index col and old first col)
            # Keep boundary 0 (left edge) as-is.
            self.__options.vrule_counts.insert(1, 0)
            self.__options.boundary_extras.insert(1, "")

            return

        # REMOVE index
        if (not include) and self.__options.include_index:
            self.__options.include_index = False

            # expected current cols = data cols + 1
            if len(self.__options.tabular_alignments) != n_data_cols + 1:
                raise ValueError(
                    "Current tabular format does not match DataFrame columns+index. "
                    f"Got {len(self.__options.tabular_alignments)} colspecs but expected {n_data_cols + 1}."
                )

            # Remove the index column spec
            self.__options.tabular_alignments.pop(0)

            # Remove boundary 1 (between old index col and first data col)
            new_ncols = n_data_cols + 1
            _ensure_extras(new_ncols)

            self.__options.vrule_counts.pop(1)
            self.__options.boundary_extras.pop(1)

            return

    def remove_index(self) -> None:
        """Remove the index from the generated latex table (if it exists)"""
        self.include_index(include=False)

    def add_hrule_above(self, index: int | list[int], count: int = 1) -> None:
        """Add horizontal rules above specified rows in the LaTeX table.

        Note: Row index 0 corresponds to the first row of data in the dataframe, so it
        will appear below the header row in the LaTeX table.

        Args:
            index (int | list[int]): Row index or list of row indices above which to add horizontal rules.
            count (int, optional): Number of horizontal rules to add. Defaults to 1.
        """
        if isinstance(index, int):
            indices = [index]
        else:
            indices = index

        if not self.__options.hrule_counts:
            self.__options.hrule_counts = [0] * len(self.df)
        if len(self.__options.hrule_counts) < len(self.df):
            self.__options.hrule_counts.extend(
                [0] * (len(self.df) - len(self.__options.hrule_counts))
            )

        for idx in indices:
            if idx < 0 or idx > len(self.df):
                raise ValueError(
                    f"Row index {idx} is out of bounds for DataFrame with {len(self.df)} rows."
                )
            self.__options.hrule_counts[idx] += count

    def add_toprule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule to the top of the LaTeX table.

        Kwargs:
            cmd (Optional[str], optional): LaTeX command for the top rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            cmd = self.__options.hrule_cmd
        self.set_toprule_command(cmd)

    def remove_toprule(self) -> None:
        """Remove the rule at the top of the LaTeX table."""
        self.__options.toprule_cmd = None

    def add_bottomrule(self, *, cmd: Optional[str] = None) -> None:
        """Add a rule to the bottom of the LaTeX table.

        Kwargs:
            cmd (Optional[str], optional): LaTeX command for the bottom rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            cmd = self.__options.hrule_cmd
        self.set_bottomrule_command(cmd)

    def remove_bottomrule(self) -> None:
        """Remove the rule at the bottom of the LaTeX table."""
        self.__options.bottomrule_cmd = None

    def add_hrule_above_all(self, count: int = 1) -> None:
        """Add horizontal rules above all rows in the LaTeX table.

        Args:
            count (int, optional): Number of horizontal rules to add above each row. Defaults to 1.
        """
        if self.__options.hrule_counts is None or len(self.__options.hrule_counts) == 0:
            self.__options.hrule_counts = [0] * len(self.df)

        for idx in range(len(self.df)):
            self.__options.hrule_counts[idx] += count

    def clear_all_hrule(self) -> None:
        """Remove all horizontal rules from the LaTeX table."""
        self.__options.hrule_counts = []

    def add_vrule_left_of(self, col_idx: int | list[int], count: int = 1) -> None:
        """Add vertical rules to the left of specified columns in the LaTeX table.

        Note: Column index 0 corresponds to the first column in the dataframe which will include
        the index column if include_index is set to True in options.

        Args:
            col_idx (int | list[int]): Column index or list of column indices to the left of which to add vertical rules.
            count (int, optional): Number of vertical rules to add. Defaults to 1.
        """
        if isinstance(col_idx, int):
            cols = [col_idx]
        else:
            cols = col_idx

        include_index_offset = 1 if self.__options.include_index else 0

        if not self.__options.vrule_counts:
            self.__options.vrule_counts = [0] * (
                len(self.df.columns) + 1 + include_index_offset
            )

        for cidx in cols:
            if cidx < 0 or cidx > len(self.df.columns) + include_index_offset:
                raise ValueError(
                    f"Column index {cidx} is out of bounds for DataFrame with "
                    f"{len(self.df.columns)} columns."
                    f"You may add vrules to the left of index 0 up to index "
                    f"{len(self.df.columns) + include_index_offset} for this dataframe."
                )
            self.__options.vrule_counts[cidx] += count

    def clear_all_vrule(self) -> None:
        """Remove all vertical rules from the LaTeX table."""
        self.__options.vrule_counts = [0] * (
            len(self.df.columns) + 1 + int(self.__options.include_index)
        )

    def add_vrule_right_of(self, col_idx: int | list[int], count: int = 1) -> None:
        """Add vertical rules to the right of specified columns in the LaTeX table.

        Note: Column index 0 corresponds to the first column in the dataframe which will include
        the index column if include_index is set to True in options.

        Args:
            col_idx (int | list[int]): Column index or list of column indices to the right of which to add vertical rules.
            count (int, optional): Number of vertical rules to add. Defaults to 1.
        """
        if isinstance(col_idx, int):
            cols = [col_idx]
        else:
            cols = col_idx

        include_index_offset = 1 if self.__options.include_index else 0

        if not self.__options.vrule_counts:
            self.__options.vrule_counts = [0] * (
                len(self.df.columns) + 1 + include_index_offset
            )

        for cidx in cols:
            if cidx < -1 or cidx > len(self.df.columns) + include_index_offset - 1:
                raise ValueError(
                    f"Column index {cidx} is out of bounds for DataFrame with "
                    f"{len(self.df.columns)} columns."
                    f"You may add vrules to the right of index -1 up to index "
                    f"{len(self.df.columns) + include_index_offset - 1} for this dataframe."
                )
            self.__options.vrule_counts[cidx + 1] += count

    def add_vrule_all(self, count: int = 1) -> None:
        """Add vertical rules between all columns in the LaTeX table.

        Args:
            count (int, optional): Number of vertical rules to add between each column.
                Defaults to 1.
        """
        total_cols = len(self.df.columns) + int(self.__options.include_index)
        if not self.__options.vrule_counts:
            self.__options.vrule_counts = [0] * (total_cols + 1)

        for idx in range(total_cols + 1):
            self.__options.vrule_counts[idx] += count

    def column_headers_text_format(
        self, bold: bool = True, italic: bool = False
    ) -> None:
        """Set whether to bold the column headers in the LaTeX table."""
        self.__options.bold_column_headers = bold
        self.__options.italic_column_headers = italic

    def group_headers_text_format(
        self, bold: bool = True, italic: bool = False
    ) -> None:
        """Set whether to bold the group headers in the LaTeX table."""
        self.__options.bold_group_headers = bold
        self.__options.italic_group_headers = italic

    def highlight_rows(
        self, rows: int | Iterable[int], color: ColorLike = "yellow"
    ) -> None:
        """Highlight specified rows in the LaTeX table.

        Args:
            row (int | Iterable[int]): Row index or iterable of row indices to highlight.
            color (ColorLike, optional): Color to use for highlighting. Can be a LaTeX color name
                (str), a hex color code (str), or an RGB tuple (tuple[float, float, float]).
                Defaults to "yellow".
        """
        row_indices: list[int]

        if isinstance(rows, int):
            row_indices = [rows]
        else:
            row_indices = list(rows)

        color_tup = None

        if isinstance(color, str):
            if re.match(r"^#?[0-9A-Fa-f]{6}$", color):
                # hex string
                color_tup = ("HTML", color.lstrip("#").upper())
            else:
                color_tup = ("NAME", color)
        elif isinstance(color, tuple) and len(color) == 3:
            if all(0.0 <= c <= 1.0 for c in color):  # type: ignore
                color_tup = ("rgb", (color[0], color[1], color[2]))
            elif all(0 <= c <= 255 for c in color):  # type: ignore
                color_tup = (
                    "RGB",
                    (int(round(color[0])), int(round(color[1])), int(round(color[2]))),
                )
            else:
                raise ValueError(
                    "RGB color components must be in the range [0.0, 1.0] or [0, 255]"
                )

        if color_tup is None:
            raise ValueError("Invalid color specification for row highlighting")

        if (
            self.__options.row_highlight_colors is None
            or len(self.__options.row_highlight_colors) == 0
        ):
            self.__options.row_highlight_colors = [("NONE", "")] * len(self.df)

        for ridx in row_indices:
            self.__options.row_highlight_colors[ridx] = color_tup

    # ==================
    #   OPTION SETTERS
    # ==================

    def set_decimal_count(self, count: int) -> None:
        """Set the number of decimal places to round float values to in the LaTeX table.

        Args:
            count (int): Number of decimal places to round to.
        """
        if count < 0:
            raise ValueError("Decimal count must be non-negative")

        self.__options.number_fmt_fn = round_decimals(count)

    def set_hrule_command(self, cmd: str) -> None:
        r"""Set the LaTeX command for horizontal rules in the table.

        Args:
            cmd (str): LaTeX command for horizontal rules (e.g., r"\hline").
        """
        self.__options.hrule_cmd = cmd

    def set_toprule_command(self, cmd: str | None = None) -> None:
        """Set the LaTeX command for the top rule in the table.

        Args:
            cmd (str | None, optional): LaTeX command for the top rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            self.__options.toprule_cmd = self.__options.hrule_cmd
        else:
            self.__options.toprule_cmd = cmd

    def set_bottomrule_command(self, cmd: str | None = None) -> None:
        """Set the LaTeX command for the bottom rule in the table.

        Args:
            cmd (str | None, optional): LaTeX command for the bottom rule. If None, uses the
                current hrule_cmd in options. Defaults to None.
        """
        if cmd is None:
            self.__options.bottomrule_cmd = self.__options.hrule_cmd
        else:
            self.__options.bottomrule_cmd = cmd

    def set_all_hrule(self, count: int) -> None:
        """Set the number of horizontal rules above all rows in the LaTeX table.

        Args:
            count (int): Number of horizontal rules to add above each row.
        """
        self.__options.hrule_counts = [count] * len(self.df)

    def set_nan_string(self, nan_str: str) -> None:
        """Set the string to represent NaN values in the LaTeX table.

        Args:
            nan_str (str): String to represent NaN values.
        """
        self.__options.nan_string = nan_str

    def set_tabular_format(self, fmt: str) -> None:
        """
        Set the tabular preamble. Supports l/c/r plus richer specs like:
        |l|p{2cm}||S[table-format=1.3]|>{...}p{..}<{...}|

        Keeps literal top-level '|' for fancy multicolumn bar formatting.
        """
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        expected_cols = self.df.shape[1] + (1 if self.__options.include_index else 0)
        if len(colspecs) != expected_cols:
            raise ValueError(
                f"Format implies {len(colspecs)} columns but expected {expected_cols} "
                f"({'with' if self.__options.include_index else 'without'} index)."
            )

        self.__options.tabular_alignments = colspecs
        self.__options.vrule_counts = vrules
        self.__options.boundary_extras = extras

    def set_group_tabular_format(self, fmt: str) -> None:
        """Set the *group header row* formatting, like a normal tabular preamble."""
        colspecs, vrules, extras = _parse_tabular_preamble(fmt)

        # group header has one cell per group-header block (+ index cell if include_index)
        group_cells = len(self.__options.groups_to_cols) + (
            1 if self.__options.include_index else 0
        )

        if len(colspecs) == group_cells - 1:
            colspecs, vrules, extras = _parse_tabular_preamble(fmt + "c")

        if len(colspecs) != group_cells:
            raise ValueError(
                f"Group-header format implies {len(colspecs)} cells but expected {group_cells}. "
                f"({('with' if self.__options.include_index else 'without')} index)."
            )

        self.__options.group_tabular_alignments = colspecs
        self.__options.group_vrule_counts = vrules
        self.__options.group_boundary_extras = extras

    def clear_header_groups(self) -> None:
        """Clear any header groups set for the LaTeX table."""
        self.__options.groups_to_cols = {"": list(self.df.columns)}

        self.__options.group_tabular_alignments = None
        self.__options.group_vrule_counts = None
        self.__options.group_boundary_extras = None

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
            str(grp): list(map(str, cols)) for grp, cols in groups_to_columns.items()
        }

        all_listed: list[str] = []
        for cols in retyped_group_to_cols.values():
            all_listed.extend(cols)

        unknown = set(all_listed) - set(df_cols)
        if unknown:
            raise ValueError(f"Unknown columns in groups_to_columns: {sorted(unknown)}")

        # preserve DF column order: any unlisted columns go into the "" group (at the end)
        missing_cols = [c for c in df_cols if c not in set(all_listed)]

        if "" in retyped_group_to_cols:
            retyped_group_to_cols[""].extend(missing_cols)
            groups_to_cols = {k: v for k, v in retyped_group_to_cols.items()}
        else:
            groups_to_cols = retyped_group_to_cols

        if len(missing_cols) > 0:
            groups_to_cols[""] = missing_cols

        self.__options.groups_to_cols = groups_to_cols

        self.__options.group_tabular_alignments = None
        self.__options.group_vrule_counts = None
        self.__options.group_boundary_extras = None

    def set_number_formatter(
        self, fmt_fn: CellWrapper | Callable[[float], str]
    ) -> None:
        """Set the number formatter function for the LaTeX table.

        Used as the default formatter for all float values in the table.

        Args:
            fmt_fn (Wrapper): Formatter function for float values.

        Raises:
            ValueError: If the provided formatter is not callable.
        """
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[Real], str], fmt_fn)

            def _wrapped(v: Real, s: str) -> tuple[Real, str]:
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self.__options.number_fmt_fn = new_fn

    def set_str_formatter(self, fmt_fn: CellWrapper | Callable[[str], str]) -> None:
        """Set the string formatter function for the LaTeX table.

        Used as the default formatter for all string values in the table.
        Args:
            fmt_fn (Wrapper): Formatter function for string values.

        Raises:
            ValueError: If the provided formatter is not callable.
        """
        if len(inspect.signature(fmt_fn).parameters) == 1:
            one_arg = cast(Callable[[Real], str], fmt_fn)

            def _wrapped(v: Real, s: str) -> tuple[Real, str]:
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self.__options.str_fmt_fn = new_fn

    def __set_single_col_formatter(
        self, col: str, fmt_fn: CellWrapper | Callable[[Any], str]
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
            one_arg = cast(Callable[[Real], str], fmt_fn)

            def _wrapped(v: Real, s: str) -> tuple[Real, str]:
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self.__options.col_formatters[col] = new_fn

    def set_col_formatter(self, col: str | list[str], fmt_fn: CellWrapper) -> None:
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
        self, row_idx: int, fmt_fn: CellWrapper | Callable[[Any], str]
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
            one_arg = cast(Callable[[Real], str], fmt_fn)

            def _wrapped(v: Real, s: str) -> tuple[Real, str]:
                return v, one_arg(v)

            new_fn: CellWrapper = _wrapped
        else:
            new_fn = cast(CellWrapper, fmt_fn)

        self.__options.row_formatters[row_idx] = new_fn

    def set_row_formatter(self, row_idx: int, fmt_fn: CellWrapper) -> None:
        """Set a specific row formatter function for the LaTeX table.

        Args:
            row_idx (int): Row index to set the formatter for.
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
        header_string = f"{{{self.__options.column_format}}}"
        if self.__options.toprule_cmd is not None:
            header_string += "\n" + self.__options.toprule_cmd + "\n"

        column_titles = []
        if set(self.__options.groups_to_cols.keys()) != set({""}):
            header_string += f"\n{self.__options.multicolumn_format}"

        if self.__options.include_index:
            index_name = (
                self.__options.index_name
                if self.__options.index_name is not None
                else (self.df.index.name if self.df.index.name is not None else "")
            )
            column_titles.append(latex_escape(str(index_name)))

        for _, col_list in self.__options.groups_to_cols.items():
            for col in col_list:
                col_title = latex_escape(col)
                if self.__options.bold_column_headers:
                    col_title = rf"\textbf{{{col_title}}}"
                if self.__options.italic_column_headers:
                    col_title = rf"\textit{{{col_title}}}"

                column_titles.append(col_title)

        header_string += "\n" + " & ".join(column_titles) + r" \\"

        return rf"\begin{{tabular}}{header_string}" + "\n"

    def _generate_body(self) -> str:
        """Generate the LaTeX table body string.

        Will format each cell according to the specified formatters in options and only
        touches the data rows of the internal dataframe.

        Returns:
            str: LaTeX table body string.
        """
        column_ordering = list(self.df.columns)

        if set(self.__options.groups_to_cols.keys()) != {""}:
            column_ordering = []
            for cols in self.__options.groups_to_cols.values():
                column_ordering.extend(cols)

        body_string = ""
        for row_idx, (df_row_idx, row) in enumerate(self.df.iterrows()):
            if (
                len(self.__options.hrule_counts) > 0
                and self.__options.hrule_counts[row_idx] > 0
            ):
                body_string += (
                    self.__options.hrule_cmd * self.__options.hrule_counts[row_idx]
                    + "\n"
                )

            color_type, color_value = self.__options.row_highlight_colors[row_idx]
            match color_type:
                case "NAME":
                    if not isinstance(color_value, str):
                        raise ValueError(f"Found invalid color value '{color_value}'.")
                    body_string += r"\rowcolor{" + str(color_value) + "}\n"
                case "HTML":
                    if not isinstance(color_value, str):
                        raise ValueError(
                            f"Found invalid hex color value '{color_value}'."
                        )

                    if not re.match(r"^#?[0-9A-Fa-f]{6}$", color_value):
                        raise ValueError(
                            f"Invalid hex color value '{color_value}'. "
                            "Must be 6 hexadecimal digits."
                        )
                    color_value = color_value.lstrip("#").upper()
                    body_string += r"\rowcolor[HTML]{" + color_value + "}\n"

                case "RGB":
                    r_val, g_val, b_val = color_value
                    body_string += r"\rowcolor[RGB]{" f"{r_val},{g_val},{b_val}" + "}\n"

                case "rgb":
                    r_val, g_val, b_val = color_value
                    body_string += (
                        r"\rowcolor[rgb]{"
                        f"{r_val:.3f},{g_val:.3f},{b_val:.3f}" + "}\n"
                    )

                case "NONE":
                    pass

                case _:
                    warnings.warn(
                        f"Unsupported color type '{color_type}' for row highlighting. "
                        f"Skipping row highlighting for row '{row_idx}'.",
                        stacklevel=2,
                    )

            row_items = []
            if self.__options.include_index:
                row_items.append(latex_escape(str(df_row_idx)))

            for col in column_ordering:
                cell_value = row[col]
                if pd.isna(cell_value):
                    cell_str = self.__options.nan_string
                else:
                    if col in self.__options.col_formatters:
                        cell_str = self.__options.col_formatters[col](
                            cell_value, str(cell_value)
                        )[1]
                    elif (
                        isinstance(cell_value, float)
                        and self.__options.number_fmt_fn is not None
                    ):
                        cell_str = self.__options.number_fmt_fn(
                            cell_value, str(cell_value)
                        )[1]
                    elif (
                        isinstance(cell_value, str)
                        and self.__options.str_fmt_fn is not None
                    ):
                        cell_str = self.__options.str_fmt_fn(cell_value, cell_value)[1]
                    else:
                        cell_str_raw = latex_escape(str(cell_value))
                        cell_str = (
                            self.__options.str_fmt_fn(cell_str_raw, cell_str_raw)[1]
                            if self.__options.str_fmt_fn is not None
                            else cell_str_raw
                        )

                row_items.append(cell_str)
            body_string += " & ".join(row_items) + r" \\" + "\n"

        return body_string

    def _generate_footer(self) -> str:
        """Generate the LaTeX table footer string."""
        footer_str = ""
        if self.__options.bottomrule_cmd is not None:
            footer_str += "\n" + self.__options.bottomrule_cmd + "\n"
        footer_str += r"\end{tabular}"
        return footer_str

    def _generate_latex(self) -> str:
        """Generate the complete LaTeX table string."""
        tex_string = ""
        tex_string += self._generate_header()
        tex_string += self._generate_body()
        tex_string += self._generate_footer()
        return tex_string
