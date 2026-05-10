# %% [markdown]
# # TikzTable Tutorial
#
# `TikzTable` generates LaTeX tables using a TikZ `matrix of nodes` instead
# of a `tabular` environment.  Because every cell is a proper TikZ node, the
# table can be styled and annotated with the full power of TikZ — uniform cell
# grids, per-cell borders, gradient heat-maps, and custom draw commands are all
# first-class features.
#
# **Prerequisites:** a working LaTeX installation with `pdflatex` (or `lualatex`)
# and the packages listed in `TexDocument`'s default preamble.  The `preview()`
# calls below render a PDF and display it inline in Jupyter.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd

from gerrytools.latex import TikzTable
from gerrytools.latex.formatters import (
    compose_formatters,
    diverging_gradient_formatter,
    highlight_between,
    highlight_ge,
    highlight_gt,
    highlight_lt,
    round_decimals,
    wrap_with_tex_command,
)

# Reproducible sample data ------------------------------------------------
rng = np.random.default_rng(42)
N = 10
df = pd.DataFrame(
    {
        "Score A": rng.uniform(0, 1, N),
        "Score B": rng.uniform(0, 1, N),
        "Score C": rng.uniform(0, 1, N),
        "Score D": rng.uniform(0, 1, N),
        "Score E": rng.uniform(0, 1, N),
        "Label": [f"item {i}" for i in range(N)],
    }
)
# Inject a NaN to show how it is handled
df.loc[3, "Score A"] = float("nan")

df

# %% [markdown]
# ## 1. Basic construction
#
# Pass a DataFrame to `TikzTable`.  By default:
# - all numeric columns are rounded to **4 decimal places**
# - column headers are shown (not bolded by default)
# - a double `\hline` separates headers from data
# - columns auto-size to their widest content

# %%
table = TikzTable(df)
print(table)
table.preview()

# %% [markdown]
# ## 2. Column alignment and tabular format
#
# `set_tabular_format` accepts the same string as a LaTeX tabular preamble.
# The number of column-spec tokens must equal the number of DataFrame columns
# (plus one if the index is included).
#
# Use `l`, `c`, `r` for left/center/right alignment.
# Fixed-width columns use `p{<dim>}`, `m{<dim>}`, or `b{<dim>}`.
# Vertical rules (`|`) and decorator specs (`>{}`, `<{}`) are supported.

# %%
table = TikzTable(df)
table.set_tabular_format("l c c c c r")  # left, four center, right
print(table)
table.preview()

# %%
# Wrap every cell in the first numeric column in \textbf using >{}/<{}
table = TikzTable(df)
table.set_tabular_format(r">{\bfseries}r c c c c r")
print(table)
table.preview()

# %% [markdown]
# ## 3. The index column
#
# By default the DataFrame index is hidden.  Call `include_index` to show it.
# You can pass a custom header name and an alignment token.

# %%
table = TikzTable(df)
table.include_index(name="Row", alignment="r")
print(table)
table.preview()

# %% [markdown]
# ## 4. Column header styling
#
# Headers can be made **bold** or *italic* independently.

# %%
table = TikzTable(df)
table.set_column_headers_text_format(bold=True, italic=False)
print(table)
table.preview()

# %%
table = TikzTable(df)
table.set_column_headers_text_format(bold=False, italic=True)
table.preview()

# %%
# Remove column headers entirely
table = TikzTable(df)
table.remove_column_headers()
table.preview()

# %% [markdown]
# ## 5. Header groups
#
# `set_header_groups` adds a second header row above the column headers that
# spans groups of columns.  The dict maps group labels to lists of column
# names.  Any column not mentioned is silently placed into an unlabelled group.

# %%
table = TikzTable(df)
table.set_header_groups(
    {
        "Group A": ["Score A", "Score B", "Score C"],
        "Group B": ["Score D", "Score E"],
    }
)
table.set_column_headers_text_format(bold=True)
table.set_group_headers_text_format(bold=True, italic=True)
print(table)
table.preview()

# %%
# Group headers can also be given their own alignment
table = TikzTable(df)
table.set_header_groups(
    {
        "Group A": ["Score A", "Score B", "Score C"],
        "Group B": ["Score D", "Score E"],
    }
)
table.set_group_tabular_format("c c c")  # three group cells (two named + unlabelled Label)
table.preview()

# %%
# Clear groups to go back to a flat header
table.clear_header_groups()
table.preview()

# %% [markdown]
# ## 6. Horizontal rules
#
# Horizontal rules are drawn as TikZ `\draw` commands after the matrix, so
# they span the **true column boundaries** regardless of cell content width.

# %%
table = TikzTable(df)

# Single rule above data row 0 (i.e., under the column header)
table.add_hrule_above(0)
# Double rule above data row 3
table.add_hrule_above(3, count=2)

table.preview()

# %%
# add_hrule_above_all puts a single rule above every data row
table = TikzTable(df)
table.add_hrule_above_all()
table.preview()

# %%
# Top-rule and bottom-rule (analogous to booktabs \toprule / \bottomrule)
table = TikzTable(df)
table.add_toprule()
table.add_bottomrule()
table.add_hrule_above(0)  # rule below the header
table.preview()

# %%
# Style the rules — any TikZ draw option string is accepted
table = TikzTable(df)
table.add_toprule(cmd="line width=1.5pt")
table.add_bottomrule(cmd="line width=1.5pt")
table.set_hrule_command("dashed, gray")
table.add_hrule_above_all()
table.preview()

# %% [markdown]
# ## 7. Vertical rules
#
# Vertical rules are placed at column *boundaries* (0 = left edge, n = right
# edge).  The helper methods accept 0-based **column indices**.

# %%
table = TikzTable(df)
table.add_vrule_left_of(0)  # left outer border
table.add_vrule_right_of(4)  # right outer border (after last numeric col)
table.add_vrule_left_of(3)  # separator before Score D
table.preview()

# %%
# Full box around the table
table = TikzTable(df)
table.add_vrule_all()
table.add_toprule()
table.add_bottomrule()
table.add_hrule_above(0)
table.preview()

# %% [markdown]
# ## 8. Row highlighting
#
# `highlight_rows` fills entire rows with a background colour on the TikZ
# background layer — it spans the true full table width, including
# inter-column gaps.

# %%
table = TikzTable(df)
table.highlight_rows(0, color="lightblue")
table.highlight_rows([2, 4, 6, 8], color="lavenderblush")
table.preview()

# %%
# Alternating row shading
table = TikzTable(df)
table.highlight_rows(list(range(0, N, 2)), color="azure")
table.preview()

# %% [markdown]
# ## 9. NaN handling
#
# By default NaN is shown as `NaN`.  Override with `set_nan_string`.

# %%
table = TikzTable(df)
table.set_nan_string(r"$-$")  # em-dash as a LaTeX math expression
table.preview()

# %% [markdown]
# ## 10. Number formatting
#
# `set_decimal_count` is a convenience shortcut.  For full control, pass a
# `CellWrapper` (a callable taking `(value, rendered_string)` and returning
# the same pair) to `set_number_formatter`.

# %%
table = TikzTable(df)
table.set_decimal_count(2)
table.preview()


# %%
# Percent display: multiply by 100 and append a percent sign
def as_percent(value, rendered):
    if 0 <= value <= 1:
        return value, rf"{value * 100:.1f}\%"
    return value, rendered


table = TikzTable(df)
table.set_number_formatter(as_percent)
table.preview()

# %% [markdown]
# ## 11. String formatting
#
# `set_string_formatter` applies to every string-valued cell.

# %%
table = TikzTable(df)
table.set_string_formatter(lambda v: v.upper())
table.preview()

# %%
# Italic strings using a LaTeX command
table = TikzTable(df)
table.set_string_formatter(lambda v, s: (v, rf"\textit{{{s}}}"))
table.preview()

# %% [markdown]
# ## 12. Composing formatters
#
# `compose_formatters(f, g, h)` applies formatters right-to-left:
# `f(g(h(value, string)))`.  This lets you layer rounding, conditional
# highlighting, and custom formatting in any order.


# %%
def make_percent(value, rendered):
    if isinstance(value, float) and 0 <= value <= 1:
        return value, rf"{value * 100:.1f}\%"
    return value, rendered


table = TikzTable(df)
table.set_number_formatter(
    compose_formatters(
        highlight_gt(0.7, color="teal", command_prefix=None),  # applied last (outermost)
        make_percent,
        round_decimals(2),  # applied first (innermost)
    )
)
table.preview()

# %% [markdown]
# ## 13. Conditional cell highlighting
#
# All highlight helpers — `highlight_gt`, `highlight_ge`, `highlight_lt`,
# `highlight_le`, `highlight_between` — produce `CellWrapper`s that prepend
# a `\cellcolor` when the comparison is true.  In `TikzTable` these are
# automatically converted to full-column-width post-matrix fills.

# %%
table = TikzTable(df)
table.set_number_formatter(
    compose_formatters(
        highlight_between(0.3, 0.7, color="lemonchiffon", command_prefix=None),
        highlight_ge(0.7, color="teal", command_prefix=None),
        highlight_lt(0.3, color="salmon", command_prefix=None),
        round_decimals(3),
    )
)
table.preview()

# %% [markdown]
# ## 14. Column-specific formatters
#
# `set_column_formatter` overrides formatting for one or more named columns.
# Column formatters take priority over the global number/string formatters.

# %%
table = TikzTable(df)
table.set_decimal_count(3)
# Score E gets its own treatment: highlight above 0.8 in green
table.set_column_formatter(
    "Score E",
    compose_formatters(
        highlight_gt(0.8, color="limegreen", command_prefix=None),
        round_decimals(3),
    ),
)
# Label column: italicise
table.set_column_formatter("Label", lambda v, s: (v, rf"\textit{{{s}}}"))
table.preview()

# %% [markdown]
# ## 15. Row-specific formatters
#
# `set_row_formatter` overrides formatting for individual data rows (0-based).
# Row formatters take priority over the global formatters but are overridden
# by column formatters.

# %%
table = TikzTable(df)
table.set_decimal_count(3)

# Row 5 (item 5): highlight values above 0.5 in pink
table.set_row_formatter(
    5,
    compose_formatters(
        highlight_gt(0.5, color="cherryblossompink", command_prefix=None),
        round_decimals(3),
    ),
)
# Row 5 also gets a light-blue background so the pink cells stand out
table.highlight_rows(5, color="lightblue")
table.preview()

# %% [markdown]
# ## 16. Diverging gradient heat-map
#
# `diverging_gradient_formatter` computes the gradient colour in Python and
# prepends a `\cellcolor[HTML]{RRGGBB}` prefix.  In `TikzTable` this is
# converted to a post-matrix TikZ `\fill` that spans the full column width.
# Works in both `TikzTable` and `TexTable` without any extra LaTeX command
# definitions.

# %%
table = TikzTable(df)
table.set_header_groups(
    {
        "Group A": ["Score A", "Score B", "Score C"],
        "Group B": ["Score D", "Score E"],
    }
)
table.set_number_formatter(
    compose_formatters(
        diverging_gradient_formatter(
            lo=0.0,
            mid=0.5,
            hi=1.0,
            color_lo="darkpastelgreen",
            color_mid="white",
            color_hi="richlavender",
            command_name=None,
        ),
        round_decimals(3),
    )
)
table.preview()

# %%
# Custom colour stops — e.g. blue → white → red
table = TikzTable(df)
table.set_number_formatter(
    compose_formatters(
        diverging_gradient_formatter(
            color_lo="steelblue",
            color_mid="white",
            color_hi="tomato",
            command_name=None,
        ),
        round_decimals(3),
    )
)
table.preview()

# %% [markdown]
# ## 17. Uniform-cell grid (all cells the same size)
#
# `set_cell_size(width, height)` sets a `minimum width` and `minimum height`
# for **every** node in the matrix.  Because TikZ nodes always expand to at
# least their minimum size, all cells become exactly `width × height` (unless
# content is wider/taller).  Pass `""` for the width to keep auto-sizing.

# %%
table = TikzTable(df)
table.set_cell_size("2cm", "0.9cm")  # uniform 2 cm × 0.9 cm grid
table.set_decimal_count(3)
table.preview()

# %%
# Tight grid: small cells, reduced inner sep
table = TikzTable(df)
table.set_cell_size("1.5cm", "0.65cm")
table.set_inner_sep("2pt")
table.set_decimal_count(2)
table.preview()

# %% [markdown]
# ## 18. Per-column widths and per-row heights
#
# `set_col_width` and `set_row_height` set minimums for individual
# columns/rows (0-based index).  These override the global `cell_width` /
# `cell_height` for the specified column or row.

# %%
table = TikzTable(df)
# Make the Label column wider, keep numeric columns at default
table.set_col_width(5, "3cm")  # column index 5 = "Label"
table.set_decimal_count(3)
table.preview()

# %%
# Taller first data row (e.g. to accommodate a header-like note)
table = TikzTable(df)
table.set_row_height(0, "1.4cm")  # row index 0 = first data row
table.set_decimal_count(3)
table.preview()

# %% [markdown]
# ## 19. Inner sep and node shape
#
# `inner sep` is the padding inside every node (default `3pt`).
# `set_node_shape` changes the TikZ node shape (e.g. `"circle"`,
# `"rounded rectangle"`).
# `set_extra_node_style` appends arbitrary TikZ style options to every node.

# %%
table = TikzTable(df)
table.set_inner_sep("6pt")  # more breathing room
table.set_decimal_count(2)
table.preview()

# %%
table = TikzTable(df)
table.set_cell_size("1cm", "1cm")
table.remove_all_headers()
# table.set_node_shape("circle")
table.set_decimal_count(2)
table.set_all_hrule(1)
table.add_vrule_all()
print(table.document)
table.preview()

# %%
# Draw a light border around every node via extra_node_style
table = TikzTable(df)
table.set_extra_node_style("draw=gray!40, line width=0.3pt")
table.set_decimal_count(2)
table.preview()

# %% [markdown]
# ## 20. Per-cell borders
#
# `set_cell_border` draws borders on individual cells.  Row and column
# indices are **1-based TikZ matrix coordinates** — row 1 is the first
# rendered row (the group-header row if present, otherwise the column-header
# row).
#
# `sides` can be `"top"`, `"bottom"`, `"left"`, `"right"`, `"all"`, or a
# list of those strings.

# %%
table = TikzTable(df)
table.set_decimal_count(3)
# Box around cells in data rows 1-3, column 2 (Score A is col 1; row 1 = header)
# With default headers: TikZ row 1 = column header, TikZ rows 2-11 = data rows
table.set_cell_border(row=list(range(2, 12)), col=1, sides="all")
table.preview()

# %%
# Highlight a 2×2 block of data cells with a box
table = TikzTable(df)
table.set_decimal_count(3)
header_offset = 1  # 1 header row (no group headers in this example)
for r in range(header_offset + 1, header_offset + 4):  # data rows 1-3
    for c in range(1, 4):  # columns 1-3
        table.set_cell_border(r, c, "all")
table.preview()

# %% [markdown]
# ## 21. Custom TikZ draw commands
#
# `add_draw` appends a raw TikZ line after the matrix.  The matrix is
# named `table`; cell (i, j) (1-based) is `(table-i-j)`.  This is an
# escape hatch for any annotation not covered by the higher-level API.

# %%
table = TikzTable(df)
table.set_decimal_count(2)
# Draw a red diagonal across the entire table
table.add_draw(r"\draw[red, thick] (table-2-1.north west) -- (table-11-5.south east);")
table.preview()

# %%
# Annotate the maximum value cell with a circle
# (using table-row-col coordinates for a known position)
table = TikzTable(df)
table.set_decimal_count(2)
table.set_cell_size("1.6cm", "0.8cm")
table.add_draw(r"\draw[orange, thick] (table-3-1) circle (0.35cm);")  # row 3 = data row 2, col 1
table.preview()

# %% [markdown]
# ## 22. The TexDocument object
#
# `table.document` is a `TexDocument` that wraps the `tikzpicture` in a
# complete standalone LaTeX document.  You can add custom preamble commands,
# `\usepackage` declarations, or arbitrary LaTeX snippets.

# %%
print(table.document)

# %%
from gerrytools.latex.commands import tex_gradient_command

table = TikzTable(df)
table.set_decimal_count(2)

# Add a custom LaTeX command to the preamble
table.document.add_command(tex_gradient_command("myheat"))

# Now use it to wrap cells (note: for TikzTable prefer diverging_gradient_formatter;
# this demonstrates the document.add_command pattern)
print(table.document)

# %% [markdown]
# ## 23. Putting it all together
#
# A polished table combining groups, alignment, toprule/bottomrule, row
# highlighting, conditional formatting, and a diverging gradient.

# %%
table = TikzTable(df)

# Structure
table.include_index(name="", alignment="r")
table.set_header_groups(
    {
        "Scores": ["Score A", "Score B", "Score C", "Score D", "Score E"],
    }
)
table.set_tabular_format("r c c c c c l")

# Styling
table.set_column_headers_text_format(bold=True)
table.set_group_headers_text_format(bold=True, italic=False)

# Rules
table.add_toprule()
table.add_bottomrule()
table.add_hrule_above_all()

# Alternating row shading
table.highlight_rows(list(range(0, N, 2)), color="gray!50")

# Diverging gradient on all numeric columns
table.set_number_formatter(
    compose_formatters(
        diverging_gradient_formatter(
            color_lo="steelblue",
            color_mid="white",
            color_hi="firebrick",
            command_name=None,
        ),
        round_decimals(3),
    )
)

# Label column: small caps
table.set_column_formatter("Label", lambda v, s: (v, rf"\textsc{{{s}}}"))

# NaN display
table.set_nan_string(r"$-$")

# Cell sizing — uniform enough to look grid-like
table.set_cell_size("1.6cm", "0.75cm")
table.set_inner_sep("3pt")

print(table)
table.preview()

# %% [markdown]
# ## 24. Resetting options
#
# `clear_options()` resets all options back to the constructor defaults
# without touching the underlying DataFrame.

# %%
table = TikzTable(df)
table.set_cell_size("3cm", "2cm")
table.add_toprule()
table.set_decimal_count(1)

# ... now reset everything
table.clear_options()
print(table)  # back to defaults

# %%
