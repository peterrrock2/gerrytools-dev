# %% [markdown]
# # TikzTable Tutorial
#
# `TikzTable` generates LaTeX tables from pandas DataFrames using the
# `nicematrix` package's `NiceTabular` environment.  Layout is done by LaTeX's
# own tabular engine, so the output is visually identical to `TexTable` for
# every shared feature — same rules, same spacing, same double rules.  On top
# of that, nicematrix exposes every cell as a PGF/TikZ node and provides a
# boundary lattice, so the table can be annotated with arbitrary TikZ:
# per-cell borders, custom draw commands, and exact-extent fills.
#
# This tutorial mirrors the TexTable tutorial section-for-section so the two
# can be compared side by side; TikZ-only features are at the end.
#
# **Prerequisites:** a working LaTeX installation with `pdflatex` (or another
# supported engine) plus the `nicematrix` and `tikz` packages.  nicematrix
# needs **two compile passes** (cell-node positions live in the aux file);
# `TikzTable.document` configures that automatically via
# `TexDocument.compile_passes`.  The `preview()` calls below render a PDF and
# display it inline in Jupyter.

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
# - column headers are shown and bolded
# - a double `\hline` separates headers from data
# - all columns are center-aligned

# %%
table = TikzTable(df)
print(table)
table.preview()

# %% [markdown]
# ## 2. Starting without defaults
#
# Pass `use_defaults=False`, or call `clear_options()`, when you want a plain
# table and explicit control over every option.

# %%
table = TikzTable(df, use_defaults=False)
print(table)
table.preview()

# %%
table = TikzTable(df)
table.clear_options()
print(table)
table.preview()

# %% [markdown]
# ## 3. Column alignment and tabular format
#
# `set_tabular_format` accepts a LaTeX tabular preamble.  The number of column
# specs must equal the number of DataFrame columns, plus one if the index is
# included.
#
# Use `l`, `c`, `r` for left/center/right alignment.  Fixed-width columns use
# `p{<dim>}`, `m{<dim>}`, or `b{<dim>}`.  Vertical rules (`|`) and decorator
# specs (`>{...}` / `<{...}`) are preserved in the generated tabular preamble.

# %%
table = TikzTable(df)
table.set_tabular_format("l c c c c r")  # left, four center, right
print(table)
table.preview()

# %%
# Use vertical rules directly in the tabular preamble.
table = TikzTable(df)
table.set_tabular_format("|l|c c c c|r|")
print(table)
table.preview()

# %%
# Wrap every cell in the first numeric column in \bfseries using >{}/<{}.
table = TikzTable(df)
table.set_tabular_format(r">{\bfseries}r c c c c r")
print(table)
table.preview()

# %% [markdown]
# ## 4. Fixed-width text columns
#
# Since `TikzTable` emits a normal `tabular`, column sizing is controlled through
# the tabular preamble.  Use `p{...}`, `m{...}`, or `b{...}` columns for wrapped
# text.  The `array` package is already included by the default document.

# %%
wide_df = df.copy()
wide_df["Label"] = [f"item {i}: a longer label for wrapped text" for i in range(N)]

table = TikzTable(wide_df)
table.set_tabular_format(r"c c c c c >{\raggedright\arraybackslash}p{3cm}")
table.set_decimal_count(2)
table.preview()

# %% [markdown]
# ## 5. The index column
#
# By default the DataFrame index is hidden.  Call `include_index` to show it.
# You can pass a custom header name and an alignment token.

# %%
table = TikzTable(df)
table.include_index(name="Row", alignment="r")
print(table)
table.preview()

# %%
# Index values can also be formatted.
table = TikzTable(df)
table.include_index(name="ID", alignment="r")
table.set_index_formatter(lambda v, s: (v, rf"\texttt{{{s}}}"))
table.preview()

# %% [markdown]
# ## 6. Column header styling
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
# Remove column headers entirely.
table = TikzTable(df)
table.remove_column_headers()
table.preview()

# %% [markdown]
# ## 7. Header groups
#
# `set_header_groups` adds a second header row above the column headers that
# spans groups of columns.  The dict maps group labels to lists of column names.
# Any column not mentioned is placed into an unlabelled group.

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
# Group headers can be given their own alignment and vertical rules.
table = TikzTable(df)
table.set_header_groups(
    {
        "Group A": ["Score A", "Score B", "Score C"],
        "Group B": ["Score D", "Score E"],
    }
)
table.set_group_tabular_format("|c|c|c|")  # two named groups + unlabelled Label group
table.preview()

# %%
# Clear groups to go back to a flat header.
table.clear_header_groups()
table.preview()

# %% [markdown]
# ## 8. Horizontal rules
#
# Horizontal rules are literal LaTeX commands inserted into the tabular body.
# Row index 0 means "above the first data row", i.e. directly below the header.

# %%
table = TikzTable(df)

# Single rule above data row 0 (i.e., under the column header).
table.add_hrule_above(0)
# Double rule above data row 3.
table.add_hrule_above(3, count=2)

table.preview()

# %%
# add_hrule_above_all puts a single rule above every data row.
table = TikzTable(df)
table.add_hrule_above_all()
table.preview()

# %%
# Top-rule and bottom-rule from booktabs.
table = TikzTable(df)
table.add_toprule(cmd=r"\toprule")
table.add_bottomrule(cmd=r"\bottomrule")
table.set_hrule_command(r"\midrule")
table.add_hrule_above(0)
table.preview()

# %% [markdown]
# ## 9. Vertical rules
#
# Vertical rules are represented by `|` tokens in the tabular preamble.  The
# helper methods accept 0-based column indices.

# %%
table = TikzTable(df)
table.add_vrule_left_of(0)  # left outer border
table.add_vrule_right_of(5)  # right outer border (after Label)
table.add_vrule_left_of(3)  # separator before Score D
print(table)
table.preview()

# %%
# Full box around the table, with horizontal rules as well.
table = TikzTable(df)
table.add_vrule_all()
table.add_toprule()
table.add_bottomrule()
table.add_hrule_above(0)
table.preview()

# %% [markdown]
# ## 10. Row highlighting
#
# `highlight_rows` emits nicematrix `\rowcolor` commands in the table's
# `\CodeBefore` block, filling each row's exact extent.  Colors may be xcolor
# names/expressions, hex strings, or RGB tuples.

# %%
table = TikzTable(df)
table.highlight_rows(0, color="lightblue")
table.highlight_rows([2, 4, 6, 8], color="lavenderblush")
table.preview()

# %%
# Alternating row shading.
table = TikzTable(df)
table.highlight_rows(list(range(0, N, 2)), color="gray!20")
table.preview()

# %%
# Hex colors are converted to \rowcolor[HTML]{...}.
table = TikzTable(df)
table.highlight_rows([1, 3, 5], color="#f6e8c3")
table.preview()

# %% [markdown]
# ## 11. NaN handling
#
# By default NaN is shown as `NaN`.  Override with `set_nan_string`.

# %%
table = TikzTable(df)
table.set_nan_string(r"$-$")
table.preview()

# %% [markdown]
# ## 12. Number formatting
#
# `set_decimal_count` is a convenience shortcut.  For full control, pass a
# `CellWrapper` (a callable taking `(value, rendered_string)` and returning the
# same pair) to `set_number_formatter`.

# %%
table = TikzTable(df)
table.set_decimal_count(2)
table.preview()


# %%
# Percent display: multiply by 100 and append a percent sign.
def as_percent(value, rendered):
    if 0 <= value <= 1:
        return value, rf"{value * 100:.1f}\%"
    return value, rendered


table = TikzTable(df)
table.set_number_formatter(as_percent)
table.preview()

# %% [markdown]
# ## 13. String formatting
#
# `set_string_formatter` applies to every string-valued cell.

# %%
table = TikzTable(df)
table.set_string_formatter(lambda v: v.upper())
table.preview()

# %%
# Italic strings using a LaTeX command.
table = TikzTable(df)
table.set_string_formatter(lambda v, s: (v, rf"\textit{{{s}}}"))
table.preview()

# %% [markdown]
# ## 14. Composing formatters
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
# ## 15. Conditional cell highlighting
#
# All highlight helpers -- `highlight_gt`, `highlight_ge`, `highlight_lt`,
# `highlight_le`, `highlight_between` -- produce `CellWrapper`s.  With
# `TikzTable`, pass `command_prefix=None` so they emit literal
# `\cellcolor` prefixes: nicematrix is not compatible with `colortbl`, so
# `TikzTable` converts those prefixes into exact-extent `\CodeBefore`
# background fills instead (the compact-command path is TexTable-only).

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
# ## 16. Column-specific formatters
#
# `set_column_formatter` overrides formatting for one or more named columns.
# Column formatters take priority over the global number/string formatters.

# %%
table = TikzTable(df)
table.set_decimal_count(3)
# Score E gets its own treatment: highlight above 0.8 in green.
table.set_column_formatter(
    "Score E",
    compose_formatters(
        highlight_gt(0.8, color="limegreen", command_prefix=None), round_decimals(3)
    ),
)
# Label column: italicise.
table.set_column_formatter("Label", lambda v, s: (v, rf"\textit{{{s}}}"))
table.preview()

# %% [markdown]
# ## 17. Row-specific formatters
#
# `set_row_formatter` overrides formatting for individual data rows (0-based).
# Row formatters take priority over the global formatters but are overridden by
# column formatters.

# %%
table = TikzTable(df)
table.set_decimal_count(3)

# Row 5 (item 5): highlight values above 0.5 in pink.
table.set_row_formatter(
    5,
    compose_formatters(
        highlight_gt(0.5, color="cherryblossompink", command_prefix=None), round_decimals(3)
    ),
)
# Row 5 also gets a light-blue background so the pink cells stand out.
table.highlight_rows(5, color="lightblue")
table.preview()

# %% [markdown]
# ## 18. Diverging gradient heat-map
#
# With `command_name=None`, `diverging_gradient_formatter` computes each
# cell's color in Python and emits a literal `\cellcolor[HTML]{...}` prefix.
# `TikzTable` strips those prefixes and turns them into named colors plus
# `\CodeBefore` cell fills that tile each cell exactly.

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
            precision=3,
            command_name=None,
        ),
        round_decimals(3),
    )
)
table.preview()

# %%
# Custom color stops, for example blue to white to red.
table = TikzTable(df)
table.set_number_formatter(
    compose_formatters(
        diverging_gradient_formatter(
            color_lo="steelblue",
            color_mid="white",
            color_hi="tomato",
            precision=3,
            command_name=None,
        ),
        round_decimals(3),
    )
)
table.preview()

# %% [markdown]
# ## 19. Command-based gradients
#
# `TexTable` supports a compact command-based path (`\divgrad{0.774}` calls
# with the gradient computed in LaTeX via `xfp`).  That path relies on
# `colortbl`'s in-cell `\cellcolor`, which is **not compatible with
# nicematrix** — so for `TikzTable`, always use `command_name=None`.  The
# rendered output is identical; only the generated LaTeX differs (named
# `\definecolor` entries plus `\CodeBefore` fills instead of preamble
# commands).

# %%
table = TikzTable(df)
table.set_number_formatter(
    compose_formatters(
        diverging_gradient_formatter(
            lo=0.0,
            mid=0.5,
            hi=1.0,
            color_lo="steelblue",
            color_mid="white",
            color_hi="firebrick",
            command_name=None,
        ),
        round_decimals(3),
    )
)
print(table.document)
table.preview()

# %% [markdown]
# ## 20. Wrapping values in LaTeX commands
#
# `wrap_with_tex_command` is useful when you want a formatter to wrap rendered
# cell text in a LaTeX command while leaving the raw value available for later
# formatters.

# %%
table = TikzTable(df)
table.set_number_formatter(
    compose_formatters(
        wrap_with_tex_command("textbf"),
        round_decimals(2),
    )
)
table.preview()

# %%
# Apply the wrapper to one column only.
table = TikzTable(df)
table.set_decimal_count(2)
table.set_column_formatter(
    "Score A",
    compose_formatters(wrap_with_tex_command("textbf"), round_decimals(2)),
)
table.preview()

# %% [markdown]
# ## 21. Header visibility and resetting options
#
# You can hide group headers, column headers, or both.  `clear_options()` resets
# all table options without changing the underlying DataFrame.

# %%
table = TikzTable(df)
table.set_header_groups({"Scores": ["Score A", "Score B", "Score C", "Score D", "Score E"]})
table.remove_group_headers()
table.preview()

# %%
table = TikzTable(df)
table.remove_all_headers()
table.preview()

# %%
table = TikzTable(df)
table.set_tabular_format("|l|c|c|c|c|r|")
table.add_toprule()
table.set_decimal_count(1)

# ... now reset everything.
table.clear_options()
print(table)  # back to plain, no-default options

# %% [markdown]
# ## 22. The TexDocument object
#
# `table.document` is a `TexDocument` that wraps the `tabular` in a complete
# standalone LaTeX document.  You can add custom preamble commands,
# `\usepackage` declarations, colors, or arbitrary LaTeX snippets.

# %%
print(table.document)

# %%
table = TikzTable(df)
table.set_decimal_count(2)

# Register a custom color on the document and use it from a formatter.
table.document.add_color("accent", "denim!60!white")
table.set_column_formatter("Score A", lambda v, s: (v, rf"\textcolor{{accent}}{{{v:.2f}}}"))
print(table.document)
table.preview()

# %% [markdown]
# ## 23. Putting it all together
#
# A polished table combining groups, alignment, booktabs rules, row
# highlighting, conditional formatting, and a command-based diverging gradient.

# %%
table = TikzTable(df)

# Structure.
table.include_index(name="", alignment="r")
table.set_header_groups(
    {
        "Scores": ["Score A", "Score B", "Score C", "Score D", "Score E"],
    }
)
table.set_tabular_format(r"r c c c c c >{\raggedright\arraybackslash}p{2cm}")

# Styling.
table.set_column_headers_text_format(bold=True)
table.set_group_headers_text_format(bold=True, italic=False)

# Rules.
table.add_toprule(cmd=r"\toprule")
table.add_bottomrule(cmd=r"\bottomrule")
table.set_hrule_command(r"\midrule")
table.add_hrule_above(0)

# Alternating row shading.
table.highlight_rows(list(range(0, N, 2)), color="gray!15")

# Diverging gradient on all numeric columns, computed in Python and applied
# as exact-extent \CodeBefore cell fills.
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

# Label column: small caps.
table.set_column_formatter("Label", lambda v, s: (v, rf"\textsc{{{s}}}"))

# NaN display.
table.set_nan_string(r"$-$")

print(table.document)
table.preview()

# %%

# %% [markdown]
# ## 24. Per-cell borders (TikZ-only)
#
# `set_cell_border` draws borders on individual cells.  Coordinates are
# 1-based and count header rows: with one header row, data row 0 is table
# row 2.  Borders are drawn on nicematrix's boundary lattice, so shared
# edges between adjacent cells are drawn exactly once and always meet the
# table's other rules.

# %%
table = TikzTable(df)
table.set_decimal_count(2)
table.set_cell_border(3, 2, "all")  # box around one cell
table.set_cell_border(5, [3, 4], "bottom")  # underline two cells
table.preview()

# %% [markdown]
# ## 25. Custom TikZ draw commands (TikZ-only)
#
# `add_draw` appends raw TikZ drawn over the finished table.  The content
# node of cell *(i, j)* is `(table-i-j)`, and the boundary lattice is
# available as `(row-i)` / `(col-j)` coordinates (combine them with the
# TikZ `-|` syntax).

# %%
table = TikzTable(df)
table.set_decimal_count(2)
# Circle one cell's content node.
table.add_draw(r"\draw[red, thick] (table-4-2) circle (4mm);")
# Underline the header row across the first three columns, on the lattice.
table.add_draw(r"\draw[blue, very thick] (row-2-|col-1) -- (row-2-|col-4);")
table.preview()

# %% [markdown]
# ## 26. Cell spacing and the two-pass compile (TikZ-only)
#
# `set_cell_space_limits` controls nicematrix's minimal vertical space
# around cell content (the analogue of a cell "inner sep").  And because
# nicematrix records cell-node positions in the aux file, the document
# compiles twice — `TikzTable` sets `document.compile_passes = 2` for you,
# which `save_pdf`, `save_png`, and `preview` all honor.

# %%
table = TikzTable(df)
table.set_decimal_count(2)
table.set_cell_space_limits("4pt")
print(table.document.compile_passes)
table.preview()

# %%
