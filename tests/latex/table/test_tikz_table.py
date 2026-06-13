"""Tests for the nicematrix-based TikzTable LaTeX generation."""

from gerrytools.latex import TikzTable
from gerrytools.latex.formatters import (
    compose_formatters,
    diverging_gradient_formatter,
    round_decimals,
)


class TestTikzTableNiceTabularGeneration:
    def test_emits_nicetabular_with_tabular_dialect_rules(self, df):
        table = TikzTable(df)
        table.add_vrule_all()
        table.add_hrule_above_all()
        table.add_toprule()
        table.add_bottomrule()

        latex = str(table)

        # Real tabular column spec: rules live in the preamble, so they meet
        # horizontal rules by construction (the old matrix-of-nodes emitter
        # drew rules between node anchors and left visible gaps).
        assert r"\begin{NiceTabular}{|c|c|c|c|c|c|}[name=table" in latex
        assert latex.count(r"\hline") >= len(df)  # one interior rule per data row
        assert latex.strip().endswith(r"\end{NiceTabular}")

    def test_header_double_rule_pushed_into_header(self, df):
        # The header double rule (count == 2) is split: a single \hline at the
        # header/data boundary plus an upper rule drawn in \CodeAfter, with a
        # depth strut on the header row supplying the \doublerulesep gap. This
        # keeps the first data row's colour band the same height as the others
        # (a plain \hline\hline absorbs the gap into the first shaded band).
        table = TikzTable(df)  # use_defaults=True -> header double rule

        latex = str(table)

        # No verbatim double rule; the pair is reconstructed instead.
        assert "\\hline\n\\hline" not in latex
        assert r"\rule[-2.4pt]{0pt}{0pt} \\" in latex  # header depth strut
        # Upper rule of the pair, drawn above the boundary across the full width
        # (df has 6 columns -> col-7 is the right boundary; data starts at row 2).
        assert (
            r"\draw[line width=0.4pt] "
            r"([yshift=2pt]row-2-|col-1) -- ([yshift=2pt]row-2-|col-7);" in latex
        )
        assert r"\fill[white]" not in latex

    def test_rules_never_perturbed_by_fills(self, df):
        # The header rule geometry is identical whether or not rows are coloured,
        # so the (row-i) lattice stays stable for custom \draw commands.
        table = TikzTable(df)
        table.highlight_rows(0, color="lightblue")

        latex = str(table)

        assert r"\rule[-2.4pt]{0pt}{0pt} \\" in latex
        assert (
            r"\draw[line width=0.4pt] "
            r"([yshift=2pt]row-2-|col-1) -- ([yshift=2pt]row-2-|col-7);" in latex
        )
        assert r"\fill[white]" not in latex

    def test_single_and_custom_rules_emitted_verbatim(self, df):
        table = TikzTable(df, use_defaults=False)
        table.add_hrule_above_all()  # count 1 everywhere
        table.set_hrule_command(r"\midrule")

        latex = str(table)

        assert r"\midrule" in latex
        assert r"\Hline[tikz=" not in latex

    def test_row_and_cell_fills_emitted_in_code_before(self, df):
        table = TikzTable(df, use_defaults=False)
        table.highlight_rows([0], color="gray!50")
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

        latex = str(table)

        assert r"\CodeBefore" in latex
        assert r"\Body" in latex
        # Row highlight targets the first data row (row 2: one header row).
        assert r"\rowcolor{gray!50}{2}" in latex
        # Gradient cell colours are emitted inline as \cellcolor[HTML]{...}.
        assert r"\cellcolor[HTML]{" in latex
        # Regression: NO body-level \definecolor. nicematrix reserves spurious
        # horizontal space for every \definecolor in the document body, which
        # shifted gradient tables far to the right.
        assert r"\definecolor" not in latex

    def test_no_body_definecolor_for_hex_row_highlight(self, df):
        # Same regression guard for hex-coloured row highlights.
        table = TikzTable(df, use_defaults=False)
        table.highlight_rows([1, 3], color="#F6E8C3")

        latex = str(table)

        assert r"\rowcolor[HTML]{F6E8C3}{3}" in latex
        assert r"\definecolor" not in latex

    def test_group_tabular_format_draws_group_row_vrules(self, df):
        # set_group_tabular_format adds vertical rules that span only the
        # group-header row (matching \multicolumn{n}{|c|}{} in TexTable),
        # drawn on nicematrix's boundary lattice in \CodeAfter.
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Group A": ["Column 1", "Column 2", "Column 3"], "Group B": ["Column 4", "Column 5"]}
        )
        table.set_group_tabular_format("|c|c|c|")

        latex = str(table)

        assert r"\CodeAfter" in latex
        # Boundaries at columns 0, 3, 5, 6 -> lattice col-1, col-4, col-6, col-7,
        # each spanning the group-header row (row-1 to row-2).
        assert r"\draw (row-1-|col-1) -- (row-2-|col-1);" in latex
        assert r"\draw (row-1-|col-4) -- (row-2-|col-4);" in latex
        assert r"\draw (row-1-|col-6) -- (row-2-|col-6);" in latex
        assert r"\draw (row-1-|col-7) -- (row-2-|col-7);" in latex

    def test_group_headers_use_block_spans(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_header_groups(
            {"Scores": ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"]}
        )

        latex = str(table)

        assert r"\Block[c]{1-5}{\textbf{Scores}}" in latex
        # "Names" column is outside any group: empty cell, no Block.
        assert latex.count(r"\Block") == 1

    def test_cell_borders_draw_on_boundary_lattice(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_cell_border(3, 2, "all")

        latex = str(table)

        assert r"\CodeAfter" in latex
        # Four sides of cell (3,2) on nicematrix's (row-i)/(col-j) lattice.
        assert r"\draw (row-3-|col-2) -- (row-3-|col-3);" in latex  # top
        assert r"\draw (row-4-|col-2) -- (row-4-|col-3);" in latex  # bottom
        assert r"\draw (row-3-|col-2) -- (row-4-|col-2);" in latex  # left
        assert r"\draw (row-3-|col-3) -- (row-4-|col-3);" in latex  # right

    def test_adjacent_cell_borders_share_edges(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_cell_border(3, 2, "right")
        table.set_cell_border(3, 3, "left")  # same physical edge

        latex = str(table)

        assert latex.count(r"\draw (row-3-|col-3) -- (row-4-|col-3);") == 1

    def test_extra_draws_render_inside_code_after(self, df):
        table = TikzTable(df, use_defaults=False)
        draw = r"\draw[red] (table-2-1.north west) rectangle (table-2-1.south east);"
        table.add_draw(draw)

        latex = str(table)

        assert r"\CodeAfter" in latex
        assert r"\begin{tikzpicture}" in latex
        assert draw in latex

    def test_boundary_extras_reconstruct_valid_preamble_ordering(self, df):
        table = TikzTable(df, use_defaults=False)
        table.set_tabular_format(r">{\bfseries}c|c c c c l")
        table.add_vrule_left_of(0)

        latex = str(table)

        # ``>{...}`` opens its column after any vrule at the same boundary.
        assert r"\begin{NiceTabular}{|>{\bfseries}c|c" in latex

    def test_document_requires_nicematrix_and_two_passes(self, df):
        table = TikzTable(df, use_defaults=False)

        doc = table.document

        assert "nicematrix" in doc.package_list
        assert doc.compile_passes == 2
