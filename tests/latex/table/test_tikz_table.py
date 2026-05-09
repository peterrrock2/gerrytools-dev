"""Tests for TikzTable geometry and LaTeX generation."""

from gerrytools.latex import TikzTable
from gerrytools.latex.formatters import (
    compose_formatters,
    diverging_gradient_formatter,
    round_decimals,
)


class TestTikzTableGeometry:
    def test_cell_size_normalizes_text_metrics_for_visual_centering(self, df):
        table = TikzTable(df, use_defaults=False)

        table.remove_all_headers()
        table.set_cell_size("1cm", "1cm")
        table.set_decimal_count(2)

        latex = str(table)

        assert r"minimum width=1cm" in latex
        assert r"minimum height=1cm" in latex
        # Dimension constants are defined once at the top of the picture...
        assert r"\def\gtboxwidth{\dimexpr 1cm-3pt*2\relax}" in latex
        assert r"\def\gtraiseheight{\dimexpr 1cm/2-3pt\relax}" in latex
        assert r"\def\gtraiseshift{\dimexpr (\dp\strutbox-\ht\strutbox)/2\relax}" in latex
        # ...and the \gtcell macro wraps each cell in raisebox+smash+makebox.
        assert (
            r"\def\gtcell#1#2{\raisebox{\gtraiseshift}"
            r"[\gtraiseheight][\gtraiseheight]"
            r"{\smash{\strut \makebox[\gtboxwidth][#1]{#2}}}}" in latex
        )
        assert r"\gtcell{c}{" in latex

    def test_exact_cell_size_wraps_cells_in_fixed_size_boxes(self, df):
        table = TikzTable(df, use_defaults=False)

        table.remove_all_headers()
        table.set_exact_cell_size("1cm", "1cm")
        table.set_decimal_count(2)

        latex = str(table)

        assert r"column sep=0pt" in latex
        assert r"minimum width=1cm" in latex
        assert r"minimum height=1cm" in latex
        assert r"\def\gtboxwidth{\dimexpr 1cm-3pt*2\relax}" in latex
        assert r"\def\gtraiseheight{\dimexpr 1cm/2-3pt\relax}" in latex
        assert r"\def\gtraiseshift{\dimexpr (\dp\strutbox-\ht\strutbox)/2\relax}" in latex
        assert r"\gtcell{c}{" in latex

    def test_uses_canonical_row_and_column_fit_nodes_for_fills_and_rules(self, df):
        table = TikzTable(df, use_defaults=False)

        table.include_index(name="", alignment="r")
        table.set_header_groups(
            {
                "Scores": ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"],
            }
        )
        table.set_tabular_format("r c c c c c l")
        table.add_hrule_above_all()
        table.highlight_rows([0], color="gray!50")
        table.set_number_formatter(
            compose_formatters(
                diverging_gradient_formatter(
                    color_lo="steelblue",
                    color_mid="white",
                    color_hi="firebrick",
                ),
                round_decimals(3),
            )
        )

        latex = str(table)

        # colfit / rowfit foreach blocks are unchanged
        assert r"\foreach \c in {1,...,7} {" in latex
        assert (
            r"\node[fit=(table-1-\c)(table-2-\c)(table-3-\c)(table-4-\c)(table-5-\c)"
            r"(table-6-\c)(table-7-\c)(table-8-\c)(table-9-\c)(table-10-\c)"
            r"(table-11-\c)(table-12-\c), inner sep=0pt] (colfitV\c) {};" in latex
        )
        # Row highlights go through the \gtrowfill macro inside a foreach.
        # Color is wrapped in {fill={...}} so inline rgb specs (which
        # contain commas/semicolons) survive pgfkeys parsing.
        assert r"\foreach \r in {3} {" in latex
        assert r"\gtrowfill{gray!50}{\r}" in latex
        assert (
            r"\def\gtrowfill#1#2{\fill[fill={#1}] "
            r"(colfitV1.west |- rowfitH#2.north)"
            r" rectangle (colfitV7.east |- rowfitH#2.south);}" in latex
        )
        # Per-cell hex highlights are grouped per row into one
        # \gtcellrowfills{row}{col1/name1, col2/name2, ...} call.  Each
        # name is derived from the source value (e.g. value 0.55 -> gradc_55)
        # and resolves to a \definecolor entry in the colour-definitions
        # block above.
        assert r"\gtcellrowfills{3}{" in latex
        assert "\\definecolor{tikzcc" not in latex
        # Value-derived gradient names appear both in the call and as
        # \definecolor entries.
        import re

        assert re.search(r"\\definecolor\{gradc_\d+", latex)
        assert re.search(r"\\gtcellrowfills\{3\}\{[^}]*gradc_\d+", latex)
        assert (
            r"\def\gtcellrowfills#1#2{\foreach \gtc/\gtn in {#2} {"
            r"\fill[fill={\gtn}]"
            r" (colfitV\gtc.west |- rowfitH#1.north) rectangle"
            r" (colfitV\gtc.east |- rowfitH#1.south);}}" in latex
        )
        # Hrules use \gthline / \gthlinestyled macros instead of inline \draw
        assert r"\gthline{3}" in latex
        assert (
            r"\def\gthline#1{\draw (colfitV1.west |- rowfitH#1.north)"
            r" -- (colfitV7.east |- rowfitH#1.north);}" in latex
        )
