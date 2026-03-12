"""Tests for LaTeX TableOptions column format and multicolumn format generation."""

import pytest

from gerrytools.latex.table import TableOptions


# ===================
# == COLUMN FORMAT ==
# ===================
class TestColumnFormat:
    def test_column_format_basic_no_boundary_extras(self):
        opts = TableOptions(
            tabular_alignments=["l", "c", "r"],
            vrule_counts=[1, 0, 2, 0],
            boundary_extras=[],
        )

        expected = "|lc||r"
        assert opts.column_format == expected

    def test_column_format_with_boundary_extras_and_vrules(self):
        opts = TableOptions(
            tabular_alignments=["c", "c"],
            vrule_counts=[1, 0, 1],  # 3 boundaries
            boundary_extras=["A", "B", "C"],
        )

        expected = "|AcBc|C"
        assert opts.column_format == expected

    def test_column_format_bad_boundary_extras_length_raises(self):
        opts = TableOptions(
            tabular_alignments=["l", "r"],
            vrule_counts=[0, 0, 0],
            boundary_extras=["only_one"],
        )

        with pytest.raises(ValueError, match="boundary_extras must have length ncols\\+1"):
            _ = opts.column_format


# ===============================
# == MULTICOLUMN FORMAT MODE A ==
# ===============================
class TestMulticolumnFormatModeA:
    def test_multicolumn_format_mode_a_simple_no_index(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[1, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[1, 0],
            group_boundary_extras=["", ""],
        )

        out = opts.multicolumn_format
        assert out == r"\multicolumn{1}{|c}{\textbf{Group1}} \\"

    def test_multicolumn_format_mode_a_with_index(self):
        opts = TableOptions(
            include_index=True,
            index_alignment="l",
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[1, 0],
            group_boundary_extras=["", ""],
        )

        out = opts.multicolumn_format

        expected = r"\multicolumn{1}{l|}{}" r" & " r"\multicolumn{1}{c}{\textbf{Group1}} \\"
        assert out == expected

    def test_multicolumn_format_mode_a_bold_and_italic_group_headers(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
            bold_group_headers=True,
            italic_group_headers=True,
        )

        out = opts.multicolumn_format

        assert r"\textit{\textbf{Group1}}" in out

    def test_multicolumn_format_mode_a_no_bold_no_italic(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
            bold_group_headers=False,
            italic_group_headers=False,
        )

        out = opts.multicolumn_format

        assert "Group1" in out
        assert r"\textbf{Group1}" not in out
        assert r"\textit{Group1}" not in out

    def test_multicolumn_format_mode_a_span_zero_group_is_skipped(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Empty": [], "Full": ["c1"]},
            group_tabular_alignments=["c", "c"],
            group_vrule_counts=[0, 0, 0],
            group_boundary_extras=["", "", ""],
        )

        out = opts.multicolumn_format

        assert "Empty" not in out
        assert r"\textbf{Full}" in out
        assert out.count(r"\multicolumn{1}") == 1

    def test_multicolumn_format_mode_a_italic_only_group_headers(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
            bold_group_headers=False,
            italic_group_headers=True,
        )

        out = opts.multicolumn_format

        assert r"\textit{Group1}" in out
        assert r"\textbf{Group1}" not in out

    def test_multicolumn_format_mode_a_partial_group_preamble_raises(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=None,
            group_boundary_extras=None,
        )

        with pytest.raises(
            ValueError,
            match="group_\\* preamble is partially set; set alignments, vrules, and extras together\\.",
        ):
            _ = opts.multicolumn_format

    def test_multicolumn_format_mode_a_group_vrules_bad_length_raises(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c", "c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", "", ""],
        )

        with pytest.raises(ValueError, match="vrule_counts must have length ncols\\+1"):
            _ = opts.multicolumn_format

    def test_multicolumn_format_mode_a_group_boundary_extras_bad_length_raises(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["col1"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["only_one"],
        )

        with pytest.raises(
            ValueError,
            match="boundary_extras must have length ncols\\+1",
        ):
            _ = opts.multicolumn_format

    def test_multicolumn_format_mode_a_group_header_cell_count_mismatch_raises(self):
        opts = TableOptions(
            tabular_alignments=["l", "c"],
            vrule_counts=[0, 0, 0],
            groups_to_cols={"G1": ["c1"], "G2": ["c2"]},
            group_tabular_alignments=["c"],
            group_vrule_counts=[0, 0],
            group_boundary_extras=["", ""],
        )

        with pytest.raises(
            ValueError,
            match=r"Group-header preamble has 1 cells but expected 2\.",
        ):
            _ = opts.multicolumn_format


# ===============================
# == MULTICOLUMN FORMAT MODE B ==
# ===============================
class TestMulticolumnFormatModeB:
    def test_multicolumn_format_mode_b_simple_no_index(self):
        opts = TableOptions(
            tabular_alignments=["l", "c", "r"],
            vrule_counts=[1, 0, 0, 0],
            groups_to_cols={"G1": ["c1", "c2"], "G2": ["c3"]},
        )

        out = opts.multicolumn_format

        expected = r"\multicolumn{2}{|c}{\textbf{G1}}" r" & " r"\multicolumn{1}{r}{\textbf{G2}} \\"

        assert out == expected

    def test_multicolumn_format_mode_b_italic_only_group_headers(self):
        opts = TableOptions(
            tabular_alignments=["c"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            bold_group_headers=False,
            italic_group_headers=True,
        )

        out = opts.multicolumn_format

        expected = r"\multicolumn{1}{c}{\textit{Group1}} \\"
        assert out == expected

    def test_multicolumn_format_mode_b_bold_and_italic_group_headers(self):
        opts = TableOptions(
            tabular_alignments=["c"],
            vrule_counts=[0, 0],
            groups_to_cols={"Group1": ["c1"]},
            bold_group_headers=True,
            italic_group_headers=True,
        )

        out = opts.multicolumn_format

        expected = r"\multicolumn{1}{c}{\textit{\textbf{Group1}}} \\"
        assert out == expected

    def test_multicolumn_format_mode_b_span_zero_group_is_skipped(self):
        opts = TableOptions(
            tabular_alignments=["l"],
            vrule_counts=[0, 0],
            groups_to_cols={"Empty": [], "Used": ["c1"]},
        )

        out = opts.multicolumn_format

        assert "Empty" not in out
        assert r"\textbf{Used}" in out
        assert out.count(r"\multicolumn{1}") == 1

    def test_multicolumn_format_mode_b_vrules_bad_length_raises(self):
        opts = TableOptions(
            tabular_alignments=["l", "c"],  # ncols_total = 2
            vrule_counts=[0, 0],  # length 2 (should be 3)
            groups_to_cols={"G1": ["c1"], "G2": ["c2"]},
        )

        with pytest.raises(ValueError, match="vrule_counts must have length ncols\\+1"):
            _ = opts.multicolumn_format
