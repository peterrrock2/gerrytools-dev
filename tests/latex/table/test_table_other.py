import math

import pandas as pd
import pytest

from gerrytools.latex.table import TexTable


def test_default_and_clear_options_are_the_same(df):
    table_1 = TexTable(df)
    table_1.clear_options()
    table_2 = TexTable(df, use_defaults=False)

    assert str(table_1) == str(table_2)


def test_include_index_resizes_existing_boundary_extras(table_defaults):
    table = table_defaults
    n_data_cols = table.df.shape[1]

    assert not table._options.include_index
    assert len(table._options.tabular_alignments) == n_data_cols

    table._options.boundary_extras = ["existing"] * 1  # != ncols + 1

    table.include_index(include=True)

    assert len(table._options.boundary_extras) == n_data_cols + 2


def test_string_ops_idempotent(table_defaults):
    orig_str = str(table_defaults)
    table_defaults.include_index(include=True)
    table_defaults.remove_index()
    assert str(table_defaults) == orig_str


# =============================
#   CHECK FOR ERRORS IN TABLE
# =============================


def test_include_index_add_mismatched_tabular_alignments_raises(table_defaults):
    table = table_defaults
    n_data_cols = table.df.shape[1]

    table._options.tabular_alignments = ["c"] * (n_data_cols + 1)

    with pytest.raises(
        ValueError,
        match=r"Current tabular format does not match DataFrame columns\.",
    ):
        table.include_index(include=True)


def test_include_index_remove_mismatched_tabular_alignments_raises(table_defaults):
    table = table_defaults
    n_data_cols = table.df.shape[1]

    table.include_index(include=True)

    table._options.tabular_alignments = ["c"] * (n_data_cols + 2)

    with pytest.raises(
        ValueError,
        match=r"Current tabular format does not match DataFrame columns\+index\.",
    ):
        table.include_index(include=False)


def test_add_hrule_above_negative_index_raises(table_defaults):
    table = table_defaults
    with pytest.raises(
        ValueError,
        match=r"Row index -1 is out of bounds for DataFrame with \d+ rows\.",
    ):
        table.add_hrule_above(-1)


def test_add_hrule_above_index_too_large_raises(table_defaults):
    table = table_defaults
    too_big = len(table.df) + 1
    with pytest.raises(
        ValueError,
        match=rf"Row index {too_big} is out of bounds for DataFrame with {len(table.df)} rows\.",
    ):
        table.add_hrule_above(too_big)


def test_add_vrule_left_of_negative_index_raises(table_defaults):
    table = table_defaults
    with pytest.raises(
        ValueError,
        match=r"Column index -1 is out of bounds for DataFrame with \d+ columns\.",
    ):
        table.add_vrule_left_of(-1)


def test_add_vrule_left_of_index_too_large_raises(table_defaults):
    table = table_defaults
    include_index_offset = int(table._options.include_index)
    too_big = len(table.df.columns) + include_index_offset + 1

    with pytest.raises(
        ValueError,
        match=rf"Column index {too_big} is out of bounds for DataFrame with "
        rf"{len(table.df.columns)} columns\.",
    ):
        table.add_vrule_left_of(too_big)


def test_add_vrule_right_of_index_too_small_raises(table_defaults):
    table = table_defaults
    with pytest.raises(
        ValueError,
        match=r"Column index -2 is out of bounds for DataFrame with \d+ columns\.",
    ):
        table.add_vrule_right_of(-2)


def test_add_vrule_right_of_index_too_large_raises(table_defaults):
    table = table_defaults
    include_index_offset = int(table._options.include_index)
    too_big = len(table.df.columns) + include_index_offset

    with pytest.raises(
        ValueError,
        match=rf"Column index {too_big} is out of bounds for DataFrame with "
        rf"{len(table.df.columns)} columns\.",
    ):
        table.add_vrule_right_of(too_big)


def test_highlight_rows_invalid_rgb_range_raises(table_defaults):
    table = table_defaults
    bad_color = (-1, 0, 0)

    with pytest.raises(
        ValueError,
        match=r"RGB color components must be in the range \[0\.0, 1\.0\] or \[0, 255\]",
    ):
        table.highlight_rows(0, color=bad_color)


def test_highlight_rows_invalid_color_spec_raises(table_defaults):
    table = table_defaults
    with pytest.raises(
        ValueError,
        match=r"Invalid color specification for row highlighting",
    ):
        table.highlight_rows(0, color=(1, 2))


def test_generate_body_invalid_name_color_value_type_raises(table_defaults):
    table = table_defaults
    # Force bad ("NAME", non-str) entry
    table._options.row_highlight_colors[0] = ("NAME", 123)

    with pytest.raises(
        ValueError,
        match=r"Found invalid color value '123'\.",
    ):
        table._generate_body()


def test_generate_body_invalid_html_color_value_type_raises(table_defaults):
    table = table_defaults
    table._options.row_highlight_colors[0] = ("HTML", 123)

    with pytest.raises(
        ValueError,
        match=r"Found invalid hex color value '123'\.",
    ):
        table._generate_body()


def test_generate_body_invalid_html_color_value_bad_hex_raises(table_defaults):
    table = table_defaults
    table._options.row_highlight_colors[0] = ("HTML", "#12345")  # 5 digits

    with pytest.raises(
        ValueError,
        match=r"Invalid hex color value '#12345'\. Must be 6 hexadecimal digits\.",
    ):
        table._generate_body()


def test_generate_body_unsupported_color_type_warns(table_defaults):
    table = table_defaults
    table._options.row_highlight_colors[0] = ("WEIRD", "blue")

    with pytest.warns(UserWarning, match=r"Unsupported color type 'WEIRD'"):
        table._generate_body()


def test_set_decimal_count_negative_raises(table_defaults):
    table = table_defaults
    with pytest.raises(
        ValueError,
        match=r"Decimal count must be non-negative",
    ):
        table.set_decimal_count(-1)


def test_set_tabular_format_column_count_mismatch_raises(table_defaults):
    table = table_defaults
    fmt = "c" * max(1, table.df.shape[1] - 1)

    with pytest.raises(
        ValueError,
        match=r"Format implies \d+ columns but expected \d+ ",
    ):
        table.set_tabular_format(fmt)


def test_set_group_tabular_format_cell_count_mismatch_raises(table_defaults):
    table = table_defaults
    fmt = "cc"

    with pytest.raises(
        ValueError,
        match=r"Group-header format implies \d+ cells but expected \d+",
    ):
        table.set_group_tabular_format(fmt)


def test_set_header_groups_unknown_columns_raises(table_defaults):
    table = table_defaults

    with pytest.raises(
        ValueError,
        match=r"Unknown columns in groups_to_columns: \['not_a_col'\]",
    ):
        table.set_header_groups({"GroupA": ["not_a_col"]})


def test_set_column_formatter_unknown_column_raises(table_defaults):
    table = table_defaults

    def fmt(v, s):
        return v, s

    with pytest.raises(
        ValueError,
        match=r"Column 'not_a_col' does not exist in DataFrame\.",
    ):
        table.set_column_formatter("not_a_col", fmt)


def test_set_row_formatter_row_too_large_raises(table_defaults):
    table = table_defaults

    def fmt(v, s):
        return v, s

    bad_idx = len(table.df)

    with pytest.raises(
        ValueError,
        match=rf"Row index {bad_idx} is out of bounds for DataFrame with {len(table.df)} rows\.",
    ):
        table.set_row_formatter(bad_idx, fmt)


def test_set_row_formatter_negative_index_raises(table_defaults):
    table = table_defaults

    def fmt(v, s):
        return v, s

    with pytest.raises(
        ValueError,
        match=r"Row index -1 is out of bounds for DataFrame with \d+ rows\.",
    ):
        table.set_row_formatter(-1, fmt)


# =======================
#   MISCELLANEOUS TESTS
# =======================


def test_add_hrule_above_initializes_and_extends(table_defaults):
    table = table_defaults

    table.clear_all_hrule()
    assert table._options.hrule_counts == []

    table.add_hrule_above(0, count=2)

    assert len(table._options.hrule_counts) == len(table.df)
    assert table._options.hrule_counts[0] == 2


def test_add_hrule_above_extends_existing_short_hrule_counts():
    df = pd.DataFrame({"a": [1, 2, 3]})
    table = TexTable(df)

    table._options.hrule_counts = [5]

    table.add_hrule_above(2, count=1)

    assert len(table._options.hrule_counts) == 3
    assert table._options.hrule_counts[0] == 5
    assert table._options.hrule_counts[2] == 1


def test_add_hrule_above_all_initializes_if_empty(table_defaults):
    table = table_defaults

    table._options.hrule_counts = []
    table.add_hrule_above_all(count=3)

    assert table._options.hrule_counts == [3] * len(table.df)


def test_clear_all_hrule(table_defaults):
    table = table_defaults

    table.add_hrule_above_all(count=1)
    assert any(c > 0 for c in table._options.hrule_counts)

    table.clear_all_hrule()
    assert table._options.hrule_counts == []


def test_add_toprule_default_and_remove(table_defaults):
    table = table_defaults

    table._options.hrule_cmd = r"\midrule"
    table._options.toprule_cmd = None

    table.add_toprule()
    assert table._options.toprule_cmd == r"\midrule"

    header = table._generate_header()
    assert r"\midrule" in header

    table.remove_toprule()
    assert table._options.toprule_cmd is None


def test_add_bottomrule_default_and_remove(table_defaults):
    table = table_defaults

    table._options.hrule_cmd = r"\midrule"
    table._options.bottomrule_cmd = None

    table.add_bottomrule()
    assert table._options.bottomrule_cmd == r"\midrule"

    footer = table._generate_footer()
    assert r"\midrule" in footer

    table.remove_bottomrule()
    assert table._options.bottomrule_cmd is None


def test_add_vrule_left_of_initializes_and_updates(table_defaults):
    table = table_defaults
    ncols = len(table.df.columns)
    include_index_offset = int(table._options.include_index)

    table._options.vrule_counts = []

    table.add_vrule_left_of(0, count=2)

    assert len(table._options.vrule_counts) == ncols + 1 + include_index_offset
    assert table._options.vrule_counts[0] == 2


def test_clear_all_vrule_sets_correct_length(table_defaults):
    table = table_defaults

    table.include_index(include=True)
    table.clear_all_vrule()

    expected_len = len(table.df.columns) + 1 + int(table._options.include_index)
    assert len(table._options.vrule_counts) == expected_len
    assert all(c == 0 for c in table._options.vrule_counts)


def test_add_vrule_right_of_initializes_and_updates(table_defaults):
    table = table_defaults
    ncols = len(table.df.columns)
    include_index_offset = int(table._options.include_index)

    table._options.vrule_counts = []

    table.add_vrule_right_of(0, count=3)

    assert len(table._options.vrule_counts) == ncols + 1 + include_index_offset
    assert table._options.vrule_counts[1] == 3


def test_add_vrule_all_initializes_and_increments(table_defaults):
    table = table_defaults

    table._options.vrule_counts = []
    table.add_vrule_all(count=1)

    total_cols = len(table.df.columns) + int(table._options.include_index)
    assert len(table._options.vrule_counts) == total_cols + 1
    assert table._options.vrule_counts == [1] * (total_cols + 1)


def test_highlight_rows_name_color_and_generate_body(table_defaults):
    table = table_defaults
    table.highlight_rows(0, color="yellow")

    body = table._generate_body()
    assert r"\rowcolor{yellow}" in body


def test_highlight_rows_xcolor_expression_preserved(table_defaults):
    table = table_defaults
    table.highlight_rows(0, color="amber!80!gray")

    assert table._options.row_highlight_colors[0] == ("NAME", "amber!80!gray")

    body = table._generate_body()
    assert r"\rowcolor{amber!80!gray}" in body


def test_highlight_rows_non_latex_named_color_converts_to_html(table_defaults):
    table = table_defaults
    table.highlight_rows(0, color="tab:blue")

    color_type, color_value = table._options.row_highlight_colors[0]
    assert color_type == "HTML"
    assert isinstance(color_value, str)
    assert len(color_value) == 6

    body = table._generate_body()
    assert r"\rowcolor[HTML]{" in body


def test_highlight_rows_initializes_row_highlight_colors_if_empty():
    df = pd.DataFrame({"a": [1, 2]})
    table = TexTable(df)

    table._options.row_highlight_colors = []

    table.highlight_rows(1, color="yellow")

    assert len(table._options.row_highlight_colors) == len(df)
    assert table._options.row_highlight_colors[1] == ("NAME", "yellow")


def test_highlight_rows_hex_color_and_generate_body(table_defaults):
    table = table_defaults
    table.highlight_rows(0, color="#abcdef")

    body = table._generate_body()
    assert r"\rowcolor[HTML]{abcdef}" in body


def test_highlight_rows_rgb_float_and_generate_body(table_defaults):
    # Use our own one-row df for clean expectations
    df = pd.DataFrame({"a": [1.0]})
    table = TexTable(df)

    table.highlight_rows(0, color=(0.1, 0.2, 0.3))

    body = table._generate_body()
    # formatted to 3 decimals
    assert r"\rowcolor[rgb]{0.100,0.200,0.300}" in body


def test_highlight_rows_RGB_int_and_generate_body(table_defaults):
    df = pd.DataFrame({"a": [1.0]})
    table = TexTable(df)

    table.highlight_rows(0, color=(10, 20, 30))

    body = table._generate_body()
    assert r"\rowcolor[RGB]{10,20,30}" in body


def test_set_column_and_group_header_text_format(table_defaults):
    table = table_defaults

    table.set_column_headers_text_format(bold=True, italic=True)
    table.set_group_headers_text_format(bold=False, italic=True)

    header = table._generate_header()
    assert r"\textbf{" in header
    assert r"\textit{" in header


def test_set_decimal_count_positive_path_and_used_in_body():
    df = pd.DataFrame({"a": [math.pi]})
    table = TexTable(df)

    table.set_decimal_count(2)
    body = table._generate_body()
    # Rounded to 2 decimals
    assert "3.14" in body


def test_set_hrule_and_all_hrule(table_defaults):
    table = table_defaults

    table.set_hrule_command(r"\hline")
    table.set_all_hrule(2)

    assert table._options.hrule_cmd == r"\hline"
    assert table._options.hrule_counts == [2] * len(table.df)


def test_set_nan_string_used_in_body():
    df = pd.DataFrame({"a": [1.0, float("nan")]})
    table = TexTable(df)

    table.set_nan_string("NA")
    body = table._generate_body()
    assert "NA" in body


def test_set_tabular_format_success_without_index(table_defaults):
    table = table_defaults
    ncols = len(table.df.columns)

    fmt = "c" * ncols
    table.set_tabular_format(fmt)

    assert table._options.tabular_alignments == ["c"] * ncols
    assert len(table._options.vrule_counts) == ncols + 1
    assert len(table._options.boundary_extras) == ncols + 1


def test_set_tabular_format_success_with_index():
    df = pd.DataFrame({"a": [1, 2]})
    table = TexTable(df)
    table.include_index(include=True)

    fmt = "cc"
    table.set_tabular_format(fmt)

    assert table._options.tabular_alignments == ["c", "c"]


def test_set_group_tabular_format_autocompletes_short_fmt():
    df = pd.DataFrame({"a": [1], "b": [2]})
    table = TexTable(df)

    table.set_header_groups({"G1": ["a"], "G2": ["b"]})
    table.set_group_tabular_format("c")

    assert table._options.group_tabular_alignments == ["c", "c"]
    assert table._options.group_vrule_counts is not None
    assert len(table._options.group_vrule_counts) == 3  # 2 cells + 1 boundary


def test_clear_header_groups_resets_state():
    df = pd.DataFrame({"a": [1], "b": [2]})
    table = TexTable(df)

    table.set_header_groups({"G1": ["a"], "G2": ["b"]})
    table.set_group_tabular_format("cc")
    table.clear_header_groups()

    assert table._options.groups_to_cols == {"": ["a", "b"]}
    assert table._options.group_tabular_alignments is None
    assert table._options.group_vrule_counts is None
    assert table._options.group_boundary_extras is None


def test_set_header_groups_with_missing_cols_and_blank_group_key():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    table = TexTable(df)

    table.set_header_groups(
        {
            "G1": ["a"],
            "": ["b"],
        }
    )
    g2c = table._options.groups_to_cols
    assert g2c[""][-1] == "c"


def test_set_header_groups_with_missing_cols_and_no_blank_group():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    table = TexTable(df)

    table.set_header_groups({"G1": ["a"]})
    g2c = table._options.groups_to_cols

    assert g2c["G1"] == ["a"]
    assert g2c[""] == ["b", "c"]


def test_set_header_groups_covering_all_cols_no_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    table = TexTable(df)

    table.set_header_groups({"G1": ["a"], "G2": ["b"]})
    g2c = table._options.groups_to_cols

    assert set(g2c.keys()) == {"G1", "G2"}
    assert "" not in g2c


def test_set_number_formatter_one_arg_and_used():
    df = pd.DataFrame({"a": [2.0]})
    table = TexTable(df)

    table.set_number_formatter(lambda x: f"{x:.1f}")
    body = table._generate_body()
    assert "2.0" in body or "2.00" in body


def test_set_number_formatter_two_arg_and_used():
    df = pd.DataFrame({"a": [2.0]})
    table = TexTable(df)

    def fmt(v, s):
        return v, f"VAL={s}"

    table.set_number_formatter(fmt)
    body = table._generate_body()
    assert "VAL=" in body


def test_set_string_formatter_one_arg_and_used():
    df = pd.DataFrame({"a": ["hello"]})
    table = TexTable(df)

    table.set_string_formatter(lambda s: s.upper())
    body = table._generate_body()
    assert "HELLO" in body


def test_set_string_formatter_two_arg_and_used():
    df = pd.DataFrame({"a": ["hello"]})
    table = TexTable(df)

    def fmt(v, s):
        return v, s + "!"

    table.set_string_formatter(fmt)
    body = table._generate_body()
    assert "hello!" in body


def test_set_column_formatter_list_branch_and_wrapping():
    df = pd.DataFrame({"a": [1], "b": [2]})
    table = TexTable(df)

    def fmt(v, s):
        return v, f"C{v}"

    table.set_column_formatter(["a", "b"], fmt)
    body = table._generate_body()
    assert "C1" in body
    assert "C2" in body


def test_set_column_formatter_one_arg_wrapper():
    df = pd.DataFrame({"a": [1]})
    table = TexTable(df)

    table.set_column_formatter("a", lambda v: f"{v}X")
    body = table._generate_body()
    assert "1X" in body


def test_set_row_formatter_list_branch_and_wrapping():
    df = pd.DataFrame({"a": [1, 2]})
    table = TexTable(df)

    def fmt(v, s):
        return v, f"R{v}"

    table.set_row_formatter([0, 1], fmt)

    body = table._generate_body()
    assert "R1" in body
    assert "R2" in body


def test_set_row_formatter_one_arg_wrapper():
    df = pd.DataFrame({"a": [1]})
    table = TexTable(df)

    table.set_row_formatter(0, lambda v: f"R{v}")
    body = table._generate_body()
    assert "R1" in body


def test_generate_header_with_groups_and_index():
    df = pd.DataFrame({"A": [1], "B": [2]})
    table = TexTable(df)

    table.include_index(name="idx", include=True)
    table.set_header_groups({"G1": ["A"], "G2": ["B"]})
    table.add_toprule()

    header = table._generate_header()

    assert r"\begin{tabular}" in header
    assert "idx" in header
    assert r"\multicolumn" in header or "G1" in header or "G2" in header


def test_generate_body_with_hrule_and_various_paths():
    df = pd.DataFrame(
        {
            "f": [1.23, 4.56],
            "s": ["x&y", "z"],
        }
    )
    table = TexTable(df)

    table.set_all_hrule(0)
    table.add_hrule_above(0, count=1)

    table.set_column_formatter("f", lambda v, s: (v, f"F={s}"))
    table.set_row_formatter(1, lambda v, s: (v, f"R={s}"))
    table.set_string_formatter(lambda s: s.replace("&", r"\&"))

    body = table._generate_body()

    assert "F=1.23" in body or "F=1.2300" in body
    assert "R=z" in body
    assert r"\&" in body


def test_generate_footer_and_full_latex(table_defaults):
    table = table_defaults
    table.add_bottomrule()

    footer = table._generate_footer()
    assert r"\end{tabular}" in footer

    full = table._generate_latex()
    assert full.startswith(r"\begin{tabular}")
    assert full.strip().endswith(r"\end{tabular}")
