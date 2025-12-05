from gerrytools.latex.table import _parse_tabular_preamble
import pytest


# =========================================
#   TEST TABULAR PREABLE PARSING FUNCTION
# =========================================


@pytest.mark.timeout(1)
def test_basic_letters_no_rules():
    cols, vr, ex = _parse_tabular_preamble("lcr")
    assert cols == ["l", "c", "r"]
    assert vr == [0, 0, 0, 0]
    assert ex == ["", "", "", ""]


@pytest.mark.timeout(1)
def test_splits_repeated_letters():
    cols, vr, ex = _parse_tabular_preamble("lccr")
    assert cols == ["l", "c", "c", "r"]
    assert vr == [0, 0, 0, 0, 0]
    assert ex == ["", "", "", "", ""]


@pytest.mark.timeout(1)
def test_counts_vertical_rules_per_boundary():
    cols, vr, ex = _parse_tabular_preamble(r"|l||c|r|")
    assert cols == ["l", "c", "r"]
    # boundaries: before l, between l-c, between c-r, after r
    assert vr == [1, 2, 1, 1]
    assert ex == ["", "", "", ""]


@pytest.mark.timeout(1)
def test_boundary_extras_gt_applies_to_next_column_boundary():
    cols, vr, ex = _parse_tabular_preamble(r"l||>{\scriptsize}cc|cc||c")
    assert cols == ["l", "c", "c", "c", "c", "c"]
    assert vr == [0, 2, 0, 1, 0, 2, 0]
    assert ex == ["", r">{\scriptsize}", "", "", "", "", ""]


@pytest.mark.timeout(1)
def test_boundary_extras_lt_applies_to_boundary_before_column():
    # l  <{...} c  means the <{...} is attached to the boundary before that c
    cols, vr, ex = _parse_tabular_preamble(r"l<{X}c")
    assert cols == ["l", "c"]
    assert vr == [0, 0, 0]
    assert ex == ["", r"<{X}", ""]


@pytest.mark.timeout(1)
def test_boundary_extras_at_and_bang_preserved():
    cols, vr, ex = _parse_tabular_preamble(r"@{}l!{\hspace{2pt}}c@{}")
    assert cols == ["l", "c"]
    assert vr == [0, 0, 0]
    assert ex == [r"@{}", r"!{\hspace{2pt}}", r"@{}"]


@pytest.mark.timeout(1)
def test_mixed_rules_and_extras():
    cols, vr, ex = _parse_tabular_preamble(r"|@{}l||>{\bfseries}c|r@{}|")
    assert cols == ["l", "c", "r"]
    assert vr == [1, 2, 1, 1]
    assert ex == [r"@{}", r">{\bfseries}", "", r"@{}"]


@pytest.mark.timeout(1)
def test_p_m_b_columns_are_single_column_each():
    cols, vr, ex = _parse_tabular_preamble(r"p{2cm}|m{1in}b{3em}")
    assert cols == [r"p{2cm}", r"m{1in}", r"b{3em}"]
    assert vr == [0, 1, 0, 0]
    assert ex == ["", "", "", ""]


@pytest.mark.timeout(1)
def test_S_column_with_options_is_single_column():
    cols, vr, ex = _parse_tabular_preamble(r"l|S[table-format=2.3]|r")
    assert cols == ["l", r"S[table-format=2.3]", "r"]
    assert vr == [0, 1, 1, 0]
    assert ex == ["", "", "", ""]


@pytest.mark.timeout(1)
def test_D_column_is_single_column():
    cols, vr, ex = _parse_tabular_preamble(r"l|D{.}{.}{-1}|r")
    assert cols == ["l", r"D{.}{.}{-1}", "r"]
    assert vr == [0, 1, 1, 0]
    assert ex == ["", "", "", ""]


@pytest.mark.timeout(1)
def test_unbalanced_braces_raises():
    with pytest.raises(ValueError):
        _parse_tabular_preamble(r"l|p{2cm|r")  # missing closing }


@pytest.mark.timeout(1)
def test_stray_brace_raises_instead_of_hanging():
    import pytest

    with pytest.raises(ValueError):
        _parse_tabular_preamble(r"l|{c}|r")


@pytest.mark.timeout(1)
def test_invalid_char_raises():
    with pytest.raises(ValueError):
        _parse_tabular_preamble("l$cr")  # $ not valid in preamble outside braces


@pytest.mark.timeout(1)
def test_parse_complex_preamble():
    fmt_complex = r"|@{}l||>{\scriptsize\bfseries\color{purpleheart}}c<{\,}!{\hspace{2pt}}S[table-format=2.3,table-number-alignment=center]|D{.}{\cdot}{-1}||>{\raggedleft\arraybackslash}p{2.5cm}|m{1.8cm}b{3em}|r@{}|"
    expected_cols = [
        "l",
        "c",
        r"S[table-format=2.3,table-number-alignment=center]",
        r"D{.}{\cdot}{-1}",
        r"p{2.5cm}",
        r"m{1.8cm}",
        r"b{3em}",
        "r",
    ]

    expected_vrules = [1, 2, 0, 1, 2, 1, 0, 1, 1]

    expected_extras = [
        r"@{}",
        r">{\scriptsize\bfseries\color{purpleheart}}",
        r"<{\,}!{\hspace{2pt}}",
        "",
        r">{\raggedleft\arraybackslash}",
        "",
        "",
        "",
        r"@{}",
    ]
    cols, vr, ex = _parse_tabular_preamble(fmt_complex)
    assert cols == expected_cols
    assert vr == expected_vrules
    assert ex == expected_extras
    assert len(vr) == len(cols) + 1
    assert len(ex) == len(cols) + 1
