"""Tests for LaTeX color helper utilities."""

import pytest

from gerrytools.latex._colors import (
    cellcolor_prefix,
    is_hex_color,
    is_latex_color_expression,
    normalize_hex_color,
    to_latex_color_spec,
    to_latex_xcolor_or_html_spec,
)


# ===================
# == HEX UTILITIES ==
# ===================
class TestHexUtilities:
    def test_is_hex_color_accepts_hash_and_whitespace(self):
        assert is_hex_color(" #Aa00Ff ")
        assert is_hex_color("00FF00")

    def test_is_hex_color_rejects_invalid_strings(self):
        assert not is_hex_color("")
        assert not is_hex_color("#12345")
        assert not is_hex_color("#1234567")
        assert not is_hex_color("#12GG56")

    def test_normalize_hex_color_lowercases_and_strips_hash(self):
        assert normalize_hex_color(" #Aa00Ff ") == "aa00ff"

    def test_normalize_hex_color_invalid_raises(self):
        with pytest.raises(ValueError, match="HEX string"):
            normalize_hex_color("not-a-color")


# ========================
# == XCOLOR EXPRESSIONS ==
# ========================
class TestXcolorExpressions:
    def test_is_latex_color_expression_accepts_single_name(self):
        assert is_latex_color_expression("denim")

    def test_is_latex_color_expression_accepts_compound_expression(self):
        assert is_latex_color_expression("denim!25!amber!50!white")

    def test_is_latex_color_expression_rejects_unknown_names(self):
        assert not is_latex_color_expression("notacolor!25!amber")

    def test_is_latex_color_expression_rejects_out_of_range_percentages(self):
        assert not is_latex_color_expression("denim!101!amber")
        assert not is_latex_color_expression("denim!-1!amber")

    def test_is_latex_color_expression_rejects_empty_or_malformed_tokens(self):
        assert not is_latex_color_expression("")
        assert not is_latex_color_expression("!!!")
        assert not is_latex_color_expression("denim!!amber")


# ========================
# == COLOR SPEC PARSING ==
# ========================
class TestColorSpecParsing:
    def test_to_latex_color_spec_handles_name_and_hex(self):
        assert to_latex_color_spec("denim") == ("NAME", "denim")
        assert to_latex_color_spec("#AABBCC") == ("HTML", "aabbcc")

    def test_to_latex_color_spec_handles_unit_rgb_tuple(self):
        assert to_latex_color_spec((0.1, 0.2, 0.3)) == ("rgb", (0.1, 0.2, 0.3))

    def test_to_latex_color_spec_handles_255_scale_rgb_tuple(self):
        assert to_latex_color_spec((12, 34, 56)) == ("RGB", (12, 34, 56))
        assert to_latex_color_spec((255.0, 127.6, 0.4)) == ("RGB", (255, 128, 0))

    def test_to_latex_color_spec_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="tuple of length 3"):
            to_latex_color_spec((1, 2))  # type: ignore[arg-type]

    def test_to_latex_color_spec_invalid_range_raises(self):
        with pytest.raises(ValueError, match="range \\[0\\.0, 1\\.0\\] or \\[0, 255\\]"):
            to_latex_color_spec((300, 0, 0))

    def test_to_latex_xcolor_or_html_spec_preserves_xcolor_expressions(self):
        assert to_latex_xcolor_or_html_spec("denim!25!amber") == ("NAME", "denim!25!amber")

    def test_to_latex_xcolor_or_html_spec_converts_named_colors_to_html(self):
        assert to_latex_xcolor_or_html_spec("tab:blue") == ("HTML", "1f77b4")

    def test_to_latex_xcolor_or_html_spec_preserves_none(self):
        assert to_latex_xcolor_or_html_spec("none") == ("NAME", "none")


# ======================
# == CELLCOLOR PREFIX ==
# ======================
class TestCellcolorPrefix:
    def test_cellcolor_prefix_supports_name_html_rgb_and_RGB(self):
        assert cellcolor_prefix("denim!20!amber") == r"\cellcolor{denim!20!amber}"
        assert cellcolor_prefix("#ABCDEF") == r"\cellcolor[HTML]{abcdef}"
        assert cellcolor_prefix((0.1, 0.2, 0.3)) == r"\cellcolor[rgb]{0.10,0.20,0.30}"
        assert cellcolor_prefix((10, 20, 30)) == r"\cellcolor[RGB]{10,20,30}"
