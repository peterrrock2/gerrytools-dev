import pytest

from gerrytools.colors.latex import (
    _hex_to_rgb,
    _norm_hex,
    _rgb_to_hex,
    _xcolor_mix_hex,
    get_color_from_latex_string,
)

# =========================
# == HEX NORMALIZATION ==
# =========================


class TestNormalizeHex:
    def test_norm_hex_expands_three_digit_values(self):
        assert _norm_hex("#AbC") == "#aabbcc"

    def test_norm_hex_lowercases_six_digit_values(self):
        assert _norm_hex("A1B2C3") == "#a1b2c3"

    def test_norm_hex_warns_and_strips_alpha_for_four_digit_values(self):
        with pytest.warns(UserWarning, match="Ignoring alpha channel"):
            assert _norm_hex("#abcd") == "#aabbcc"

    def test_norm_hex_warns_and_strips_alpha_for_eight_digit_values(self):
        with pytest.warns(UserWarning, match="Ignoring alpha channel"):
            assert _norm_hex("#11223344") == "#112233"

    def test_norm_hex_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Not a valid hex color"):
            _norm_hex("xyz")


# ======================
# == HEX/RGB HELPERS ==
# ======================


class TestHexRgbHelpers:
    def test_hex_to_rgb_returns_rgb_tuple(self):
        assert _hex_to_rgb("#123456") == (18, 52, 86)

    def test_rgb_to_hex_rounds_and_clamps_components(self):
        assert _rgb_to_hex((255.0, 127.6, -5.0)) == "#ff8000"

    def test_xcolor_mix_hex_applies_xcolor_weighting(self):
        assert _xcolor_mix_hex(["#ff0000", "#ffffff"], [50]) == "#ff8080"

    def test_xcolor_mix_hex_requires_one_more_color_than_percentages(self):
        with pytest.raises(ValueError, match="one more than number of percentages"):
            _xcolor_mix_hex(["#ff0000"], [50])


# ========================
# == LATEX RESOLUTION ==
# ========================


class TestGetColorFromLatexString:
    def test_single_name_uses_bright_green_override(self):
        assert get_color_from_latex_string("green") == "#00ff00"

    def test_name_with_percent_defaults_to_white(self):
        assert get_color_from_latex_string("red!50") == "#ff8080"

    def test_explicit_two_color_mix(self):
        assert get_color_from_latex_string("red!0!blue") == "#0000ff"

    def test_three_color_left_fold_mix(self):
        # "red!100!blue!0!white" == left-fold: (red!100!blue)!0!white
        # red!100!blue = pure red; then red!0!white = white
        assert get_color_from_latex_string("red!100!blue!0!white") == "#ffffff"

    def test_unknown_color_name_raises(self):
        with pytest.raises(KeyError, match="Unknown color name"):
            get_color_from_latex_string("notacolor!50!white")

    def test_out_of_range_percentage_raises(self):
        with pytest.raises(ValueError, match="Percentages must be in \\[0,100\\]"):
            get_color_from_latex_string("red!101!white")

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="Empty color expression"):
            get_color_from_latex_string("   ")
