"""Tests for LaTeX cell formatter helpers."""

import math

from gerrytools.latex.formatters import (
    _safe_round,
    boxed_center,
    compose_formatters,
    diverging_gradient_formatter,
    highlight_between,
    highlight_ge,
    highlight_gt,
    highlight_le,
    highlight_lt,
    latex_commands_for,
    round_decimals,
    wrap_between,
    wrap_ge,
    wrap_gt,
    wrap_le,
    wrap_lt,
    wrap_with_tex_command,
)


# ======================
# == BASIC FORMATTERS ==
# ======================
class TestBasicFormatters:
    def test_boxed_center_uses_width_for_default_height(self):
        formatter = boxed_center(4)
        value, rendered = formatter("raw", "X")

        assert value == "raw"
        assert rendered.startswith(r"\parbox[c][4mm][c]{4}")

    def test_boxed_center_wraps_content_and_preserves_value(self):
        formatter = boxed_center(3, height=5, unit="mm")
        value, rendered = formatter(7, "X")

        assert value == 7
        assert rendered.startswith(r"\parbox[c][5mm][c]{3}")
        assert r"\centering\strut X" in rendered

    def test_wrap_with_tex_command_wraps_rendered_string(self):
        formatter = wrap_with_tex_command("textbf")
        value, rendered = formatter("raw", "formatted")

        assert value == "raw"
        assert rendered == r"\textbf{formatted}"

    def test_compose_formatters_applies_right_to_left(self):
        formatter = compose_formatters(
            wrap_with_tex_command("textbf"),
            wrap_with_tex_command("emph"),
        )
        _value, rendered = formatter("raw", "hello")

        assert rendered == r"\textbf{\emph{hello}}"

    def test_round_decimals_formats_numeric_values_only(self):
        formatter = round_decimals(2)

        assert formatter(1.2349, "ignored") == (1.2349, "1.23")
        assert formatter("text", "text") == ("text", "text")

    def test_compose_formatters_preserves_latex_command_metadata(self):
        formatter = compose_formatters(
            diverging_gradient_formatter(),
            round_decimals(3),
        )

        commands = latex_commands_for(formatter)

        assert len(commands) == 1
        assert r"\newcommand{\divgrad}[1]{%" in commands[0]

    def test_diverging_gradient_formatter_defaults_to_divgrad_command(self):
        formatter = diverging_gradient_formatter(precision=3)

        value, rendered = formatter(0.75, "ignored")

        assert value == 0.75
        assert rendered == r"\divgrad{0.750}"
        assert r"\newcommand{\divgrad}[1]{%" in latex_commands_for(formatter)[0]

    def test_diverging_gradient_formatter_can_emit_compact_command_call(self):
        formatter = diverging_gradient_formatter(
            command_name="scoreheat",
            color_lo="steelblue",
            color_mid="white",
            color_hi="firebrick",
            precision=3,
        )

        value, rendered = formatter(0.75, "0.750")

        assert value == 0.75
        assert rendered == r"\scoreheat{0.750}"
        assert r"\colorlet{scoreheatLosteelblue}{steelblue}%" in latex_commands_for(formatter)[0]

    def test_diverging_gradient_command_uses_precision_without_rounding_formatter(self):
        formatter = diverging_gradient_formatter(command_name="scoreheat", precision=2)

        assert formatter(0.756, "0.756")[1] == r"\scoreheat{0.76}"

    def test_diverging_gradient_formatter_can_emit_literal_cellcolor(self):
        formatter = diverging_gradient_formatter(command_name=None)

        assert formatter(0.75, "0.750")[1].startswith(r"\cellcolor[HTML]{")
        assert latex_commands_for(formatter) == ()


# ======================
# == ROUNDING HELPERS ==
# ======================
class TestSafeRound:
    def test_safe_round_rounds_finite_numeric_values(self):
        assert _safe_round(1.2349, 2) == 1.23

    def test_safe_round_preserves_nan_and_infinities(self):
        assert math.isnan(_safe_round(float("nan"), 2))  # ty: ignore[invalid-argument-type]
        assert _safe_round(float("inf"), 2) == float("inf")
        assert _safe_round(float("-inf"), 2) == float("-inf")

    def test_safe_round_preserves_non_numeric_values(self):
        assert _safe_round("hello", 2) == "hello"
        assert _safe_round(1.2349, None) == 1.2349


# ==========================
# == NUMERIC HIGHLIGHTERS ==
# ==========================
class TestNumericHighlighters:
    def test_highlight_gt_wraps_matching_values_with_command(self):
        formatter = highlight_gt(10, color="denim")
        assert formatter(11, "11") == (11, r"\gta{11}")
        assert formatter(10, "10") == (10, "10")
        assert latex_commands_for(formatter) == (r"\newcommand{\gta}[1]{\cellcolor{denim}#1}",)

    def test_highlight_ge_respects_rounding(self):
        formatter = highlight_ge(1.23, color="#00FF00", round_to=2)
        assert formatter(1.234, "1.234") == (1.234, r"\gea{1.234}")
        assert formatter(1.225, "1.225") == (1.225, r"\gea{1.225}")
        assert latex_commands_for(formatter) == (
            r"\newcommand{\gea}[1]{\cellcolor[HTML]{00ff00}#1}",
        )

    def test_highlight_ge_can_emit_literal_cellcolor(self):
        formatter = highlight_ge(1.23, color="#00FF00", round_to=2, command_prefix=None)
        assert formatter(1.234, "1.234") == (1.234, r"\cellcolor[HTML]{00ff00}1.234")
        assert latex_commands_for(formatter) == ()

    def test_highlight_lt_and_le_leave_strings_unchanged(self):
        assert highlight_lt(5)("text", "text") == ("text", "text")
        assert highlight_le(5)("text", "text") == ("text", "text")

    def test_highlight_between_respects_inclusive_flags(self):
        inclusive = highlight_between(1, 2, color="amber")
        exclusive = highlight_between(1, 2, color="amber", include_lower=False, include_upper=False)

        assert inclusive(1, "1") == (1, r"\btwa{1}")
        assert inclusive(2, "2") == (2, r"\btwa{2}")
        assert exclusive(1, "1") == (1, "1")
        assert exclusive(2, "2") == (2, "2")


# ======================
# == NUMERIC WRAPPERS ==
# ======================
class TestNumericWrappers:
    def test_wrap_gt_and_ge_wrap_matching_values(self):
        assert wrap_gt(5, "textbf")(6, "6") == (6, r"\textbf{6}")
        assert wrap_ge(5, "emph")(5, "5") == (5, r"\emph{5}")

    def test_wrap_lt_and_le_wrap_matching_values(self):
        assert wrap_lt(5, "textit")(4, "4") == (4, r"\textit{4}")
        assert wrap_le(5, "underline")(5, "5") == (5, r"\underline{5}")

    def test_wrap_between_respects_exclusive_bounds(self):
        formatter = wrap_between(0, 1, "textbf", include_upper=False)
        assert formatter(0.5, "0.5") == (0.5, r"\textbf{0.5}")
        assert formatter(1.0, "1.0") == (1.0, "1.0")

    def test_wrap_between_can_exclude_lower_bound(self):
        formatter = wrap_between(0, 1, "textbf", include_lower=False)

        assert formatter(0.0, "0.0") == (0.0, "0.0")
        assert formatter(0.5, "0.5") == (0.5, r"\textbf{0.5}")
