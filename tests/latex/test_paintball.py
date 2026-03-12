"""Tests for non-visual LaTeX paintball generation."""

import pytest

from gerrytools.latex.paintball import PaintBall, PaintBallLine, PaintBallOptions


# ============================
# == DATA CLASSES & OPTIONS ==
# ============================
class TestPaintBallDataclasses:
    def test_paintball_line_accepts_valid_tikz_linestyle(self):
        line = PaintBallLine(slope=1.0, linecolor="black", linewidth=1.5, linestyle="dashed")
        assert line.linestyle == "dashed"

    def test_paintball_line_rejects_invalid_tikz_linestyle(self):
        with pytest.raises(ValueError, match="Invalid linestyle"):
            PaintBallLine(
                slope=1.0,
                linecolor="black",
                linewidth=1.0,
                linestyle="zigzag",  # type: ignore[arg-type]
            )

    def test_paintball_options_validate_numeric_ranges(self):
        options = PaintBallOptions()

        with pytest.raises(ValueError, match="markersize must be positive"):
            options.markersize = 0

        with pytest.raises(ValueError, match="markeralpha must be in \\[0\\.0, 1\\.0\\]"):
            options.markeralpha = 2

        with pytest.raises(ValueError, match="xlim\\[0\\] must be less than xlim\\[1\\]"):
            options.xlim = (1.0, 0.0)

        with pytest.raises(AttributeError, match="Unknown PaintBallOptions attribute"):
            options.unknown = 1

    def test_paintball_options_validate_additional_ranges_and_none_assignments(self):
        options = PaintBallOptions()

        options.hullalpha = None
        options.hulledgewidth = None
        options.hulledgealpha = None

        assert options.hullalpha is None
        assert options.hulledgewidth is None
        assert options.hulledgealpha is None

        with pytest.raises(ValueError, match="markeredgewidth must be non-negative"):
            options.markeredgewidth = -1

        with pytest.raises(ValueError, match="markeredgealpha must be in \\[0\\.0, 1\\.0\\]"):
            options.markeredgealpha = 2

        with pytest.raises(ValueError, match="hullalpha must be in \\[0\\.0, 1\\.0\\]"):
            options.hullalpha = 2

        with pytest.raises(ValueError, match="hulledgewidth must be non-negative"):
            options.hulledgewidth = -1

        with pytest.raises(ValueError, match="hulledgealpha must be in \\[0\\.0, 1\\.0\\]"):
            options.hulledgealpha = 2

        with pytest.raises(ValueError, match="crosshair_width must be non-negative"):
            options.crosshair_width = -1

        with pytest.raises(ValueError, match="xscale must be positive"):
            options.xscale = 0

        with pytest.raises(ValueError, match="yscale must be positive"):
            options.yscale = 0

        with pytest.raises(ValueError, match="ylim\\[0\\] must be less than ylim\\[1\\]"):
            options.ylim = (1.0, 0.0)


# =========================
# == CONSTRUCTION & DATA ==
# =========================
class TestPaintBallConstruction:
    def test_initialization_normalizes_seat_counts(self):
        plot = PaintBall(
            voteshare_data=[0.4, 0.6],
            seats_data=[2, 3],
            maximum_seats=4,
        )

        assert plot._voteshare_data == [0.4, 0.6]
        assert plot._seatshare_data == [0.5, 0.75]

    def test_initialization_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            PaintBall(voteshare_data=[0.5], seats_data=[0.5, 0.6])

    def test_initialization_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="at least one element"):
            PaintBall(voteshare_data=[], seats_data=[])

    def test_initialization_rejects_invalid_seatshares_without_maximum_seats(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            PaintBall(voteshare_data=[0.5], seats_data=[1.5])

    def test_initialization_rejects_invalid_scaled_seatshares(self):
        with pytest.raises(ValueError, match="After scaling by maximum_seats"):
            PaintBall(voteshare_data=[0.5], seats_data=[8], maximum_seats=4)

    def test_add_voteshare_seatshare_data_extends_existing_data(self):
        plot = PaintBall(voteshare_data=[0.4], seats_data=[0.5])
        plot.add_voteshare_seatshare_data([0.6], [0.75])

        assert plot._voteshare_data == [0.4, 0.6]
        assert plot._seatshare_data == [0.5, 0.75]


# =====================
# == LINE MANAGEMENT ==
# =====================
class TestPaintBallLines:
    def test_add_lines_with_slope_tracks_named_and_unnamed_lines(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5], include_efficiency_gap_line=False)
        plot.add_lines_with_slope([0.5, 2.0], linecolor="denim", linestyle="dotted")
        plot.add_lines_with_slope([1.0], linecolor="amber", name="custom")

        assert 0.5 in plot._lines
        assert 2.0 in plot._lines
        assert "custom" in plot._nammed_lines

    def test_clear_lines_clears_named_and_unnamed_lines(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        plot.add_lines_with_slope([0.5], name="custom")
        plot.clear_lines()

        assert plot._lines == {}
        assert plot._nammed_lines == {}


# =======================
# == OPTION MANAGEMENT ==
# =======================
class TestPaintBallOptionsManagement:
    def test_clear_options_resets_overrides(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        plot.set_marker_options(size=12, color="black", alpha=0.4)
        plot.set_crosshair_options(color="red", width=2.0)
        plot.clear_options()

        assert plot.options.markersize == 8.0
        assert plot.options.markercolor == "cadmiumgreen"
        assert plot.options.crosshair_color == "gray!50"
        assert plot.options.crosshair_width == 5.0

    def test_set_limits_with_rescale_updates_scales(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        plot.set_xlim(0.25, 0.75, rescale=True)
        plot.set_ylim(0.2, 0.8, rescale=True)

        assert plot.options.xlim == (0.25, 0.75)
        assert plot.options.ylim == (0.2, 0.8)
        assert plot.options.xscale == pytest.approx(20.0)
        assert plot.options.yscale == pytest.approx(16.6667, rel=1e-4)

    def test_set_scale_and_option_setters_update_all_edge_and_hull_values(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5])

        plot.set_scale(xscale=12.0, yscale=8.0)
        plot.set_marker_options(
            edgecolor="amber",
            edgewidth=2.0,
            edgealpha=0.6,
        )
        plot.set_hull_options(
            color="denim",
            alpha=0.3,
            edgecolor="black",
            edgewidth=1.5,
            edgealpha=0.9,
        )

        assert plot.options.xscale == 12.0
        assert plot.options.yscale == 8.0
        assert plot.options.markeredgecolor == "amber"
        assert plot.options.markeredgewidth == 2.0
        assert plot.options.markeredgealpha == 0.6
        assert plot.options.hullcolor == "denim"
        assert plot.options.hullalpha == 0.3
        assert plot.options.hulledgecolor == "black"
        assert plot.options.hulledgewidth == 1.5
        assert plot.options.hulledgealpha == 0.9


# =======================
# == STRING GENERATION ==
# =======================
class TestPaintBallStringGeneration:
    def test_to_latex_color_handles_none_expression_and_auto_colors(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5])

        assert plot._to_latex_color("none", prefix="pb") == "none"
        assert plot._to_latex_color("denim!20!amber", prefix="pb") == "denim!20!amber"

        auto_name = plot._to_latex_color("#123456", prefix="pb")
        assert auto_name.startswith("pb")
        assert plot.document.color_dict[auto_name] == ("HTML", "123456")

    def test_to_latex_color_handles_none_via_base_alpha_tuple(self):
        plot = PaintBall(voteshare_data=[0.5], seats_data=[0.5])

        assert plot._to_latex_color(None, prefix="pb") == "none"  # type: ignore[arg-type]

    def test_document_properties_update_body_string(self):
        plot = PaintBall(voteshare_data=[0.4, 0.6], seats_data=[0.5, 0.75])

        point_document = plot.document
        hull_document = plot.hull_document

        assert r"\begin{tikzpicture}" in point_document.body_string
        assert r"\begin{tikzpicture}" in hull_document.body_string
        assert r"\draw [fill=" in hull_document.body_string

    def test_print_emits_point_or_hull_body(self, capsys):
        plot = PaintBall(voteshare_data=[0.4], seats_data=[0.5])

        plot.print()
        out = capsys.readouterr().out
        assert r"\begin{tikzpicture}" in out
        assert r"\foreach \votes/\seats in {" in out

        plot.print(hull=True)
        out = capsys.readouterr().out
        assert r"\draw [fill=" in out

    def test_generate_latex_includes_crosshairs_lines_and_points(self):
        plot = PaintBall(voteshare_data=[0.4], seats_data=[0.5])
        plot.add_lines_with_slope([0.5], linecolor="denim", linestyle="dotted")

        latex = str(plot)

        assert r"\begin{tikzpicture}" in latex
        assert r"\clip [draw]" in latex
        assert "line width=5.0pt" in latex
        assert "dotted" in latex
        assert r"\foreach \votes/\seats in {" in latex

    def test_hull_string_uses_hull_specific_overrides(self):
        plot = PaintBall(voteshare_data=[0.4, 0.6], seats_data=[0.5, 0.75])
        plot.set_hull_options(
            color="denim",
            alpha=0.3,
            edgecolor="amber",
            edgewidth=1.5,
            edgealpha=0.9,
        )

        hull = plot._paintball_hull_str()
        assert "fill=denim" in hull
        assert "fill opacity=0.3" in hull
        assert "line width=1.5" in hull
        assert "color=amber" in hull
        assert "draw opacity=0.9" in hull

    def test_hull_string_tracks_min_and_max_x_for_duplicate_y_values(self):
        plot = PaintBall(
            voteshare_data=[0.5, 0.8, 0.2],
            seats_data=[0.5, 0.5, 0.5],
        )

        hull = plot._paintball_hull_str()

        assert "fill=cadmiumgreen" in hull
        assert "line width=2.0" in hull
        assert "(0.2,0.5)--" in hull
        assert "(0.8,0.5);" in hull
