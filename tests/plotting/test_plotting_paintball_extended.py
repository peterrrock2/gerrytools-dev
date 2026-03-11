"""Extended tests for PaintBall plot behavior.

Covers: data validation, add_voteshare_seatshare_data, line management,
crosshair/marker/hull option setters with validation, scale setters,
axis limit setters, coordinate transformation, horizontal hull vertices,
clear_options, hull_ax property, legend handles in point vs hull mode,
default line inclusion flags.
"""

import math

import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.paintball import PaintBall, PaintBallLine


# ============
# == HELPER ==
# ============
def _simple_paintball(**kwargs):
    """Create a PaintBall with minimal valid data."""
    defaults = dict(
        voteshare_data=[0.4, 0.5, 0.6],
        seats_data=[0.3, 0.5, 0.7],
    )
    defaults.update(kwargs)
    return PaintBall(**defaults)  # ty: ignore[invalid-argument-type]


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallLineDataclass:
    def test_valid_construction(self):
        line = PaintBallLine(slope=2.0, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.slope == 2.0
        assert line.linewidth == 1.0

    def test_nan_slope_raises_valueerror(self):
        with pytest.raises(ValueError, match="NaN"):
            PaintBallLine(slope=float("nan"), linecolor="black", linewidth=1.0, linestyle="-")

    def test_infinite_slope_is_allowed(self):
        line = PaintBallLine(slope=float("inf"), linecolor="black", linewidth=1.0, linestyle="-")
        assert math.isinf(line.slope)

    def test_negative_slope_is_allowed(self):
        line = PaintBallLine(slope=-1.5, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.slope == -1.5

    def test_zero_slope_is_allowed(self):
        line = PaintBallLine(slope=0.0, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.slope == 0.0

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            PaintBallLine(slope=1.0, linecolor="black", linewidth=float("inf"), linestyle="-")

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            PaintBallLine(slope=1.0, linecolor="black", linewidth=-1.0, linestyle="-")

    def test_zero_linewidth_is_allowed(self):
        line = PaintBallLine(slope=1.0, linecolor="black", linewidth=0.0, linestyle="-")
        assert line.linewidth == 0.0

    def test_slope_coerced_to_float(self):
        line = PaintBallLine(slope=2, linecolor="black", linewidth=1.0, linestyle="-")
        assert isinstance(line.slope, float)

    def test_linewidth_coerced_to_float(self):
        line = PaintBallLine(slope=1.0, linecolor="black", linewidth=3, linestyle="-")
        assert isinstance(line.linewidth, float)

    def test_zorder_coerced_to_int(self):
        line = PaintBallLine(
            slope=1.0,
            linecolor="black",
            linewidth=1.0,
            linestyle="-",
            zorder=2.7,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(line.zorder, int)
        assert line.zorder == 2

    def test_default_label_is_none(self):
        line = PaintBallLine(slope=1.0, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.label is None

    def test_frozen_prevents_mutation(self):
        line = PaintBallLine(slope=1.0, linecolor="black", linewidth=1.0, linestyle="-")
        with pytest.raises(AttributeError):
            line.slope = 5.0  # ty: ignore[invalid-assignment]


# ======================================
# == CONSTRUCTION AND DATA VALIDATION ==
# ======================================
class TestPaintBallConstruction:
    def test_minimal_valid_construction(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        assert len(pb._voteshare_data) == 1
        assert len(pb._seatshare_data) == 1

    def test_data_stored_as_floats(self):
        pb = PaintBall(voteshare_data=[1, 0], seats_data=[1, 0])
        assert all(isinstance(v, float) for v in pb._voteshare_data)
        assert all(isinstance(s, float) for s in pb._seatshare_data)

    def test_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same length"):
            PaintBall(voteshare_data=[0.5], seats_data=[0.5, 0.6])

    def test_empty_data_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one element"):
            PaintBall(voteshare_data=[], seats_data=[])

    def test_voteshare_above_one_raises_valueerror(self):
        with pytest.raises(ValueError, match="vote-share values must be in"):
            PaintBall(voteshare_data=[1.1], seats_data=[0.5])

    def test_voteshare_below_zero_raises_valueerror(self):
        with pytest.raises(ValueError, match="vote-share values must be in"):
            PaintBall(voteshare_data=[-0.1], seats_data=[0.5])

    def test_seatshare_above_one_without_max_seats_raises_valueerror(self):
        with pytest.raises(ValueError, match="seat-share values must be in"):
            PaintBall(voteshare_data=[0.5], seats_data=[1.5])

    def test_seatshare_below_zero_without_max_seats_raises_valueerror(self):
        with pytest.raises(ValueError, match="seat-share values must be in"):
            PaintBall(voteshare_data=[0.5], seats_data=[-0.1])

    def test_maximum_seats_normalizes_seat_counts(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[5], maximum_seats=10)
        assert pb._seatshare_data[0] == pytest.approx(0.5)

    def test_maximum_seats_zero_raises_valueerror(self):
        with pytest.raises(ValueError, match="positive integer"):
            PaintBall(voteshare_data=[0.5], seats_data=[5], maximum_seats=0)

    def test_maximum_seats_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="positive integer"):
            PaintBall(voteshare_data=[0.5], seats_data=[5], maximum_seats=-1)

    def test_seat_count_exceeds_max_seats_raises_valueerror(self):
        with pytest.raises(ValueError, match="seat-share values must be in"):
            PaintBall(voteshare_data=[0.5], seats_data=[20], maximum_seats=10)

    def test_boundary_voteshare_zero_is_valid(self):
        pb = PaintBall(voteshare_data=[0.0], seats_data=[0.5])
        assert pb._voteshare_data[0] == 0.0

    def test_boundary_voteshare_one_is_valid(self):
        pb = PaintBall(voteshare_data=[1.0], seats_data=[0.5])
        assert pb._voteshare_data[0] == 1.0

    def test_boundary_seatshare_zero_is_valid(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.0])
        assert pb._seatshare_data[0] == 0.0

    def test_boundary_seatshare_one_is_valid(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[1.0])
        assert pb._seatshare_data[0] == 1.0


# ==================================
# == DEFAULT LINE INCLUSION FLAGS ==
# ==================================
class TestPaintBallDefaultLines:
    def test_default_includes_efficiency_gap_and_proportionality(self):
        pb = _simple_paintball()
        assert "Efficiency Gap" in pb._named_lines
        assert "Proportionality" in pb._named_lines

    def test_efficiency_gap_line_has_slope_two(self):
        pb = _simple_paintball()
        assert pb._named_lines["Efficiency Gap"].slope == 2.0

    def test_proportionality_line_has_slope_one(self):
        pb = _simple_paintball()
        assert pb._named_lines["Proportionality"].slope == 1.0

    def test_disable_efficiency_gap_line(self):
        pb = _simple_paintball(include_efficiency_gap_line=False)
        assert "Efficiency Gap" not in pb._named_lines

    def test_disable_proportionality_line(self):
        pb = _simple_paintball(include_proportionality_line=False)
        assert "Proportionality" not in pb._named_lines

    def test_disable_both_default_lines(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        assert len(pb._named_lines) == 0

    def test_default_legend_is_false(self):
        pb = _simple_paintball()
        assert pb.include_legend is False


# ==================================
# == ADD VOTESHARE SEATSHARE DATA ==
# ==================================
class TestAddVoteshareSeatshareData:
    def test_adds_to_existing_data(self):
        pb = _simple_paintball()
        original_length = len(pb._voteshare_data)
        pb.add_voteshare_seatshare_data([0.45], [0.55])
        assert len(pb._voteshare_data) == original_length + 1

    def test_new_data_appended_not_replaced(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        pb.add_voteshare_seatshare_data([0.6], [0.7])
        assert pb._voteshare_data == [0.5, 0.6]
        assert pb._seatshare_data == [0.5, 0.7]

    def test_add_with_maximum_seats_normalization(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        pb.add_voteshare_seatshare_data([0.6], [9], maximum_seats=18)
        assert pb._seatshare_data[-1] == pytest.approx(0.5)

    def test_add_invalid_data_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="vote-share values must be in"):
            pb.add_voteshare_seatshare_data([1.5], [0.5])

    def test_add_mismatched_length_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="same length"):
            pb.add_voteshare_seatshare_data([0.5, 0.6], [0.5])

    def test_add_empty_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="at least one element"):
            pb.add_voteshare_seatshare_data([], [])


# =====================
# == LINE MANAGEMENT ==
# =====================
class TestPaintBallLineManagement:
    def test_add_named_line(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[1.5], name="Custom Line")
        assert "Custom Line" in pb._named_lines
        assert pb._named_lines["Custom Line"].slope == 1.5

    def test_add_anonymous_line(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[3.0], linecolor="red")
        assert 3.0 in pb._lines
        assert len(pb._lines[3.0]) == 1

    def test_add_multiple_slopes_at_once(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[1.0, 2.0, 3.0], linecolor="blue")
        assert 1.0 in pb._lines
        assert 2.0 in pb._lines
        assert 3.0 in pb._lines

    def test_add_multiple_slopes_with_name_stores_last_only(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[1.0, 2.0], name="Shared Name")
        # Named lines use dict, so last slope overwrites
        assert pb._named_lines["Shared Name"].slope == 2.0

    def test_add_line_with_custom_properties(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(
            slopes=[1.0],
            linecolor="red",
            linewidth=2.5,
            linestyle="--",
            linealpha=0.7,
            zorder=5,
            name="Styled",
        )
        line = pb._named_lines["Styled"]
        assert line.linewidth == 2.5
        assert line.linestyle == "--"
        assert line.zorder == 5

    def test_clear_lines_removes_all(self):
        pb = _simple_paintball()
        pb.add_lines_with_slope(slopes=[3.0])
        pb.clear_lines()
        assert len(pb._named_lines) == 0
        assert len(pb._lines) == 0

    def test_clear_lines_leaves_data_intact(self):
        pb = _simple_paintball()
        original_data_len = len(pb._voteshare_data)
        pb.clear_lines()
        assert len(pb._voteshare_data) == original_data_len


# =========================
# == SET XLIM / SET YLIM ==
# =========================
class TestPaintBallAxisLimits:
    def test_set_xlim_valid(self):
        pb = _simple_paintball()
        pb.set_xlim(0.2, 0.8)
        assert pb._x_limits == (0.2, 0.8)

    def test_set_ylim_valid(self):
        pb = _simple_paintball()
        pb.set_ylim(0.1, 0.9)
        assert pb._y_limits == (0.1, 0.9)

    def test_set_xlim_left_equals_right_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_xlim(0.5, 0.5)

    def test_set_xlim_left_greater_than_right_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_xlim(0.8, 0.2)

    def test_set_ylim_bottom_equals_top_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_ylim(0.5, 0.5)

    def test_set_ylim_bottom_greater_than_top_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_ylim(0.9, 0.1)


# =========================================
# == SET XSCALE / SET YSCALE / SET SCALE ==
# =========================================
class TestPaintBallScaleSetters:
    def test_set_xscale_valid(self):
        pb = _simple_paintball()
        pb.set_xscale(5.0)
        assert pb.xscale == 5.0

    def test_set_yscale_valid(self):
        pb = _simple_paintball()
        pb.set_yscale(5.0)
        assert pb.yscale == 5.0

    def test_set_xscale_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_xscale(0.0)

    def test_set_xscale_negative_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_xscale(-1.0)

    def test_set_xscale_infinite_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_xscale(float("inf"))

    def test_set_xscale_nan_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_xscale(float("nan"))

    def test_set_yscale_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_yscale(0.0)

    def test_set_yscale_negative_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_yscale(-1.0)

    def test_set_yscale_infinite_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_yscale(float("inf"))

    def test_set_scale_sets_both(self):
        pb = _simple_paintball()
        pb.set_scale(xscale=7.0, yscale=3.0)
        assert pb.xscale == 7.0
        assert pb.yscale == 3.0

    def test_set_scale_xscale_only(self):
        pb = _simple_paintball()
        original_yscale = pb.yscale
        pb.set_scale(xscale=4.0)
        assert pb.xscale == 4.0
        assert pb.yscale == original_yscale

    def test_set_scale_yscale_only(self):
        pb = _simple_paintball()
        original_xscale = pb.xscale
        pb.set_scale(yscale=4.0)
        assert pb.yscale == 4.0
        assert pb.xscale == original_xscale

    def test_set_scale_neither_is_noop(self):
        pb = _simple_paintball()
        original_x = pb.xscale
        original_y = pb.yscale
        pb.set_scale()
        assert pb.xscale == original_x
        assert pb.yscale == original_y

    def test_set_xscale_coerces_int_to_float(self):
        pb = _simple_paintball()
        pb.set_xscale(5)
        assert isinstance(pb.xscale, float)
        assert pb.xscale == 5.0


# ===========================
# == SET CROSSHAIR OPTIONS ==
# ===========================
class TestPaintBallCrosshairOptions:
    def test_set_color_only(self):
        pb = _simple_paintball()
        original_width = pb.crosshair_width
        pb.set_crosshair_options(color="red")
        assert pb.crosshair_color == "red"
        assert pb.crosshair_width == original_width

    def test_set_width_only(self):
        pb = _simple_paintball()
        original_color = pb.crosshair_color
        pb.set_crosshair_options(width=3.0)
        assert pb.crosshair_width == 3.0
        assert pb.crosshair_color == original_color

    def test_set_alpha_only(self):
        pb = _simple_paintball()
        pb.set_crosshair_options(alpha=0.5)
        assert pb.crosshair_alpha == 0.5

    def test_set_all_crosshair_options(self):
        pb = _simple_paintball()
        pb.set_crosshair_options(color="blue", width=2.0, alpha=0.3)
        assert pb.crosshair_color == "blue"
        assert pb.crosshair_width == 2.0
        assert pb.crosshair_alpha == 0.3

    def test_width_infinite_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_crosshair_options(width=float("inf"))

    def test_width_negative_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_crosshair_options(width=-1.0)

    def test_width_zero_is_valid(self):
        pb = _simple_paintball()
        pb.set_crosshair_options(width=0.0)
        assert pb.crosshair_width == 0.0

    def test_alpha_above_one_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_crosshair_options(alpha=1.5)

    def test_alpha_below_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_crosshair_options(alpha=-0.1)

    def test_alpha_boundary_zero_is_valid(self):
        pb = _simple_paintball()
        pb.set_crosshair_options(alpha=0.0)
        assert pb.crosshair_alpha == 0.0

    def test_alpha_boundary_one_is_valid(self):
        pb = _simple_paintball()
        pb.set_crosshair_options(alpha=1.0)
        assert pb.crosshair_alpha == 1.0


# ========================
# == SET MARKER OPTIONS ==
# ========================
class TestPaintBallMarkerOptions:
    def test_set_size(self):
        pb = _simple_paintball()
        pb.set_marker_options(size=20.0)
        assert pb.markersize == 20.0

    def test_set_color(self):
        pb = _simple_paintball()
        pb.set_marker_options(color="red")
        assert pb.markercolor == "red"

    def test_set_marker_string(self):
        pb = _simple_paintball()
        pb.set_marker_options(marker="s")
        assert pb.marker == "s"

    def test_set_alpha(self):
        pb = _simple_paintball()
        pb.set_marker_options(alpha=0.5)
        assert pb.markeralpha == 0.5

    def test_set_edgecolor(self):
        pb = _simple_paintball()
        pb.set_marker_options(edgecolor="blue")
        assert pb.markeredgecolor == "blue"

    def test_set_edgewidth(self):
        pb = _simple_paintball()
        pb.set_marker_options(edgewidth=2.0)
        assert pb.markeredgewidth == 2.0

    def test_set_edgealpha(self):
        pb = _simple_paintball()
        pb.set_marker_options(edgealpha=0.3)
        assert pb.markeredgealpha == 0.3

    def test_size_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_marker_options(size=0.0)

    def test_size_negative_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_marker_options(size=-1.0)

    def test_size_infinite_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_marker_options(size=float("inf"))

    def test_alpha_above_one_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_marker_options(alpha=1.1)

    def test_alpha_below_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_marker_options(alpha=-0.1)

    def test_edgewidth_negative_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_marker_options(edgewidth=-0.5)

    def test_edgewidth_infinite_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_marker_options(edgewidth=float("inf"))

    def test_edgewidth_zero_is_valid(self):
        pb = _simple_paintball()
        pb.set_marker_options(edgewidth=0.0)
        assert pb.markeredgewidth == 0.0

    def test_edgealpha_above_one_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_marker_options(edgealpha=1.5)

    def test_edgealpha_below_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_marker_options(edgealpha=-0.1)

    def test_partial_update_preserves_other_fields(self):
        pb = _simple_paintball()
        original_marker = pb.marker
        original_alpha = pb.markeralpha
        pb.set_marker_options(size=99.0)
        assert pb.marker == original_marker
        assert pb.markeralpha == original_alpha


# ======================
# == SET HULL OPTIONS ==
# ======================
class TestPaintBallHullOptions:
    def test_set_hull_color(self):
        pb = _simple_paintball()
        pb.set_hull_options(color="green")
        assert pb.hullcolor == "green"

    def test_set_hull_alpha(self):
        pb = _simple_paintball()
        pb.set_hull_options(alpha=0.4)
        assert pb.hullalpha == 0.4

    def test_set_hull_edgecolor(self):
        pb = _simple_paintball()
        pb.set_hull_options(edgecolor="black")
        assert pb.hulledgecolor == "black"

    def test_set_hull_edgewidth(self):
        pb = _simple_paintball()
        pb.set_hull_options(edgewidth=3.0)
        assert pb.hulledgewidth == 3.0

    def test_set_hull_edgealpha(self):
        pb = _simple_paintball()
        pb.set_hull_options(edgealpha=0.7)
        assert pb.hulledgealpha == 0.7

    def test_hull_alpha_above_one_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_hull_options(alpha=1.5)

    def test_hull_alpha_below_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_hull_options(alpha=-0.1)

    def test_hull_edgewidth_negative_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_hull_options(edgewidth=-1.0)

    def test_hull_edgewidth_infinite_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_hull_options(edgewidth=float("inf"))

    def test_hull_edgewidth_zero_is_valid(self):
        pb = _simple_paintball()
        pb.set_hull_options(edgewidth=0.0)
        assert pb.hulledgewidth == 0.0

    def test_hull_edgealpha_above_one_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_hull_options(edgealpha=1.5)

    def test_hull_edgealpha_below_zero_raises_valueerror(self):
        pb = _simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_hull_options(edgealpha=-0.1)

    def test_default_hull_color_is_none(self):
        pb = _simple_paintball()
        assert pb.hullcolor is None

    def test_default_hull_edgecolor_is_none(self):
        pb = _simple_paintball()
        assert pb.hulledgecolor is None

    def test_partial_hull_update_preserves_other_fields(self):
        pb = _simple_paintball()
        pb.set_hull_options(color="red", edgewidth=4.0)
        original_edgewidth = pb.hulledgewidth
        pb.set_hull_options(alpha=0.5)
        assert pb.hullcolor == "red"
        assert pb.hulledgewidth == original_edgewidth


# ===================
# == CLEAR OPTIONS ==
# ===================
class TestPaintBallClearOptions:
    def test_clear_options_resets_markersize(self):
        pb = _simple_paintball()
        pb.set_marker_options(size=99.0)
        pb.clear_options()
        assert pb.markersize == 16.0

    def test_clear_options_resets_marker_to_circle(self):
        pb = _simple_paintball()
        pb.set_marker_options(marker="s")
        pb.clear_options()
        assert pb.marker == "o"

    def test_clear_options_resets_crosshair_width(self):
        pb = _simple_paintball()
        pb.set_crosshair_options(width=0.5)
        pb.clear_options()
        assert pb.crosshair_width == 5.0

    def test_clear_options_resets_hull_color_to_none(self):
        pb = _simple_paintball()
        pb.set_hull_options(color="red")
        pb.clear_options()
        assert pb.hullcolor is None

    def test_clear_options_resets_scale(self):
        pb = _simple_paintball()
        pb.set_scale(xscale=5.0, yscale=3.0)
        pb.clear_options()
        assert pb.xscale == 10.0
        assert pb.yscale == 10.0

    def test_clear_options_does_not_remove_data(self):
        pb = _simple_paintball()
        original_len = len(pb._voteshare_data)
        pb.clear_options()
        assert len(pb._voteshare_data) == original_len


# ==========================================
# == PAINTBALL COORDINATES TRANSFORMATION ==
# ==========================================
class TestPaintBallCoordinates:
    def test_coordinates_are_one_minus_data(self):
        pb = PaintBall(voteshare_data=[0.3, 0.7], seats_data=[0.2, 0.8])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.7), pytest.approx(0.3)]
        assert ys == [pytest.approx(0.8), pytest.approx(0.2)]

    def test_center_point_maps_to_center(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.5)]
        assert ys == [pytest.approx(0.5)]

    def test_boundary_zero_maps_to_one(self):
        pb = PaintBall(voteshare_data=[0.0], seats_data=[0.0])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(1.0)]
        assert ys == [pytest.approx(1.0)]

    def test_boundary_one_maps_to_zero(self):
        pb = PaintBall(voteshare_data=[1.0], seats_data=[1.0])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.0)]
        assert ys == [pytest.approx(0.0)]


# ==============================
# == HORIZONTAL HULL VERTICES ==
# ==============================
class TestHorizontalHullVertices:
    def test_single_point_hull(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        vertices = pb._horizontal_hull_vertices()
        # Single point: one y level, left_side + right_side = 2 vertices
        assert len(vertices) == 2
        # Both vertices should be the same point
        assert vertices[0] == vertices[1]

    def test_two_points_same_y(self):
        # Both have same seat share (so same transformed y)
        pb = PaintBall(voteshare_data=[0.3, 0.7], seats_data=[0.5, 0.5])
        vertices = pb._horizontal_hull_vertices()
        assert len(vertices) == 2
        # min-x on left side, max-x on right side
        xs = [v[0] for v in vertices]
        assert min(xs) <= max(xs)

    def test_three_points_different_y_produces_hull(self):
        pb = PaintBall(voteshare_data=[0.3, 0.5, 0.7], seats_data=[0.2, 0.5, 0.8])
        vertices = pb._horizontal_hull_vertices()
        # 3 unique y-levels -> left_side(3) + right_side(3) = 6 vertices
        assert len(vertices) == 6

    def test_hull_vertices_are_tuples_of_floats(self):
        pb = _simple_paintball()
        vertices = pb._horizontal_hull_vertices()
        for v in vertices:
            assert isinstance(v, tuple)
            assert len(v) == 2


# =========================
# == BUILD AND RENDERING ==
# =========================
class TestPaintBallBuild:
    def test_build_point_view(self):
        pb = _simple_paintball()
        ax = pb.ax
        assert ax is not None

    def test_build_hull_view(self):
        pb = _simple_paintball()
        ax = pb.hull_ax
        assert ax is not None

    def test_hull_ax_restores_draw_hull_flag(self):
        pb = _simple_paintball()
        assert pb._draw_hull is False
        _ = pb.hull_ax
        # After hull_ax returns, _draw_hull should be restored
        assert pb._draw_hull is False

    def test_build_with_no_default_lines(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False,
            include_proportionality_line=False,
        )
        ax = pb.ax
        assert ax is not None

    def test_build_with_custom_scale(self):
        pb = _simple_paintball()
        pb.set_scale(xscale=5.0, yscale=15.0)
        ax = pb.ax
        assert ax is not None

    def test_build_with_single_point(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        ax = pb.ax
        assert ax is not None

    def test_hull_view_with_single_point(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        ax = pb.hull_ax
        assert ax is not None

    def test_hull_view_with_two_colinear_points(self):
        # Two points with same y -> hull degenerates to a line (< 3 vertices)
        pb = PaintBall(voteshare_data=[0.3, 0.7], seats_data=[0.5, 0.5])
        ax = pb.hull_ax
        assert ax is not None


# ====================
# == LEGEND HANDLES ==
# ====================
class TestPaintBallLegendHandles:
    def test_point_view_legend_has_plan_outcomes(self):
        pb = _simple_paintball()
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Plan Outcomes" in labels

    def test_point_view_legend_has_named_lines(self):
        pb = _simple_paintball()
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Efficiency Gap" in labels
        assert "Proportionality" in labels

    def test_hull_view_legend_has_horizontal_hull(self):
        pb = _simple_paintball()
        pb._draw_hull = True
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Horizontal Hull" in labels

    def test_hull_view_legend_excludes_plan_outcomes(self):
        pb = _simple_paintball()
        pb._draw_hull = True
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Plan Outcomes" not in labels

    def test_no_named_lines_means_fewer_legend_handles(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False,
            include_proportionality_line=False,
        )
        handles = pb._legend_handles
        # Only the Plan Outcomes handle
        assert len(handles) == 1
        assert handles[0].get_label() == "Plan Outcomes"

    def test_anonymous_lines_excluded_from_legend(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False,
            include_proportionality_line=False,
        )
        pb.add_lines_with_slope(slopes=[3.0])  # no name
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        # Only Plan Outcomes, no label for the anonymous line
        assert len(labels) == 1

    def test_custom_named_line_appears_in_legend(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False,
            include_proportionality_line=False,
        )
        pb.add_lines_with_slope(slopes=[1.5], name="My Guide")
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "My Guide" in labels
