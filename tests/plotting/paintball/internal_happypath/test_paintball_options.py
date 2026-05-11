import matplotlib

matplotlib.use("Agg")

import pytest

from tests.plotting.paintball._helpers import simple_paintball


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallAxisLimits:
    def test_set_xlim_valid(self):
        pb = simple_paintball()
        pb.set_xlim(0.2, 0.8)
        assert pb._x_limits == (0.2, 0.8)

    def test_set_ylim_valid(self):
        pb = simple_paintball()
        pb.set_ylim(0.1, 0.9)
        assert pb._y_limits == (0.1, 0.9)

    def test_set_xlim_left_equals_right_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_xlim(0.5, 0.5)

    def test_set_xlim_left_greater_than_right_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_xlim(0.8, 0.2)

    def test_set_ylim_bottom_equals_top_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_ylim(0.5, 0.5)

    def test_set_ylim_bottom_greater_than_top_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="less than"):
            pb.set_ylim(0.9, 0.1)


# =========================================
# == SET XSCALE / SET YSCALE / SET SCALE ==
# =========================================


class TestPaintBallScaleSetters:
    def test_set_xscale_valid(self):
        pb = simple_paintball()
        pb.set_xscale(5.0)
        assert pb.xscale == 5.0

    def test_set_yscale_valid(self):
        pb = simple_paintball()
        pb.set_yscale(5.0)
        assert pb.yscale == 5.0

    def test_set_xscale_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_xscale(0.0)

    def test_set_xscale_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_xscale(-1.0)

    def test_set_xscale_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_xscale(float("inf"))

    def test_set_xscale_nan_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_xscale(float("nan"))

    def test_set_yscale_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_yscale(0.0)

    def test_set_yscale_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_yscale(-1.0)

    def test_set_yscale_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_yscale(float("inf"))

    def test_set_scale_sets_both(self):
        pb = simple_paintball()
        pb.set_scale(xscale=7.0, yscale=3.0)
        assert pb.xscale == 7.0
        assert pb.yscale == 3.0

    def test_set_scale_xscale_only(self):
        pb = simple_paintball()
        original_yscale = pb.yscale
        pb.set_scale(xscale=4.0)
        assert pb.xscale == 4.0
        assert pb.yscale == original_yscale

    def test_set_scale_yscale_only(self):
        pb = simple_paintball()
        original_xscale = pb.xscale
        pb.set_scale(yscale=4.0)
        assert pb.yscale == 4.0
        assert pb.xscale == original_xscale

    def test_set_scale_neither_is_noop(self):
        pb = simple_paintball()
        original_x = pb.xscale
        original_y = pb.yscale
        pb.set_scale()
        assert pb.xscale == original_x
        assert pb.yscale == original_y

    def test_set_xscale_coerces_int_to_float(self):
        pb = simple_paintball()
        pb.set_xscale(5)
        assert isinstance(pb.xscale, float)
        assert pb.xscale == 5.0


# ===========================
# == SET CROSSHAIR OPTIONS ==
# ===========================


class TestPaintBallCrosshairOptions:
    def test_set_color_only(self):
        pb = simple_paintball()
        original_width = pb.crosshair_width
        pb.set_crosshair_options(color="red")
        assert pb.crosshair_color == "red"
        assert pb.crosshair_width == original_width

    def test_set_width_only(self):
        pb = simple_paintball()
        original_color = pb.crosshair_color
        pb.set_crosshair_options(width=3.0)
        assert pb.crosshair_width == 3.0
        assert pb.crosshair_color == original_color

    def test_set_alpha_only(self):
        pb = simple_paintball()
        pb.set_crosshair_options(alpha=0.5)
        assert pb.crosshair_alpha == 0.5

    def test_set_all_crosshair_options(self):
        pb = simple_paintball()
        pb.set_crosshair_options(color="blue", width=2.0, alpha=0.3)
        assert pb.crosshair_color == "blue"
        assert pb.crosshair_width == 2.0
        assert pb.crosshair_alpha == 0.3

    def test_width_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_crosshair_options(width=float("inf"))

    def test_width_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_crosshair_options(width=-1.0)

    def test_width_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_crosshair_options(width=0.0)
        assert pb.crosshair_width == 0.0

    def test_alpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_crosshair_options(alpha=1.5)

    def test_alpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_crosshair_options(alpha=-0.1)

    def test_alpha_boundary_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_crosshair_options(alpha=0.0)
        assert pb.crosshair_alpha == 0.0

    def test_alpha_boundary_one_is_valid(self):
        pb = simple_paintball()
        pb.set_crosshair_options(alpha=1.0)
        assert pb.crosshair_alpha == 1.0


# ========================
# == SET MARKER OPTIONS ==
# ========================


class TestPaintBallMarkerOptions:
    def test_set_size(self):
        pb = simple_paintball()
        pb.set_marker_options(size=20.0)
        assert pb.markersize == 20.0

    def test_set_color(self):
        pb = simple_paintball()
        pb.set_marker_options(color="red")
        assert pb.markerfacecolor == "red"

    def test_set_marker_string(self):
        pb = simple_paintball()
        pb.set_marker_options(marker="s")
        assert pb.marker == "s"

    def test_set_alpha(self):
        pb = simple_paintball()
        pb.set_marker_options(alpha=0.5)
        assert pb.markerfacealpha == 0.5

    def test_set_edgecolor(self):
        pb = simple_paintball()
        pb.set_marker_options(edgecolor="blue")
        assert pb.markeredgecolor == "blue"

    def test_set_edgewidth(self):
        pb = simple_paintball()
        pb.set_marker_options(edgewidth=2.0)
        assert pb.markeredgewidth == 2.0

    def test_set_edgealpha(self):
        pb = simple_paintball()
        pb.set_marker_options(edgealpha=0.3)
        assert pb.markeredgealpha == 0.3

    def test_size_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_marker_options(size=0.0)

    def test_size_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_marker_options(size=-1.0)

    def test_size_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_marker_options(size=float("inf"))

    def test_alpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_marker_options(alpha=1.1)

    def test_alpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_marker_options(alpha=-0.1)

    def test_edgewidth_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_marker_options(edgewidth=-0.5)

    def test_edgewidth_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_marker_options(edgewidth=float("inf"))

    def test_edgewidth_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_marker_options(edgewidth=0.0)
        assert pb.markeredgewidth == 0.0

    def test_edgealpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_marker_options(edgealpha=1.5)

    def test_edgealpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_marker_options(edgealpha=-0.1)

    def test_partial_update_preserves_other_fields(self):
        pb = simple_paintball()
        original_marker = pb.marker
        original_alpha = pb.markerfacealpha
        pb.set_marker_options(size=99.0)
        assert pb.marker == original_marker
        assert pb.markerfacealpha == original_alpha


# ======================
# == SET HULL OPTIONS ==
# ======================


class TestPaintBallHullOptions:
    def test_set_hull_color(self):
        pb = simple_paintball()
        pb.set_hull_options(color="green")
        assert pb.hullcolor == "green"

    def test_set_hull_alpha(self):
        pb = simple_paintball()
        pb.set_hull_options(alpha=0.4)
        assert pb.hullalpha == 0.4

    def test_set_hull_edgecolor(self):
        pb = simple_paintball()
        pb.set_hull_options(edgecolor="black")
        assert pb.hulledgecolor == "black"

    def test_set_hull_edgewidth(self):
        pb = simple_paintball()
        pb.set_hull_options(edgewidth=3.0)
        assert pb.hulledgewidth == 3.0

    def test_set_hull_edgealpha(self):
        pb = simple_paintball()
        pb.set_hull_options(edgealpha=0.7)
        assert pb.hulledgealpha == 0.7

    def test_hull_alpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_hull_options(alpha=1.5)

    def test_hull_alpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_hull_options(alpha=-0.1)

    def test_hull_edgewidth_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_hull_options(edgewidth=-1.0)

    def test_hull_edgewidth_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_hull_options(edgewidth=float("inf"))

    def test_hull_edgewidth_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_hull_options(edgewidth=0.0)
        assert pb.hulledgewidth == 0.0

    def test_hull_edgealpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_hull_options(edgealpha=1.5)

    def test_hull_edgealpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_hull_options(edgealpha=-0.1)

    def test_default_hull_color_is_none(self):
        pb = simple_paintball()
        assert pb.hullcolor is None

    def test_default_hull_edgecolor_is_none(self):
        pb = simple_paintball()
        assert pb.hulledgecolor is None

    def test_partial_hull_update_preserves_other_fields(self):
        pb = simple_paintball()
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
        pb = simple_paintball()
        pb.set_marker_options(size=99.0)
        pb.clear_options()
        assert pb.markersize == 16.0

    def test_clear_options_resets_marker_to_circle(self):
        pb = simple_paintball()
        pb.set_marker_options(marker="s")
        pb.clear_options()
        assert pb.marker == "o"

    def test_clear_options_resets_crosshair_width(self):
        pb = simple_paintball()
        pb.set_crosshair_options(width=0.5)
        pb.clear_options()
        assert pb.crosshair_width == 5.0

    def test_clear_options_resets_hull_color_to_none(self):
        pb = simple_paintball()
        pb.set_hull_options(color="red")
        pb.clear_options()
        assert pb.hullcolor is None

    def test_clear_options_resets_scale(self):
        pb = simple_paintball()
        pb.set_scale(xscale=5.0, yscale=3.0)
        pb.clear_options()
        assert pb.xscale == 10.0
        assert pb.yscale == 10.0

    def test_clear_options_does_not_remove_data(self):
        pb = simple_paintball()
        original_len = len(pb._voteshare_data)
        pb.clear_options()
        assert len(pb._voteshare_data) == original_len


# ==========================================
# == PAINTBALL COORDINATES TRANSFORMATION ==
# ==========================================
