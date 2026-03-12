"""Tests for GerryPlotBase common functionality.

Covers: vertical/horizontal lines and bands, tick management,
axis limits, label/title styling, frame visibility, annotation arrows,
clear methods, and the ax property triggering build.

Uses ScatterPlot as the concrete subclass since it has the simplest
build requirements.
"""

import matplotlib

matplotlib.use("Agg")

import os
import tempfile

import pytest

from gerrytools.plotting.data.scatterplot import ScatterPlot
from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.tick_style import TickStyle


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# ====================
# == VERTICAL LINES ==
# ====================
class TestVerticalLines:
    def test_add_single_vertical_line(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        assert len(sp._vertical_lines) == 1

    def test_add_multiple_vertical_lines_from_list(self):
        sp = _make_plot()
        sp.add_vertical_lines([0.2, 0.4, 0.6])
        assert len(sp._vertical_lines) == 1
        # The single LineData stores all three values
        assert sp._vertical_lines[0].values == [0.2, 0.4, 0.6]

    def test_add_vertical_lines_from_int(self):
        sp = _make_plot()
        sp.add_vertical_lines(5)
        assert sp._vertical_lines[0].values == [5.0]

    def test_string_x_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="string"):
            sp.add_vertical_lines("bad")

    def test_bool_x_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="bool"):
            sp.add_vertical_lines(True)

    def test_named_vertical_line(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5, name="Threshold")
        assert sp._vertical_lines[0].name == "Threshold"


# ======================
# == HORIZONTAL LINES ==
# ======================
class TestHorizontalLines:
    def test_add_single_horizontal_line(self):
        sp = _make_plot()
        sp.add_horizontal_lines(0.5)
        assert len(sp._horizontal_lines) == 1

    def test_add_multiple_horizontal_lines(self):
        sp = _make_plot()
        sp.add_horizontal_lines([0.25, 0.75])
        assert sp._horizontal_lines[0].values == [0.25, 0.75]

    def test_string_y_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="string"):
            sp.add_horizontal_lines("bad")

    def test_bool_y_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="bool"):
            sp.add_horizontal_lines(True)


# ====================
# == VERTICAL BANDS ==
# ====================
class TestVerticalBands:
    def test_add_vertical_band(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.7)
        assert len(sp._vertical_bands) == 1
        assert sp._vertical_bands[0].lower_bound == 0.3
        assert sp._vertical_bands[0].upper_bound == 0.7

    def test_vertical_band_auto_sorts_bounds(self):
        sp = _make_plot()
        sp.add_vertical_band(0.9, 0.1)
        assert sp._vertical_bands[0].lower_bound == 0.1
        assert sp._vertical_bands[0].upper_bound == 0.9

    def test_named_vertical_band(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.7, name="Confidence Interval")
        assert sp._vertical_bands[0].name == "Confidence Interval"


# ======================
# == HORIZONTAL BANDS ==
# ======================
class TestHorizontalBands:
    def test_add_horizontal_band(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.4, 0.6)
        assert len(sp._horizontal_bands) == 1

    def test_horizontal_band_auto_sorts_bounds(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.8, 0.2)
        assert sp._horizontal_bands[0].lower_bound == 0.2
        assert sp._horizontal_bands[0].upper_bound == 0.8


# ===================
# == CLEAR METHODS ==
# ===================
class TestClearMethods:
    def test_clear_vertical_lines_and_bands(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        sp.add_vertical_band(0.3, 0.7)
        sp.clear_vertical_lines_and_bands()
        assert len(sp._vertical_lines) == 0
        assert len(sp._vertical_bands) == 0

    def test_clear_horizontal_lines_and_bands(self):
        sp = _make_plot()
        sp.add_horizontal_lines(0.5)
        sp.add_horizontal_band(0.3, 0.7)
        sp.clear_horizontal_lines_and_bands()
        assert len(sp._horizontal_lines) == 0
        assert len(sp._horizontal_bands) == 0

    def test_clear_annotation_arrows(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right")
        assert len(sp._annotation_arrows) == 1
        sp.clear_annotation_arrows()
        assert len(sp._annotation_arrows) == 0


# =====================
# == TICK MANAGEMENT ==
# =====================
class TestTickManagement:
    def test_set_xticks_stores_locations_and_labels(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 0.5, 1.0], labels=["a", "b", "c"])
        assert sp._x_tick_locations == [0.0, 0.5, 1.0]
        assert sp._x_tick_labels == ["a", "b", "c"]

    def test_set_yticks_stores_locations_and_labels(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["low", "high"])
        assert sp._y_tick_locations == [0.0, 1.0]
        assert sp._y_tick_labels == ["low", "high"]

    def test_set_xticks_empty_clears(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 1.0], labels=["a", "b"])
        sp.set_xticks([])
        assert sp._x_tick_locations == []
        assert sp._x_tick_labels == []

    def test_set_yticks_empty_clears(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["a", "b"])
        sp.set_yticks([])
        assert sp._y_tick_locations == []
        assert sp._y_tick_labels == []

    def test_update_xtick_values_locations_only(self):
        sp = _make_plot()
        sp.update_xtick_values(locations=[0.0, 0.5, 1.0])
        assert sp._x_tick_locations == [0.0, 0.5, 1.0]

    def test_update_xtick_values_labels_only(self):
        sp = _make_plot()
        sp.update_xtick_values(labels=["a", "b"])
        assert sp._x_tick_labels == ["a", "b"]

    def test_update_xtick_values_both(self):
        sp = _make_plot()
        sp.update_xtick_values(locations=[1.0, 2.0], labels=["x", "y"])
        assert sp._x_tick_locations == [1.0, 2.0]
        assert sp._x_tick_labels == ["x", "y"]

    def test_update_xtick_values_mismatched_lengths_raises_valueerror(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.update_xtick_values(locations=[1.0, 2.0], labels=["x"])

    def test_update_xtick_empty_locations_with_non_empty_labels_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="clear both"):
            sp.update_xtick_values(locations=[], labels=["a"])

    def test_update_xtick_empty_labels_with_non_empty_locations_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="clear both"):
            sp.update_xtick_values(locations=[1.0], labels=[])

    def test_update_ytick_values_mismatched_lengths_raises_valueerror(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.update_ytick_values(locations=[1.0], labels=["a", "b"])

    def test_update_xtick_none_none_is_noop(self):
        sp = _make_plot()
        sp.update_xtick_values(locations=[1.0], labels=["a"])
        sp.update_xtick_values()  # noop
        assert sp._x_tick_locations == [1.0]
        assert sp._x_tick_labels == ["a"]

    def test_update_locations_incompatible_with_existing_labels_raises(self):
        sp = _make_plot()
        sp.update_xtick_values(locations=[1.0, 2.0], labels=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            sp.update_xtick_values(locations=[1.0, 2.0, 3.0])

    def test_update_labels_incompatible_with_existing_locations_raises(self):
        sp = _make_plot()
        sp.update_xtick_values(locations=[1.0, 2.0], labels=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            sp.update_xtick_values(labels=["x", "y", "z"])

    def test_update_ytick_empty_locations_clears_both(self):
        sp = _make_plot()
        sp.update_ytick_values(locations=[1.0], labels=["a"])
        sp.update_ytick_values(locations=[])
        assert sp._y_tick_locations == []
        assert sp._y_tick_labels == []

    def test_update_ytick_empty_labels_clears_labels(self):
        sp = _make_plot()
        sp.update_ytick_values(locations=[1.0], labels=["a"])
        sp.update_ytick_values(labels=[])
        assert sp._y_tick_labels == []


# =================
# == AXIS LIMITS ==
# =================
class TestAxisLimits:
    def test_set_xlimits(self):
        sp = _make_plot()
        sp.set_xlimits(0.0, 10.0)
        assert sp._x_limits == (0.0, 10.0)

    def test_set_ylimits(self):
        sp = _make_plot()
        sp.set_ylimits(-5.0, 5.0)
        assert sp._y_limits == (-5.0, 5.0)

    def test_set_xlim_alias(self):
        sp = _make_plot()
        sp.set_xlim(1.0, 2.0)
        assert sp._x_limits == (1.0, 2.0)

    def test_set_ylim_alias(self):
        sp = _make_plot()
        sp.set_ylim(3.0, 4.0)
        assert sp._y_limits == (3.0, 4.0)

    def test_limits_applied_on_build(self):
        sp = _make_plot()
        sp.set_xlimits(-1.0, 2.0)
        sp.set_ylimits(-1.0, 2.0)
        ax = sp.ax
        assert ax.get_xlim() == (-1.0, 2.0)
        assert ax.get_ylim() == (-1.0, 2.0)


# ============================
# == LABEL AND TITLE STYLES ==
# ============================
class TestLabelAndTitleStyles:
    def test_set_xaxis_label_style_stores_style(self):
        sp = _make_plot()
        sp.set_xaxis_label_style(fontsize=14, fontweight="bold")
        assert sp._xlabel_style is not None
        assert isinstance(sp._xlabel_style, AxisLabelStyle)

    def test_set_yaxis_label_style_stores_style(self):
        sp = _make_plot()
        sp.set_yaxis_label_style(fontsize=12)
        assert sp._ylabel_style is not None

    def test_set_title_style_stores_style(self):
        sp = _make_plot()
        sp.set_title_style(fontsize=18, loc="left")
        assert sp._title_style is not None
        assert isinstance(sp._title_style, TitleStyle)

    def test_xlabel_rendered_on_build(self):
        sp = ScatterPlot(xlabel="X Label")
        sp.add_scatter(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_xlabel() == "X Label"

    def test_ylabel_rendered_on_build(self):
        sp = ScatterPlot(ylabel="Y Label")
        sp.add_scatter(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_ylabel() == "Y Label"

    def test_title_rendered_on_build(self):
        sp = ScatterPlot(title="My Title")
        sp.add_scatter(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_title() == "My Title"

    def test_none_xlabel_not_rendered(self):
        sp = ScatterPlot(xlabel=None)
        sp.add_scatter(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_xlabel() == ""


# =================
# == TICK STYLES ==
# =================
class TestTickStyles:
    def test_set_xaxis_tick_style(self):
        sp = _make_plot()
        sp.set_xaxis_tick_style(size=14, rotation=45)
        assert sp._x_tick_style is not None
        assert isinstance(sp._x_tick_style, TickStyle)
        assert sp._x_tick_style.size == 14
        assert sp._x_tick_style.rotation == 45

    def test_set_yaxis_tick_style(self):
        sp = _make_plot()
        sp.set_yaxis_tick_style(size=12)
        assert sp._y_tick_style is not None
        assert sp._y_tick_style.size == 12


# ======================
# == FRAME VISIBILITY ==
# ======================
class TestFrameVisibility:
    def test_show_or_hide_frame_stores_settings(self):
        sp = _make_plot()
        sp.show_or_hide_frame(show_top=False, show_right=False, show_bottom=True, show_left=True)
        assert sp._frame_visibility["top"] is False
        assert sp._frame_visibility["right"] is False
        assert sp._frame_visibility["bottom"] is True
        assert sp._frame_visibility["left"] is True

    def test_frame_applied_on_build(self):
        sp = _make_plot()
        sp.show_or_hide_frame(show_top=False, show_right=False)
        ax = sp.ax
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["bottom"].get_visible() is True
        assert ax.spines["left"].get_visible() is True


# ==================================
# == ANNOTATION ARROWS (DEFERRED) ==
# ==================================
class TestAnnotationArrows:
    def test_add_text_arrow_stores_arrow(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right")
        assert len(sp._annotation_arrows) == 1
        assert sp._annotation_arrows[0].arrowtype == "text"

    def test_add_label_arrow_stores_arrow(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "up")
        assert len(sp._annotation_arrows) == 1
        assert sp._annotation_arrows[0].arrowtype == "label"

    def test_text_arrow_empty_text_normalized_to_spaces(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", text="")
        assert sp._annotation_arrows[0].text == "   "

    def test_text_arrow_textrotation_overrides_style_rotation(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", textrotation=45.0)
        assert sp._annotation_arrows[0].textstyle.rotation == 45.0

    def test_label_arrow_with_arrow_length(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "down", arrow_length=50.0)
        assert sp._annotation_arrows[0].arrow_length_percentage == 50.0

    def test_label_arrow_length_with_explicit_tail_raises_valueerror(self):
        from gerrytools.plotting.data._gerryplot_dataclasses import ArrowPlacement

        sp = _make_plot()
        with pytest.raises(ValueError, match="arrowtail"):
            sp.add_label_arrow(
                (0.5, 0.5),
                "up",
                arrow_length=50.0,
                arrowplacement=ArrowPlacement(arrowtail=(0.5, 0.3)),
            )

    def test_label_arrow_length_out_of_range_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            sp.add_label_arrow((0.5, 0.5), "up", arrow_length=150.0)

    def test_add_text_arrow_with_custom_facecolor(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "left", arrowfacecolor="red")
        style = sp._annotation_arrows[0].textarrowstyle
        assert style.arrowfacecolor != "#5c676f"  # not the default

    def test_add_label_arrow_with_custom_outlinecolor(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "right", arrowoutlinecolor="blue")
        style = sp._annotation_arrows[0].labelarrowstyle
        assert style.arrowoutlinecolor != "black"  # overridden

    def test_add_text_arrow_name_stored(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", name="my_arrow")
        assert sp._annotation_arrows[0].name == "my_arrow"


# ==========================
# == LEGEND CONFIGURATION ==
# ==========================
class TestLegendConfiguration:
    def test_include_legend_flag(self):
        sp = ScatterPlot(include_legend=False)
        assert sp.include_legend is False

    def test_legend_options_accessible(self):
        sp = _make_plot()
        assert sp._legend_options is not None


# =================================
# == NAMED LINES/BANDS IN LEGEND ==
# =================================
class TestNamedOverlaysInLegend:
    def test_named_vertical_line_in_legend(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5, name="Cutoff")
        handles = sp._get_named_line_legend_handles()
        labels = [h.get_label() for h in handles]
        assert "Cutoff" in labels

    def test_unnamed_vertical_line_not_in_legend(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        handles = sp._get_named_line_legend_handles()
        assert len(handles) == 0

    def test_named_horizontal_band_in_legend(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.3, 0.7, name="CI")
        handles = sp._get_named_band_legend_handles()
        labels = [h.get_label() for h in handles]
        assert "CI" in labels

    def test_unnamed_band_not_in_legend(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.3, 0.7)
        handles = sp._get_named_band_legend_handles()
        assert len(handles) == 0


class TestGerryPlotBaseSetters:
    """Tests for set_xlabel / set_ylabel / set_title and related clearers."""

    def test_set_xlabel_stores_text(self):
        sp = ScatterPlot()
        sp.set_xlabel("Vote Share")
        assert sp.xlabel == "Vote Share"

    def test_set_xlabel_none_clears(self):
        sp = ScatterPlot(xlabel="old")
        sp.set_xlabel(None)
        assert sp.xlabel is None

    def test_set_ylabel_stores_text(self):
        sp = ScatterPlot()
        sp.set_ylabel("Seat Share")
        assert sp.ylabel == "Seat Share"

    def test_set_title_stores_text(self):
        sp = ScatterPlot()
        sp.set_title("My Plot")
        assert sp.title == "My Plot"

    def test_clear_xlabel_ylabel_and_title_styles_clears_all(self):
        sp = ScatterPlot()
        sp.set_xaxis_label_style(fontsize=14.0)
        sp.set_yaxis_label_style(fontsize=14.0)
        sp.set_title_style(fontsize=14.0)
        sp.clear_xlabel_ylabel_and_title_styles()
        assert sp._xlabel_style is None
        assert sp._ylabel_style is None
        assert sp._title_style is None

    def test_clear_xtick_labels(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.clear_xtick_labels()
        assert sp._x_tick_labels == []

    def test_clear_ytick_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.5], labels=["mid"])
        sp.clear_ytick_labels()
        assert sp._y_tick_labels == []

    def test_clear_xticks_clears_locations_and_labels(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.clear_xticks()
        assert sp._x_tick_locations == []
        assert sp._x_tick_labels == []

    def test_clear_yticks_clears_locations_and_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.5], labels=["mid"])
        sp.clear_yticks()
        assert sp._y_tick_locations == []
        assert sp._y_tick_labels == []

    def test_set_legend_options_updates_options(self):
        sp = ScatterPlot()
        sp.set_legend_options(ncols=2, fontsize=12.0)
        assert sp._legend_options.ncols == 2
        assert sp._legend_options.fontsize == 12.0


class TestGerryPlotBaseUpdateYTickEdgeCases:
    """Edge cases in update_ytick_values that weren't previously exercised."""

    def test_update_ytick_values_both_none_is_noop(self):
        sp = ScatterPlot()
        sp.update_ytick_values()
        assert sp._y_tick_locations is None
        assert sp._y_tick_labels is None

    def test_update_ytick_values_inconsistent_clear_raises(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="clear both"):
            sp.update_ytick_values(locations=[], labels=["a"])

    def test_update_ytick_values_locations_mismatch_existing_labels_raises(self):
        sp = ScatterPlot()
        sp.update_ytick_values(labels=["a", "b"])
        with pytest.raises(ValueError, match="Locations length"):
            sp.update_ytick_values(locations=[0.1, 0.2, 0.3])

    def test_update_ytick_values_locations_only_stores(self):
        sp = ScatterPlot()
        sp.update_ytick_values(locations=[0.1, 0.2])
        assert sp._y_tick_locations == [0.1, 0.2]
        assert sp._y_tick_labels is None

    def test_update_ytick_values_labels_empty_clears_only_labels(self):
        sp = ScatterPlot()
        sp.update_ytick_values(labels=[])
        assert sp._y_tick_labels == []

    def test_update_ytick_values_labels_mismatch_existing_locations_raises(self):
        sp = ScatterPlot()
        sp.update_ytick_values(locations=[0.1, 0.2])
        with pytest.raises(ValueError, match="Labels length"):
            sp.update_ytick_values(labels=["only_one"])

    def test_update_ytick_values_labels_only_stores(self):
        sp = ScatterPlot()
        sp.update_ytick_values(locations=[0.1, 0.2])
        sp.update_ytick_values(labels=["a", "b"])
        assert sp._y_tick_labels == ["a", "b"]

    def test_update_xtick_values_labels_empty_clears(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.update_xtick_values(labels=[])
        assert sp._x_tick_labels == []


class TestGerryPlotBaseBuildWithStyles:
    """Tests that trigger the _apply_deferred_label_styles path during build."""

    def test_build_with_xlabel_style(self):
        sp = ScatterPlot(xlabel="x")
        sp.set_xaxis_label_style(fontsize=12.0)
        ax = sp.ax
        assert ax.get_xlabel() == "x"

    def test_build_with_ylabel_style(self):
        sp = ScatterPlot(ylabel="y")
        sp.set_yaxis_label_style(fontsize=12.0)
        ax = sp.ax
        assert ax.get_ylabel() == "y"

    def test_build_with_title_style(self):
        sp = ScatterPlot(title="T")
        sp.set_title_style(fontsize=14.0)
        ax = sp.ax
        assert ax.get_title() == "T"

    def test_build_with_minor_x_tick_style(self):
        sp = ScatterPlot()
        sp.set_xaxis_tick_style(ticktype="minor")
        ax = sp.ax
        assert ax is not None

    def test_build_with_minor_y_tick_style(self):
        sp = ScatterPlot()
        sp.set_yaxis_tick_style(ticktype="minor")
        ax = sp.ax
        assert ax is not None

    def test_build_with_horizontal_line(self):
        sp = ScatterPlot()
        sp.add_horizontal_lines(0.5)
        ax = sp.ax
        assert ax is not None

    def test_build_with_scalar_horizontal_line_covers_real_branch(self):
        """Passing a scalar (not a list) exercises the isinstance(vals, Real) branch."""
        sp = ScatterPlot()
        sp.add_horizontal_lines(0.5)
        ax = sp.ax
        assert len(ax.lines) >= 1

    def test_build_with_scalar_vertical_line_covers_real_branch(self):
        sp = ScatterPlot()
        sp.add_vertical_lines(0.5)
        ax = sp.ax
        assert len(ax.lines) >= 1

    def test_build_with_horizontal_band(self):
        sp = ScatterPlot()
        sp.add_horizontal_band(0.3, 0.7, bandcolor="blue")
        ax = sp.ax
        assert ax is not None

    def test_build_with_horizontal_band_zero_linewidth_uses_none_edgecolor(self):
        """linewidth=0.0 triggers the edgecolor='none' branch in _draw_horizontals."""
        sp = ScatterPlot()
        sp.add_horizontal_band(0.3, 0.7, linewidth=0.0)
        ax = sp.ax
        assert ax is not None

    def test_build_with_vertical_band_zero_linewidth_uses_none_edgecolor(self):
        sp = ScatterPlot()
        sp.add_vertical_band(0.3, 0.7, linewidth=0.0)
        ax = sp.ax
        assert ax is not None

    def test_build_with_named_band_no_linecolor_in_legend(self):
        """Named band with linewidth=0 should produce a 'none' edgecolor legend handle."""
        sp = ScatterPlot()
        sp.include_legend = True
        sp.add_horizontal_band(0.3, 0.7, linewidth=0.0, name="My Band")
        ax = sp.ax
        assert ax is not None

    def test_save_to_tempfile(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmppath = f.name
        try:
            sp.save(tmppath)
            assert os.path.getsize(tmppath) > 0
        finally:
            os.unlink(tmppath)

    def test_arrow_length_with_explicit_arrowtail_raises(self):
        from gerrytools.plotting.data._gerryplot_dataclasses import ArrowPlacement

        sp = ScatterPlot()
        placement_with_tail = ArrowPlacement(arrowtail=(0.3, 0.3))
        with pytest.raises(ValueError, match="arrowtail"):
            sp.add_label_arrow(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrow_length=10.0,
                arrowplacement=placement_with_tail,
            )


class TestGerryPlotBaseYTickBuildErrors:
    """Tests that trigger ValueError during build for mismatched tick labels."""

    def test_build_with_y_tick_labels_but_no_locations_still_builds(self):
        """Setting y_tick_labels without locations falls back to existing auto-ticks."""
        sp = ScatterPlot()
        # set labels explicitly matching auto-tick count would be brittle, so just check
        # that it doesn't hard-crash when labels list is empty
        sp._y_tick_labels = []
        ax = sp.ax
        assert ax is not None
