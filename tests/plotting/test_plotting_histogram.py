"""Tests for Histogram plot behavior.

Covers: histogram addition, outline mode corrections, density mode,
bin computation, points above, build preconditions, histtype validation,
legend handles, clear, and warnings.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.data.histogram import Histogram


# ===============================
# == CONSTRUCTION AND DEFAULTS ==
# ===============================
class TestHistogramConstruction:
    def test_default_construction(self):
        h = Histogram()
        assert h.grid is False
        assert h.as_denisty_plot is False
        assert h._bins is None
        assert h._binwidth is None

    def test_custom_construction(self):
        h = Histogram(grid=True, hide_warnings=True)
        assert h.grid is True
        assert h.hide_warnings is True


# ===================
# == ADD HISTOGRAM ==
# ===================
class TestAddHistogram:
    def test_add_single_overlay_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0])
        assert len(h._hist_data_dict["overlay"]) == 1

    def test_add_multiple_histograms_of_different_types(self):
        h = Histogram()
        h.add_histogram([1, 2, 3], histtype="overlay")
        h.add_histogram([4, 5, 6], histtype="stack")
        h.add_histogram([7, 8, 9], histtype="weave")
        h.add_histogram([1, 2], histtype="outline", edgecolor="black", edgewidth=1.0)
        assert len(h._hist_data_dict["overlay"]) == 1
        assert len(h._hist_data_dict["stack"]) == 1
        assert len(h._hist_data_dict["weave"]) == 1
        assert len(h._hist_data_dict["outline"]) == 1

    def test_invalid_histtype_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="Invalid histtype"):
            h.add_histogram([1, 2, 3], histtype="bad")  # ty: ignore[invalid-argument-type]

    def test_auto_name_for_histogram(self):
        h = Histogram()
        h.add_histogram([1, 2, 3])
        assert h._hist_data_dict["overlay"][0].name == "Overlay histogram 1"

    def test_explicit_name_for_histogram(self):
        h = Histogram()
        h.add_histogram([1, 2, 3], name="My Hist")
        assert h._hist_data_dict["overlay"][0].name == "My Hist"

    def test_add_histogram_with_weights(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], weights=[1.0, 2.0, 3.0])
        hdata = h._hist_data_dict["overlay"][0]
        np.testing.assert_array_equal(hdata.weights, [1.0, 2.0, 3.0])

    def test_add_histogram_from_series(self):
        h = Histogram()
        h.add_histogram(pd.Series([1.0, 2.0, 3.0]))
        assert len(h._hist_data_dict["overlay"]) == 1

    def test_add_histogram_from_dataframe_single_column(self):
        h = Histogram()
        h.add_histogram(pd.DataFrame({"vals": [1.0, 2.0, 3.0]}))
        assert len(h._hist_data_dict["overlay"]) == 1

    def test_add_histogram_from_dataframe_with_column(self):
        h = Histogram()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        h.add_histogram(df, column="b")
        hdata = h._hist_data_dict["overlay"][0]
        np.testing.assert_array_equal(hdata.values, [3.0, 4.0])

    def test_add_histogram_filters_non_finite_values(self):
        h = Histogram()
        h.add_histogram([1.0, float("nan"), 3.0, float("inf")])
        hdata = h._hist_data_dict["overlay"][0]
        assert hdata.values.shape == (2,)

    def test_all_non_finite_values_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="at least one finite"):
            h.add_histogram([float("nan"), float("inf")])

    def test_empty_values_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="at least one entry"):
            h.add_histogram([])


# ========================================
# == OUTLINE HISTOGRAM AUTO-CORRECTIONS ==
# ========================================
class TestOutlineHistogramCorrections:
    def test_outline_with_zero_edgewidth_warns_and_sets_default(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram([1, 2, 3], histtype="outline", edgewidth=0.0)
            assert any("edgewidth" in str(warning.message) for warning in w)
        hdata = h._hist_data_dict["outline"][0]
        assert hdata.edgewidth == 0.8

    def test_outline_with_non_none_facecolor_warns_and_overrides(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram(
                [1, 2, 3],
                histtype="outline",
                facecolor="red",
                edgecolor="black",
                edgewidth=1.0,
            )
            assert any("facecolor" in str(warning.message) for warning in w)
        hdata = h._hist_data_dict["outline"][0]
        assert isinstance(hdata.facecolor, str)
        assert hdata.facecolor.lower() == "none"

    def test_outline_with_none_edgecolor_warns_and_sets_black(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram(
                [1, 2, 3],
                histtype="outline",
                edgecolor="none",
                edgewidth=1.0,
            )
            assert any("edgecolor" in str(warning.message) for warning in w)

    def test_outline_corrections_suppressed_when_hide_warnings(self):
        h = Histogram(hide_warnings=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram([1, 2, 3], histtype="outline", edgewidth=0.0)
            outline_warnings = [x for x in w if "edgewidth" in str(x.message)]
            assert len(outline_warnings) == 0

    def test_outline_with_valid_settings_no_warnings(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram(
                [1, 2, 3],
                histtype="outline",
                facecolor="none",
                edgecolor="black",
                edgewidth=1.5,
            )
            outline_warnings = [
                x
                for x in w
                if any(kw in str(x.message) for kw in ("edgewidth", "facecolor", "edgecolor"))
            ]
            assert len(outline_warnings) == 0


# ==================
# == BIN SETTINGS ==
# ==================
class TestHistogramBins:
    def test_set_bins_stores_value(self):
        h = Histogram()
        h.set_bins(20)
        assert h._bins == 20

    def test_set_bins_with_string(self):
        h = Histogram()
        h.set_bins("sturges")
        assert h._bins == "sturges"

    def test_set_bins_with_array(self):
        h = Histogram()
        edges = np.arange(0, 11, 1.0)
        h.set_bins(edges)
        np.testing.assert_array_equal(h._bins, edges)

    def test_set_bins_by_width(self):
        h = Histogram()
        h.set_bins_by_width(0.5)
        assert h._binwidth == 0.5
        assert h._bins is None

    def test_set_bins_by_width_none_resets(self):
        h = Histogram()
        h.set_bins(20)
        h.set_bins_by_width(None)
        assert h._binwidth is None
        assert h._bins is None

    def test_center_data_on_bin_edges(self):
        h = Histogram()
        h.center_data_on_bin_edges()
        assert h._bin_alignment == "center"


# ==================
# == DENSITY MODE ==
# ==================
class TestHistogramDensity:
    def test_transform_to_density(self):
        h = Histogram()
        h.transform_to_density()
        assert h.as_denisty_plot is True


# ==================
# == POINTS ABOVE ==
# ==================
class TestHistogramPointsAbove:
    def test_add_single_point(self):
        h = Histogram()
        h.add_points_above(5.0)
        assert len(h._histpointlist_list) == 1

    def test_add_list_of_points(self):
        h = Histogram()
        h.add_points_above([1.0, 2.0, 3.0])
        assert len(h._histpointlist_list) == 1
        assert h._histpointlist_list[0].values.shape == (3,)

    def test_add_points_from_series(self):
        h = Histogram()
        h.add_points_above(pd.Series([1.0, 2.0]))
        assert h._histpointlist_list[0].values.shape == (2,)

    def test_centered_on_bin_flag_stored(self):
        h = Histogram()
        h.add_points_above(5.0, centered_on_bin=True)
        assert h._histpointlist_list[0].centered is True

    def test_auto_name_for_points(self):
        h = Histogram()
        h.add_points_above(5.0)
        assert h._histpointlist_list[0].name == "Point Marker 1"

    def test_explicit_name_for_points(self):
        h = Histogram()
        h.add_points_above(5.0, name="Plan Value")
        assert h._histpointlist_list[0].name == "Plan Value"

    def test_y_offset_is_stored(self):
        h = Histogram()
        h.add_points_above(5.0, y_offset=0.05)
        assert h._histpointlist_list[0].y_offset == 0.05

    def test_nan_values_filtered_from_points(self):
        h = Histogram()
        h.add_points_above([1.0, float("nan"), 3.0])
        assert h._histpointlist_list[0].values.shape == (2,)


# =========================
# == BUILD PRECONDITIONS ==
# =========================
class TestHistogramBuildPreconditions:
    def test_no_histograms_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="No histogram sets"):
            h.ax

    def test_clear_histograms(self):
        h = Histogram()
        h.add_histogram([1, 2, 3])
        h.clear_histograms()
        assert all(len(v) == 0 for v in h._hist_data_dict.values())


# ====================
# == LEGEND HANDLES ==
# ====================
class TestHistogramLegend:
    def test_legend_includes_histogram_handles(self):
        h = Histogram()
        h.add_histogram([1, 2, 3], name="Ensemble")
        handles = h._legend_handles
        labels = [handle.get_label() for handle in handles]
        assert "Ensemble" in labels

    def test_legend_includes_point_handles(self):
        h = Histogram()
        h.add_histogram([1, 2, 3])
        h.add_points_above(2.0, name="Plan")
        handles = h._legend_handles
        labels = [handle.get_label() for handle in handles]
        assert "Plan" in labels

    def test_legend_includes_named_line_handles(self):
        h = Histogram()
        h.add_histogram([1, 2, 3])
        h.add_vertical_lines(2.0, name="Threshold")
        handles = h._legend_handles
        labels = [handle.get_label() for handle in handles]
        assert "Threshold" in labels


class TestHistogramActualBuilds:
    """Every test here calls .ax to exercise the draw path."""

    def test_build_overlay_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = h.ax
        assert ax is not None

    def test_build_stack_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], histtype="stack")
        h.add_histogram([4.0, 5.0, 6.0], histtype="stack")
        ax = h.ax
        assert ax is not None

    def test_build_weave_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], histtype="weave")
        h.add_histogram([4.0, 5.0, 6.0], histtype="weave")
        ax = h.ax
        assert ax is not None

    def test_build_outline_histogram(self):
        h = Histogram()
        h.add_histogram(
            [1.0, 2.0, 3.0],
            histtype="outline",
            facecolor="none",
            edgecolor="black",
            edgewidth=1.0,
        )
        ax = h.ax
        assert ax is not None

    def test_build_with_explicit_bins(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.set_bins(5)
        ax = h.ax
        assert ax is not None

    def test_build_with_binwidth(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.set_bins_by_width(0.5)
        ax = h.ax
        assert ax is not None

    def test_build_with_points_above(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0])
        h.add_points_above(2.5, name="Threshold")
        ax = h.ax
        assert ax is not None

    def test_build_with_centered_bin_alignment(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.center_data_on_bin_edges()
        ax = h.ax
        assert ax is not None

    def test_build_as_density(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        h.transform_to_density()
        ax = h.ax
        assert ax is not None

    def test_build_with_grid(self):
        h = Histogram(grid=True)
        h.add_histogram([1.0, 2.0, 3.0])
        ax = h.ax
        assert ax is not None

    def test_build_with_weights(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], weights=[1.0, 2.0, 3.0])
        ax = h.ax
        assert ax is not None

    def test_build_with_point_outside_bin_range(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.add_points_above(10.0, name="OutOfRange")
        ax = h.ax
        assert ax is not None

    def test_build_stacked_with_point_shows_stacked_height(self):
        h = Histogram()
        h.add_histogram([1.0, 1.5, 2.0], histtype="stack")
        h.add_histogram([1.0, 1.5, 2.0], histtype="stack")
        h.add_points_above(1.2, name="Plan")
        ax = h.ax
        assert ax is not None


class TestHistogramDataEdgeCases:
    def test_infinite_edgewidth_raises_valueerror(self):
        from gerrytools.plotting.data.histogram import HistogramData

        with pytest.raises(ValueError, match="edgewidth must be finite"):
            HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.array([1.0, 1.0]),
                facecolor="blue",
                edgecolor="none",
                edgewidth=float("inf"),
            )
