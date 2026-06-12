import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.data.histogram import Histogram


# ===============================
# == CONSTRUCTION AND DEFAULTS ==
# ===============================
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

    def test_set_bins_clears_prior_binwidth(self):
        # Mirror of the reset above: set_bins and set_bins_by_width are
        # mutually exclusive, so each clears the other's setting.
        h = Histogram()
        h.set_bins_by_width(0.5)
        h.set_bins(20)
        assert h._bins == 20
        assert h._binwidth is None

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
        assert h.as_density_plot is True


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
