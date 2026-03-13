"""Tests for ScatterPlot behavior.

Covers: add_scatter input modes, add_point, validation, build,
legend handles.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.plotting.data.scatterplot import ScatterPlot


# ==================
# == CONSTRUCTION ==
# ==================
class TestScatterPlotConstruction:
    def test_default_construction(self):
        sp = ScatterPlot()
        assert sp._scatter_data_list == []
        assert sp.include_legend is True


# =================
# == ADD SCATTER ==
# =================
class TestAddScatter:
    def test_add_scatter_with_x_and_y(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0, 2.0], y=[3.0, 4.0])
        assert len(sp._scatter_data_list) == 1

    def test_add_scatter_with_xy_pairs(self):
        sp = ScatterPlot()
        sp.add_scatter(xy_pairs=[(1.0, 2.0), (3.0, 4.0)])
        assert len(sp._scatter_data_list) == 1
        sd = sp._scatter_data_list[0]
        np.testing.assert_array_equal(sd.x, [1.0, 3.0])
        np.testing.assert_array_equal(sd.y, [2.0, 4.0])

    def test_add_scatter_xy_pairs_and_x_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="not both"):
            sp.add_scatter(x=[1.0], xy_pairs=[(1.0, 2.0)])

    def test_add_scatter_xy_pairs_and_y_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="not both"):
            sp.add_scatter(y=[1.0], xy_pairs=[(1.0, 2.0)])

    def test_add_scatter_neither_x_nor_xy_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must be provided"):
            sp.add_scatter()

    def test_add_scatter_x_only_no_y_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must be provided"):
            sp.add_scatter(x=[1.0])

    def test_add_scatter_y_only_no_x_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must be provided"):
            sp.add_scatter(y=[1.0])

    def test_add_scatter_mismatched_lengths_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="same shape"):
            sp.add_scatter(x=[1.0, 2.0], y=[3.0])

    def test_add_scatter_empty_arrays_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="not be empty"):
            sp.add_scatter(x=[], y=[])

    def test_add_scatter_with_label(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0], label="Point A")
        assert sp._scatter_data_list[0].label == "Point A"

    def test_add_scatter_without_label(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0])
        assert sp._scatter_data_list[0].label is None

    def test_default_markeredgecolor_is_none_string(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0])
        # When markeredgecolor is None at the call site, it becomes "none"
        sd = sp._scatter_data_list[0]
        assert isinstance(sd.marker_options.markeredgecolor, str)
        assert sd.marker_options.markeredgecolor.lower() == "none"

    def test_explicit_markeredgecolor_is_preserved(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0], markeredgecolor="red")
        sd = sp._scatter_data_list[0]
        assert sd.marker_options.markeredgecolor != "none"


# ===============
# == ADD POINT ==
# ===============
class TestAddPoint:
    def test_add_single_point(self):
        sp = ScatterPlot()
        sp.add_point(0.5, 0.5, label="Center")
        assert len(sp._scatter_data_list) == 1
        sd = sp._scatter_data_list[0]
        np.testing.assert_array_equal(sd.x, [0.5])
        np.testing.assert_array_equal(sd.y, [0.5])
        assert sd.label == "Center"

    def test_add_multiple_points(self):
        sp = ScatterPlot()
        sp.add_point(0.0, 0.0, label="Origin")
        sp.add_point(1.0, 1.0, label="Corner")
        assert len(sp._scatter_data_list) == 2


# =======================
# == BUILD AND DRAWING ==
# =======================
class TestScatterPlotBuild:
    def test_build_with_no_data_does_not_raise(self):
        sp = ScatterPlot()
        # ScatterPlot._build_plot calls _draw_points which early-returns on empty
        ax = sp.ax  # should not raise
        assert ax is not None

    def test_build_with_data_succeeds(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], label="data")
        ax = sp.ax
        assert ax is not None


# ====================
# == LEGEND HANDLES ==
# ====================
class TestScatterPlotLegend:
    def test_labeled_scatter_appears_in_legend(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0], label="A")
        handles = sp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "A" in labels

    def test_unlabeled_scatter_excluded_from_legend(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0])  # no label
        handles = sp._legend_handles
        assert len(handles) == 0

    def test_mix_of_labeled_and_unlabeled(self):
        sp = ScatterPlot()
        sp.add_scatter(x=[1.0], y=[2.0], label="Labeled")
        sp.add_scatter(x=[3.0], y=[4.0])  # no label
        handles = sp._legend_handles
        assert len(handles) == 1
        assert handles[0].get_label() == "Labeled"
