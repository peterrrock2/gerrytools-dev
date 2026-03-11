"""Tests for ViolinPlot behavior.

Covers: construction, data addition, label sync, build preconditions,
property accessors, legend handles.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.violin import ViolinPlot


class TestViolinPlotConstruction:
    def test_default_construction(self):
        vp = ViolinPlot()
        assert vp.violinplot_group_width == 0.7
        assert vp.violinplot_width_scale == 0.8

    def test_custom_width_params(self):
        vp = ViolinPlot(violinplot_group_width=0.5, violinplot_width_scale=0.6)
        assert vp.violinplot_group_width == 0.5
        assert vp.violinplot_width_scale == 0.6


class TestViolinPlotDataAddition:
    def test_add_single_dataset(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert len(vp._violinplot_data_list) == 1
        assert vp._labels == ["A", "B"]

    def test_add_multiple_datasets_same_labels(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1, 2], "B": [3, 4]})
        vp.add_violinplot_datasets({"A": [5, 6], "B": [7, 8]})
        assert len(vp._violinplot_data_list) == 2

    def test_different_labels_raises_valueerror(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="labels must match"):
            vp.add_violinplot_datasets({"X": [3], "Y": [4]})

    def test_add_extra_labels_with_different_labels(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1], "B": [2]})
        vp.add_violinplot_datasets({"A": [3], "C": [4]}, add_extra_labels=True)
        assert vp._labels == ["A", "B", "C"]

    def test_auto_name_increments(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1]})
        vp.add_violinplot_datasets({"A": [2]})
        assert vp._violinplot_data_list[0].name == "Set 1"
        assert vp._violinplot_data_list[1].name == "Set 2"


class TestViolinPlotBuildPreconditions:
    def test_no_labels_raises_valueerror(self):
        vp = ViolinPlot()
        with pytest.raises(ValueError, match="No labels"):
            vp.ax

    def test_no_datasets_raises_valueerror(self):
        vp = ViolinPlot()
        vp._labels = ["A"]
        with pytest.raises(ValueError, match="No violinplot sets"):
            vp.ax


class TestViolinPlotProperties:
    def test_group_width_setter(self):
        vp = ViolinPlot()
        vp.violinplot_group_width = 0.9
        assert vp.violinplot_group_width == 0.9

    def test_width_scale_setter(self):
        vp = ViolinPlot()
        vp.violinplot_width_scale = 0.5
        assert vp.violinplot_width_scale == 0.5


class TestViolinPlotLegend:
    def test_legend_includes_violin_handles(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1, 2, 3]}, name="Ensemble")
        handles = vp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Ensemble" in labels

    def test_legend_includes_pointset_handles(self):
        vp = ViolinPlot()
        vp.add_violinplot_datasets({"A": [1, 2, 3]})
        vp.add_pointset({"A": 1.5}, name="Enacted")
        handles = vp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Enacted" in labels
