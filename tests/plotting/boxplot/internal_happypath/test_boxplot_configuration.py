import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot


# =============================================
# == CONVERT DISTRIBUTION DATA TO DICTIONARY ==
# =============================================
class TestCategoricalDistributionBaseInit:
    def test_zero_group_width_raises_valueerror(self):
        with pytest.raises(ValueError, match="positive"):
            BoxPlot(boxplot_group_width=0.0)

    def test_negative_group_width_raises_valueerror(self):
        with pytest.raises(ValueError, match="positive"):
            BoxPlot(boxplot_group_width=-0.1)

    def test_group_width_greater_than_one_raises_valueerror(self):
        with pytest.raises(ValueError, match="<= 1.0"):
            BoxPlot(boxplot_group_width=1.5)

    def test_zero_width_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="width_scale"):
            BoxPlot(boxplot_width_scale=0.0)

    def test_negative_width_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="width_scale"):
            BoxPlot(boxplot_width_scale=-0.1)

    def test_width_scale_greater_than_one_raises_valueerror(self):
        with pytest.raises(ValueError, match="width_scale"):
            BoxPlot(boxplot_width_scale=1.5)

    def test_width_scale_one_is_valid(self):
        bp = BoxPlot(boxplot_width_scale=1.0)
        assert bp.boxplot_width_scale == 1.0

    def test_group_width_one_is_valid(self):
        bp = BoxPlot(boxplot_group_width=1.0)
        assert bp.boxplot_group_width == 1.0


# ===========================
# == LABEL SYNCHRONIZATION ==
# ===========================


class TestBoxPlotProperties:
    def test_boxplot_group_width_getter_and_setter(self):
        bp = BoxPlot(boxplot_group_width=0.5)
        assert bp.boxplot_group_width == 0.5
        bp.boxplot_group_width = 0.8
        assert bp.boxplot_group_width == 0.8

    def test_boxplot_width_scale_getter_and_setter(self):
        bp = BoxPlot(boxplot_width_scale=0.6)
        assert bp.boxplot_width_scale == 0.6
        bp.boxplot_width_scale = 0.9
        assert bp.boxplot_width_scale == 0.9


# =================================
# == BOXPLOT BUILD PRECONDITIONS ==
# =================================


class TestBoxPlotCategoryCenters:
    def test_category_centers_are_1_indexed(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2], "C": [3]})
        centers = bp._boxplot_centers
        np.testing.assert_array_equal(centers, [1.0, 2.0, 3.0])

    def test_no_labels_returns_empty_array(self):
        bp = BoxPlot()
        assert len(bp._boxplot_centers) == 0


# ==========================
# == BOXPLOT GROUP VLINES ==
# ==========================


class TestBoxPlotGroupVlines:
    def test_remove_group_vlines_disables(self):
        bp = BoxPlot()
        bp.enable_boxplot_group_vlines()
        assert bp._include_group_vlines is True
        bp.remove_group_vlines()
        assert bp._include_group_vlines is False

    def test_update_group_vline_settings_enables(self):
        bp = BoxPlot()
        bp.update_group_vline_settings(linecolor="red")
        assert bp._include_group_vlines is True

    def test_clear_vertical_lines_and_bands_disables_vlines(self):
        bp = BoxPlot()
        bp.enable_boxplot_group_vlines()
        bp.clear_verticals()
        assert bp._include_group_vlines is False
