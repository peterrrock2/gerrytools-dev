"""Tests for BoxPlot and CategoricalDistributionPlotBase behavior.

Covers: data conversion, label synchronization, pointset handling,
group width/width scale validation, build-plot preconditions, and
property accessors.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot


# =============================================
# == CONVERT DISTRIBUTION DATA TO DICTIONARY ==
# =============================================
class TestConvertDistributionDataToDictionary:
    def test_dict_input_passthrough(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary({"A": [1, 2], "B": [3, 4]})
        assert result == {"A": [1, 2], "B": [3, 4]}

    def test_dict_keys_coerced_to_strings(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary(
            {1: [10], 2: [20]}  # ty: ignore[invalid-argument-type]
        )
        assert "1" in result and "2" in result

    def test_dataframe_uses_columns_as_labels(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        result = BoxPlot._convert_boxplot_data_to_dictionary(df)
        assert set(result.keys()) == {"X", "Y"}
        assert result["X"] == [1.0, 2.0]

    def test_dataframe_drops_nan(self):
        df = pd.DataFrame({"X": [1.0, float("nan"), 3.0]})
        result = BoxPlot._convert_boxplot_data_to_dictionary(df)
        assert result["X"] == [1.0, 3.0]

    def test_flat_list_with_labels_returns_single_entry_dict(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary([10, 20, 30], scores_labels=["A"])
        assert "A" in result
        assert result["A"] == [10, 20, 30]

    def test_nested_list_with_labels(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary(
            [[1, 2], [3, 4]], scores_labels=["A", "B"]
        )
        assert result == {"A": [1, 2], "B": [3, 4]}

    def test_list_without_labels_raises_valueerror(self):
        with pytest.raises(ValueError, match="labels"):
            BoxPlot._convert_boxplot_data_to_dictionary([1, 2, 3])

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            BoxPlot._convert_boxplot_data_to_dictionary([], scores_labels=["A"])

    def test_labels_count_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="length"):
            BoxPlot._convert_boxplot_data_to_dictionary(
                [[1, 2], [3, 4], [5, 6]], scores_labels=["A", "B"]
            )

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="dict"):
            BoxPlot._convert_boxplot_data_to_dictionary(42)  # ty: ignore[invalid-argument-type]


# ======================================================
# == CATEGORICALDISTRIBUTIONPLOTBASE: INIT VALIDATION ==
# ======================================================
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
class TestLabelSynchronization:
    def test_first_dataset_defines_labels(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        assert bp._labels == ["A", "B"]

    def test_second_dataset_same_labels_succeeds(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        bp.add_boxplot_datasets({"A": [3], "B": [4]})
        assert len(bp._boxplot_data_list) == 2

    def test_second_dataset_different_labels_raises_valueerror(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="labels must match"):
            bp.add_boxplot_datasets({"C": [3], "D": [4]})

    def test_add_extra_labels_merges_labels(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        bp.add_boxplot_datasets({"A": [3], "B": [4], "C": [5]}, add_extra_labels=True)
        assert bp._labels is not None
        assert sorted(bp._labels) == ["A", "B", "C"]

    def test_add_extra_labels_preserves_original_order(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        bp.add_boxplot_datasets({"C": [5], "A": [3]}, add_extra_labels=True)
        # Original order A, B maintained, with C appended
        assert bp._labels is not None
        assert bp._labels[:2] == ["A", "B"]


# ================================
# == BOXPLOT PROPERTY ACCESSORS ==
# ================================
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
class TestBoxPlotBuildPreconditions:
    def test_build_with_no_labels_raises_valueerror(self):
        bp = BoxPlot()
        with pytest.raises(ValueError, match="No labels"):
            bp.ax  # triggers build

    def test_build_with_no_datasets_raises_valueerror(self):
        bp = BoxPlot()
        # Manually set labels but add no data
        bp._labels = ["A", "B"]
        with pytest.raises(ValueError, match="No boxplot sets"):
            bp.ax


# ===============================
# == BOXPLOT POINTSET HANDLING ==
# ===============================
class TestBoxPlotPointset:
    def test_add_pointset_from_dict(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1, 2], "B": [3, 4]})
        bp.add_pointset({"A": 1.5, "B": 3.5}, name="Enacted")
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_list_uses_existing_labels(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        bp.add_pointset([1.5, 2.5])
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_list_without_labels_raises_when_no_data(self):
        bp = BoxPlot()
        with pytest.raises(ValueError, match="labels"):
            bp.add_pointset([1.5, 2.5])

    def test_add_pointset_length_mismatch_raises_valueerror(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="length"):
            bp.add_pointset([1.5])  # only 1 value for 2 labels

    def test_add_pointset_from_series(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        ser = pd.Series({"A": 1.5, "B": 2.5})
        bp.add_pointset(ser)
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_dataframe_single_column(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        df = pd.DataFrame(
            {"val": [1.5, 2.5]}, index=["A", "B"]  # ty: ignore[invalid-argument-type]
        )
        bp.add_pointset(df)
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_dataframe_multi_column_no_column_raises(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        df = pd.DataFrame(
            {"v1": [1.5, 2.5], "v2": [3.5, 4.5]},
            index=["A", "B"],  # ty: ignore[invalid-argument-type]
        )
        with pytest.raises(ValueError, match="exactly one"):
            bp.add_pointset(df)

    def test_add_pointset_from_dataframe_with_column_param(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1], "B": [2]})
        df = pd.DataFrame(
            {"v1": [1.5, 2.5], "v2": [3.5, 4.5]},
            index=["A", "B"],  # ty: ignore[invalid-argument-type]
        )
        bp.add_pointset(df, column="v2")
        assert len(bp._pointset_data_list) == 1


# =========================
# == BOXPLOT AUTO-NAMING ==
# =========================
class TestBoxPlotAutoNaming:
    def test_auto_name_for_boxplot_dataset(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1]})
        assert bp._boxplot_data_list[0].name == "Set 1"

    def test_explicit_name_for_boxplot_dataset(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1]}, name="Custom Name")
        assert bp._boxplot_data_list[0].name == "Custom Name"

    def test_auto_name_for_pointset(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1]})
        bp.add_pointset({"A": 1.5})
        assert bp._pointset_data_list[0].name == "Point Set 1"

    def test_explicit_name_for_pointset(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1]})
        bp.add_pointset({"A": 1.5}, name="Enacted Plan")
        assert bp._pointset_data_list[0].name == "Enacted Plan"


# ==============================
# == BOXPLOT CATEGORY CENTERS ==
# ==============================
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
        bp = BoxPlot(include_boxplot_group_vlines=True)
        assert bp._include_group_vlines is True
        bp.remove_group_vlines()
        assert bp._include_group_vlines is False

    def test_update_group_vline_settings_enables(self):
        bp = BoxPlot(include_boxplot_group_vlines=False)
        bp.update_group_vline_settings(linecolor="red")
        assert bp._include_group_vlines is True

    def test_clear_vertical_lines_and_bands_disables_vlines(self):
        bp = BoxPlot(include_boxplot_group_vlines=True)
        bp.clear_vertical_lines_and_bands()
        assert bp._include_group_vlines is False


# ============================
# == BOXPLOT LEGEND HANDLES ==
# ============================
class TestBoxPlotLegendHandles:
    def test_legend_handles_include_boxplot_and_pointset(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1, 2]}, name="Ensemble")
        bp.add_pointset({"A": 1.5}, name="Enacted")
        handles = bp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Ensemble" in labels
        assert "Enacted" in labels

    def test_named_lines_appear_in_legend(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1, 2]})
        bp.add_vertical_lines(1.5, name="Threshold")
        handles = bp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Threshold" in labels

    def test_unnamed_lines_do_not_appear_in_legend(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1, 2]})
        bp.add_vertical_lines(1.5)  # no name
        handles = bp._legend_handles
        # Should only have the boxplot handle
        assert len(handles) == 1


class TestBoxPlotActualBuilds:
    def test_build_single_dataset(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        ax = bp.ax
        assert ax is not None

    def test_build_two_datasets_grouped(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, name="Set1")
        bp.add_boxplot_datasets({"A": [2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0]}, name="Set2")
        ax = bp.ax
        assert ax is not None

    def test_build_with_pointset_overlay(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        bp.add_pointset({"A": 2.0, "B": 5.0}, name="Enacted")
        ax = bp.ax
        assert ax is not None

    def test_build_with_group_vlines(self):
        bp = BoxPlot(include_boxplot_group_vlines=True)
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 3.0]})
        ax = bp.ax
        assert ax is not None

    def test_build_with_vlines_disabled(self):
        bp = BoxPlot(include_boxplot_group_vlines=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 3.0]})
        ax = bp.ax
        assert ax is not None

    def test_build_with_fliers_shown(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 10.0, 3.0]}, showfliers=True)
        ax = bp.ax
        assert ax is not None

    def test_build_category_tick_labels_populated(self):
        bp = BoxPlot()
        bp.add_boxplot_datasets({"Alpha": [1.0, 2.0], "Beta": [3.0, 4.0]})
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Alpha" in tick_labels
        assert "Beta" in tick_labels
