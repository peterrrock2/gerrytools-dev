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

    def test_flat_list_without_labels_is_single_numeric_category(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary([1, 2, 3])
        assert result == {"0": [1, 2, 3]}

    def test_nested_list_without_labels_auto_numbers_categories(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary([[1, 2], [3, 4], [5, 6]])
        assert result == {"0": [1, 2], "1": [3, 4], "2": [5, 6]}

    def test_2d_array_without_labels_auto_numbers_categories(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result == {"0": [1.0, 2.0], "1": [3.0, 4.0]}

    def test_1d_array_without_labels_auto_numbers_categories(self):
        result = BoxPlot._convert_boxplot_data_to_dictionary(np.array([1.0, 2.0, 3.0]))
        assert result == {"0": [1.0], "1": [2.0], "2": [3.0]}

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


class TestLabelSynchronization:
    def test_first_dataset_defines_labels(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1], "B": [2]})
        assert bp._labels == ["A", "B"]

    def test_second_dataset_same_labels_succeeds(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1], "B": [2]})
        bp.add_boxplot_dataset({"A": [3], "B": [4]})
        assert len(bp._boxplot_data_list) == 2

    def test_second_dataset_different_labels_raises_valueerror(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="labels must match"):
            bp.add_boxplot_dataset({"C": [3], "D": [4]})

    def test_add_extra_labels_merges_labels(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1], "B": [2]})
        bp.add_boxplot_dataset({"A": [3], "B": [4], "C": [5]}, add_extra_labels=True)
        assert bp._labels is not None
        assert sorted(bp._labels) == ["A", "B", "C"]

    def test_add_extra_labels_preserves_original_order(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1], "B": [2]})
        bp.add_boxplot_dataset({"C": [5], "A": [3]}, add_extra_labels=True)
        # Original order A, B maintained, with C appended
        assert bp._labels is not None
        assert bp._labels[:2] == ["A", "B"]


# ================================
# == BOXPLOT PROPERTY ACCESSORS ==
# ================================


class TestBoxPlotAutoNaming:
    def test_auto_name_for_boxplot_dataset(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1]})
        assert bp._boxplot_data_list[0].name == "Set 1"

    def test_explicit_name_for_boxplot_dataset(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1]}, name="Custom Name")
        assert bp._boxplot_data_list[0].name == "Custom Name"

    def test_auto_name_for_pointset(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1]})
        bp.add_pointset({"A": 1.5})
        assert bp._pointset_data_list[0].name == "Point Set 1"

    def test_explicit_name_for_pointset(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1]})
        bp.add_pointset({"A": 1.5}, name="Enacted Plan")
        assert bp._pointset_data_list[0].name == "Enacted Plan"


# ===============================
# == BOXPLOT COLOR RESOLUTION ===
# ===============================


class TestBoxPlotColorResolution:
    def test_facecolor_none_resolves_to_none(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1, 2, 3]}, facecolor=None)
        assert bp._boxplot_data_list[0].facecolor == "none"

    def test_edgecolor_none_resolves_to_none_and_drops_edgewidth(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1, 2, 3]}, edgecolor=None)
        set_data = bp._boxplot_data_list[0]
        assert set_data.edgecolor == "none"
        assert set_data.edgewidth == 0.0

    def test_omitted_facecolor_uses_options_default(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1, 2, 3]})
        # The default "denim" resolves to its hex form.
        assert bp._boxplot_data_list[0].facecolor == "#1560bd"

    def test_explicit_facecolor_still_resolves(self):
        bp = BoxPlot()
        bp.add_boxplot_dataset({"A": [1, 2, 3]}, facecolor="red")
        assert bp._boxplot_data_list[0].facecolor == "#ff0000"


# ==============================
# == BOXPLOT CATEGORY CENTERS ==
# ==============================
