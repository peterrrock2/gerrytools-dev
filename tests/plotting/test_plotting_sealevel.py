"""Tests for SeaLevel plot behavior.

Covers: construction, data conversion, jitter configuration,
format_ylabels_as_fractions, build preconditions, label validation,
marker color defaults.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.data.sealevel import SeaLevel


# ==================
# == CONSTRUCTION ==
# ==================
class TestSeaLevelConstruction:
    def test_default_construction(self):
        sl = SeaLevel()
        assert sl._labels is None
        assert sl.jitter_rng_seed is None

    def test_construction_with_seed(self):
        sl = SeaLevel(jitter_rng_seed=42)
        assert sl.jitter_rng_seed == 42

    def test_construction_with_rng(self):
        rng = np.random.default_rng(99)
        sl = SeaLevel(jitter_rng=rng)
        assert sl.jitter_rng_seed is None

    def test_seed_and_rng_raises_valueerror(self):
        with pytest.raises(ValueError, match="not both"):
            SeaLevel(jitter_rng_seed=42, jitter_rng=np.random.default_rng(0))

    def test_seed_setter(self):
        sl = SeaLevel()
        sl.jitter_rng_seed = 123
        assert sl.jitter_rng_seed == 123

    def test_seed_setter_none(self):
        sl = SeaLevel(jitter_rng_seed=42)
        sl.jitter_rng_seed = None
        assert sl.jitter_rng_seed is None


# ======================================
# == CONVERT SCORE DATA TO DICTIONARY ==
# ======================================
class TestSeaLevelDataConversion:
    def test_dict_input(self):
        sl = SeaLevel()
        result = sl._convert_score_data_to_dictionary({"A": 0.5, "B": 0.7})
        assert result == {"A": 0.5, "B": 0.7}

    def test_list_input_with_labels(self):
        sl = SeaLevel()
        result = sl._convert_score_data_to_dictionary([0.5, 0.7], scores_labels=["A", "B"])
        assert result == {"A": 0.5, "B": 0.7}

    def test_list_without_labels_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="scores_labels"):
            sl._convert_score_data_to_dictionary([0.5, 0.7])

    def test_list_length_mismatch_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="match length"):
            sl._convert_score_data_to_dictionary([0.5, 0.7, 0.9], scores_labels=["A", "B"])

    def test_series_input(self):
        sl = SeaLevel()
        result = sl._convert_score_data_to_dictionary(pd.Series({"A": 0.5, "B": 0.7}))
        assert result == {"A": 0.5, "B": 0.7}

    def test_dataframe_input_with_row_index(self):
        sl = SeaLevel()
        df = pd.DataFrame(
            {"A": [0.5, 0.6], "B": [0.7, 0.8]},
            index=["row1", "row2"],  # ty: ignore[invalid-argument-type]
        )
        result = sl._convert_score_data_to_dictionary(df, df_row_index="row1")
        assert result == {"A": 0.5, "B": 0.7}

    def test_dataframe_without_row_index_raises_valueerror(self):
        sl = SeaLevel()
        df = pd.DataFrame({"A": [0.5]})
        with pytest.raises(ValueError, match="df_row_index"):
            sl._convert_score_data_to_dictionary(df)

    def test_dataframe_missing_row_index_raises_valueerror(self):
        sl = SeaLevel()
        df = pd.DataFrame(
            {"A": [0.5]},
            index=["row1"],  # ty: ignore[invalid-argument-type]
        )
        with pytest.raises(ValueError, match="not found"):
            sl._convert_score_data_to_dictionary(df, df_row_index="bad_row")

    def test_non_finite_values_raise_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="finite"):
            sl._convert_score_data_to_dictionary({"A": float("nan")})

    def test_infinite_values_raise_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="finite"):
            sl._convert_score_data_to_dictionary({"A": float("inf")})

    def test_empty_dict_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="empty"):
            sl._convert_score_data_to_dictionary({})

    def test_unsupported_type_raises_typeerror(self):
        sl = SeaLevel()
        with pytest.raises(TypeError):
            sl._convert_score_data_to_dictionary(42)  # ty: ignore[invalid-argument-type]


# ===========================================
# == ADD SEALEVEL SET AND LABEL VALIDATION ==
# ===========================================
class TestAddSeaLevelSet:
    def test_first_set_defines_labels(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5, "B": 0.7})
        assert sl._labels == ["A", "B"]

    def test_second_set_same_labels_succeeds(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5, "B": 0.7})
        sl.add_sealevel_set({"A": 0.3, "B": 0.6})
        assert len(sl._sealevel_data_list) == 2

    def test_second_set_different_labels_raises_valueerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5, "B": 0.7})
        with pytest.raises(ValueError, match="same labels"):
            sl.add_sealevel_set({"X": 0.3, "Y": 0.6})

    def test_auto_name_increments(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        sl.add_sealevel_set({"A": 0.6})
        assert sl._sealevel_data_list[0].name == "Set 1"
        assert sl._sealevel_data_list[1].name == "Set 2"

    def test_explicit_name(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5}, name="Enacted Plan")
        assert sl._sealevel_data_list[0].name == "Enacted Plan"

    def test_markerfacecolor_defaults_to_linecolor(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5}, linecolor="red")
        ms = sl._sealevel_data_list[0].markersettings
        # markerfacecolor should have been derived from linecolor
        assert ms.markerfacecolor != "none"

    def test_markeredgecolor_defaults_to_markerfacecolor(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5}, linecolor="blue")
        ms = sl._sealevel_data_list[0].markersettings
        # Both face and edge should be derived from linecolor
        mpl_dict = ms.to_mpl_settings_dict()
        assert mpl_dict["markerfacecolor"] == mpl_dict["markeredgecolor"]


# ==========================
# == JITTER CONFIGURATION ==
# ==========================
class TestSeaLevelJitter:
    def _make_sealevel_with_labels(self):
        sl = SeaLevel(jitter_rng_seed=42)
        sl.add_sealevel_set({"A": 0.5, "B": 0.7, "C": 0.3})
        return sl

    def test_set_vertical_jitter_per_category(self):
        sl = self._make_sealevel_with_labels()
        sl.set_max_vertical_jitter_per_category(jitter_per_category={"A": 0.1, "B": 0.05})
        assert sl._maximum_vertical_jitter_per_category["A"] == 0.1
        assert "C" not in sl._maximum_vertical_jitter_per_category

    def test_set_vertical_jitter_all(self):
        sl = self._make_sealevel_with_labels()
        sl.set_max_vertical_jitter_all(0.1)
        assert all(v == 0.1 for v in sl._maximum_vertical_jitter_per_category.values())

    def test_set_horizontal_jitter_per_category(self):
        sl = self._make_sealevel_with_labels()
        sl.set_max_horizontal_jitter_per_category(jitter_per_category={"A": 0.05})
        assert sl._maximum_horizontal_jitter_per_category["A"] == 0.05

    def test_set_horizontal_jitter_all(self):
        sl = self._make_sealevel_with_labels()
        sl.set_max_horizontal_jitter_all(0.2)
        assert all(v == 0.2 for v in sl._maximum_horizontal_jitter_per_category.values())

    def test_jitter_before_labels_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_max_vertical_jitter_all(0.1)

    def test_jitter_before_labels_horizontal_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_max_horizontal_jitter_all(0.1)

    def test_jitter_extra_keys_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="Extra keys"):
            sl.set_max_vertical_jitter_per_category(jitter_per_category={"A": 0.1, "Z": 0.2})

    def test_negative_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_max_vertical_jitter_per_category(jitter_per_category={"A": -0.1})

    def test_infinite_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="finite"):
            sl.set_max_vertical_jitter_per_category(jitter_per_category={"A": float("inf")})

    def test_non_numeric_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="real numbers"):
            sl.set_max_vertical_jitter_per_category(jitter_per_category={"A": "big"})

    def test_non_dict_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="dictionary"):
            sl.set_max_vertical_jitter_per_category(jitter_per_category=[0.1, 0.2, 0.3])

    def test_negative_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_max_vertical_jitter_all(-0.1)

    def test_infinite_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_max_vertical_jitter_all(float("inf"))

    def test_zero_jitter_is_valid(self):
        sl = self._make_sealevel_with_labels()
        sl.set_max_vertical_jitter_all(0.0)
        assert all(v == 0.0 for v in sl._maximum_vertical_jitter_per_category.values())


# =================================
# == FORMAT YLABELS AS FRACTIONS ==
# =================================
class TestFormatYLabelsAsFractions:
    def test_basic_fraction_labels(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        sl.format_ylabels_as_fractions(4)
        assert sl._y_tick_locations == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert sl._y_tick_labels == ["0/4", "1/4", "2/4", "3/4", "4/4"]

    def test_with_minimum_numerator(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        sl.format_ylabels_as_fractions(4, minimum_numerator=1)
        assert sl._y_tick_locations is not None
        assert sl._y_tick_locations[0] == 0.25

    def test_with_maximum_numerator(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        sl.format_ylabels_as_fractions(4, maximum_numerator=3)
        assert sl._y_tick_locations is not None
        assert sl._y_tick_locations[-1] == 0.75

    def test_non_int_denominator_raises_typeerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        with pytest.raises(TypeError, match="integer"):
            sl.format_ylabels_as_fractions(4.5)  # ty: ignore[invalid-argument-type]

    def test_zero_denominator_raises_valueerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        with pytest.raises(ValueError, match="positive"):
            sl.format_ylabels_as_fractions(0)

    def test_negative_denominator_raises_valueerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        with pytest.raises(ValueError, match="positive"):
            sl.format_ylabels_as_fractions(-4)

    def test_max_numerator_exceeds_denominator_raises_valueerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        with pytest.raises(ValueError, match="exceed"):
            sl.format_ylabels_as_fractions(4, maximum_numerator=5)

    def test_min_exceeds_max_raises_valueerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        with pytest.raises(ValueError, match="exceed"):
            sl.format_ylabels_as_fractions(4, minimum_numerator=3, maximum_numerator=2)

    def test_min_exceeds_denominator_when_max_is_none_raises_valueerror(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5})
        with pytest.raises(ValueError, match="exceed"):
            sl.format_ylabels_as_fractions(4, minimum_numerator=5)


# =========================
# == BUILD PRECONDITIONS ==
# =========================
class TestSeaLevelBuildPreconditions:
    def test_no_labels_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="No labels"):
            sl.ax

    def test_no_datasets_raises_valueerror(self):
        sl = SeaLevel()
        sl._labels = ["A"]
        with pytest.raises(ValueError, match="No sealevel sets"):
            sl.ax


# ======================
# == CATEGORY CENTERS ==
# ======================
class TestSeaLevelCategoryCenters:
    def test_centers_are_1_indexed(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5, "B": 0.7, "C": 0.3})
        centers = sl._sealevel_centers
        np.testing.assert_array_equal(centers, [1.0, 2.0, 3.0])

    def test_no_labels_returns_empty(self):
        sl = SeaLevel()
        assert len(sl._sealevel_centers) == 0
