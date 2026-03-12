"""Tests for plotting utility functions.

Covers: _coerce_real_iter, sort_elections, resolve_numpy_rng, spawn_child_seeds,
line_segment_through_unit_square, build_legend_options, save_legend_handles,
_coerce_to_1d_float_array, _coerce_values_and_weights.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from numpy.random import Generator

from gerrytools.plotting._legend_utils import build_legend_options, save_legend_handles
from gerrytools.plotting._rng import resolve_numpy_rng, spawn_child_seeds
from gerrytools.plotting.data._geometry import line_segment_through_unit_square
from gerrytools.plotting.data.histogram import (
    _coerce_to_1d_finite_float_array,
    _coerce_to_1d_float_array,
    _coerce_values_and_weights,
)
from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.plotting.utils import _coerce_real_iter, sort_elections


# ======================
# == COERCE REAL ITER ==
# ======================
class TestCoerceRealIter:
    def test_scalar_int_returns_single_element_list(self):
        assert _coerce_real_iter(5, field="x") == [5.0]

    def test_scalar_float_returns_single_element_list(self):
        assert _coerce_real_iter(3.14, field="x") == [3.14]

    def test_list_of_ints_returns_float_list(self):
        result = _coerce_real_iter([1, 2, 3], field="x")
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)

    def test_list_of_floats_passthrough(self):
        assert _coerce_real_iter([1.5, 2.5], field="x") == [1.5, 2.5]

    def test_string_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="string"):
            _coerce_real_iter("hello", field="x")  # ty: ignore[invalid-argument-type]

    def test_bytes_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="string"):
            _coerce_real_iter(b"hello", field="x")

    def test_bool_scalar_raises_typeerror(self):
        with pytest.raises(TypeError):
            _coerce_real_iter(True, field="x")

    def test_list_containing_bool_raises_typeerror(self):
        with pytest.raises(TypeError, match="real numbers"):
            _coerce_real_iter([1.0, True], field="x")

    def test_list_containing_string_raises_typeerror(self):
        with pytest.raises(TypeError, match="real numbers"):
            _coerce_real_iter([1.0, "two"], field="x")  # ty: ignore[invalid-argument-type]

    def test_non_iterable_non_numeric_raises_typeerror(self):
        with pytest.raises(TypeError, match="must be a number"):
            _coerce_real_iter(object(), field="x")  # ty: ignore[invalid-argument-type]

    def test_empty_list_returns_empty(self):
        assert _coerce_real_iter([], field="x") == []

    def test_tuple_input_works(self):
        assert _coerce_real_iter((1, 2, 3), field="x") == [1.0, 2.0, 3.0]

    def test_numpy_scalar_is_accepted(self):
        result = _coerce_real_iter(np.float64(3.0), field="x")
        assert result == [3.0]

    def test_generator_expression_is_accepted(self):
        result = _coerce_real_iter((x for x in [1, 2, 3]), field="x")
        assert result == [1.0, 2.0, 3.0]


# ====================
# == SORT ELECTIONS ==
# ====================
class TestSortElections:
    def test_sorts_by_year_suffix(self):
        result = sort_elections(["SEN20", "PRES16", "GOV18"])
        assert result == ["PRES16", "GOV18", "SEN20"]

    def test_single_election_returns_unchanged(self):
        assert sort_elections(["SEN22"]) == ["SEN22"]

    def test_empty_list_returns_empty(self):
        assert sort_elections([]) == []

    def test_same_year_preserves_alphabetical_order(self):
        result = sort_elections(["GOV20", "SEN20", "PRES20"])
        # All have suffix "20", sorted by full name first then by year
        assert all(e.endswith("20") for e in result)

    def test_two_digit_year_ordering_wraps_correctly(self):
        # 02 < 18 < 20 in two-digit year comparison
        result = sort_elections(["SEN20", "GOV02", "PRES18"])
        assert result == ["GOV02", "PRES18", "SEN20"]


# =======================
# == RESOLVE NUMPY RNG ==
# =======================
class TestResolveNumpyRng:
    def test_seed_none_rng_none_returns_generator_and_none_seed(self):
        rng, seed = resolve_numpy_rng()
        assert isinstance(rng, Generator)
        assert seed is None

    def test_seed_int_returns_generator_and_same_seed(self):
        rng, seed = resolve_numpy_rng(seed=42)
        assert isinstance(rng, Generator)
        assert seed == 42

    def test_explicit_rng_returns_same_rng(self):
        my_rng = np.random.default_rng(123)
        rng, seed = resolve_numpy_rng(rng=my_rng)
        assert rng is my_rng
        assert seed is None

    def test_both_seed_and_rng_raises_valueerror(self):
        with pytest.raises(ValueError, match="not both"):
            resolve_numpy_rng(seed=42, rng=np.random.default_rng(0))

    def test_bool_seed_raises_typeerror(self):
        with pytest.raises(TypeError, match="integer"):
            resolve_numpy_rng(seed=True)

    def test_float_seed_raises_typeerror(self):
        with pytest.raises(TypeError, match="integer"):
            resolve_numpy_rng(seed=3.14)  # ty: ignore[invalid-argument-type]

    def test_string_seed_raises_typeerror(self):
        with pytest.raises(TypeError, match="integer"):
            resolve_numpy_rng(seed="42")  # ty: ignore[invalid-argument-type]

    def test_same_seed_produces_identical_generators(self):
        rng1, _ = resolve_numpy_rng(seed=99)
        rng2, _ = resolve_numpy_rng(seed=99)
        assert rng1.random() == rng2.random()

    def test_different_seeds_produce_different_generators(self):
        rng1, _ = resolve_numpy_rng(seed=1)
        rng2, _ = resolve_numpy_rng(seed=2)
        assert rng1.random() != rng2.random()

    def test_negative_seed_is_not_accepted(self):
        # numpy does allow negative seeds
        with pytest.raises(ValueError, match="expected non-negative integer"):
            rng, seed = resolve_numpy_rng(seed=-1)
            assert isinstance(rng, Generator)

    def test_zero_seed_is_accepted(self):
        rng, seed = resolve_numpy_rng(seed=0)
        assert isinstance(rng, Generator)
        assert seed == 0

    def test_custom_field_name_in_error(self):
        with pytest.raises(ValueError, match="my_seed"):
            resolve_numpy_rng(seed=42, rng=np.random.default_rng(0), field_name="my_seed")


# =======================
# == SPAWN CHILD SEEDS ==
# =======================
class TestSpawnChildSeeds:
    def test_returns_expected_count(self):
        rng = np.random.default_rng(42)
        seeds = spawn_child_seeds(rng, 5)
        assert len(seeds) == 5

    def test_all_seeds_are_integers(self):
        rng = np.random.default_rng(0)
        seeds = spawn_child_seeds(rng, 3)
        assert all(isinstance(s, int) for s in seeds)

    def test_count_zero_returns_empty(self):
        rng = np.random.default_rng(0)
        assert spawn_child_seeds(rng, 0) == []

    def test_count_negative_returns_empty(self):
        rng = np.random.default_rng(0)
        assert spawn_child_seeds(rng, -5) == []

    def test_deterministic_with_same_rng_state(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert spawn_child_seeds(rng1, 3) == spawn_child_seeds(rng2, 3)

    def test_seeds_are_nonnegative(self):
        rng = np.random.default_rng(0)
        seeds = spawn_child_seeds(rng, 100)
        assert all(s >= 0 for s in seeds)


# ======================================
# == LINE SEGMENT THROUGH UNIT SQUARE ==
# ======================================
class TestLineSegmentThroughUnitSquare:
    def test_slope_zero_returns_horizontal_line(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(0)
        assert x0 == 0.0 and x1 == 1.0
        assert y0 == 0.5 and y1 == 0.5

    def test_slope_positive_infinity_returns_vertical_line(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(float("inf"))
        assert x0 == 0.5 and x1 == 0.5
        assert y0 == 0.0 and y1 == 1.0

    def test_slope_one_returns_diagonal(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(1.0)
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(1.0)
        assert y1 == pytest.approx(1.0)

    def test_slope_negative_one_returns_anti_diagonal(self):
        # For slope = -1 in the range -1 < slope < 1: False, so it falls in else branch
        # slope = -1 is in the else branch (slope <= -1)
        x0, y0, x1, y1 = line_segment_through_unit_square(-1.0)
        # All endpoints should be in [0, 1]
        for val in (x0, y0, x1, y1):
            assert 0.0 <= val <= 1.0

    def test_slope_two_stays_in_unit_square(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(2.0)
        for val in (x0, y0, x1, y1):
            assert 0.0 <= val <= 1.0

    def test_center_point_lies_on_line_for_various_slopes(self):
        for slope in [0.0, 0.5, 1.0, 2.0, 5.0, -0.5, -2.0]:
            x0, y0, x1, y1 = line_segment_through_unit_square(slope)
            # The line through (0.5, 0.5) should pass through the center
            # Verify that (0.5, 0.5) is on the segment
            if x1 != x0:
                t = (0.5 - x0) / (x1 - x0)
                y_at_center = y0 + t * (y1 - y0)
                assert y_at_center == pytest.approx(0.5, abs=1e-10)

    def test_negative_infinity_slope_returns_vertical_line(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(float("-inf"))
        assert x0 == 0.5 and x1 == 0.5

    def test_small_positive_slope_extends_full_width(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(0.1)
        assert x0 == pytest.approx(0.0)
        assert x1 == pytest.approx(1.0)

    def test_very_large_slope_clips_to_unit_square(self):
        x0, y0, x1, y1 = line_segment_through_unit_square(100.0)
        for val in (x0, y0, x1, y1):
            assert 0.0 <= val <= 1.0


# ==========================
# == BUILD LEGEND OPTIONS ==
# ==========================
class TestBuildLegendOptions:
    def test_returns_legend_options_instance(self):
        lo = build_legend_options()
        assert isinstance(lo, LegendOptions)

    def test_default_values(self):
        lo = build_legend_options()
        assert lo.loc == "center left"
        assert lo.bbox_to_anchor == (1.01, 0.5)
        assert lo.ncols == 1

    def test_custom_values_propagated(self):
        lo = build_legend_options(
            loc="upper right",
            ncols=2,
            fontsize=12.0,
            title="My Legend",
        )
        assert lo.loc == "upper right"
        assert lo.ncols == 2
        assert lo.fontsize == 12.0
        assert lo.title == "My Legend"


# =========================
# == SAVE LEGEND HANDLES ==
# =========================
class TestSaveLegendHandles:
    def test_empty_handles_raises_valueerror(self):
        with pytest.raises(ValueError, match="No legend handles"):
            save_legend_handles(
                handles=[],
                legend_options=LegendOptions(),
                filepath="/tmp/test_legend.png",
            )

    def test_saves_legend_to_file(self, tmp_path):
        from matplotlib.patches import Patch

        handle = Patch(facecolor="red", label="Test")
        filepath = str(tmp_path / "legend.png")
        result = save_legend_handles(
            handles=[handle],
            legend_options=build_legend_options(),
            filepath=filepath,
        )
        assert result is True
        assert (tmp_path / "legend.png").exists()


# ==============================
# == COERCE TO 1D FLOAT ARRAY ==
# ==============================
class TestCoerceTo1dFloatArray:
    def test_numpy_array_passthrough(self):
        arr = _coerce_to_1d_float_array(np.array([1, 2, 3]), field="test")
        assert arr.shape == (3,)
        assert arr.dtype == np.float64

    def test_pandas_series_coerced(self):
        ser = pd.Series([1.0, 2.0, 3.0])
        arr = _coerce_to_1d_float_array(ser, field="test")
        assert arr.shape == (3,)

    def test_pandas_dataframe_single_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        arr = _coerce_to_1d_float_array(df, field="test")
        assert arr.shape == (2,)

    def test_pandas_dataframe_multi_column_no_column_raises_valueerror(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="exactly one column"):
            _coerce_to_1d_float_array(df, field="test")

    def test_pandas_dataframe_with_column_param(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        arr = _coerce_to_1d_float_array(df, column="y", field="test")
        assert arr.shape == (1,)
        assert arr[0] == 2.0

    def test_pandas_dataframe_missing_column_raises_valueerror(self):
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            _coerce_to_1d_float_array(df, column="z", field="test")

    def test_scalar_input_returns_length_one_array(self):
        arr = _coerce_to_1d_float_array(5.0, field="test")
        assert arr.shape == (1,)
        assert arr[0] == 5.0

    def test_list_input_coerced(self):
        arr = _coerce_to_1d_float_array([1, 2, 3], field="test")
        assert arr.shape == (3,)

    def test_tuple_input_coerced(self):
        arr = _coerce_to_1d_float_array((1, 2), field="test")
        assert arr.shape == (2,)

    def test_none_input_raises_valueerror(self):
        with pytest.raises(ValueError, match="None"):
            _coerce_to_1d_float_array(None, field="test")  # ty: ignore[invalid-argument-type]

    def test_generator_input_coerced(self):
        arr = _coerce_to_1d_float_array((x for x in [1, 2, 3]), field="test")
        assert arr.shape == (3,)

    def test_2d_array_is_flattened(self):
        arr = _coerce_to_1d_float_array(np.array([[1, 2], [3, 4]]), field="test")
        assert arr.ndim == 1
        assert arr.shape == (4,)

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError, match="iterable"):
            _coerce_to_1d_float_array(object(), field="test")  # ty: ignore[invalid-argument-type]


# =====================================
# == COERCE TO 1D FINITE FLOAT ARRAY ==
# =====================================
class TestCoerceTo1dFiniteFloatArray:
    def test_filters_out_nan(self):
        arr = _coerce_to_1d_finite_float_array(np.array([1.0, float("nan"), 3.0]), field="test")
        assert arr.shape == (2,)
        np.testing.assert_array_equal(arr, [1.0, 3.0])

    def test_filters_out_inf(self):
        arr = _coerce_to_1d_finite_float_array(np.array([1.0, float("inf"), 3.0]), field="test")
        assert arr.shape == (2,)

    def test_all_finite_passthrough(self):
        arr = _coerce_to_1d_finite_float_array(np.array([1.0, 2.0, 3.0]), field="test")
        assert arr.shape == (3,)


# ===============================
# == COERCE VALUES AND WEIGHTS ==
# ===============================
class TestCoerceValuesAndWeights:
    def test_values_only_returns_unit_weights(self):
        vals, wts = _coerce_values_and_weights([1.0, 2.0, 3.0], weights=None, column=None)
        assert vals.shape == (3,)
        np.testing.assert_array_equal(wts, [1.0, 1.0, 1.0])

    def test_values_and_weights_aligned(self):
        vals, wts = _coerce_values_and_weights([1.0, 2.0], weights=[0.5, 1.5], column=None)
        assert vals.shape == (2,)
        np.testing.assert_array_equal(wts, [0.5, 1.5])

    def test_non_finite_values_filtered_with_shared_masking(self):
        vals, wts = _coerce_values_and_weights(
            [1.0, float("nan"), 3.0], weights=[1.0, 2.0, 3.0], column=None
        )
        assert vals.shape == (2,)
        assert wts.shape == (2,)
        np.testing.assert_array_equal(vals, [1.0, 3.0])
        np.testing.assert_array_equal(wts, [1.0, 3.0])

    def test_empty_values_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one entry"):
            _coerce_values_and_weights([], weights=None, column=None)

    def test_all_non_finite_values_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one finite"):
            _coerce_values_and_weights([float("nan"), float("inf")], weights=None, column=None)

    def test_weights_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same length"):
            _coerce_values_and_weights([1.0, 2.0, 3.0], weights=[1.0, 2.0], column=None)

    def test_non_finite_weights_at_finite_values_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _coerce_values_and_weights([1.0, 2.0], weights=[1.0, float("nan")], column=None)

    def test_non_finite_weights_at_non_finite_values_are_ok(self):
        # nan value -> masked out, so nan weight at that position is fine
        vals, wts = _coerce_values_and_weights(
            [1.0, float("nan")], weights=[1.0, float("nan")], column=None
        )
        assert vals.shape == (1,)
        np.testing.assert_array_equal(vals, [1.0])
        np.testing.assert_array_equal(wts, [1.0])


class TestSaveFigure:
    def test_save_figure_writes_nonempty_file(self):
        import matplotlib.pyplot as plt

        from gerrytools.plotting._figure_io import save_figure

        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmppath = f.name
        try:
            save_figure(fig, tmppath)
            assert os.path.getsize(tmppath) > 0
        finally:
            plt.close(fig)
            os.unlink(tmppath)

    def test_save_figure_accepts_custom_dpi(self):
        import matplotlib.pyplot as plt

        from gerrytools.plotting._figure_io import save_figure

        fig = plt.figure()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmppath = f.name
        try:
            save_figure(fig, tmppath, dpi=72)
            assert os.path.getsize(tmppath) > 0
        finally:
            plt.close(fig)
            os.unlink(tmppath)

    def test_save_figure_default_bbox_inches_is_tight(self):
        import matplotlib.pyplot as plt

        from gerrytools.plotting._figure_io import save_figure

        fig = plt.figure()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmppath = f.name
        try:
            save_figure(fig, tmppath)
        finally:
            plt.close(fig)
            os.unlink(tmppath)
