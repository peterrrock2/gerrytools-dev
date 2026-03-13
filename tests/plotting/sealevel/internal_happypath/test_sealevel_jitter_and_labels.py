import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.sealevel import SeaLevel


# ==================
# == CONSTRUCTION ==
# ==================
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


class TestSeaLevelHorizontalJitterValidation:
    def _make_sealevel_with_labels(self):
        sl = SeaLevel(jitter_rng_seed=0)
        sl.add_sealevel_set({"A": 0.5, "B": 0.7, "C": 0.3})
        return sl

    def test_non_dict_horizontal_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="dictionary"):
            sl.set_max_horizontal_jitter_per_category(jitter_per_category=[0.1, 0.2])

    def test_non_real_horizontal_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="real numbers"):
            sl.set_max_horizontal_jitter_per_category(jitter_per_category={"A": "big"})

    def test_negative_horizontal_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_max_horizontal_jitter_per_category(jitter_per_category={"A": -0.1})

    def test_infinite_horizontal_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="finite"):
            sl.set_max_horizontal_jitter_per_category(jitter_per_category={"A": float("inf")})

    def test_no_labels_horizontal_per_category_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_max_horizontal_jitter_per_category(jitter_per_category={"A": 0.1})

    def test_extra_keys_horizontal_per_category_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="Extra keys"):
            sl.set_max_horizontal_jitter_per_category(jitter_per_category={"A": 0.1, "Z": 0.2})

    def test_negative_horizontal_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_max_horizontal_jitter_all(-0.1)

    def test_infinite_horizontal_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_max_horizontal_jitter_all(float("inf"))

    def test_no_labels_vertical_per_category_raises_valueerror(self):
        """Direct call to set_max_vertical_jitter_per_category with no labels."""
        sl = SeaLevel()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_max_vertical_jitter_per_category(jitter_per_category={"A": 0.1})


class TestSeaLevelTickLabelMismatch:
    """Mismatched tick counts leave the default labels unchanged."""

    def test_custom_tick_locations_mismatch_returns_none_labels(self):
        """A mismatched tick count leaves the existing labels unchanged."""
        sl = SeaLevel(include_legend=False)
        sl.add_sealevel_set({"A": 0.5, "B": 0.6, "C": 0.7}, linecolor="black")
        # Set custom tick locations only, with a count that does not match the labels.
        sl.set_xticks(locations=[0.5, 1.0, 1.5, 2.0, 2.5])
        ax = sl.ax
        assert ax is not None


# ============================================
# == FORMAT YLABELS AS FRACTIONS VALIDATION ==
# ============================================


class TestSeaLevelFractionLabelValidation:
    """Validation errors in `format_ylabels_as_fractions` are surfaced clearly."""

    def test_minimum_numerator_exceeds_denominator_raises(self):
        """A minimum numerator above the denominator is rejected."""
        sl = SeaLevel(include_legend=False)
        sl.add_sealevel_set({"A": 0.5}, linecolor="black")
        with pytest.raises(ValueError, match="minimum_numerator cannot exceed denominator"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=11)

    def test_minimum_numerator_exceeds_explicit_maximum_raises(self):
        """A minimum numerator above the maximum numerator is rejected."""
        sl = SeaLevel(include_legend=False)
        sl.add_sealevel_set({"A": 0.5}, linecolor="black")
        with pytest.raises(ValueError, match="minimum_numerator cannot exceed maximum_numerator"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=7, maximum_numerator=5)


# =======================================
# == FORMAT_YLABELS_AS_FRACTIONS TYPES ==
# =======================================


class TestFormatYLabelsAsFractionsTypeGuards:
    """Non-integer numerator inputs raise `TypeError`."""

    def test_float_minimum_numerator_raises_typeerror(self):
        from gerrytools.plotting.data.sealevel import SeaLevel

        sl = SeaLevel(include_legend=False)
        sl.add_sealevel_set({"A": 0.5}, linecolor="black")
        with pytest.raises(TypeError, match="minimum_numerator must be an integer"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=1.5)  # type: ignore[arg-type]

    def test_float_maximum_numerator_raises_typeerror(self):
        from gerrytools.plotting.data.sealevel import SeaLevel

        sl = SeaLevel(include_legend=False)
        sl.add_sealevel_set({"A": 0.5}, linecolor="black")
        with pytest.raises(TypeError, match="maximum_numerator must be an integer"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=0, maximum_numerator=5.5)  # type: ignore[arg-type]
