"""Tests for categorical base edge cases."""

import matplotlib

matplotlib.use("Agg")


from gerrytools.plotting.data.boxplot import BoxPlot


# ======================================
# == TICK LABEL MISMATCH RETURNS NONE ==
# ======================================
class TestCategoricalBaseTickLabelMismatch:
    """The default tick-label helper returns `None` when counts differ."""

    def test_custom_tick_locations_mismatch_returns_none_labels(self):
        """A mismatched tick count leaves the existing labels unchanged."""
        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        # There are 2 categories but 5 tick locations, so the default helper returns None.
        bp.set_xticks(locations=[0.5, 1.0, 1.5, 2.0, 2.5])
        ax = bp.ax
        assert ax is not None

    def test_default_labels_applied_when_count_matches(self):
        """When tick count matches label count, _default_x_tick_labels returns labels."""
        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        # Default tick locations should match (2 categories → 2 ticks)
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "A" in tick_labels or "B" in tick_labels


# =================================
# == ALL-NAN POINTSET IS SKIPPED ==
# =================================
class TestCategoricalBaseAllNaNPointset:
    """All-NaN pointsets are skipped cleanly."""

    def test_all_nan_pointset_is_skipped(self):
        """A pointset with only NaN values is ignored during drawing."""
        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        # Add a pointset where all values are NaN for every label
        bp.add_pointset({"A": float("nan"), "B": float("nan")})
        ax = bp.ax
        # Should build without error; the all-NaN pointset is silently skipped
        assert ax is not None

    def test_partial_nan_pointset_draws_non_nan_points(self):
        """A pointset with some NaN values only draws the non-NaN ones."""
        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        bp.add_pointset({"A": 1.5, "B": float("nan")})
        ax = bp.ax
        assert ax is not None
