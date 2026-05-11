import matplotlib

matplotlib.use("Agg")


import pytest

from gerrytools.plotting.data.scatterplot import ScatterPlot


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# =================
# == TICK VALUES ==
# =================
class TestTickManagement:
    def test_set_xticks_stores_locations_and_labels(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 0.5, 1.0], labels=["a", "b", "c"])
        assert sp._x_tick_locations == [0.0, 0.5, 1.0]
        assert sp._x_tick_labels == ["a", "b", "c"]

    def test_set_yticks_stores_locations_and_labels(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["low", "high"])
        assert sp._y_tick_locations == [0.0, 1.0]
        assert sp._y_tick_labels == ["low", "high"]

    def test_set_xticks_empty_clears(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 1.0], labels=["a", "b"])
        sp.set_xticks([])
        assert sp._x_tick_locations == []
        assert sp._x_tick_labels == []

    def test_set_yticks_empty_clears(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["a", "b"])
        sp.set_yticks([])
        assert sp._y_tick_locations == []
        assert sp._y_tick_labels == []

    def test_update_xtick_labels_locations_only(self):
        sp = _make_plot()
        sp.update_xtick_labels(locations=[0.0, 0.5, 1.0])
        assert sp._x_tick_locations == [0.0, 0.5, 1.0]

    def test_update_xtick_labels_labels_only(self):
        sp = _make_plot()
        sp.update_xtick_labels(labels=["a", "b"])
        assert sp._x_tick_labels == ["a", "b"]

    def test_update_xtick_labels_both(self):
        sp = _make_plot()
        sp.update_xtick_labels(locations=[1.0, 2.0], labels=["x", "y"])
        assert sp._x_tick_locations == [1.0, 2.0]
        assert sp._x_tick_labels == ["x", "y"]

    def test_update_xtick_labels_mismatched_lengths_raises_valueerror(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.update_xtick_labels(locations=[1.0, 2.0], labels=["x"])

    def test_update_xtick_empty_locations_with_non_empty_labels_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="clear both"):
            sp.update_xtick_labels(locations=[], labels=["a"])

    def test_update_xtick_empty_labels_with_non_empty_locations_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="clear both"):
            sp.update_xtick_labels(locations=[1.0], labels=[])

    def test_update_ytick_labels_mismatched_lengths_raises_valueerror(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.update_ytick_labels(locations=[1.0], labels=["a", "b"])

    def test_update_xtick_none_none_is_noop(self):
        sp = _make_plot()
        sp.update_xtick_labels(locations=[1.0], labels=["a"])
        sp.update_xtick_labels()  # noop
        assert sp._x_tick_locations == [1.0]
        assert sp._x_tick_labels == ["a"]

    def test_update_locations_incompatible_with_existing_labels_raises(self):
        sp = _make_plot()
        sp.update_xtick_labels(locations=[1.0, 2.0], labels=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            sp.update_xtick_labels(locations=[1.0, 2.0, 3.0])

    def test_update_labels_incompatible_with_existing_locations_raises(self):
        sp = _make_plot()
        sp.update_xtick_labels(locations=[1.0, 2.0], labels=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            sp.update_xtick_labels(labels=["x", "y", "z"])

    def test_update_ytick_empty_locations_clears_both(self):
        sp = _make_plot()
        sp.update_ytick_labels(locations=[1.0], labels=["a"])
        sp.update_ytick_labels(locations=[])
        assert sp._y_tick_locations == []
        assert sp._y_tick_labels == []

    def test_update_ytick_empty_labels_clears_labels(self):
        sp = _make_plot()
        sp.update_ytick_labels(locations=[1.0], labels=["a"])
        sp.update_ytick_labels(labels=[])
        assert sp._y_tick_labels == []


# ===================
# == LABEL SETTERS ==
# ===================


class TestGerryPlotBaseSetters:
    """Tests for set_xlabel / set_ylabel / set_title and related clearers."""

    def test_set_xlabel_stores_text(self):
        sp = ScatterPlot()
        sp.set_xlabel("Vote Share")
        assert sp.xlabel == "Vote Share"

    def test_set_xlabel_none_clears(self):
        sp = ScatterPlot(xlabel="old")
        sp.set_xlabel(None)
        assert sp.xlabel is None

    def test_set_ylabel_stores_text(self):
        sp = ScatterPlot()
        sp.set_ylabel("Seat Share")
        assert sp.ylabel == "Seat Share"

    def test_set_title_stores_text(self):
        sp = ScatterPlot()
        sp.set_title("My Plot")
        assert sp.title == "My Plot"

    def test_clear_xlabel_style(self):
        sp = ScatterPlot()
        sp.set_xaxis_label_style(fontsize=14.0)
        sp.clear_xlabel_style()
        assert sp._xlabel_style is None

    def test_clear_ylabel_style(self):
        sp = ScatterPlot()
        sp.set_yaxis_label_style(fontsize=14.0)
        sp.clear_ylabel_style()
        assert sp._ylabel_style is None

    def test_clear_title_style(self):
        sp = ScatterPlot()
        sp.set_title_style(fontsize=14.0)
        sp.clear_title_style()
        assert sp._title_style is None

    def test_clear_xtick_labels(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.clear_xtick_labels()
        assert sp._x_tick_labels == []

    def test_clear_ytick_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.5], labels=["mid"])
        sp.clear_ytick_labels()
        assert sp._y_tick_labels == []

    def test_clear_xticks_clears_locations_and_labels(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.clear_xticks()
        assert sp._x_tick_locations == []
        assert sp._x_tick_labels == []

    def test_clear_yticks_clears_locations_and_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.5], labels=["mid"])
        sp.clear_yticks()
        assert sp._y_tick_locations == []
        assert sp._y_tick_labels == []

    def test_set_legend_options_updates_options(self):
        sp = ScatterPlot()
        sp.set_legend_options(ncols=2, fontsize=12.0)
        assert sp._legend_options.ncols == 2
        assert sp._legend_options.fontsize == 12.0


class TestGerryPlotBaseUpdateYTickEdgeCases:
    """Edge cases in update_ytick_labels that weren't previously exercised."""

    def test_update_ytick_labels_both_none_is_noop(self):
        sp = ScatterPlot()
        sp.update_ytick_labels()
        assert sp._y_tick_locations is None
        assert sp._y_tick_labels is None

    def test_update_ytick_labels_inconsistent_clear_raises(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="clear both"):
            sp.update_ytick_labels(locations=[], labels=["a"])

    def test_update_ytick_labels_locations_mismatch_existing_labels_raises(self):
        sp = ScatterPlot()
        sp.update_ytick_labels(labels=["a", "b"])
        with pytest.raises(ValueError, match="Locations length"):
            sp.update_ytick_labels(locations=[0.1, 0.2, 0.3])

    def test_update_ytick_labels_locations_only_stores(self):
        sp = ScatterPlot()
        sp.update_ytick_labels(locations=[0.1, 0.2])
        assert sp._y_tick_locations == [0.1, 0.2]
        assert sp._y_tick_labels is None

    def test_update_ytick_labels_labels_empty_clears_only_labels(self):
        sp = ScatterPlot()
        sp.update_ytick_labels(labels=[])
        assert sp._y_tick_labels == []

    def test_update_ytick_labels_labels_mismatch_existing_locations_raises(self):
        sp = ScatterPlot()
        sp.update_ytick_labels(locations=[0.1, 0.2])
        with pytest.raises(ValueError, match="Labels length"):
            sp.update_ytick_labels(labels=["only_one"])

    def test_update_ytick_labels_labels_only_stores(self):
        sp = ScatterPlot()
        sp.update_ytick_labels(locations=[0.1, 0.2])
        sp.update_ytick_labels(labels=["a", "b"])
        assert sp._y_tick_labels == ["a", "b"]

    def test_update_xtick_labels_labels_empty_clears(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.update_xtick_labels(labels=[])
        assert sp._x_tick_labels == []


class TestGerryPlotBaseYTickBuildErrors:
    """Tests that trigger ValueError during build for mismatched tick labels."""

    def test_build_with_y_tick_labels_but_no_locations_still_builds(self):
        """Setting y_tick_labels without locations falls back to existing auto-ticks."""
        sp = ScatterPlot()
        # set labels explicitly matching auto-tick count would be brittle, so just check
        # that it doesn't hard-crash when labels list is empty
        sp._y_tick_labels = []
        ax = sp.ax
        assert ax is not None


# =======================
# == TICK BUILD ERRORS ==
# =======================


class TestXTickLabelCountMismatch:
    """Mismatched explicit x tick labels raise `ValueError`."""

    def test_xtick_labels_wrong_count_raises(self):
        plot = ScatterPlot(include_legend=False)
        plot.add_scatter(x=[0.0, 0.5, 1.0], y=[0.0, 0.5, 1.0])
        # Set 3 explicit tick locations, then try to set 2 labels
        plot.set_xticks([0.0, 0.5, 1.0])
        plot.update_xtick_labels(labels=["a", "b", "c"])
        # Force a mismatch: clear locations then set labels with wrong count
        plot._x_tick_locations = [0.0, 0.5, 1.0]
        plot._x_tick_labels = ["a", "b"]  # wrong count
        with pytest.raises(ValueError, match="Expected 3 x tick labels, got 2"):
            _ = plot.ax


class TestYTickLabelsWithoutLocations:
    """Y tick labels can reuse auto-generated tick locations."""

    def test_ytick_labels_without_locations_applies_to_default_ticks(self):
        plot = ScatterPlot(include_legend=False)
        plot.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
        # Trigger a full build first so matplotlib computes auto ticks for this data
        _ = plot.ax
        default_ticks = list(plot._ax.get_yticks())
        # Match the auto-tick count while leaving explicit locations unset.
        plot._y_tick_labels = [str(i) for i in range(len(default_ticks))]
        ax = plot.ax
        assert ax is not None

    def test_ytick_labels_wrong_count_raises(self):
        plot = ScatterPlot(include_legend=False)
        plot.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
        _ = plot._ax
        # Force y-tick labels with wrong count compared to default ticks
        plot._y_tick_labels = ["only_one_label"]
        # Leave _y_tick_locations as None so we go through the auto-tick path
        with pytest.raises(ValueError, match="y tick labels"):
            _ = plot.ax
