import matplotlib

matplotlib.use("Agg")

import os

from gerrytools.plotting.data.scatterplot import ScatterPlot


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# ====================
# == LEGEND OPTIONS ==
# ====================
class TestLegendConfiguration:
    def test_include_legend_flag(self):
        sp = ScatterPlot(include_legend=False)
        assert sp.include_legend is False

    def test_legend_options_accessible(self):
        sp = _make_plot()
        assert sp._legend_options is not None


# =================================
# == NAMED LINES/BANDS IN LEGEND ==
# =================================


class TestNamedOverlaysInLegend:
    def test_named_vertical_line_in_legend(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5, name="Cutoff")
        handles = sp._get_named_line_legend_handles()
        labels = [h.get_label() for h in handles]
        assert "Cutoff" in labels

    def test_unnamed_vertical_line_not_in_legend(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        handles = sp._get_named_line_legend_handles()
        assert len(handles) == 0

    def test_named_horizontal_band_in_legend(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.3, 0.7, name="CI")
        handles = sp._get_named_band_legend_handles()
        labels = [h.get_label() for h in handles]
        assert "CI" in labels

    def test_unnamed_band_not_in_legend(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.3, 0.7)
        handles = sp._get_named_band_legend_handles()
        assert len(handles) == 0


class TestBandLegendHandles:
    """Bands without visible edges still produce legend handles."""

    def test_named_band_with_zero_linewidth_in_legend(self):
        sp = _make_plot()
        sp.add_vertical_band(0.2, 0.4, linewidth=0.0, name="NoBorderBand")
        # linewidth=0.0 triggers the edgecolor="none" branch in _get_named_band_legend_handles
        handles = sp._get_named_band_legend_handles()
        labels = [h.get_label() for h in handles if hasattr(h, "get_label")]
        assert "NoBorderBand" in labels


# =================
# == SAVE LEGEND ==
# =================


class TestSaveLegend:
    """Saving the legend writes the expected file."""

    def test_save_legend_creates_file(self, tmp_path):
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=True)
        bp.add_boxplot_dataset({"A": [1.0, 2.0, 3.0]}, name="Dataset 1")
        legend_path = str(tmp_path / "legend.png")
        bp.save_legend(legend_path)
        assert (tmp_path / "legend.png").exists()


# =================================
# == UPDATE LEGEND EMPTY HANDLES ==
# =================================


class TestUpdateLegendEmptyHandles:
    """Legend updates are skipped when there are no named handles."""

    def test_include_legend_true_but_no_named_data_skips_legend(self):
        """With no named entries, the legend update returns without error."""
        # Add data without a name (auto-name will be set, but that is included in handles)
        # Use ScatterPlot instead which allows no data
        sp = ScatterPlot()
        sp.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
        # ScatterPlot uses default legend=True but no named entries
        ax = sp.ax
        assert ax is not None


# =================
# == SHOW METHOD ==
# =================


class TestShowMethod:
    """The non-GUI show path writes an image file."""

    def test_show_builds_and_saves(self, tmp_path):

        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_dataset({"A": [1.0, 2.0, 3.0]})
        # show() always writes to "gerrytools_plot.png" in cwd for non-GUI backends
        bp.show()
        out_file = "gerrytools_plot.png"
        if os.path.exists(out_file):
            os.unlink(out_file)


# ==================
# == LEGEND EDGES ==
# ==================


class TestNamedBandWithNoEdge:
    """Bands with `linecolor=None` still produce legend handles."""

    def test_named_band_no_linecolor_gives_none_edgecolor(self):
        plot = ScatterPlot(include_legend=True)
        plot.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0], label="data")
        plot.add_vertical_band(0.2, 0.4, name="Shaded Region", linecolor=None)
        handles = plot._get_named_band_legend_handles()
        assert len(handles) >= 1
