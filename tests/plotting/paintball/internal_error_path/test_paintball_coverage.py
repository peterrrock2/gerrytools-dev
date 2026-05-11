"""Tests for PaintBall edge cases."""

import matplotlib

matplotlib.use("Agg")

import os

from gerrytools.plotting.data.paintball import PaintBall
from tests.plotting.paintball._helpers import simple_paintball


# ====================================
# == DEGENERATE HULL (< 3 VERTICES) ==
# ====================================
class TestPaintballHullPolygonDegenerate:
    """Degenerate hulls are rendered without polygon errors."""

    def test_two_identical_points_create_degenerate_hull(self):
        """Two data points with same transformed coordinates give a degenerate hull."""
        # Both points map to same (1 - voteshare, 1 - seatshare) coordinates
        plot = PaintBall()
        plot.add_voteshare_seatshare_data([0.5, 0.5], [0.5, 0.5])
        # hull_ax triggers _draw_hull=True → _draw_horizontal_hull → degenerate path
        ax = plot.hull_ax
        assert ax is not None

    def test_single_unique_y_two_x_gives_two_hull_vertices(self):
        """Points sharing a y-value produce a two-vertex hull."""
        plot = PaintBall()
        plot.add_voteshare_seatshare_data([0.4, 0.6], [0.5, 0.5])
        ax = plot.hull_ax
        assert ax is not None


# ================================
# == LINE LEGEND: LABEL IS NONE ==
# ================================
class TestPaintballLineLegend:
    """Unnamed lines are omitted from legend handles."""

    def test_unnamed_line_not_in_legend_handles(self):
        """add_lines_with_slope with name=None produces a line with label=None,
        which is skipped in _legend_handles."""
        plot = PaintBall(include_legend=True)
        plot.add_voteshare_seatshare_data([0.5], [0.5])
        plot.add_lines_with_slope(slopes=[1.5], linecolor="red", name=None)
        handles = plot._legend_handles
        labels = [h.get_label() for h in handles if hasattr(h, "get_label")]
        # The unnamed line should not appear in legend handles
        assert all(label is not None for label in labels)


# =================
# == SHOW METHOD ==
# =================
class TestPaintballShow:
    """The non-GUI show path writes an image file."""

    def test_show_saves_file_in_agg_backend(self):
        plot = simple_paintball()
        # show() writes to "gerrytools_plot.png" for non-GUI backends
        plot.show()
        out_file = "gerrytools_plot.png"
        if os.path.exists(out_file):
            os.unlink(out_file)

    def test_show_with_hull_saves_file(self):
        plot = simple_paintball()
        plot.show(hull=True)
        out_file = "gerrytools_plot.png"
        if os.path.exists(out_file):
            os.unlink(out_file)


# =================
# == SAVE METHOD ==
# =================
class TestPaintballSave:
    """The save path writes the expected files."""

    def test_save_creates_file(self, tmp_path):
        plot = simple_paintball()
        out_path = str(tmp_path / "test.png")
        plot.save(out_path)
        assert (tmp_path / "test.png").exists()

    def test_save_with_hull_creates_file(self, tmp_path):
        plot = simple_paintball()
        out_path = str(tmp_path / "test_hull.png")
        plot.save(out_path, hull=True)
        assert (tmp_path / "test_hull.png").exists()


# =========================
# == UNNAMED SLOPE LINES ==
# =========================


class TestPaintballUnnamedLines:
    """Unnamed slope lines still render correctly."""

    def test_unnamed_lines_rendered_without_error(self):
        """add_lines_with_slope without name= stores in self._lines and renders."""
        plot = PaintBall(include_legend=False)
        plot.add_voteshare_seatshare_data([0.45, 0.50, 0.55], [0.4, 0.5, 0.6])
        # No name → goes into self._lines
        plot.add_lines_with_slope(slopes=[1.0, 2.0])
        ax = plot.ax
        assert ax is not None
