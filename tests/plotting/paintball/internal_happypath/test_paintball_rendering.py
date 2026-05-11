import matplotlib

matplotlib.use("Agg")


from gerrytools.plotting.data.paintball import PaintBall
from tests.plotting.paintball._helpers import simple_paintball


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallBuild:
    def test_build_point_view(self):
        pb = simple_paintball()
        ax = pb.ax
        assert ax is not None

    def test_build_hull_view(self):
        pb = simple_paintball()
        ax = pb.hull_ax
        assert ax is not None

    def test_hull_ax_restores_draw_hull_flag(self):
        pb = simple_paintball()
        assert pb._draw_hull is False
        _ = pb.hull_ax
        # After hull_ax returns, _draw_hull should be restored
        assert pb._draw_hull is False

    def test_build_with_no_default_lines(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        ax = pb.ax
        assert ax is not None

    def test_build_with_custom_scale(self):
        pb = simple_paintball()
        pb.set_scale(xscale=5.0, yscale=15.0)
        ax = pb.ax
        assert ax is not None

    def test_build_with_single_point(self):
        pb = PaintBall()
        pb.add_voteshare_seatshare_data([0.5], [0.5])
        ax = pb.ax
        assert ax is not None

    def test_hull_view_with_single_point(self):
        pb = PaintBall()
        pb.add_voteshare_seatshare_data([0.5], [0.5])
        ax = pb.hull_ax
        assert ax is not None

    def test_hull_view_with_two_colinear_points(self):
        # Two points with same y -> hull degenerates to a line (< 3 vertices)
        pb = PaintBall()
        pb.add_voteshare_seatshare_data([0.3, 0.7], [0.5, 0.5])
        ax = pb.hull_ax
        assert ax is not None


# ====================
# == LEGEND HANDLES ==
# ====================


class TestPaintBallLegendHandles:
    def test_point_view_legend_has_plan_outcomes(self):
        pb = simple_paintball()
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Plan Outcomes" in labels

    def test_point_view_legend_has_named_lines(self):
        pb = simple_paintball()
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Efficiency Gap" in labels
        assert "Proportionality" in labels

    def test_hull_view_legend_has_horizontal_hull(self):
        pb = simple_paintball()
        pb._draw_hull = True
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Horizontal Hull" in labels

    def test_hull_view_legend_excludes_plan_outcomes(self):
        pb = simple_paintball()
        pb._draw_hull = True
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Plan Outcomes" not in labels

    def test_no_named_lines_means_fewer_legend_handles(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        handles = pb._legend_handles
        # Only the Plan Outcomes handle
        assert len(handles) == 1
        assert handles[0].get_label() == "Plan Outcomes"

    def test_anonymous_lines_excluded_from_legend(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        pb.add_lines_with_slope(slopes=[3.0])  # no name
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        # Only Plan Outcomes, no label for the anonymous line
        assert len(labels) == 1

    def test_custom_named_line_appears_in_legend(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        pb.add_lines_with_slope(slopes=[1.5], name="My Guide")
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "My Guide" in labels
