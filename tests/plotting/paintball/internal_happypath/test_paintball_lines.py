import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.paintball import PaintBall


def _simple_paintball(**kwargs):
    """Create a PaintBall with minimal valid data."""
    defaults = dict(
        voteshare_data=[0.4, 0.5, 0.6],
        seats_data=[0.3, 0.5, 0.7],
    )
    defaults.update(kwargs)
    return PaintBall(**defaults)  # ty: ignore[invalid-argument-type]


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallLineManagement:
    def test_add_named_line(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[1.5], name="Custom Line")
        assert "Custom Line" in pb._named_lines
        assert pb._named_lines["Custom Line"].slope == 1.5

    def test_add_anonymous_line(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[3.0], linecolor="red")
        assert 3.0 in pb._lines
        assert len(pb._lines[3.0]) == 1

    def test_add_multiple_slopes_at_once(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[1.0, 2.0, 3.0], linecolor="blue")
        assert 1.0 in pb._lines
        assert 2.0 in pb._lines
        assert 3.0 in pb._lines

    def test_add_multiple_slopes_with_name_stores_last_only(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(slopes=[1.0, 2.0], name="Shared Name")
        # Named lines use dict, so last slope overwrites
        assert pb._named_lines["Shared Name"].slope == 2.0

    def test_add_line_with_custom_properties(self):
        pb = _simple_paintball(
            include_efficiency_gap_line=False, include_proportionality_line=False
        )
        pb.add_lines_with_slope(
            slopes=[1.0],
            linecolor="red",
            linewidth=2.5,
            linestyle="--",
            linealpha=0.7,
            zorder=5,
            name="Styled",
        )
        line = pb._named_lines["Styled"]
        assert line.linewidth == 2.5
        assert line.linestyle == "--"
        assert line.zorder == 5

    def test_clear_lines_removes_all(self):
        pb = _simple_paintball()
        pb.add_lines_with_slope(slopes=[3.0])
        pb.clear_lines()
        assert len(pb._named_lines) == 0
        assert len(pb._lines) == 0

    def test_clear_lines_leaves_data_intact(self):
        pb = _simple_paintball()
        original_data_len = len(pb._voteshare_data)
        pb.clear_lines()
        assert len(pb._voteshare_data) == original_data_len


# =========================
# == SET XLIM / SET YLIM ==
# =========================


class TestPaintBallCoordinates:
    def test_coordinates_are_one_minus_data(self):
        pb = PaintBall(voteshare_data=[0.3, 0.7], seats_data=[0.2, 0.8])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.7), pytest.approx(0.3)]
        assert ys == [pytest.approx(0.8), pytest.approx(0.2)]

    def test_center_point_maps_to_center(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.5)]
        assert ys == [pytest.approx(0.5)]

    def test_boundary_zero_maps_to_one(self):
        pb = PaintBall(voteshare_data=[0.0], seats_data=[0.0])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(1.0)]
        assert ys == [pytest.approx(1.0)]

    def test_boundary_one_maps_to_zero(self):
        pb = PaintBall(voteshare_data=[1.0], seats_data=[1.0])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.0)]
        assert ys == [pytest.approx(0.0)]


# ==============================
# == HORIZONTAL HULL VERTICES ==
# ==============================


class TestHorizontalHullVertices:
    def test_single_point_hull(self):
        pb = PaintBall(voteshare_data=[0.5], seats_data=[0.5])
        vertices = pb._horizontal_hull_vertices()
        # Single point: one y level, left_side + right_side = 2 vertices
        assert len(vertices) == 2
        # Both vertices should be the same point
        assert vertices[0] == vertices[1]

    def test_two_points_same_y(self):
        # Both have same seat share (so same transformed y)
        pb = PaintBall(voteshare_data=[0.3, 0.7], seats_data=[0.5, 0.5])
        vertices = pb._horizontal_hull_vertices()
        assert len(vertices) == 2
        # min-x on left side, max-x on right side
        xs = [v[0] for v in vertices]
        assert min(xs) <= max(xs)

    def test_three_points_different_y_produces_hull(self):
        pb = PaintBall(voteshare_data=[0.3, 0.5, 0.7], seats_data=[0.2, 0.5, 0.8])
        vertices = pb._horizontal_hull_vertices()
        # 3 unique y-levels -> left_side(3) + right_side(3) = 6 vertices
        assert len(vertices) == 6

    def test_hull_vertices_are_tuples_of_floats(self):
        pb = _simple_paintball()
        vertices = pb._horizontal_hull_vertices()
        for v in vertices:
            assert isinstance(v, tuple)
            assert len(v) == 2


# =========================
# == BUILD AND RENDERING ==
# =========================
