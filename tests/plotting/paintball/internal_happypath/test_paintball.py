import pytest

from gerrytools.plotting.data.paintball import PaintBall


def test_paintball_builds_point_and_hull_views():
    plot = PaintBall(include_legend=True)
    plot.add_voteshare_seatshare_data(
        [0.4925, 0.5233, 0.4960, 0.5259],
        [5, 10, 9, 9],
        maximum_seats=18,
    )
    plot.set_marker_options(
        size=9.0,
        color="alizarin",
        alpha=0.4,
        edgecolor="black",
        edgewidth=1.0,
        edgealpha=0.8,
        marker="s",
    )
    plot.set_hull_options(
        color="teagreen",
        alpha=0.5,
        edgecolor="green",
        edgewidth=1.2,
        edgealpha=0.9,
    )
    plot.add_lines_with_slope(slopes=[0.75], linealpha=0.5, zorder=-5, name="Custom")

    ax = plot.ax
    assert len(ax.lines) >= 3
    assert any(line.get_marker() == "s" for line in ax.lines)
    assert any(line.get_zorder() == -5 for line in ax.lines)

    hull_ax = plot.hull_ax
    assert len(hull_ax.patches) >= 1


def test_paintball_validates_data_shapes_and_ranges():
    with pytest.raises(ValueError, match="same length"):
        _pb = PaintBall()
        _pb.add_voteshare_seatshare_data([0.5], [0.5, 0.6])
    with pytest.raises(ValueError, match="vote-share values must be in \\[0, 1\\]"):
        _pb = PaintBall()
        _pb.add_voteshare_seatshare_data([1.2], [0.5])
    with pytest.raises(ValueError, match="maximum_seats must be a positive integer"):
        _pb = PaintBall()
        _pb.add_voteshare_seatshare_data([0.5], [2], maximum_seats=0)


def test_paintball_set_crosshair_options_supports_partial_updates():
    plot = PaintBall()
    plot.add_voteshare_seatshare_data([0.5], [0.5])
    original_width = plot.crosshair_width
    original_alpha = plot.crosshair_alpha

    plot.set_crosshair_options(color="black")
    assert plot.crosshair_color == "black"
    assert plot.crosshair_width == original_width
    assert plot.crosshair_alpha == original_alpha
