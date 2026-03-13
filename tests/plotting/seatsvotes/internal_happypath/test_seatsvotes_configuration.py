import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.seatsvotes import SeatsVotes


def _simple_sv():
    """Create a SeatsVotes with one dataset added."""
    sv = SeatsVotes()
    sv.add_seat_votes_data([0.3, 0.6, 0.55], name="SEN20")
    return sv


# ===========================
# == CONSTRUCTION DEFAULTS ==
# ===========================
class TestSeatsVotesConstruction:
    def test_default_figure_size_is_square(self):
        sv = SeatsVotes()
        assert sv.fig.get_size_inches()[0] == sv.fig.get_size_inches()[1]

    def test_default_axis_limits_are_unit_square(self):
        sv = SeatsVotes()
        assert sv._x_limits == (0.0, 1.0)
        assert sv._y_limits == (0.0, 1.0)

    def test_crosshair_enabled_by_default(self):
        sv = SeatsVotes()
        assert sv._crosshair_settings is not None

    def test_default_election_markers_shown(self):
        sv = SeatsVotes()
        assert sv._display_election_markers is True

    def test_default_linewidth(self):
        sv = SeatsVotes()
        assert sv.linewidth == 2.5

    def test_default_markersize(self):
        sv = SeatsVotes()
        assert sv.markersize == 8.0


# =========================
# == ADD SEAT VOTES DATA ==
# =========================


class TestSeatsVotesCrosshairs:
    def test_update_crosshair_settings(self):
        sv = SeatsVotes()
        sv.update_crosshair_settings(x_width=0.05, y_width=0.05)
        assert sv._crosshair_settings is not None
        # Verify the bounds are correct
        x_settings = sv._crosshair_settings["x"]
        assert x_settings["xmin"] == pytest.approx(0.475)
        assert x_settings["xmax"] == pytest.approx(0.525)

    def test_remove_crosshairs(self):
        sv = SeatsVotes()
        sv.remove_crosshairs()
        assert sv._crosshair_settings is None


# ====================================
# == MARKER/LINE VISIBILITY TOGGLES ==
# ====================================


class TestSeatsVotesVisibility:
    def test_hide_election_markers(self):
        sv = SeatsVotes()
        sv.hide_election_markers()
        assert sv._display_election_markers is False

    def test_show_election_markers(self):
        sv = SeatsVotes()
        sv.hide_election_markers()
        sv.show_election_markers()
        assert sv._display_election_markers is True

    def test_hide_additional_lines_in_legend(self):
        sv = SeatsVotes()
        sv.hide_additional_lines_in_legend()
        assert sv._display_line_legend is False

    def test_show_additional_lines_in_legend(self):
        sv = SeatsVotes()
        sv.hide_additional_lines_in_legend()
        sv.show_additional_lines_in_legend()
        assert sv._display_line_legend is True


# =======================
# == FONT/SIZE SETTERS ==
# =======================


class TestSeatsVotesSetters:
    def test_set_tick_fontsize(self):
        sv = SeatsVotes()
        sv.set_tick_fontsize(20.0)
        assert sv._x_tick_style is not None
        assert sv._y_tick_style is not None
        assert sv._x_tick_style.size == 20.0
        assert sv._y_tick_style.size == 20.0

    def test_set_fontsize_sets_ticks_and_legend(self):
        sv = SeatsVotes()
        sv.set_fontsize(18.0)
        assert sv._x_tick_style is not None
        assert sv._x_tick_style.size == 18.0
        assert sv._legend_options.fontsize == 18.0

    def test_set_markersize(self):
        sv = SeatsVotes()
        sv.set_markersize(12.0)
        assert sv.markersize == 12.0

    def test_set_linewidth(self):
        sv = SeatsVotes()
        sv.set_linewidth(3.0)
        assert sv.linewidth == 3.0


# ================================
# == LEGEND HANDLES COMPOSITION ==
# ================================
