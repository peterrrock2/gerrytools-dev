"""Extended tests for SeatsVotes plot behavior.

Covers: construction defaults, add_seat_votes_data validation,
vote share validation, custom lines, crosshair settings, marker/line
visibility toggles, fontsize/linewidth/markersize setters, and legend
handles composition.
"""

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
class TestAddSeatVotesData:
    def test_add_with_vote_shares_only(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.3, 0.6, 0.55])
        assert len(sv._sv_data_list) == 1

    def test_add_with_vote_counts_and_total_counts(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([300, 600, 550], total_vote_shares=[1000, 1000, 1000])
        assert len(sv._sv_data_list) == 1

    def test_vote_shares_out_of_range_without_total_raises_valueerror(self):
        sv = SeatsVotes()
        with pytest.raises(ValueError, match="vote shares"):
            sv.add_seat_votes_data([0.3, 1.5, 0.55])

    def test_negative_vote_shares_without_total_raises_valueerror(self):
        sv = SeatsVotes()
        with pytest.raises(ValueError, match="vote shares"):
            sv.add_seat_votes_data([-0.1, 0.6, 0.55])

    def test_auto_name_default(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6])
        assert sv._sv_data_list[0].name == "Election Seats-Votes Curve"

    def test_explicit_name(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6], name="GOV22")
        assert sv._sv_data_list[0].name == "GOV22"

    def test_auto_markerlabel_default(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6])
        assert sv._sv_data_list[0].markerlabel == "Election Result"

    def test_explicit_markerlabel(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6], markerlabel="2020 Outcome")
        assert sv._sv_data_list[0].markerlabel == "2020 Outcome"

    def test_linecolor_defaults_to_standard_color(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6])
        assert sv._sv_data_list[0].linecolor == sv.standard_election_color

    def test_markercolor_defaults_to_standard_color(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6])
        assert sv._sv_data_list[0].markercolor == sv.standard_marker_color

    def test_custom_linecolor(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.5, 0.6], linecolor="red")
        assert sv._sv_data_list[0].linecolor == "red"

    def test_add_multiple_datasets(self):
        sv = SeatsVotes()
        sv.add_seat_votes_data([0.3, 0.6], name="A")
        sv.add_seat_votes_data([0.4, 0.7], name="B")
        assert len(sv._sv_data_list) == 2


# ==================
# == CUSTOM LINES ==
# ==================
class TestSeatsVotesLines:
    def test_add_proportionality_line(self):
        sv = SeatsVotes()
        sv.add_proportionality_line()
        assert len(sv._line_data_list) == 1
        assert sv._line_data_list[0].slope == 1.0
        assert sv._line_data_list[0].label == "Proportionality"

    def test_add_efficiency_gap_line(self):
        sv = SeatsVotes()
        sv.add_efficiency_gap_line()
        assert len(sv._line_data_list) == 1
        assert sv._line_data_list[0].slope == 2.0
        assert sv._line_data_list[0].label == "Efficiency Gap"

    def test_add_custom_line(self):
        sv = SeatsVotes()
        sv.add_custom_line(3.0, linecolor="red", linestyle="--", linewidth=1.5)
        assert len(sv._line_data_list) == 1
        assert sv._line_data_list[0].slope == 3.0

    def test_custom_line_name_and_label_conflict_raises(self):
        sv = SeatsVotes()
        with pytest.raises(ValueError, match="must match"):
            sv.add_custom_line(
                1.0,
                linecolor="red",
                linestyle="-",
                linewidth=1.0,
                name="foo",
                label="bar",
            )

    def test_custom_line_name_and_label_same_is_ok(self):
        sv = SeatsVotes()
        sv.add_custom_line(
            1.0,
            linecolor="red",
            linestyle="-",
            linewidth=1.0,
            name="same",
            label="same",
        )
        assert sv._line_data_list[0].label == "same"

    def test_custom_line_name_only_sets_label(self):
        sv = SeatsVotes()
        sv.add_custom_line(1.0, linecolor="red", linestyle="-", linewidth=1.0, name="myline")
        assert sv._line_data_list[0].label == "myline"

    def test_custom_line_label_only_sets_label(self):
        sv = SeatsVotes()
        sv.add_custom_line(1.0, linecolor="red", linestyle="-", linewidth=1.0, label="myline")
        assert sv._line_data_list[0].label == "myline"

    def test_proportionality_line_custom_name(self):
        sv = SeatsVotes()
        sv.add_proportionality_line(name="1:1")
        assert sv._line_data_list[0].label == "1:1"


# ========================
# == CROSSHAIR SETTINGS ==
# ========================
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
class TestSeatsVotesLegend:
    def test_legend_includes_sv_curve_handle(self):
        sv = _simple_sv()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "SEN20" in labels

    def test_legend_includes_marker_handle_when_shown(self):
        sv = _simple_sv()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Election Result" in labels

    def test_legend_excludes_marker_handle_when_hidden(self):
        sv = _simple_sv()
        sv.hide_election_markers()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Election Result" not in labels

    def test_legend_includes_line_handles_when_shown(self):
        sv = _simple_sv()
        sv.add_proportionality_line()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Proportionality" in labels

    def test_legend_excludes_line_handles_when_hidden(self):
        sv = _simple_sv()
        sv.add_proportionality_line()
        sv.hide_additional_lines_in_legend()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Proportionality" not in labels

    def test_unlabeled_lines_excluded_from_legend(self):
        sv = _simple_sv()
        sv.add_custom_line(1.5, linecolor="red", linestyle="-", linewidth=1.0)  # no label
        line_handles = sv._get_line_legend_handles()
        assert len(line_handles) == 0
