import numpy as np
import pytest
from matplotlib.lines import Line2D

from gerrytools.plotting.data.seatsvotes import SeatsVotes


def test_seatsvotes_supports_per_series_line_and_marker_options():
    plot = SeatsVotes(include_legend=True)
    plot.add_seat_votes_data(
        pov_party_vote_shares=[0.46, 0.52, 0.57],
        name="Election A",
        linecolor="denim",
        linealpha=0.8,
        linestyle="--",
        linewidth=3.5,
        zorder=3,
        markerfacecolor="alizarin",
        markerfacealpha=0.6,
        marker="s",
        markersize=9.0,
        markeredgecolor="black",
        markeredgealpha=1.0,
        markeredgewidth=1.2,
        markerzorder=4,
        markerlabel="Result A",
    )
    plot.add_proportionality_line(
        color="grey", linealpha=0.5, linestyle=":", linewidth=1.5, zorder=-2, name="Prop"
    )
    plot.add_efficiency_gap_line(
        color="black", linealpha=0.4, linestyle="-", linewidth=1.0, zorder=-3, name="EG"
    )
    plot.add_custom_line(
        slope=0.5,
        linecolor="green",
        linealpha=0.7,
        linestyle="-.",
        linewidth=2.0,
        zorder=-4,
        name="Custom",
    )

    ax = plot.ax
    assert len(ax.lines) >= 5

    handles = plot._legend_handles
    labels = {h.get_label() for h in handles}
    assert {"Election A", "Result A", "Prop", "EG", "Custom"}.issubset(labels)

    marker_handle = next(h for h in handles if h.get_label() == "Result A")
    assert isinstance(marker_handle, Line2D)
    assert marker_handle.get_marker() == "s"
    assert marker_handle.get_markersize() == pytest.approx(9.0)
    assert marker_handle.get_markeredgewidth() == pytest.approx(1.2)

    curve_handle = next(h for h in handles if h.get_label() == "Election A")
    assert isinstance(curve_handle, Line2D)
    assert curve_handle.get_linestyle() == "--"
    assert curve_handle.get_linewidth() == pytest.approx(3.5)


def test_seatsvotes_rejects_conflicting_custom_line_name_and_label():
    plot = SeatsVotes()
    with pytest.raises(ValueError, match="name and label must match"):
        plot.add_custom_line(
            slope=1.0,
            linecolor="black",
            linestyle="-",
            linewidth=1.0,
            label="Label A",
            name="Label B",
        )


class TestSeatsVotesDataEdgeCases:
    def _base_kwargs(self):
        return dict(
            pov_party_vote_counts=np.array([400.0, 600.0]),
            total_vote_counts=np.array([1000.0, 1000.0]),
            name="Test",
            linecolor="blue",
            markerfacecolor="red",
            markerlabel="Result",
        )

    def test_infinite_markersize_raises_valueerror(self):
        from gerrytools.plotting.data.seatsvotes import SeatsVotesData

        with pytest.raises(ValueError, match="markersize must be finite"):
            SeatsVotesData(**self._base_kwargs(), markersize=float("inf"))

    def test_infinite_markeredgewidth_raises_valueerror(self):
        from gerrytools.plotting.data.seatsvotes import SeatsVotesData

        with pytest.raises(ValueError, match="markeredgewidth must be finite"):
            SeatsVotesData(**self._base_kwargs(), markeredgewidth=float("inf"))
