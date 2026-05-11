import math

import numpy as np
import pytest

from gerrytools.plotting.data.paintball import PaintBallLine
from gerrytools.plotting.data.seatsvotes import SeatsVotesData, SVPlotLine


# ==============
# == LINEDATA ==
# ==============
class TestSeatsVotesData:
    def _make_basic(self, **overrides):
        defaults = dict(
            pov_party_vote_counts=np.array([300, 400, 600]),
            total_vote_counts=np.array([1000, 1000, 1000]),
            name="SEN20",
            linecolor="blue",
            markerfacecolor="gold",
            markerlabel="Result",
        )
        defaults.update(overrides)
        return SeatsVotesData(**defaults)  # ty: ignore[invalid-argument-type]

    def test_default_construction(self):
        svd = self._make_basic()
        assert svd.zorder == 1
        assert svd.markerzorder == 2

    def test_linealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="linealpha"):
            self._make_basic(linealpha=1.5)

    def test_linealpha_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="linealpha"):
            self._make_basic(linealpha=-0.1)

    def test_linewidth_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._make_basic(linewidth=-1.0)

    def test_linewidth_infinite_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            self._make_basic(linewidth=float("inf"))

    def test_markerfacealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="markerfacealpha"):
            self._make_basic(markerfacealpha=2.0)

    def test_markersize_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._make_basic(markersize=-1.0)

    def test_markeredgealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="markeredgealpha"):
            self._make_basic(markeredgealpha=-0.5)

    def test_markeredgewidth_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            self._make_basic(markeredgewidth=-0.5)

    def test_resolved_linewidth_uses_override_when_set(self):
        svd = self._make_basic(linewidth=5.0)
        assert svd.resolved_linewidth(default_linewidth=2.0) == 5.0

    def test_resolved_linewidth_falls_back_to_default(self):
        svd = self._make_basic(linewidth=None)
        assert svd.resolved_linewidth(default_linewidth=2.5) == 2.5

    def test_resolved_markersize_uses_override_when_set(self):
        svd = self._make_basic(markersize=10.0)
        assert svd.resolved_markersize(default_markersize=5.0) == 10.0

    def test_resolved_markersize_falls_back_to_default(self):
        svd = self._make_basic(markersize=None)
        assert svd.resolved_markersize(default_markersize=5.0) == 5.0

    def test_resolved_markeredgecolor_defaults_to_markercolor(self):
        svd = self._make_basic(markeredgecolor=None)
        assert svd.resolved_markeredgecolor() == svd.markerfacecolor

    def test_resolved_markeredgecolor_uses_override(self):
        svd = self._make_basic(markeredgecolor="red")
        assert svd.resolved_markeredgecolor() == "red"

    def test_resolved_markeredgealpha_defaults_to_markeralpha(self):
        svd = self._make_basic(markerfacealpha=0.7, markeredgealpha=None)
        assert svd.resolved_markeredgealpha() == 0.7

    def test_seats_votes_curve_values_positive_total_votes(self):
        svd = self._make_basic()
        vote_shares, seat_shares = svd.seats_votes_curve_values()
        assert vote_shares[0] == 0.0
        assert vote_shares[-1] == 1.0
        assert seat_shares[0] == 0.0

    def test_seats_votes_curve_rejects_zero_total_votes(self):
        svd = self._make_basic(total_vote_counts=np.array([0, 1000, 1000]))
        with pytest.raises(ValueError, match="positive"):
            svd.seats_votes_curve_values()

    def test_seats_votes_curve_shape_mismatch_raises_valueerror(self):
        svd = SeatsVotesData(
            pov_party_vote_counts=np.array([300, 400]),
            total_vote_counts=np.array([1000, 1000, 1000]),
            name="bad",
            linecolor="blue",
            markerfacecolor="gold",
            markerlabel="Result",
        )
        with pytest.raises(ValueError, match="same shape"):
            svd.seats_votes_curve_values()

    def test_none_linealpha_is_valid(self):
        svd = self._make_basic(linealpha=None)
        assert svd.linealpha is None

    def test_zero_linewidth_is_valid(self):
        svd = self._make_basic(linewidth=0.0)
        assert svd.linewidth == 0.0


# ================
# == SVPLOTLINE ==
# ================


class TestSVPlotLine:
    def test_default_construction(self):
        line = SVPlotLine(slope=1.0, linecolor="grey", linewidth=2.0, linestyle="--")
        assert line.slope == 1.0
        assert line.zorder == -1

    def test_nan_slope_raises_valueerror(self):
        with pytest.raises(ValueError, match="NaN"):
            SVPlotLine(slope=float("nan"), linecolor="grey", linewidth=1.0, linestyle="-")

    def test_infinite_slope_is_valid(self):
        line = SVPlotLine(slope=float("inf"), linecolor="grey", linewidth=1.0, linestyle="-")
        assert math.isinf(line.slope)

    def test_negative_infinite_slope_is_valid(self):
        line = SVPlotLine(slope=float("-inf"), linecolor="grey", linewidth=1.0, linestyle="-")
        assert line.slope == float("-inf")

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            SVPlotLine(slope=1.0, linecolor="grey", linewidth=-1.0, linestyle="-")

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            SVPlotLine(slope=1.0, linecolor="grey", linewidth=float("inf"), linestyle="-")

    def test_linealpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="linealpha"):
            SVPlotLine(slope=1.0, linecolor="grey", linewidth=1.0, linestyle="-", linealpha=2.0)

    def test_zero_slope_is_valid(self):
        line = SVPlotLine(slope=0.0, linecolor="grey", linewidth=1.0, linestyle="-")
        assert line.slope == 0.0


# ===================
# == PAINTBALLLINE ==
# ===================


class TestPaintBallLine:
    def test_default_construction(self):
        line = PaintBallLine(slope=2.0, linecolor="gray", linewidth=1.0, linestyle="-")
        assert line.slope == 2.0

    def test_nan_slope_raises_valueerror(self):
        with pytest.raises(ValueError, match="NaN"):
            PaintBallLine(slope=float("nan"), linecolor="gray", linewidth=1.0, linestyle="-")

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            PaintBallLine(slope=1.0, linecolor="gray", linewidth=-1.0, linestyle="-")

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            PaintBallLine(slope=1.0, linecolor="gray", linewidth=float("inf"), linestyle="-")

    def test_zorder_coerced_to_int(self):
        line = PaintBallLine(
            slope=1.0,
            linecolor="gray",
            linewidth=1.0,
            linestyle="-",
            zorder=-2.5,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(line.zorder, int)
