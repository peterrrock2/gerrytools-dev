import pytest

from gerrytools.plotting.data._gerryplot_dataclasses import (
    BandData,
    LineData,
    PointSetData,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions


# ==============
# == LINEDATA ==
# ==============
class TestLineData:
    def test_default_construction_produces_valid_line(self):
        line = LineData(values=0.5)
        assert line.linewidth == 1.0
        assert line.zorder == 3
        assert line.name is None

    def test_linewidth_coerced_from_int_to_float(self):
        line = LineData(values=0.5, linewidth=2)
        assert isinstance(line.linewidth, float)
        assert line.linewidth == 2.0

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LineData(values=0.5, linewidth=-1.0)

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LineData(values=0.5, linewidth=float("inf"))

    def test_nan_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LineData(values=0.5, linewidth=float("nan"))

    def test_linecolor_none_with_positive_width_resets_width_to_zero(self):
        line = LineData(values=0.5, linecolor="none", linewidth=3.0)
        assert line.linewidth == 0.0

    def test_linecolor_none_string_is_case_insensitive(self):
        line = LineData(values=0.5, linecolor="None", linewidth=3.0)
        assert line.linewidth == 0.0

    def test_zorder_coerced_to_int(self):
        line = LineData(values=0.5, zorder=5.9)  # ty: ignore[invalid-argument-type]
        assert isinstance(line.zorder, int)
        assert line.zorder == 5

    def test_multiple_values_accepted_as_iterable(self):
        line = LineData(values=[0.2, 0.4, 0.6])
        assert line.values == [0.2, 0.4, 0.6]

    def test_named_line_stores_name(self):
        line = LineData(values=0.5, name="threshold")
        assert line.name == "threshold"

    def test_zero_linewidth_is_valid(self):
        line = LineData(values=0.5, linewidth=0.0)
        assert line.linewidth == 0.0


# ==============
# == BANDDATA ==
# ==============


class TestBandData:
    def test_default_construction_produces_valid_band(self):
        band = BandData(lower_bound=0.4, upper_bound=0.6)
        assert band.lower_bound == 0.4
        assert band.upper_bound == 0.6

    def test_bounds_are_auto_sorted(self):
        band = BandData(lower_bound=0.8, upper_bound=0.2)
        assert band.lower_bound == 0.2
        assert band.upper_bound == 0.8

    def test_infinite_bounds_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            BandData(lower_bound=float("inf"), upper_bound=0.5)

    def test_nan_bound_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            BandData(lower_bound=float("nan"), upper_bound=0.5)

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            BandData(lower_bound=0.0, upper_bound=1.0, linewidth=-1.0)

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            BandData(lower_bound=0.0, upper_bound=1.0, linewidth=float("inf"))

    def test_linecolor_defaults_to_bandcolor_when_none(self):
        band = BandData(lower_bound=0.0, upper_bound=1.0, bandcolor="red", linecolor=None)
        # linecolor should default to bandcolor when not explicitly set
        assert band.linecolor != "none"

    def test_linecolor_none_with_bandcolor_none_falls_back_to_grey(self):
        band = BandData(lower_bound=0.0, upper_bound=1.0, bandcolor="none", linecolor=None)
        assert isinstance(band.linecolor, str)
        assert band.linecolor.lower() != "none"

    def test_linecolor_none_string_resets_linewidth_to_zero(self):
        band = BandData(lower_bound=0.0, upper_bound=1.0, linecolor="none", linewidth=2.0)
        assert band.linewidth == 0.0

    def test_zorder_coerced_to_int(self):
        band = BandData(
            lower_bound=0.0,
            upper_bound=1.0,
            zorder=7.2,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(band.zorder, int)

    def test_equal_bounds_after_sorting_are_valid(self):
        band = BandData(lower_bound=0.5, upper_bound=0.5)
        assert band.lower_bound == band.upper_bound == 0.5

    def test_negative_bounds_are_valid(self):
        band = BandData(lower_bound=-1.0, upper_bound=-0.5)
        assert band.lower_bound == -1.0
        assert band.upper_bound == -0.5


# ==================
# == POINTSETDATA ==
# ==================


class TestPointSetData:
    def test_basic_construction(self):
        ps = PointSetData(
            name="Enacted",
            values_dict={"A": 0.5, "B": 0.7},
            point_data=PointMarkerOptions(),
        )
        assert ps.name == "Enacted"
        assert ps.x_offset is None

    def test_x_offset_stores_value(self):
        ps = PointSetData(
            name="Plan",
            values_dict={"X": 1.0},
            point_data=PointMarkerOptions(),
            x_offset=0.15,
        )
        assert ps.x_offset == 0.15

    def test_frozen_cannot_be_mutated(self):
        ps = PointSetData(
            name="A",
            values_dict={"A": 1.0},
            point_data=PointMarkerOptions(),
        )
        with pytest.raises(AttributeError):
            ps.name = "B"  # ty: ignore[invalid-assignment]


# ====================
# == ARROWTEXTSTYLE ==
# ====================
