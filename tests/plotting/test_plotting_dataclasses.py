"""Tests for plotting data classes and their __post_init__ validation logic.

Covers: LineData, BandData, PointSetData, ArrowTextStyle, TextArrowStyle,
LabelArrowStyle, ArrowPlacement, ArrowData, BoxPlotSetData, ViolinPlotSetData,
SeaLevelSetData, HistogramData, ScatterData, SeatsVotesData, SVPlotLine,
PaintBallLine.
"""

import math

import numpy as np
import pytest

from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowData,
    ArrowPlacement,
    ArrowTextStyle,
    BandData,
    LabelArrowStyle,
    LineData,
    PointSetData,
    TextArrowStyle,
)
from gerrytools.plotting.data.boxplot import BoxPlotSetData
from gerrytools.plotting.data.histogram import HistogramData
from gerrytools.plotting.data.paintball import PaintBallLine
from gerrytools.plotting.data.scatterplot import ScatterData
from gerrytools.plotting.data.sealevel import SeaLevelSetData
from gerrytools.plotting.data.seatsvotes import SeatsVotesData, SVPlotLine
from gerrytools.plotting.data.violin import ViolinPlotSetData
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
class TestArrowTextStyle:
    def test_default_construction(self):
        ts = ArrowTextStyle()
        assert ts.fontsize == 10.0
        assert ts.fontcolor == "#ffffff"
        assert ts.fontoutlinewidth == 0.5

    def test_negative_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowTextStyle(fontsize=-1.0)

    def test_nan_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(fontsize=float("nan"))

    def test_infinite_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(fontsize=float("inf"))

    def test_negative_fontoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowTextStyle(fontoutlinewidth=-0.1)

    def test_infinite_fontoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(fontoutlinewidth=float("inf"))

    def test_nan_rotation_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(rotation=float("nan"))

    def test_none_rotation_is_valid(self):
        ts = ArrowTextStyle(rotation=None)
        assert ts.rotation is None

    def test_rotation_coerced_to_float(self):
        ts = ArrowTextStyle(rotation=90)
        assert isinstance(ts.rotation, float)
        assert ts.rotation == 90.0

    def test_fontoutlinecolor_none_resets_fontoutlinewidth_to_zero(self):
        ts = ArrowTextStyle(fontoutlinecolor=None, fontoutlinewidth=2.0)
        assert ts.fontoutlinewidth == 0.0

    def test_fontoutlinecolor_none_string_resets_fontoutlinewidth_to_zero(self):
        ts = ArrowTextStyle(fontoutlinecolor="none", fontoutlinewidth=2.0)
        assert ts.fontoutlinewidth == 0.0

    def test_zero_fontsize_is_valid(self):
        ts = ArrowTextStyle(fontsize=0.0)
        assert ts.fontsize == 0.0


# ====================
# == TEXTARROWSTYLE ==
# ====================
class TestTextArrowStyle:
    def test_default_construction(self):
        style = TextArrowStyle()
        assert style.arrowoutlinewidth == 1.0
        assert style.boxpad == 0.3

    def test_negative_arrowoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TextArrowStyle(arrowoutlinewidth=-1.0)

    def test_infinite_arrowoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TextArrowStyle(arrowoutlinewidth=float("inf"))

    def test_negative_boxpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TextArrowStyle(boxpad=-0.1)

    def test_infinite_boxpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TextArrowStyle(boxpad=float("inf"))

    def test_outlinecolor_none_string_resets_outlinewidth_to_zero(self):
        style = TextArrowStyle(arrowoutlinecolor="none", arrowoutlinewidth=2.0)
        assert style.arrowoutlinewidth == 0.0

    def test_zero_boxpad_is_valid(self):
        style = TextArrowStyle(boxpad=0.0)
        assert style.boxpad == 0.0


# =====================
# == LABELARROWSTYLE ==
# =====================
class TestLabelArrowStyle:
    def test_default_construction(self):
        style = LabelArrowStyle()
        assert style.arrowstyle == "-|>"
        assert style.arrowhead_scale == 12.0

    def test_negative_mutation_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(arrowhead_scale=-1.0)

    def test_infinite_mutation_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LabelArrowStyle(arrowhead_scale=float("inf"))

    def test_negative_shrink_a_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(shrink_a=-1.0)

    def test_negative_shrink_b_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(shrink_b=-1.0)

    def test_infinite_shrink_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LabelArrowStyle(shrink_a=float("inf"))

    def test_negative_arrowoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(arrowoutlinewidth=-0.5)

    def test_outlinecolor_none_resets_outlinewidth_to_zero(self):
        style = LabelArrowStyle(arrowoutlinecolor="none", arrowoutlinewidth=3.0)
        assert style.arrowoutlinewidth == 0.0

    def test_zero_shrink_values_are_valid(self):
        style = LabelArrowStyle(shrink_a=0.0, shrink_b=0.0)
        assert style.shrink_a == 0.0
        assert style.shrink_b == 0.0


# ====================
# == ARROWPLACEMENT ==
# ====================
class TestArrowPlacement:
    def test_default_construction(self):
        ap = ArrowPlacement()
        assert ap.coordinate_system == "data"
        assert ap.text_offset == (0.0, 0.0)
        assert ap.tail_length == 0.08
        assert ap.zorder == 20
        assert ap.clip_on is False

    def test_negative_tail_length_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowPlacement(tail_length=-0.01)

    def test_infinite_tail_length_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(tail_length=float("inf"))

    def test_negative_label_padding_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowPlacement(label_padding=-0.001)

    def test_infinite_label_padding_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(label_padding=float("inf"))

    def test_non_finite_text_offset_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(text_offset=(float("nan"), 0.0))

    def test_non_finite_arrowtail_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(arrowtail=(0.0, float("inf")))

    def test_arrowtail_none_is_valid(self):
        ap = ArrowPlacement(arrowtail=None)
        assert ap.arrowtail is None

    def test_arrowtail_tuple_is_coerced_to_floats(self):
        ap = ArrowPlacement(arrowtail=(1, 2))
        assert ap.arrowtail == (1.0, 2.0)

    def test_text_offset_coerced_to_floats(self):
        ap = ArrowPlacement(text_offset=(1, 2))
        assert ap.text_offset == (1.0, 2.0)

    def test_zorder_coerced_to_int(self):
        ap = ArrowPlacement(zorder=10.8)  # ty: ignore[invalid-argument-type]
        assert isinstance(ap.zorder, int)


# ===============
# == ARROWDATA ==
# ===============
class TestArrowData:
    def test_text_arrow_defaults_to_textarrowstyle(self):
        ad = ArrowData(arrowtip=(0.5, 0.5), direction="right", arrowtype="text")
        assert ad.textarrowstyle is not None
        assert ad.labelarrowstyle is None

    def test_label_arrow_defaults_to_labelarrowstyle(self):
        ad = ArrowData(arrowtip=(0.5, 0.5), direction="up", arrowtype="label")
        assert ad.labelarrowstyle is not None
        assert ad.textarrowstyle is None

    def test_non_finite_arrowtip_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowData(arrowtip=(float("nan"), 0.5), direction="right")

    def test_arrowtip_coerced_to_float_tuple(self):
        ad = ArrowData(arrowtip=(1, 2), direction="right")
        assert ad.arrowtip == (1.0, 2.0)

    def test_text_arrow_rejects_labelarrowstyle(self):
        with pytest.raises(ValueError, match="cannot set labelarrowstyle"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrowtype="text",
                labelarrowstyle=LabelArrowStyle(),
            )

    def test_text_arrow_rejects_arrow_length_percentage(self):
        with pytest.raises(ValueError, match="cannot set arrow_length_percentage"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrowtype="text",
                arrow_length_percentage=50.0,
            )

    def test_text_arrow_rejects_label_position(self):
        with pytest.raises(ValueError, match="cannot set label_position"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrowtype="text",
                label_position=(0.5, 0.5),
            )

    def test_text_arrow_rejects_labelfont_options(self):
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions

        with pytest.raises(ValueError, match="cannot set labelfont_options"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrowtype="text",
                labelfont_options=LabelFontOptions(),
            )

    def test_text_arrow_rejects_labelbox_options(self):
        from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions

        with pytest.raises(ValueError, match="cannot set labelbox_options"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrowtype="text",
                labelbox_options=LabelBoxOptions(),
            )

    def test_label_arrow_rejects_textarrowstyle(self):
        with pytest.raises(ValueError, match="cannot set textarrowstyle"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrowtype="label",
                textarrowstyle=TextArrowStyle(),
            )

    def test_label_arrow_rejects_arrow_length_with_explicit_tail(self):
        with pytest.raises(ValueError, match="cannot set placement.arrowtail"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrowtype="label",
                arrow_length_percentage=50.0,
                placement=ArrowPlacement(arrowtail=(0.5, 0.3)),
            )

    def test_arrow_length_percentage_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="must be in"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrowtype="label",
                arrow_length_percentage=101.0,
            )

    def test_arrow_length_percentage_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="must be in"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrowtype="label",
                arrow_length_percentage=-1.0,
            )

    def test_non_finite_label_position_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrowtype="label",
                label_position=(float("inf"), 0.5),
            )

    def test_arrow_length_percentage_zero_is_valid(self):
        ad = ArrowData(
            arrowtip=(0.5, 0.5),
            direction="up",
            arrowtype="label",
            arrow_length_percentage=0.0,
        )
        assert ad.arrow_length_percentage == 0.0

    def test_arrow_length_percentage_100_is_valid(self):
        ad = ArrowData(
            arrowtip=(0.5, 0.5),
            direction="up",
            arrowtype="label",
            arrow_length_percentage=100.0,
        )
        assert ad.arrow_length_percentage == 100.0

    def test_all_four_directions_are_valid(self):
        for direction in ("right", "left", "up", "down"):
            ad = ArrowData(arrowtip=(0.5, 0.5), direction=direction)
            assert ad.direction == direction


# ====================
# == BOXPLOTSETDATA ==
# ====================
class TestBoxPlotSetData:
    def test_default_construction(self):
        bps = BoxPlotSetData(name="test", scores_dict={"A": [1, 2, 3]}, facecolor="blue")
        assert bps.edgewidth == 0.8
        assert bps.percentiles == (1, 99)
        assert bps.showfliers is False

    def test_percentiles_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="within"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(-1, 50),
            )

    def test_percentiles_high_greater_than_100_raises_valueerror(self):
        with pytest.raises(ValueError, match="within"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(5, 101),
            )

    def test_percentiles_low_equals_high_raises_valueerror(self):
        with pytest.raises(ValueError, match="low < high"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(50, 50),
            )

    def test_percentiles_low_greater_than_high_raises_valueerror(self):
        with pytest.raises(ValueError, match="low < high"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(99, 1),
            )

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=-1.0,
            )

    def test_infinite_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=float("inf"),
            )

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        bps = BoxPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            edgecolor="none",
            edgewidth=2.0,
        )
        assert bps.edgewidth == 0.0

    def test_zorder_coerced_to_int(self):
        bps = BoxPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            zorder=3.7,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(bps.zorder, int)

    def test_edgewidth_int_coerced_to_float(self):
        bps = BoxPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            edgewidth=2,
        )
        assert isinstance(bps.edgewidth, float)


# =======================
# == VIOLINPLOTSETDATA ==
# =======================
class TestViolinPlotSetData:
    def test_default_construction(self):
        vpsd = ViolinPlotSetData(name="test", scores_dict={"A": [1, 2, 3]}, facecolor="blue")
        assert vpsd.edgewidth == 0.8

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ViolinPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=-1.0,
            )

    def test_infinite_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ViolinPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=float("inf"),
            )

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        vpsd = ViolinPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            edgecolor="none",
            edgewidth=2.0,
        )
        assert vpsd.edgewidth == 0.0


# =====================
# == SEALEVELSETDATA ==
# =====================
class TestSeaLevelSetData:
    def test_default_construction(self):
        sl = SeaLevelSetData(name="test", scores_dict={"A": 0.5}, linecolor="black")
        assert sl.linewidth == 2.0

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            SeaLevelSetData(
                name="test",
                scores_dict={"A": 0.5},
                linecolor="black",
                linewidth=-1.0,
            )

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            SeaLevelSetData(
                name="test",
                scores_dict={"A": 0.5},
                linecolor="black",
                linewidth=float("inf"),
            )

    def test_zorder_coerced_to_int(self):
        sl = SeaLevelSetData(
            name="test",
            scores_dict={"A": 0.5},
            linecolor="black",
            zorder=4.9,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(sl.zorder, int)


# ===================
# == HISTOGRAMDATA ==
# ===================
class TestHistogramData:
    def test_default_construction_with_valid_data(self):
        hd = HistogramData(
            name="test",
            values=np.array([1.0, 2.0, 3.0]),
            weights=np.array([1.0, 1.0, 1.0]),
        )
        assert hd.values.shape == (3,)
        assert hd.weights.shape == (3,)

    def test_non_finite_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            HistogramData(
                name="test",
                values=np.array([1.0, float("nan")]),
                weights=np.array([1.0, 1.0]),
            )

    def test_infinite_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            HistogramData(
                name="test",
                values=np.array([float("inf"), 2.0]),
                weights=np.array([1.0, 1.0]),
            )

    def test_empty_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="no entries"):
            HistogramData(
                name="test",
                values=np.array([]),
                weights=np.array([]),
            )

    def test_weights_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same length"):
            HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.array([1.0]),
            )

    def test_non_finite_weights_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.array([1.0, float("nan")]),
            )

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            HistogramData(
                name="test",
                values=np.array([1.0]),
                weights=np.array([1.0]),
                edgewidth=-1.0,
            )

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        hd = HistogramData(
            name="test",
            values=np.array([1.0]),
            weights=np.array([1.0]),
            edgecolor="none",
            edgewidth=2.0,
        )
        assert hd.edgewidth == 0.0

    def test_values_are_flattened(self):
        hd = HistogramData(
            name="test",
            values=np.array([[1.0, 2.0, 3.0]]),
            weights=np.array([1.0, 1.0, 1.0]),
        )
        assert hd.values.ndim == 1
        assert hd.values.shape == (3,)


# =================
# == SCATTERDATA ==
# =================
class TestScatterData:
    def test_default_construction(self):
        sd = ScatterData(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            label="test",
            marker_options=PointMarkerOptions(),
        )
        assert sd.x.shape == (2,)
        assert sd.label == "test"

    def test_shape_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same shape"):
            ScatterData(
                x=np.array([1.0, 2.0]),
                y=np.array([3.0]),
                label="test",
                marker_options=PointMarkerOptions(),
            )

    def test_multi_dimensional_arrays_raise_valueerror(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            ScatterData(
                x=np.array([[1.0, 2.0]]),
                y=np.array([[3.0, 4.0]]),
                label="test",
                marker_options=PointMarkerOptions(),
            )

    def test_empty_arrays_raise_valueerror(self):
        with pytest.raises(ValueError, match="not be empty"):
            ScatterData(
                x=np.array([]),
                y=np.array([]),
                label="test",
                marker_options=PointMarkerOptions(),
            )

    def test_none_label_is_valid(self):
        sd = ScatterData(
            x=np.array([1.0]),
            y=np.array([2.0]),
            label=None,
            marker_options=PointMarkerOptions(),
        )
        assert sd.label is None


# ====================
# == SEATSVOTESDATA ==
# ====================
class TestSeatsVotesData:
    def _make_basic(self, **overrides):
        defaults = dict(
            pov_party_vote_counts=np.array([300, 400, 600]),
            total_vote_counts=np.array([1000, 1000, 1000]),
            name="SEN20",
            linecolor="blue",
            markercolor="gold",
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

    def test_markeralpha_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="markeralpha"):
            self._make_basic(markeralpha=2.0)

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
        assert svd.resolved_markeredgecolor() == svd.markercolor

    def test_resolved_markeredgecolor_uses_override(self):
        svd = self._make_basic(markeredgecolor="red")
        assert svd.resolved_markeredgecolor() == "red"

    def test_resolved_markeredgealpha_defaults_to_markeralpha(self):
        svd = self._make_basic(markeralpha=0.7, markeredgealpha=None)
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
            markercolor="gold",
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
