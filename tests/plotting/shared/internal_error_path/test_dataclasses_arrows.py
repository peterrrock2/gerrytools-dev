import pytest

from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowData,
    ArrowPlacement,
    ArrowTextStyle,
    LabelArrowStyle,
    TextArrowStyle,
)


# ==============
# == LINEDATA ==
# ==============
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
        assert style.arrowedgewidth == 1.0
        assert style.boxpad == 0.3

    def test_negative_arrowedgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TextArrowStyle(arrowedgewidth=-1.0)

    def test_infinite_arrowedgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TextArrowStyle(arrowedgewidth=float("inf"))

    def test_negative_boxpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TextArrowStyle(boxpad=-0.1)

    def test_infinite_boxpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TextArrowStyle(boxpad=float("inf"))

    def test_outlinecolor_none_string_resets_outlinewidth_to_zero(self):
        style = TextArrowStyle(arrowedgecolor="none", arrowedgewidth=2.0)
        assert style.arrowedgewidth == 0.0

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

    def test_negative_arrowedgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(arrowedgewidth=-0.5)

    def test_outlinecolor_none_resets_outlinewidth_to_zero(self):
        style = LabelArrowStyle(arrowedgecolor="none", arrowedgewidth=3.0)
        assert style.arrowedgewidth == 0.0

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
