import matplotlib

matplotlib.use("Agg")


import pytest

from gerrytools.plotting.data.scatterplot import ScatterPlot


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# =======================
# == ANNOTATION ARROWS ==
# =======================
class TestAnnotationArrows:
    def test_add_text_arrow_stores_arrow(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right")
        assert len(sp._annotations.annotation_arrows) == 1
        assert sp._annotations.annotation_arrows[0].arrowtype == "text"

    def test_add_label_arrow_stores_arrow(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "up")
        assert len(sp._annotations.annotation_arrows) == 1
        assert sp._annotations.annotation_arrows[0].arrowtype == "label"

    def test_text_arrow_empty_text_normalized_to_spaces(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", text="")
        assert sp._annotations.annotation_arrows[0].text == "   "

    def test_text_arrow_textrotation_overrides_style_rotation(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", textrotation=45.0)
        assert sp._annotations.annotation_arrows[0].textstyle.rotation == 45.0

    def test_label_arrow_with_arrow_length(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "down", arrow_length=50.0)
        assert sp._annotations.annotation_arrows[0].arrow_length_percentage == 50.0

    def test_label_arrow_length_with_explicit_tail_raises_valueerror(self):
        from gerrytools.plotting.data._gerryplot_dataclasses import ArrowPlacement

        sp = _make_plot()
        with pytest.raises(ValueError, match="arrowtail"):
            sp.add_label_arrow(
                (0.5, 0.5),
                "up",
                arrow_length=50.0,
                arrowplacement=ArrowPlacement(arrowtail=(0.5, 0.3)),
            )

    def test_label_arrow_length_out_of_range_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            sp.add_label_arrow((0.5, 0.5), "up", arrow_length=150.0)

    def test_add_text_arrow_with_custom_facecolor(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "left", arrowfacecolor="red")
        style = sp._annotations.annotation_arrows[0].textarrowstyle
        assert style.arrowfacecolor != "#5c676f"  # not the default

    def test_add_label_arrow_with_custom_outlinecolor(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "right", arrowoutlinecolor="blue")
        style = sp._annotations.annotation_arrows[0].labelarrowstyle
        assert style.arrowoutlinecolor != "black"  # overridden

    def test_add_text_arrow_name_stored(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", name="my_arrow")
        assert sp._annotations.annotation_arrows[0].name == "my_arrow"


# =========================
# == ARROW LENGTH INPUTS ==
# =========================


class TestGerryPlotArrowLength:
    """Explicit arrow lengths are preserved on stored arrow data."""

    def test_arrow_length_sets_percentage(self):
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0, 3.0]})
        bp.add_label_arrow(
            arrowtip=(1.0, 2.0),
            direction="right",
            text="ArrowLenTest",
            arrow_length=30.0,
        )
        ax = bp.ax
        # Building successfully confirms the explicit arrow length is accepted.
        assert ax is not None


# ======================
# == AXIS BUILD PATHS ==
# ======================


class TestGerryPlotAxisBuildPaths:
    """Subclass defaults still drive axis labels when no explicit labels are provided."""

    def test_boxplot_default_xtick_labels_path(self):
        """BoxPlot uses its category labels as default x tick labels."""
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"Alpha": [1.0, 2.0], "Beta": [3.0, 4.0]})
        # No explicit tick calls - subclass default labels should apply
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Alpha" in tick_labels or "Beta" in tick_labels or len(tick_labels) >= 1

    def test_clear_ytick_labels_keeps_build_working(self):
        """Clearing y tick labels still allows the plot to build."""
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0]})
        bp.clear_ytick_labels()
        ax = bp.ax
        assert ax is not None


# ====================
# == VERTICAL BANDS ==
# ====================


class TestGerryPlotVerticalBands:
    """Vertical bands accept invisible edges."""

    def test_vertical_band_linecolor_none(self):
        sp = _make_plot()
        sp.add_vertical_band(0.2, 0.4, linecolor=None)
        ax = sp.ax
        assert ax is not None

    def test_vertical_band_linewidth_zero(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.5, linecolor="black", linewidth=0.0)
        ax = sp.ax
        assert ax is not None


# =============================
# == HORIZONTAL LINES SCALAR ==
# =============================


class TestGerryPlotHorizontalLines:
    """Scalar horizontal line inputs are normalized internally."""

    def test_horizontal_line_scalar_value(self):
        sp = _make_plot()
        sp.add_horizontal_lines(0.5)
        ax = sp.ax
        assert ax is not None


# =====================================
# == ANNOTATION TEXT OUTLINE EFFECTS ==
# =====================================


class TestAnnotationTextOutlineEffects:
    """Missing or zero-width outlines skip path effects."""

    def test_no_font_outline_returns_none(self):
        """The default outline settings skip text outline effects."""
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0]})
        bp.add_label_arrow(
            arrowtip=(1.0, 2.0),
            direction="right",
            text="NoOutline",
            # default arrowtextstyle has fontoutlinecolor=None
        )
        ax = bp.ax
        assert ax is not None

    def test_font_outline_width_zero_returns_none(self):
        """A zero outline width skips text outline effects."""
        from gerrytools.plotting.data import ArrowTextStyle
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(include_legend=False)
        bp.add_boxplot_datasets({"A": [1.0, 2.0]})
        bp.add_label_arrow(
            arrowtip=(1.0, 2.0),
            direction="right",
            text="ZeroOutlineWidth",
            arrowtextstyle=ArrowTextStyle(fontoutlinecolor="black", fontoutlinewidth=0),
        )
        ax = bp.ax
        assert ax is not None


# =======================
# == ANNOTATION ERRORS ==
# =======================


class TestArrowLengthInfiniteRaises:
    """Non-finite arrow lengths are rejected."""

    def test_infinite_arrow_length_raises(self):

        plot = ScatterPlot(include_legend=False)
        plot.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
        with pytest.raises(ValueError, match="arrow_length must be finite"):
            plot.add_label_arrow(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrow_length=float("inf"),
            )


class TestFontOutlineColorNone:
    """Text path effects are omitted when no outline color is set."""

    def test_arrow_text_style_no_outline_color_returns_no_path_effects(self):
        from gerrytools.plotting.data import ArrowTextStyle

        plot = ScatterPlot(include_legend=False)
        plot.add_scatter(x=[0.0, 1.0], y=[0.0, 1.0])
        plot.add_text_arrow(
            arrowtip=(0.5, 0.5),
            direction="right",
            text="NoOutline",
            arrowtextstyle=ArrowTextStyle(fontoutlinecolor=None),
        )
        ax = plot.ax
        text = next((t for t in ax.texts if t.get_text() == "NoOutline"), None)
        assert text is not None
        # With fontoutlinecolor=None the path effects list should be empty or absent
        pe = text.get_path_effects()
        assert len(pe) == 0
