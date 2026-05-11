import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.mpl.label_text_options import (
    FontFamily,
    FontStyle,
    FontWeight,
    LabelBoxOptions,
    LabelFontOptions,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class PointSetData:
    """A dataclass representing a set of points to be plotted on a boxplot figure.

    Attributes:
        name (str): The name of the point set.
        values_dict (dict[str, float]): A dictionary mapping labels to point values.
        point_data (PointMarkerOptions): The settings for the points.
        x_offset (float | None): An optional absolute x-offset from category center.
    """

    name: str
    values_dict: dict[str, float]  # one value per label
    point_data: PointMarkerOptions
    x_offset: float | None = None  # optional absolute x-offset from category center


@dataclass(frozen=True)
class LineData:
    """Data class representing a line to be drawn on a plot.

    Attributes:
        values (float | Iterable[float]): The position(s) of the line on the axis.
        linecolor (Color): The color of the line.
        linealpha (float | None): The alpha transparency of the line color.
            If None, uses the alpha from the color if specified.
        linestyle (str): The style of the line (e.g., '-', '--', '-.', ':').
        linewidth (float): The width of the line.
        zorder (int): The z-order of the line.
        name (str | None): The name of the line for legend purposes.
    """

    values: float | Iterable[float]
    linecolor: Color = "#cccccc"
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = 3
    name: str | None = None

    def __post_init__(self) -> None:
        lw = float(self.linewidth)
        if lw < 0:
            raise ValueError("LineData.linewidth must be nonnegative.")
        if not math.isfinite(lw):
            raise ValueError("LineData.linewidth must be finite.")
        object.__setattr__(self, "linewidth", lw)

        resolved_lc, resolved_la = resolve_color_and_alpha(
            self.linecolor,
            self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="LineData",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_lc)
        object.__setattr__(self, "linealpha", resolved_la)

        if resolved_lc.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "LineData: linecolor is 'none' but "
                    f"linewidth is {lw}>0; setting linewidth to 0."
                ),
            )
            object.__setattr__(self, "linewidth", 0.0)

        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class BandData:
    """Data class representing a band to be drawn on a plot.

    Attributes:
        lower_bound (float): The lower bound of the band.
        upper_bound (float): The upper bound of the band.
        bandcolor (Color): The fill color of the band.
        alpha (float | None): The alpha transparency of the band color.
            If None, uses the alpha from the color if specified.
        linecolor (Color | None): The color of the bounding lines of the band.
        linealpha (float | None): The alpha transparency of the bounding lines.
        linestyle (str): The style of the bounding lines (e.g., '-', '--', '-.', ':').
        linewidth (float): The width of the bounding lines.
        zorder (int): The z-order of the band.
        name (str | None): The name of the band for legend purposes.
    """

    lower_bound: float
    upper_bound: float
    bandcolor: Color = "#cccccc"
    bandalpha: float | None = None
    linecolor: Color | None = None
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int = 3
    name: str | None = None

    def __post_init__(self) -> None:
        lb, ub = sorted([float(self.lower_bound), float(self.upper_bound)])
        if not (math.isfinite(lb) and math.isfinite(ub)):
            raise ValueError("BandData: lower_bound and upper_bound must both be finite.")
        object.__setattr__(self, "lower_bound", lb)
        object.__setattr__(self, "upper_bound", ub)

        resolved_bc, resolved_ba = resolve_color_and_alpha(
            self.bandcolor,
            self.bandalpha,
            allow_none=True,
            field="bandcolor",
            owner="BandData",
            logger=logger,
        )
        object.__setattr__(self, "bandcolor", resolved_bc)
        object.__setattr__(self, "bandalpha", resolved_ba)

        lw = float(self.linewidth)
        if lw < 0:
            raise ValueError("BandData.linewidth must be nonnegative.")
        if not math.isfinite(lw):
            raise ValueError("BandData.linewidth must be finite.")

        # Default linecolor: follow bandcolor unless band is none (then fallback)
        normalized_line_color = self.linecolor
        if normalized_line_color is None:
            normalized_line_color = resolved_bc
            if isinstance(normalized_line_color, str) and normalized_line_color.lower() == "none":
                normalized_line_color = "#cccccc"

        # Line color + alpha
        resolved_lc, resolved_la = resolve_color_and_alpha(
            normalized_line_color,
            self.linealpha,
            allow_none=True,
            field="linecolor",
            owner="BandData",
            logger=logger,
        )
        object.__setattr__(self, "linecolor", resolved_lc)
        object.__setattr__(self, "linealpha", resolved_la)

        if resolved_lc.lower() == "none" and lw > 0:
            logger.debug(
                "BandData: linecolor is 'none' but linewidth is %s>0; setting linewidth to 0.",
                lw,
            )
            lw = 0.0

        object.__setattr__(self, "linewidth", lw)
        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class ArrowTextStyle:
    """Text styling options for annotation arrows.

    Attributes:
        fontsize (float, optional): Text size in points. Defaults to ``10.0``.
        fontcolor (Color, optional): Text color. Defaults to ``"white"``.
        fontalpha (float | None, optional): Optional alpha override for ``fontcolor``.
            Defaults to None.
        fontoutlinecolor (Color | None, optional): Optional outline color for text glyphs.
            If None, no outline is drawn. Defaults to None.
        fontoutlinealpha (float | None, optional): Optional alpha override for
            ``fontoutlinecolor``. Defaults to None.
        fontoutlinewidth (float, optional): Outline width in points.
            Defaults to ``0.0``.
        fontweight (FontWeight | None, optional): Font weight (for example ``"bold"``).
            Defaults to None.
        fontstyle (FontStyle | None, optional): Font style.
            Defaults to None.
        fontfamily (FontFamily | None, optional): Font family selection.
            Defaults to None.
        rotation (float | None, optional): Text rotation in degrees. Defaults to None.
        horizontalalignment (Literal["left", "center", "right"] | None, optional):
            Horizontal text alignment. Defaults to None.
        verticalalignment (Literal["bottom", "center", "top"] | None, optional):
            Vertical text alignment. Defaults to None.
    """

    fontsize: float = 10.0
    fontcolor: Color = "white"
    fontalpha: float | None = None
    fontoutlinecolor: Color | None = "black"
    fontoutlinealpha: float | None = None
    fontoutlinewidth: float = 0.5
    fontweight: FontWeight | None = None
    fontstyle: FontStyle | None = None
    fontfamily: FontFamily | None = None
    rotation: float | None = None
    horizontalalignment: Literal["left", "center", "right"] | None = None
    verticalalignment: Literal["bottom", "center", "top"] | None = None

    def __post_init__(self) -> None:
        size = float(self.fontsize)
        if not math.isfinite(size):
            raise ValueError("AnnotationArrowTextStyle.fontsize must be finite.")
        if size < 0:
            raise ValueError("AnnotationArrowTextStyle.fontsize must be nonnegative.")
        object.__setattr__(self, "fontsize", size)

        outlinewidth = float(self.fontoutlinewidth)
        if not math.isfinite(outlinewidth):
            raise ValueError("AnnotationArrowTextStyle.fontoutlinewidth must be finite.")
        if outlinewidth < 0:
            raise ValueError("AnnotationArrowTextStyle.fontoutlinewidth must be nonnegative.")
        object.__setattr__(self, "fontoutlinewidth", outlinewidth)

        if self.rotation is not None:
            rotation = float(self.rotation)
            if not math.isfinite(rotation):
                raise ValueError("AnnotationArrowTextStyle.rotation must be finite.")
            object.__setattr__(self, "rotation", rotation)

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=False,
            field="fontcolor",
            owner="AnnotationArrowTextStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_fc)
        object.__setattr__(self, "fontalpha", resolved_fa)

        if self.fontoutlinecolor is None:
            object.__setattr__(self, "fontoutlinealpha", None)
            if outlinewidth > 0:
                logger.debug(
                    "AnnotationArrowTextStyle: fontoutlinewidth is %s but fontoutlinecolor is None; setting fontoutlinewidth to 0.",
                    outlinewidth,
                )
                object.__setattr__(self, "fontoutlinewidth", 0.0)
            return

        resolved_oc, resolved_oa = resolve_color_and_alpha(
            self.fontoutlinecolor,
            self.fontoutlinealpha,
            allow_none=True,
            field="fontoutlinecolor",
            owner="AnnotationArrowTextStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontoutlinecolor", resolved_oc)
        object.__setattr__(self, "fontoutlinealpha", resolved_oa)
        if resolved_oc.lower() == "none" and outlinewidth > 0:
            logger.debug(
                "AnnotationArrowTextStyle: fontoutlinecolor is 'none' but fontoutlinewidth is %s>0; setting fontoutlinewidth to 0.",
                outlinewidth,
            )
            object.__setattr__(self, "fontoutlinewidth", 0.0)


@dataclass(frozen=True)
class TextArrowStyle:
    """Styling options for text arrows rendered via ``Axes.text(..., bbox=...)``.

    Attributes:
        arrowfacecolor (Color, optional): Fill color of the arrow box. Defaults to ``"#5c676f"``.
        arrowfacealpha (float | None, optional): Optional alpha override for ``arrowfacecolor``.
            Defaults to None.
        arrowedgecolor (Color, optional): Edge color of the arrow box. Defaults to ``"black"``.
        arrowedgealpha (float | None, optional): Optional alpha override for
            ``arrowedgecolor``. Defaults to None.
        arrowedgewidth (float, optional): Edge width in points. Defaults to ``1.0``.
        boxpad (float, optional): Text-box pad used by Matplotlib ``boxstyle``.
            Defaults to ``0.3``.
        boxstyle (str | None, optional): Explicit boxstyle override.
            When None, GerryPlot selects a direction-aware default. Defaults to None.
    """

    arrowfacecolor: Color = "#5c676f"
    arrowfacealpha: float | None = None
    arrowedgecolor: Color = "black"
    arrowedgealpha: float | None = None
    arrowedgewidth: float = 1.0
    boxpad: float = 0.3
    boxstyle: str | None = None

    def __post_init__(self) -> None:
        outlinewidth = float(self.arrowedgewidth)
        if not math.isfinite(outlinewidth):
            raise ValueError("TextAnnotationArrowStyle.arrowedgewidth must be finite.")
        if outlinewidth < 0:
            raise ValueError("TextAnnotationArrowStyle.arrowedgewidth must be nonnegative.")
        object.__setattr__(self, "arrowedgewidth", outlinewidth)

        boxpad = float(self.boxpad)
        if not math.isfinite(boxpad):
            raise ValueError("TextAnnotationArrowStyle.boxpad must be finite.")
        if boxpad < 0:
            raise ValueError("TextAnnotationArrowStyle.boxpad must be nonnegative.")
        object.__setattr__(self, "boxpad", boxpad)

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.arrowfacecolor,
            self.arrowfacealpha,
            allow_none=False,
            field="arrowfacecolor",
            owner="TextAnnotationArrowStyle",
            logger=logger,
        )
        object.__setattr__(self, "arrowfacecolor", resolved_fc)
        object.__setattr__(self, "arrowfacealpha", resolved_fa)

        resolved_ec, resolved_ea = resolve_color_and_alpha(
            self.arrowedgecolor,
            self.arrowedgealpha,
            allow_none=True,
            field="arrowedgecolor",
            owner="TextAnnotationArrowStyle",
            logger=logger,
        )
        object.__setattr__(self, "arrowedgecolor", resolved_ec)
        object.__setattr__(self, "arrowedgealpha", resolved_ea)

        if resolved_ec.lower() == "none" and outlinewidth > 0:
            logger.debug(
                "TextAnnotationArrowStyle: arrowedgecolor is 'none' but arrowedgewidth is %s>0; setting arrowedgewidth to 0.",
                outlinewidth,
            )
            object.__setattr__(self, "arrowedgewidth", 0.0)


@dataclass(frozen=True)
class LabelArrowStyle:
    """Styling options for label arrows rendered via ``Axes.annotate``.

    Attributes:
        arrowstyle (str, optional): Matplotlib arrowstyle string.
            Defaults to ``"-|>"``.
        connectionstyle (str | None, optional): Matplotlib connectionstyle string.
            Defaults to ``"arc3"``.
        mutation_scale (float, optional): Arrow-head scale. Defaults to ``12.0``.
        shrink_a (float, optional): Shrink amount at the text/tail end in points.
            Defaults to ``0.0``.
        shrink_b (float, optional): Shrink amount at the tip end in points.
            Defaults to ``0.0``.
        arrowfacecolor (Color, optional): Face color of the arrow head.
            Defaults to ``"#5c676f"``.
        arrowfacealpha (float | None, optional): Optional alpha override for ``arrowfacecolor``.
            Defaults to None.
        arrowedgecolor (Color, optional): Outline color of the arrow.
            Defaults to ``"black"``.
        arrowedgealpha (float | None, optional): Optional alpha override for
            ``arrowedgecolor``. Defaults to None.
        arrowedgewidth (float, optional): Arrow outline width in points.
            Defaults to ``1.0``.
        linestyle (str, optional): Arrow line style. Defaults to ``"-"``.
    """

    arrowstyle: str = "-|>"
    connectionstyle: str | None = "arc3"
    arrowhead_scale: float = 12.0
    shrink_a: float = 0.0
    shrink_b: float = 0.0
    arrowfacecolor: Color = "#5c676f"
    arrowfacealpha: float | None = None
    arrowedgecolor: Color = "black"
    arrowedgealpha: float | None = None
    arrowedgewidth: float = 1.0
    linestyle: str = "-"

    def __post_init__(self) -> None:
        mutation_scale = float(self.arrowhead_scale)
        if not math.isfinite(mutation_scale):
            raise ValueError("LabelAnnotationArrowStyle.mutation_scale must be finite.")
        if mutation_scale < 0:
            raise ValueError("LabelAnnotationArrowStyle.mutation_scale must be nonnegative.")
        object.__setattr__(self, "mutation_scale", mutation_scale)

        shrink_a = float(self.shrink_a)
        shrink_b = float(self.shrink_b)
        if not (math.isfinite(shrink_a) and math.isfinite(shrink_b)):
            raise ValueError("LabelAnnotationArrowStyle shrink values must be finite.")
        if shrink_a < 0 or shrink_b < 0:
            raise ValueError("LabelAnnotationArrowStyle shrink values must be nonnegative.")
        object.__setattr__(self, "shrink_a", shrink_a)
        object.__setattr__(self, "shrink_b", shrink_b)

        outlinewidth = float(self.arrowedgewidth)
        if not math.isfinite(outlinewidth):
            raise ValueError("LabelAnnotationArrowStyle.arrowedgewidth must be finite.")
        if outlinewidth < 0:
            raise ValueError("LabelAnnotationArrowStyle.arrowedgewidth must be nonnegative.")
        object.__setattr__(self, "arrowedgewidth", outlinewidth)

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.arrowfacecolor,
            self.arrowfacealpha,
            allow_none=False,
            field="arrowfacecolor",
            owner="LabelAnnotationArrowStyle",
            logger=logger,
        )
        object.__setattr__(self, "arrowfacecolor", resolved_fc)
        object.__setattr__(self, "arrowfacealpha", resolved_fa)

        resolved_ec, resolved_ea = resolve_color_and_alpha(
            self.arrowedgecolor,
            self.arrowedgealpha,
            allow_none=True,
            field="arrowedgecolor",
            owner="LabelAnnotationArrowStyle",
            logger=logger,
        )
        object.__setattr__(self, "arrowedgecolor", resolved_ec)
        object.__setattr__(self, "arrowedgealpha", resolved_ea)

        if resolved_ec.lower() == "none" and outlinewidth > 0:
            logger.debug(
                "LabelAnnotationArrowStyle: arrowedgecolor is 'none' but arrowedgewidth is %s>0; setting arrowedgewidth to 0.",
                outlinewidth,
            )
            object.__setattr__(self, "arrowedgewidth", 0.0)


@dataclass(frozen=True)
class ArrowPlacement:
    """Placement options for annotation arrows.

    Attributes:
        coordinate_system (Literal["data", "axes fraction"], optional): Coordinate system used
            for ``arrowtip``, ``arrowtail`` and ``text_offset``. Defaults to ``"data"``.
        text_offset (tuple[float, float], optional): Offset applied to text anchor position in
            the selected coordinate system. Defaults to ``(0.0, 0.0)``.
        label_padding (float, optional): Additional distance between a label arrow tail and its
            auto-placed text anchor, applied away from the arrow direction. Ignored when
            ``label_position`` is set explicitly. Defaults to ``0.005``.
        arrowtail (tuple[float, float] | None, optional): Explicit tail coordinate for label
            arrows. If None, GerryPlot computes tail from direction and ``tail_length``.
            Defaults to None.
        tail_length (float, optional): Tail-to-tip distance used when ``arrowtail`` is None.
            Defaults to ``0.08``.
        zorder (int, optional): Draw order. Defaults to ``20``.
        clip_on (bool, optional): Whether artists should be clipped to the axes patch.
            Defaults to False.
    """

    coordinate_system: Literal["data", "axes fraction"] = "data"
    text_offset: tuple[float, float] = (0.0, 0.0)
    label_padding: float = 0.005
    arrowtail: tuple[float, float] | None = None
    tail_length: float = 0.08
    zorder: int = 20
    clip_on: bool = False

    def __post_init__(self) -> None:
        tail_length = float(self.tail_length)
        if not math.isfinite(tail_length):
            raise ValueError("AnnotationArrowPlacement.tail_length must be finite.")
        if tail_length < 0:
            raise ValueError("AnnotationArrowPlacement.tail_length must be nonnegative.")
        object.__setattr__(self, "tail_length", tail_length)
        object.__setattr__(self, "zorder", int(self.zorder))

        label_padding = float(self.label_padding)
        if not math.isfinite(label_padding):
            raise ValueError("AnnotationArrowPlacement.label_padding must be finite.")
        if label_padding < 0:
            raise ValueError("AnnotationArrowPlacement.label_padding must be nonnegative.")
        object.__setattr__(self, "label_padding", label_padding)

        text_offset = (float(self.text_offset[0]), float(self.text_offset[1]))
        if not (math.isfinite(text_offset[0]) and math.isfinite(text_offset[1])):
            raise ValueError("AnnotationArrowPlacement.text_offset components must be finite.")
        object.__setattr__(self, "text_offset", text_offset)

        if self.arrowtail is not None:
            arrowtail = (float(self.arrowtail[0]), float(self.arrowtail[1]))
            if not (math.isfinite(arrowtail[0]) and math.isfinite(arrowtail[1])):
                raise ValueError("AnnotationArrowPlacement.arrowtail components must be finite.")
            object.__setattr__(self, "arrowtail", arrowtail)


@dataclass(frozen=True)
class ArrowData:
    """Data container for deferred arrow annotations in ``GerryPlotBase``.

    Attributes:
        arrowtip (tuple[float, float]): Arrow tip coordinate in ``placement.coordinate_system``.
        direction (Literal["right", "left", "up", "down"]): Arrow direction.
        arrowtype (Literal["text", "label"]): Arrow rendering style.
        text (str | None, optional): Optional text shown with the arrow. Defaults to None.
        textstyle (AnnotationArrowTextStyle, optional): Text style options.
            Defaults to ``AnnotationArrowTextStyle()``.
        arrow_length_percentage (float | None, optional): Optional label-arrow length as a
            percent of axes span in the arrow direction. ``0`` means zero length, and ``100``
            means one full axes width (horizontal) or height (vertical). Defaults to None.
        label_position (tuple[float, float] | None, optional): Optional explicit text-anchor
            location for label arrows. If None, GerryPlot uses the computed tail
            coordinate. Defaults to None.
        labelfont_options (LabelFontOptions | None, optional): Optional geoplot-style font
            options for label arrows. When set, these options override text color and
            typography fields from ``textstyle``. Defaults to None.
        labelbox_options (LabelBoxOptions | None, optional): Optional geoplot-style text-box
            options for label arrows. Defaults to None.
        placement (AnnotationArrowPlacement, optional): Placement options.
            Defaults to ``AnnotationArrowPlacement()``.
        textarrowstyle (TextAnnotationArrowStyle | None, optional): Style for text arrows.
            Must be set when ``arrowtype="text"``. Defaults to None.
        labelarrowstyle (LabelAnnotationArrowStyle | None, optional): Style for label arrows.
            Must be set when ``arrowtype="label"``. Defaults to None.
        name (str | None, optional): Optional identifier for callers. Defaults to None.
    """

    arrowtip: tuple[float, float]
    direction: Literal["right", "left", "up", "down"]
    arrowtype: Literal["text", "label"] = "text"
    text: str | None = None
    textstyle: ArrowTextStyle = field(default_factory=ArrowTextStyle)
    arrow_length_percentage: float | None = None
    label_position: tuple[float, float] | None = None
    labelfont_options: LabelFontOptions | None = None
    labelbox_options: LabelBoxOptions | None = None
    placement: ArrowPlacement = field(default_factory=ArrowPlacement)
    textarrowstyle: TextArrowStyle | None = None
    labelarrowstyle: LabelArrowStyle | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        arrowtip = (float(self.arrowtip[0]), float(self.arrowtip[1]))
        if not (math.isfinite(arrowtip[0]) and math.isfinite(arrowtip[1])):
            raise ValueError("AnnotationArrowData.arrowtip components must be finite.")
        object.__setattr__(self, "arrowtip", arrowtip)

        if self.label_position is not None:
            label_position = (float(self.label_position[0]), float(self.label_position[1]))
            if not (math.isfinite(label_position[0]) and math.isfinite(label_position[1])):
                raise ValueError("AnnotationArrowData.label_position components must be finite.")
            object.__setattr__(self, "label_position", label_position)

        if self.arrow_length_percentage is not None:
            arrow_length_percentage = float(self.arrow_length_percentage)
            if not math.isfinite(arrow_length_percentage):
                raise ValueError("AnnotationArrowData.arrow_length_percentage must be finite.")
            if not (0.0 <= arrow_length_percentage <= 100.0):
                raise ValueError("AnnotationArrowData.arrow_length_percentage must be in [0, 100].")
            object.__setattr__(self, "arrow_length_percentage", arrow_length_percentage)

        if self.arrowtype == "text":
            if self.labelarrowstyle is not None:
                raise ValueError(
                    "AnnotationArrowData with arrowtype='text' cannot set labelarrowstyle."
                )
            if self.arrow_length_percentage is not None:
                raise ValueError(
                    "AnnotationArrowData with arrowtype='text' cannot set arrow_length_percentage."
                )
            if self.label_position is not None:
                raise ValueError(
                    "AnnotationArrowData with arrowtype='text' cannot set label_position."
                )
            if self.labelfont_options is not None:
                raise ValueError(
                    "AnnotationArrowData with arrowtype='text' cannot set labelfont_options."
                )
            if self.labelbox_options is not None:
                raise ValueError(
                    "AnnotationArrowData with arrowtype='text' cannot set labelbox_options."
                )
            if self.textarrowstyle is None:
                object.__setattr__(self, "textarrowstyle", TextArrowStyle())
        else:
            if self.textarrowstyle is not None:
                raise ValueError(
                    "AnnotationArrowData with arrowtype='label' cannot set textarrowstyle."
                )
            if self.arrow_length_percentage is not None and self.placement.arrowtail is not None:
                raise ValueError(
                    "AnnotationArrowData with arrow_length_percentage cannot set placement.arrowtail."
                )
            if self.labelarrowstyle is None:
                object.__setattr__(self, "labelarrowstyle", LabelArrowStyle())
