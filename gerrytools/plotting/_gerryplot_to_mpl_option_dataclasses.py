import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence, Union

import matplotlib.colors as mcolors
from matplotlib.colors import to_hex

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.typing import Color, TickType

logger = get_logger(__name__)


@dataclass(slots=True)
class PointMarkerOptions:
    """Settings for points on a matplotlib plot (or for functions that use similar artists).

    Attributes:
        markerfacecolor (Color): The fill color of the marker. Defaults to "none".
        markerfacealpha (float | None): The alpha transparency of the marker face color.
            If None, uses the alpha from the color if specified. Defaults to None.
        marker (str): The marker style. Defaults to "o".
        markersize (float): The size of the marker. Defaults to 6.0.
        markeredgecolor (Color): The edge color of the marker. Defaults to "black".
        markeredgealpha (float | None): The alpha transparency of the marker edge color.
            If None, uses the alpha from the color if specified. Defaults to None.
        markeredgewidth (float): The width of the marker edge. Defaults to 0.6.
        zorder (int): The z-order of the marker. Defaults to 4.
    """

    markerfacecolor: Color = "none"
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float = 6.0
    markeredgecolor: Color = "black"
    markeredgealpha: float | None = None
    markeredgewidth: float = 0.6
    zorder: int = 4

    def __post_init__(self) -> None:
        lw = float(self.markeredgewidth)
        if not math.isfinite(lw):
            raise ValueError("markeredgewidth must be finite")
        if lw < 0:
            raise ValueError("markeredgewidth must be nonnegative")
        object.__setattr__(self, "markeredgewidth", lw)

        s = float(self.markersize)
        if not math.isfinite(s):
            raise ValueError("markersize must be finite")
        if s < 0:
            raise ValueError("markersize must be nonnegative")
        object.__setattr__(self, "markersize", s)

        resolved_mfc, resolved_mfa = resolve_color_and_alpha(
            self.markerfacecolor,
            self.markerfacealpha,
            allow_none=True,
            field="markerfacecolor",
            owner="PointMarkerOptions",
            logger=logger,
        )

        object.__setattr__(self, "markerfacecolor", resolved_mfc)
        object.__setattr__(self, "markerfacealpha", resolved_mfa)

        resolved_mec, resolved_mea = resolve_color_and_alpha(
            self.markeredgecolor,
            self.markeredgealpha,
            allow_none=True,
            field="markeredgecolor",
            owner="PointMarkerOptions",
            logger=logger,
        )

        object.__setattr__(self, "markeredgecolor", resolved_mec)
        object.__setattr__(self, "markeredgealpha", resolved_mea)

        if resolved_mec.lower() == "none" and lw > 0:
            logger.log(
                level=logging.DEBUG,
                msg=(
                    "PointMarkerOptions: markeredgecolor is 'none' but "
                    f"markeredgewidth is {lw}>0; setting markeredgewidth to 0."
                ),
            )
            object.__setattr__(self, "markeredgewidth", 0.0)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert the PointMarkerOptions to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the PointMarkerOptions that
                can be passed to Matplotlib plot functions.
        """
        # Matplotlib alpha applies to the entire marker, so we need to
        # apply alpha to the facecolor and edgecolor separately to get things to
        # work as expected.
        return {
            "markerfacecolor": mcolors.to_rgba(self.markerfacecolor, alpha=self.markerfacealpha),
            "marker": self.marker,
            "markersize": self.markersize,
            "markeredgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "markeredgewidth": self.markeredgewidth,
            "zorder": self.zorder,
        }

    def to_mpl_scatter_settings_dict(self):
        """Convert the PointMarkerOptions to a dictionary for use with plt.scatter.

        Note: Does not include 'markerfacecolor' since that is typically set via the 'c' parameter
        in plt.scatter and is assigned per-point.

        Note: In Matplotlib's scatter function, the size parameter 's' is specified as the area
        of the marker in points squared. With the way that markersize is defined in
        MatPlotlib's scatter, we need to square the markersize to get the correct area.

        Returns:
            dict[str, Any]: A dictionary representation of the PointMarkerOptions that
                can be passed to Matplotlib's scatter function.
        """
        return {
            "marker": self.marker,
            "s": self.markersize**2,
            "edgecolor": mcolors.to_rgba(self.markeredgecolor, alpha=self.markeredgealpha),
            "linewidths": self.markeredgewidth,
            "zorder": self.zorder,
        }


@dataclass(frozen=True)
class TickStyle:
    """Data class representing the style of axis ticks.

    Attributes:
        size (float | int): The size of the ticks. Defaults to 10.
        rotation (float | int): The rotation angle of the tick labels in degrees. Defaults to 0.
        fontcolor (Color): The color of the tick labels. Defaults to "black".
        fontalpha (float | None): The alpha transparency of the tick label color.
            If None, uses the alpha from the color if specified. Defaults to None.
        tickcolor (Color): The color of the ticks. Defaults to "black".
        tickalpha (float | None): The alpha transparency of the tick color.
            If None, uses the alpha from the color if specified. Defaults to None.
        fontweight (str): The weight of the tick label font. Defaults to "normal".
        fontstyle (str): The style of the tick label font. Defaults to "normal".
        fontfamily (str): The family of the tick label font. Defaults to "sans-serif".
        ticktype (TickType): The type of ticks to apply the style to. Defaults to "major".
    """

    size: float | int = 10
    rotation: float | int = 0
    fontcolor: Color = "black"
    fontalpha: float | None = None
    tickcolor: Color = "black"
    tickalpha: float | None = None
    fontweight: str = "normal"
    fontstyle: str = "normal"
    fontfamily: str = "sans-serif"
    ticktype: TickType = "major"

    def __post_init__(self) -> None:
        if not isinstance(self.size, (int, float)):
            raise TypeError("TickStyle.size must be a float or int.")
        if not math.isfinite(self.size):
            raise ValueError("TickStyle.size must be finite.")
        if not float(self.size) >= 0:
            raise ValueError("TickStyle.size must be nonnegative.")

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="TickStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_fc)
        object.__setattr__(self, "fontalpha", resolved_fa)

        resolved_tc, resolved_ta = resolve_color_and_alpha(
            self.tickcolor,
            self.tickalpha,
            allow_none=True,
            field="tickcolor",
            owner="TickStyle",
            logger=logger,
        )
        object.__setattr__(self, "tickcolor", resolved_tc)
        object.__setattr__(self, "tickalpha", resolved_ta)

        if self.ticktype not in ("major", "minor", "both"):
            raise ValueError("TickStyle.ticktype must be 'major', 'minor', or 'both'.")


@dataclass
class LegendOptions:
    """A dataclass representing options for the legend in a boxplot figure.

    This is a restricted subset of the options available in Matplotlib's legend function.

    Attributes:
        legend_loc (str | int): The location of the legend. Defaults to "best".
        legend_bbox_to_anchor (tuple[float, float] | None): The bounding box anchor
            for the legend. Defaults to None.
        ncols (int): The number of columns in the legend. Defaults to 1.
        fontsize (float | str | None): The font size of the legend text. Defaults to None.

    """

    loc: str | int = "best"
    bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = None
    ncols: int = 1
    fontsize: float | str | None = None
    frameon: bool = True
    fancybox: bool = False
    shadow: bool = False
    framealpha: float | None = None
    facecolor: Color | None = None
    edgecolor: Color | None = None
    title: str | None = None
    alignment: Literal["center", "left", "right"] = "center"
    labelspacing: float = 0.5
    columnspacing: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the LegendOptions to a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the LegendOptions that
                can be passed to Matplotlib's legend function. Returns only fields
                that are not None.
        """
        output = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                output[field_name] = field_value

        return output


@dataclass(frozen=True)
class AxisLabelStyle:
    """Dataclass meant to mirror some matplotlib styling options for axis labels.

    Attributes:
        fontsize (float | int | None): Font size for the axis label text.
        fontweight (str | None): Font weight (e.g., "normal", "bold").
        fontstyle (str | None): Font style (e.g., "normal", "italic").
        fontfamily (str | None): Font family (e.g., "sans-serif", "serif").
        fontcolor (Color): Axis label text color.
        fontalpha (float | None): Axis label text alpha. If None, uses alpha from color if
            specified.
        labelpad (float | None): Padding between the label and the axis (in points).
    """

    fontsize: float | int | None = None
    fontweight: str | None = None
    fontstyle: str | None = None
    fontfamily: str | None = None

    fontcolor: Color = "black"
    fontalpha: float | None = None

    labelpad: float | None = None

    def __post_init__(self) -> None:
        if self.fontsize is not None:
            if not isinstance(self.fontsize, (int, float)):
                raise TypeError("AxisLabelStyle.fontsize must be a float or int.")
            size = float(self.fontsize)
            if not math.isfinite(size):
                raise ValueError("AxisLabelStyle.fontsize must be finite.")
            if size < 0:
                raise ValueError("AxisLabelStyle.fontsize must be nonnegative.")
            object.__setattr__(self, "fontsize", self.fontsize)

        if self.labelpad is not None:
            if not isinstance(self.labelpad, (int, float)):
                raise TypeError("AxisLabelStyle.labelpad must be a float or int.")
            pad = float(self.labelpad)
            if not math.isfinite(pad):
                raise ValueError("AxisLabelStyle.labelpad must be finite.")
            if pad < 0:
                raise ValueError("AxisLabelStyle.labelpad must be nonnegative.")
            object.__setattr__(self, "labelpad", pad)

        resolved_c, resolved_a = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="AxisLabelStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_c)
        object.__setattr__(self, "fontalpha", resolved_a)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert to Matplotlib keyword dictionary for ``Axes.set_xlabel`` /
        ``Axes.set_ylabel``.
        """
        settings_dict: dict[str, Any] = {
            "color": mcolors.to_rgba(self.fontcolor, alpha=self.fontalpha),
        }
        if self.fontsize is not None:
            settings_dict["fontsize"] = self.fontsize
        if self.fontweight is not None:
            settings_dict["fontweight"] = self.fontweight
        if self.fontstyle is not None:
            settings_dict["fontstyle"] = self.fontstyle
        if self.fontfamily is not None:
            settings_dict["fontfamily"] = self.fontfamily
        if self.labelpad is not None:
            settings_dict["labelpad"] = self.labelpad
        return settings_dict


@dataclass(frozen=True)
class TitleStyle:
    """Dataclass meant to mirror some matplotlib styling options for an axes title.

    Attributes:
        fontsize (float | int | None): Font size for the title text.
        fontweight (str | None): Font weight (e.g., "normal", "bold").
        fontstyle (str | None): Font style (e.g., "normal", "italic").
        fontfamily (str | None): Font family (e.g., "sans-serif", "serif").
        fontcolor (Color): Title text color.
        fontalpha (float | None): Title text alpha. If None, uses alpha from color if specified.
        loc (Literal["left", "center", "right"] | None): Title location.
        pad (float | None): Padding between the title and the axes (in points).
    """

    fontsize: float | int | None = None
    fontweight: str | None = None
    fontstyle: str | None = None
    fontfamily: str | None = None

    fontcolor: Color = "black"
    fontalpha: float | None = None

    loc: Literal["left", "center", "right"] | None = None
    pad: float | None = None

    def __post_init__(self) -> None:
        if self.fontsize is not None:
            if not isinstance(self.fontsize, (int, float)):
                raise TypeError("TitleStyle.fontsize must be a float or int.")
            size = float(self.fontsize)
            if not math.isfinite(size):
                raise ValueError("TitleStyle.fontsize must be finite.")
            if size < 0:
                raise ValueError("TitleStyle.fontsize must be nonnegative.")
            object.__setattr__(self, "fontsize", self.fontsize)

        if self.pad is not None:
            if not isinstance(self.pad, (int, float)):
                raise TypeError("TitleStyle.pad must be a float or int.")
            pad = float(self.pad)
            if not math.isfinite(pad):
                raise ValueError("TitleStyle.pad must be finite.")
            if pad < 0:
                raise ValueError("TitleStyle.pad must be nonnegative.")
            object.__setattr__(self, "pad", pad)

        if self.loc is not None and self.loc not in ("left", "center", "right"):
            raise ValueError("TitleStyle.loc must be one of {'left','center','right'}.")

        resolved_c, resolved_a = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="TitleStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_c)
        object.__setattr__(self, "fontalpha", resolved_a)

    def to_mpl_settings_dict(self) -> dict[str, Any]:
        """Convert to Matplotlib kwargs for ``Axes.set_title``."""
        settings_dict: dict[str, Any] = {
            "color": mcolors.to_rgba(self.fontcolor, alpha=self.fontalpha),
        }
        if self.fontsize is not None:
            settings_dict["fontsize"] = self.fontsize
        if self.fontweight is not None:
            settings_dict["fontweight"] = self.fontweight
        if self.fontstyle is not None:
            settings_dict["fontstyle"] = self.fontstyle
        if self.fontfamily is not None:
            settings_dict["fontfamily"] = self.fontfamily
        if self.loc is not None:
            settings_dict["loc"] = self.loc
        if self.pad is not None:
            settings_dict["pad"] = self.pad
        return settings_dict


FontStyle = Literal["normal", "italic", "oblique"]
"""How the glyphs are slanted.

- "normal": Upright (no slant). This is the default for most fonts.
- "italic": Uses the font's *italic face* if it exists (often a distinct, designed italic).
  This typically changes letterforms (e.g., a, f) and the slant.
- "oblique": Applies a *slant* to the regular face (or uses an oblique face if the font has one).
  Oblique is usually a geometric slant rather than a redesigned italic.
"""

FontVariant = Literal["normal", "small-caps"]
"""Glyph variant selection.

- "normal": Standard lowercase/uppercase forms.
- "small-caps": Lowercase letters are drawn as *small capital* forms (if the font supports it).
  If the font does not provide true small-caps, Matplotlib/font rendering may fall back to
  a synthetic approximation or ignore the request depending on backend/font.
"""

FontStretchName = Literal[
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "normal",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
]
FontStretch = Union[FontStretchName, int, float]
"""Width of the font face (condensed/expanded).

Named values (most common):
- "ultra-condensed": Extremely narrow.
- "extra-condensed": Very narrow.
- "condensed": Narrow.
- "semi-condensed": Slightly narrow.
- "normal": Standard width.
- "semi-expanded": Slightly wide.
- "expanded": Wide.
- "extra-expanded": Very wide.
- "ultra-expanded": Extremely wide.

Numeric values:
- Matplotlib also accepts numeric stretch values in the range 0–1000.
  (In practice, named values are more portable; numeric values depend on the font/backend.)
"""

FontWeightName = Literal[
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
]
FontWeight = Union[FontWeightName, int, float]
"""Stroke thickness / darkness of glyphs.

Named weights (portable, when a font provides them):
- "ultralight": Very thin strokes.
- "light": Thin strokes.
- "normal": Default weight.
- "regular": Synonym-ish for normal, depends on font naming.
- "book": Slightly heavier than normal for some typefaces.
- "medium": Between normal and bold.
- "roman": Often synonymous with normal/regular in some families.
- "semibold": Between medium and bold.
- "demibold" / "demi": Another naming convention for semibold-ish weights.
- "bold": Clearly heavier strokes, common emphasis.
- "heavy": Heavier than bold.
- "extra bold": Very heavy (note the space).
- "black": Heaviest strokes in many families.

Numeric weights:
- Matplotlib also accepts numeric weights in the range 0–1000.
  (Common convention: ~400 normal, ~700 bold, but exact mapping is font-dependent.)
"""

GenericFontFamily = Literal[
    "serif",
    "sans-serif",
    "sans serif",
    "sans",
    "monospace",
    "cursive",
    "fantasy",
]
FontFamily = Union[GenericFontFamily, str, Sequence[str]]
"""Font family selection.

- Generic families:
  - "serif": Fonts with serifs (e.g., DejaVu Serif, Times).
  - "sans-serif" / "sans serif" / "sans": Sans fonts (e.g., DejaVu Sans, Arial).
  - "monospace": Fixed-width fonts (e.g., DejaVu Sans Mono, Courier).
  - "cursive": Script-like fonts.
  - "fantasy": Decorative/display fonts.
- Specific font name:
  - Any installed font family name, e.g. "Nunito", "DejaVu Sans", "Arial".
- Fallback list:
  - A list/tuple of names like ["Nunito", "DejaVu Sans", "sans-serif"].
  - Matplotlib will pick the first available font from the list.
"""


@dataclass(frozen=True, slots=True)
class LabelFontOptions:
    """Font options for text labels.

    This is a thin, typed wrapper around Matplotlib’s text/font controls (used via
    `Axes.text(..., **to_mpl_text_kwargs())`).

    Face selection (what font Matplotlib will actually draw)
    --------------------------------------------------------
    Matplotlib chooses a *font face* by combining several independent knobs:

    1) `fontfamily` (which family to use)
       - You may pass a specific family name like `"Nunito"` or `"DejaVu Sans"`.
       - You may pass a generic family like `"sans-serif"`, `"serif"`, `"monospace"`, etc.
       - You may pass a *fallback list* like `["Nunito", "DejaVu Sans", "sans-serif"]`.
         Matplotlib will pick the first available entry on the current machine.
       - Important: specific font names only work if the font is installed or registered
         with Matplotlib (e.g., via `matplotlib.font_manager.fontManager.addfont()`).

    2) `fontweight` (how thick/dark the strokes are)
       - Named weights like `"normal"`, `"medium"`, `"semibold"`, `"bold"`, `"black"`, etc.
       - Or numeric weights `0–1000` (common convention: ~400 normal, ~700 bold),
         but the exact mapping is font-dependent.

    3) `fontstyle` (whether the glyphs are slanted)
       - `"normal"`: upright.
       - `"italic"`: uses the font’s designed italic face if present.
       - `"oblique"`: slants the regular face (or uses an oblique face if the font provides one).

    4) `fontvariant` (alternate glyph set)
       - `"small-caps"` requests small-cap lowercase forms if the font supports them.
         If not supported, Matplotlib/backends may approximate or ignore it.

    5) `fontstretch` (condensed/expanded width)
       - Requests narrower/wider variants like `"condensed"` or `"expanded"`, if present.
       - Or numeric stretch `0–1000`. Support varies by font.

    Notes & portability
    -------------------
    - The *same* settings can produce different results on different systems because the
      available fonts differ. If you need consistency, bundle a font (e.g. Nunito .ttf)
      and register it at runtime.
    - If a requested face (e.g., italic + semibold + condensed) does not exist in the chosen
      family, Matplotlib may fall back to the closest available face.

    Outline / halo
    --------------
    `outlinecolor` and `outlinewidth` are applied via path effects around the glyphs to
    improve legibility over busy map backgrounds.

    Attributes:
        fontcolor (Color): Fill color of the text.
        fontalpha (float | None): Alpha transparency of the text fill.
        fontsize (float): Font size (points).
        fontfamily (FontFamily | None): Specific family name, generic family, or fallback list.
        fontweight (FontWeight): Named or numeric weight (0–1000).
        fontstyle (FontStyle): Upright/italic/oblique slant selection.
        fontvariant (FontVariant): Normal vs small-caps glyph variant.
        fontstretch (FontStretch | None): Condensed/expanded variant (named or numeric 0–1000).
        outlinecolor (Color): Color of the glyph outline (halo).
        outlinewidth (float): Width of the glyph outline (halo), in points.
    """

    fontcolor: Color = "white"
    fontalpha: float | None = 1.0
    fontsize: float = 6.0

    # --- Style Options ---
    fontweight: FontWeight = "bold"
    fontstyle: FontStyle = "normal"
    fontvariant: FontVariant = "normal"
    fontstretch: FontStretch | None = None
    fontfamily: FontFamily | None = None

    outlinecolor: Color = "black"
    outlinewidth: float = 0.75

    def to_mpl_text_kwargs(self) -> dict:
        """Return kwargs to pass into `ax.text(...)` for font styling.

        This intentionally does NOT include color/alpha/zorder/ha/va/text/etc.
        """
        kw: dict = {
            "color": to_hex(
                resolve_color_and_alpha(self.fontcolor, self.fontalpha), keep_alpha=True
            ),
            "fontsize": float(self.fontsize),
            "fontweight": self.fontweight,
            "fontstyle": self.fontstyle,
            "fontvariant": self.fontvariant,
        }
        if self.fontstretch is not None:
            kw["fontstretch"] = self.fontstretch
        if self.fontfamily is not None:
            kw["fontfamily"] = self.fontfamily
        return kw


@dataclass(frozen=True, slots=True)
class LabelBoxOptions:
    """Background box options for text labels drawn via `Axes.text(..., bbox=...)`.

    This controls the *box behind the text*. The box automatically sizes to the text.

    Notes:
      - `pad` lives inside the `boxstyle` string (e.g., "round,pad=0.25") and is in
        *fraction of the font size* units (Matplotlib convention).
      - Matplotlib's `bbox` patch effectively has a single alpha; if you set separate
        face/edge alphas, the simplest thing is to apply one alpha to the whole patch.

    Attributes:
        enabled (bool): Whether to draw a background box behind the label text.
        boxstyle (str): The style of the background box. Default is "round". Options are:
              - "square"     : Plain rectangle
              - "round"      : Rectangle with rounded corners
              - "round4"     : Alternate rounded-rectangle style
              - "circle"     : Circular box around the text's bounding rectangle
              - "ellipse"    : Elliptical box around the text's bounding rectangle
        pad (float): Padding between text and box, in fraction-of-fontsize units.
        facecolor (Color): Fill color of the box (background).
        facealpha (float | None): Alpha for the box fill. If None, uses the color's inherent
            alpha (if any).
        edgecolor (Color): Edge (stroke) color of the box.
        edgealpha (float | None): Alpha for the box edge. If None, uses the color's inherent
            alpha (if any).
        edgewidth (float): Line width of the box edge, in points.
    """

    enabled: bool = True
    boxstyle: Literal["square", "round", "round4", "circle", "ellipse"] = "round4"
    pad: float = 0.25
    facecolor: Color = "black"
    facealpha: float | None = 0.6
    edgecolor: Color = "none"
    edgealpha: float | None = 0.0
    edgewidth: float = 0.8

    def to_mpl_bbox(self) -> dict | None:
        """Return a dict suitable for passing as `bbox=` to `Axes.text`.

        Returns:
            dict | None: A Matplotlib bbox properties dict if enabled; otherwise None.
        """
        if not self.enabled:
            return None

        face_color = resolve_color_and_alpha(self.facecolor, alpha=self.facealpha)
        edge_color = resolve_color_and_alpha(self.edgecolor, alpha=self.edgealpha)

        bbox = {
            "boxstyle": f"{self.boxstyle},pad={float(self.pad)}",
            "fc": to_hex(face_color, keep_alpha=True),
            "ec": to_hex(edge_color, keep_alpha=True),
            "lw": float(self.edgewidth),
        }

        return bbox
