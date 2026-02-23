from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Union

from matplotlib.colors import to_hex

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.typing import Color

FontStyle = Literal["normal", "italic", "oblique"]
"""How glyphs are slanted."""

FontVariant = Literal["normal", "small-caps"]
"""Glyph variant selection."""

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
"""Width of the font face (condensed/expanded)."""

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
"""Stroke thickness / darkness of glyphs."""

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
"""Font family selection."""


@dataclass(frozen=True, slots=True)
class LabelFontOptions:
    """Font options for text labels."""

    fontcolor: Color = "white"
    fontalpha: float | None = 1.0
    fontsize: float = 6.0

    fontweight: FontWeight = "bold"
    fontstyle: FontStyle = "normal"
    fontvariant: FontVariant = "normal"
    fontstretch: FontStretch | None = None
    fontfamily: FontFamily | None = None

    outlinecolor: Color = "black"
    outlinewidth: float = 0.75

    def to_mpl_text_kwargs(self) -> dict:
        """Return kwargs to pass into ``Axes.text`` for font styling."""
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
    """Background box options for text labels drawn via ``Axes.text(..., bbox=...)``."""

    enabled: bool = True
    boxstyle: Literal["square", "round", "round4", "circle", "ellipse"] = "round4"
    pad: float = 0.25
    facecolor: Color = "black"
    facealpha: float | None = 0.6
    edgecolor: Color = "none"
    edgealpha: float | None = 0.0
    edgewidth: float = 0.8

    def to_mpl_bbox(self) -> dict | None:
        """Return a dict suitable for passing as ``bbox`` to ``Axes.text``."""
        if not self.enabled:
            return None

        face_color = resolve_color_and_alpha(self.facecolor, alpha=self.facealpha)
        edge_color = resolve_color_and_alpha(self.edgecolor, alpha=self.edgealpha)

        return {
            "boxstyle": f"{self.boxstyle},pad={float(self.pad)}",
            "fc": to_hex(face_color, keep_alpha=True),
            "ec": to_hex(edge_color, keep_alpha=True),
            "lw": float(self.edgewidth),
        }
