from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.colors as mcolors

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.typing import Color

logger = get_logger(__name__)


@dataclass(frozen=True)
class AxisLabelStyle:
    """Dataclass mirroring key Matplotlib style options for axis labels."""

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
        """Convert to Matplotlib kwargs for ``Axes.set_xlabel``/``Axes.set_ylabel``."""
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
    """Dataclass mirroring key Matplotlib style options for axes titles."""

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
