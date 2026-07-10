from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import matplotlib.colors as mcolors

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.utils import _validated_finite, _validated_nonneg_finite
from gerrytools.typing import Color, MplKwargs, MplRGBAColor

logger = get_logger(__name__)


class AxisLabelKwargs(TypedDict, total=False):
    """Keyword arguments for ``Axes.set_xlabel`` and ``Axes.set_ylabel``."""

    color: MplRGBAColor
    fontsize: float | int
    fontweight: str
    fontstyle: Literal["normal", "italic", "oblique"]
    fontfamily: str
    labelpad: float


class TitleKwargs(TypedDict, total=False):
    """Keyword arguments for ``Axes.set_title``."""

    color: MplRGBAColor
    fontsize: float | int
    fontweight: str
    fontstyle: Literal["normal", "italic", "oblique"]
    fontfamily: str
    loc: Literal["left", "center", "right"]
    pad: float


@dataclass(frozen=True)
class _FontStyleBase:
    """Shared font fields and validation for the label/title style dataclasses."""

    fontsize: float | int | None = None
    fontweight: str | None = None
    fontstyle: Literal["normal", "italic", "oblique"] | None = None
    fontfamily: str | None = None

    fontcolor: Color = "black"
    fontalpha: float | None = None

    def __post_init__(self) -> None:
        if self.fontsize is not None:
            object.__setattr__(
                self,
                "fontsize",
                _validated_nonneg_finite(self.fontsize, field=f"{type(self).__name__}.fontsize"),
            )

        resolved_color, resolved_alpha = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner=type(self).__name__,
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_color)
        object.__setattr__(self, "fontalpha", resolved_alpha)

    def _font_settings(self) -> MplKwargs:
        """The shared font kwargs, with unset (None) fields dropped."""
        settings: MplKwargs = {"color": mcolors.to_rgba(self.fontcolor, alpha=self.fontalpha)}
        for name in ("fontsize", "fontweight", "fontstyle", "fontfamily"):
            value = getattr(self, name)
            if value is not None:
                settings[name] = value
        return settings


@dataclass(frozen=True)
class AxisLabelStyle(_FontStyleBase):
    """Dataclass mirroring key Matplotlib style options for axis labels."""

    labelpad: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.labelpad is not None:
            # Negative pads are legal in matplotlib (they pull the label inward).
            object.__setattr__(
                self,
                "labelpad",
                _validated_finite(self.labelpad, field="AxisLabelStyle.labelpad"),
            )

    def to_mpl_settings_dict(self) -> AxisLabelKwargs:
        """Convert to Matplotlib kwargs for ``Axes.set_xlabel``/``Axes.set_ylabel``."""
        settings_dict = cast("AxisLabelKwargs", self._font_settings())
        if self.labelpad is not None:
            settings_dict["labelpad"] = self.labelpad
        return settings_dict


@dataclass(frozen=True)
class TitleStyle(_FontStyleBase):
    """Dataclass mirroring key Matplotlib style options for axes titles."""

    loc: Literal["left", "center", "right"] | None = None
    pad: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pad is not None:
            # Negative pads are legal in matplotlib (they pull the title inward).
            object.__setattr__(self, "pad", _validated_finite(self.pad, field="TitleStyle.pad"))
        if self.loc is not None and self.loc not in ("left", "center", "right"):
            raise ValueError("TitleStyle.loc must be one of {'left','center','right'}.")

    def to_mpl_settings_dict(self) -> TitleKwargs:
        """Convert to Matplotlib kwargs for ``Axes.set_title``."""
        settings_dict = cast("TitleKwargs", self._font_settings())
        if self.loc is not None:
            settings_dict["loc"] = self.loc
        if self.pad is not None:
            settings_dict["pad"] = self.pad
        return settings_dict
