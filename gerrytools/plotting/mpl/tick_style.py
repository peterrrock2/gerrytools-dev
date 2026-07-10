from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.utils import _validated_nonneg_finite
from gerrytools.typing import Color, TickType

logger = get_logger(__name__)


@dataclass(frozen=True)
class TickStyle:
    """Data class representing the style of axis ticks."""

    size: float | int = 10
    rotation: float | int = 0
    fontcolor: Color = "black"
    fontalpha: float | None = None
    tickcolor: Color = "black"
    tickalpha: float | None = None
    fontweight: str = "normal"
    fontstyle: Literal["normal", "italic", "oblique"] = "normal"
    fontfamily: str = "sans-serif"
    ticktype: TickType = "major"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "size", _validated_nonneg_finite(self.size, field="TickStyle.size")
        )

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
