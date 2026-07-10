from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, TypeAlias

from gerrytools.typing import Color, MplKwargs

LegendAnchor: TypeAlias = tuple[float, float] | tuple[float, float, float, float]
"""Two- or four-coordinate anchor accepted by Matplotlib legends."""


@dataclass
class LegendOptions:
    """Restricted subset of Matplotlib legend options.

    Field defaults are the gerrytools defaults: a legend centered to the
    right of the axes.
    """

    loc: str | int = "center left"
    bbox_to_anchor: LegendAnchor | None = (1.01, 0.5)
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

    def to_dict(self) -> MplKwargs:
        """Convert to kwargs accepted by Matplotlib's ``Axes.legend``."""
        output: MplKwargs = {}
        for field in fields(self):
            field_value = getattr(self, field.name)
            if field_value is not None:
                output[field.name] = field_value
        return output
