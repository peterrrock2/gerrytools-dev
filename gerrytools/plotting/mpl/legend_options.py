from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from gerrytools.typing import Color


@dataclass
class LegendOptions:
    """Restricted subset of Matplotlib legend options."""

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
        """Convert to kwargs accepted by Matplotlib's ``Axes.legend``."""
        output: dict[str, Any] = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                output[field_name] = field_value
        return output
