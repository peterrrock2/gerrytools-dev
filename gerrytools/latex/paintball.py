import inspect
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, cast, Literal

import pandas as pd

from gerrytools.latex.document import TexDocument
from gerrytools.latex.formatters import round_decimals
from gerrytools.colors import is_latex_color_string
from gerrytools.logging import get_logger
from gerrytools.typing import CellWrapper, Color

logger = get_logger(__name__)


TikzLineStyle = Literal[
    "solid",
    "dashed",
    "dotted",
    "dashdotted",
    "loosely dashed",
    "loosely dotted",
    "loosely dashdotted",
    "densely dashed",
    "densely dotted",
    "densely dashdotted",
]


@dataclass(frozen=True)
class PaintBallOptions:
    markersize_pts: float = 0.5
    makercolor = "cadmiumgreen"
    markeropacity: float = 0.8
    crosshair_color: str = "gray!50"
    crosshair_width: float = 5.0
    efficiency_gap_linecolor: str = "gray"
    efficiency_gap_linewidth: float = 1.0
    efficiency_gap_linestyle: TikzLineStyle = "dashed"
    proportionality_linecolor: str = "gray"
    proportionality_linewidth: float = 1.0
    proportionality_linestyle: TikzLineStyle = "solid"
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0, 1)
    xscale: float = 10
    yscale: float = 10

    def __postinit__(self):
        if not (0.0 < self.markersize_pts):
            raise ValueError("markersize_pts must be positive")
        if not (0.0 <= self.markeropacity <= 1.0):
            raise ValueError("markeropacity must be in [0.0, 1.0]")
        if not (0.0 <= self.crosshair_width):
            raise ValueError("crosshair_width must be non-negative")
        if not (0.0 <= self.efficiency_gap_linewidth):
            raise ValueError("efficiency_gap_linewidth must be non-negative")
        if not (0.0 <= self.proportionality_linewidth):
            raise ValueError("proportionality_linewidth must be non-negative")
        if not (0.0 < self.xscale):
            raise ValueError("xscale must be positive")
        if not (0.0 < self.yscale):
            raise ValueError("yscale must be positive")
        if not (self.xlim[0] < self.xlim[1]):
            raise ValueError("xlim[0] must be less than xlim[1]")
        if not (self.ylim[0] < self.ylim[1]):
            raise ValueError("ylim[0] must be less than ylim[1]")
        object.__setattr__(self, "markersize_pts", round(float(self.markersize_pts), 4))
        object.__setattr__(self, "markeropacity", round(float(self.markeropacity), 4))
        object.__setattr__(self, "crosshair_width", round(float(self.crosshair_width), 4))
        object.__setattr__(
            self, "efficiency_gap_linewidth", round(float(self.efficiency_gap_linewidth), 4)
        )
        object.__setattr__(
            self, "proportionality_linewidth", round(float(self.proportionality_linewidth), 4)
        )
        object.__setattr__(self, "xscale", round(float(self.xscale), 4))
        object.__setattr__(self, "yscale", round(float(self.yscale), 4))
        object.__setattr__(
            self, "xlim", (round(float(self.xlim[0]), 4), round(float(self.xlim[1]), 4))
        )
        object.__setattr__(
            self, "ylim", (round(float(self.ylim[0]), 4), round(float(self.ylim[1]), 4))
        )


class PaintBall:
    """Class for generating paintball plots.

    Args:
        df (pd.DataFrame): The DataFrame to be converted to a LaTeX table
        use_defaults (bool, optional): Whether to initialize with default table options
            (bold headers, 4 decimal places, etc.). Defaults to True.

    Attributes:
        df (pd.DataFrame): The DataFrame to be converted to a LaTeX table
    """

    def __init__(
        self,
        voteshare_data: Iterable[float],
        seats_data: Iterable[float],
        maximum_seats: int | None,
        *,
        round_data_to: int = 4,
    ) -> None:
        self._document = TexDocument()
        self._document.add_packages("colortbl")
        self.__options = PaintBallOptions()
        self._voteshare_data, self._seats_data = self._validate_voteshare_seatshare_and_max_seats(
            list(voteshare_data),
            list(seats_data),
            maximum_seats,
            round_data_to=round_data_to,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    def __str__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    @property
    def document(self) -> TexDocument:
        """TexDocument: The LaTeX document associated with this table."""
        self._document.body_string = self._generate_latex()
        return self._document

    def clear_options(self) -> None:
        """Resets all table options to their default values."""
        self.__options = PaintBallOptions()

    def preview(self) -> None:  # pragma: no cover
        self.document.preview()

    # ====================
    #   FEATURE ADDITION
    # ====================

    def _validate_voteshare_seatshare_and_max_seats(
        self,
        voteshare_data: list[float],
        seats_data,
        maximum_seats: int | None,
        *,
        round_data_to: int = 4,
    ) -> tuple[list[float], list[float]]:
        """Adds the seats-votes data points to the paintball plot."""

        if len(voteshare_data) != len(seats_data):
            raise ValueError("voteshare_data and seats_data must have the same length")
        if len(voteshare_data) == 0:
            raise ValueError("voteshare_data and seats_data must have at least one element")

        ret_voteshare: list[float] = list(round(float(v), round_data_to) for v in voteshare_data)

        if maximum_seats is None:
            if not all(0.0 <= s <= 1.0 for s in seats_data):
                raise ValueError(
                    "If maximum_seats is not provided, all seats_data values must be in [0, 1]"
                )
            ret_seat_share: list[float] = list(round(float(s), round_data_to) for s in seats_data)
        else:
            new_seats_data: list[float] = list(
                round(float(s) / maximum_seats, round_data_to) for s in seats_data
            )
            if not all(0.0 <= s <= 1.0 for s in new_seats_data):
                raise ValueError(
                    "After scaling by maximum_seats, all seats_data values must be in [0, 1]"
                )
            ret_seat_share: list[float] = new_seats_data

        return ret_voteshare, ret_seat_share

    def add_voteshare_seatshare_data(
        self,
        voteshare_data: Iterable[float],
        seats_data: Iterable[float],
        maximum_seats: int | None,
        *,
        round_data_to: int = 4,
    ) -> None:
        """Adds the seats-votes data points to the paintball plot.

        Args:
            voteshare_data (Iterable[float]): The vote share data points
            seats_data (Iterable[float]): The seat share data points
            maximum_seats (int | None, optional): The maximum number of seats. If provided,
                seats_data will be scaled by this value to obtain seat shares. If None,
                seats_data is assumed to already be in seat share format (i.e., in [0, 1]).
            round_data_to (int, optional): The number of decimal places to round the data
                points to. Defaults to 4.
        """

        new_voteshare_data, new_seatshare_data = self._validate_voteshare_seatshare_and_max_seats(
            list(voteshare_data),
            list(seats_data),
            maximum_seats,
            round_data_to=round_data_to,
        )

        self._voteshare_data.extend(new_voteshare_data)
        self._seats_data.extend(new_seatshare_data)

    # ==================
    #   OPTION SETTERS
    # ==================

    # =====================
    #   STRING GENERATORS
    # =====================

    def _generate_latex(self) -> str:
        """Generate the complete LaTeX table string."""
        tex_string = ""
        return tex_string
