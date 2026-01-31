from dataclasses import dataclass
from typing import Iterable, Literal

from gerrytools.latex.document import TexDocument
from gerrytools.logging import get_logger

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
class PaintBallLine:
    """Dataclass for storing paintball line properties.

    Attributes:
        slope (float): The slope of the line.
        linecolor (str): The color of the line.
        linewidth (float): The width of the line.
        linestyle (TikzLineStyle): The style of the line.
    """

    slope: float
    linecolor: str
    linewidth: float
    linestyle: TikzLineStyle

    def __post_init__(self):
        linestyle_str = str(self.linestyle)
        if linestyle_str not in (
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
        ):
            raise ValueError(
                f"Invalid linestyle: {linestyle_str}. "
                "Must be a valid TikZ line style ('soilid', 'dashed', 'dotted', "
                "'dashdotted', 'loosely dashed', 'loosely dotted', "
                "'loosely dashdotted', 'densely dashed', 'densely dotted', "
                "'densely dashdotted')."
            )


class PaintBallOptions:
    """Class for storing paintball plot options


    Attributes:
        markersize (float): The size of the markers in points.
        markercolor (str): The color of the markers.
        markeralpha (float): The opacity of the markers (0.0 to 1.0).
        crosshair_color (str): The color of the crosshair lines.
        crosshair_width (float): The width of the crosshair lines.
        xlim (tuple[float, float]): The x-axis limits.
        ylim (tuple[float, float]): The y-axis limits.
        xscale (float): The x-axis scale factor.
        yscale (float): The y-axis scale factor.
    """

    markersize: float = 8
    markercolor = "cadmiumgreen"
    markeralpha: float = 0.8
    markeredgecolor: str = "cadmiumgreen"
    markeredgewidth: float = 0.5
    markeredgealpha: float = 1.0
    hullcolor: str | None = None
    hullalpha: float | None = None
    hulledgecolor: str | None = None
    hulledgewidth: float | None = 2.0
    hulledgealpha: float | None = None
    crosshair_color: str = "gray!50"
    crosshair_width: float = 5.0
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0, 1)
    xscale: float = 10
    yscale: float = 10

    def __setattr__(self, key, value):
        match key:
            case "markersize":
                if not (0.0 < value):
                    raise ValueError("markersize must be positive")
                object.__setattr__(self, key, round(float(value), 4))
            case "markercolor":
                object.__setattr__(self, key, str(value))
            case "markeralpha":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("markeralpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "markeredgewidth":
                if not (0.0 <= value):
                    raise ValueError("markeredgewidth must be non-negative")
                object.__setattr__(self, key, round(float(value), 4))
            case "markeredgecolor":
                object.__setattr__(self, key, str(value))
            case "markeredgealpha":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("markeredgealpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "hullcolor":
                object.__setattr__(self, key, str(value))
            case "hullalpha":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("hullalpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "hulledgecolor":
                object.__setattr__(self, key, str(value))
            case "hulledgewidth":
                if not (0.0 <= value):
                    raise ValueError("hulledgewidth must be non-negative")
                object.__setattr__(self, key, round(float(value), 4))
            case "hulledgealpha":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("hulledgealpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "crosshair_color":
                object.__setattr__(self, key, str(value))
            case "crosshair_width":
                if not (0.0 <= value):
                    raise ValueError("crosshair_width must be non-negative")
                object.__setattr__(self, key, round(float(value), 4))
            case "xscale":
                if not (0.0 < value):
                    raise ValueError("xscale must be positive")
                object.__setattr__(self, key, round(float(value), 4))
            case "yscale":
                if not (0.0 < value):
                    raise ValueError("yscale must be positive")
                object.__setattr__(self, key, round(float(value), 4))
            case "xlim":
                if not (value[0] < value[1]):
                    raise ValueError("xlim[0] must be less than xlim[1]")
                object.__setattr__(
                    self, key, (round(float(value[0]), 4), round(float(value[1]), 4))
                )
            case "ylim":
                if not (value[0] < value[1]):
                    raise ValueError("ylim[0] must be less than ylim[1]")
                object.__setattr__(
                    self, key, (round(float(value[0]), 4), round(float(value[1]), 4))
                )
            case _:
                raise AttributeError(f"Unknown PaintBallOptions attribute: {key}")


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
        maximum_seats: int | None = None,
        *,
        round_data_to: int = 4,
        include_efficiency_gap_line: bool = True,
        include_proportionality_line: bool = True,
    ) -> None:
        self._document = TexDocument()
        self._document.add_packages("tikz")
        self.options = PaintBallOptions()
        self._voteshare_data, self._seatshare_data = (
            self._validate_voteshare_seatshare_and_max_seats(
                list(voteshare_data),
                list(seats_data),
                maximum_seats,
                round_data_to=round_data_to,
            )
        )

        # slope to PaintBallLine
        self._nammed_lines: dict[str, PaintBallLine] = {}
        self._lines: dict[float, list[PaintBallLine]] = {}

        if include_efficiency_gap_line:
            efficiency_gap_line = PaintBallLine(
                slope=2.0,
                linecolor="gray",
                linewidth=1.0,
                linestyle="solid",
            )
            self._nammed_lines["efficiency_gap"] = efficiency_gap_line

        if include_proportionality_line:
            proportionality_line = PaintBallLine(
                slope=1.0,
                linecolor="gray",
                linewidth=1.0,
                linestyle="dashed",
            )
            self._nammed_lines["proportionality"] = proportionality_line

    def __repr__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    def __str__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    @property
    def document(self) -> TexDocument:
        """TexDocument: The LaTeX document associated with this table."""
        self._document.body_string = self._generate_latex()
        return self._document

    @property
    def hull_document(self) -> TexDocument:
        """TexDocument: The LaTeX document associated with this table, including the convex hull."""
        self._document.body_string = self._generate_latex(hull=True)
        return self._document

    def print(self, *, hull: bool = False) -> None:
        if hull:
            print(self._generate_latex(hull=True))
        else:
            print(self._generate_latex())

    def clear_options(self) -> None:
        """Resets all table options to their default values."""
        self.__options = PaintBallOptions()

    def preview(self, hull=False) -> None:  # pragma: no cover
        """Previews the LaTeX document associated with this table."""
        if hull:
            self.hull_document.preview()
        else:
            self.document.preview()

    # ====================
    #   FEATURE ADDITION
    # ====================

    def _validate_voteshare_seatshare_and_max_seats(
        self,
        voteshare_data: list[float],
        seats_data,
        maximum_seats: int | None = None,
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
        self._seatshare_data.extend(new_seatshare_data)

    def add_lines_with_slope(
        self,
        slopes: Iterable[float],
        linecolor: str = "black",
        linewidth: float = 1.0,
        linestyle: TikzLineStyle = "solid",
        *,
        name: str | None = None,
    ) -> None:
        """Adds lines with specified slopes to the paintball plot.

        Args:
            slopes (Iterable[float]): The slopes of the lines to be added.
            linecolor (str, optional): The color of the lines. Defaults to "black".
            linewidth (float, optional): The width of the lines. Defaults to 1.0
            linestyle (TikzLineStyle, optional): The style of the lines. Defaults to "solid".
            name (str | None, optional): An optional name for the line. If provided,
                the line can be referenced later by this name. Defaults to None.
        """
        for slope in slopes:
            line = PaintBallLine(
                slope=slope,
                linecolor=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            if name is not None:
                self._nammed_lines[name] = line
            else:
                self._lines.setdefault(slope, []).append(line)

    def clear_lines(self) -> None:
        """Clears all added lines from the paintball plot."""
        self._lines = {}
        self._nammed_lines = {}

    # ==================
    #   OPTION SETTERS
    # ==================

    def set_xlim(self, xmin: float, xmax: float, rescale=False) -> None:
        """Sets the x-axis limits for the paintball plot.

        Args:
            xmin (float): The minimum x-axis limit.
            xmax (float): The maximum x-axis limit.
            rescale (bool): Whether to rescale the x-axis to fit the new limits. Defaults to False.
        """
        self.options.xlim = (xmin, xmax)
        if rescale:
            self.set_xscale(self.options.xscale * (1.0 / (xmax - xmin)))

    def set_ylim(self, ymin: float, ymax: float, rescale=False) -> None:
        """Sets the y-axis limits for the paintball plot.

        Args:
            ymin (float): The minimum y-axis limit.
            ymax (float): The maximum y-axis limit.
            rescale (bool): Whether to rescale the y-axis to fit the new limits. Defaults to False.
        """
        self.options.ylim = (ymin, ymax)
        if rescale:
            self.set_yscale(self.options.yscale * (1.0 / (ymax - ymin)))

    def set_xscale(self, xscale: float) -> None:
        """Sets the x-axis scale for the paintball plot.

        Args:
            xscale (float): The x-axis scale factor.
        """
        self.options.xscale = xscale

    def set_yscale(self, yscale: float) -> None:
        """Sets the y-axis scale for the paintball plot.

        Args:
            yscale (float): The y-axis scale factor.
        """
        self.options.yscale = yscale

    def set_scale(self, xscale: float | None = None, yscale: float | None = None) -> None:
        """Sets both the x-axis and y-axis scales for the paintball plot.

        Args:
            xscale (float): The x-axis scale factor.
            yscale (float): The y-axis scale factor.
        """
        if xscale is not None:
            self.set_xscale(xscale)
        if yscale is not None:
            self.set_yscale(yscale)

    def set_crosshair_options(self, color: str, width: float) -> None:
        """Sets the crosshair options for the paintball plot.

        Args:
            color (str): The color of the crosshair.
            width (float): The width of the crosshair lines.
        """
        self.options.crosshair_color = color
        self.options.crosshair_width = width

    def set_marker_options(
        self,
        size: float | None = None,
        color: str | None = None,
        alpha: float | None = None,
        edgecolor: str | None = None,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Sets the marker options for the paintball plot.

        Args:
            size_pts (float | None): The size of the markers in points. If None, the size is not
                changed from the previous setting. Defaults to None.
            color (str | None): The color of the markers. If None, the color is not changed from
                the previous setting. Defaults to None.
            alpha (float): The opacity of the markers (0.0 to 1.0). If None, the opacity is not
                changed from the previous setting. Defaults to None.
            edgecolor (str | None): The edge color of the markers. If None, the edge color is not
                changed from the previous setting. Defaults to None.
            edgewidth (float | None): The edge width of the markers. If None, the edge width is not
                changed from the previous setting. Defaults to None.
            edgealpha (float | None): The edge opacity of the markers (0.0 to 1.0). If None, the edge
                opacity is not changed from the previous setting. Defaults to None.
        """
        if size is not None:
            self.options.markersize = size
        if color is not None:
            self.options.markercolor = color
        if alpha is not None:
            self.options.markeralpha = alpha
        if edgecolor is not None:
            self.options.markeredgecolor = edgecolor
        if edgewidth is not None:
            self.options.markeredgewidth = edgewidth
        if edgealpha is not None:
            self.options.markeredgealpha = edgealpha

    def set_hull_options(
        self,
        color: str | None = None,
        alpha: float | None = None,
        edgecolor: str | None = None,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Sets the hull options for the paintball plot.

        Args:
            color (str | None): The color of the hull. If None, the color is not changed from
                the previous setting. Defaults to None.
            alpha (float | None): The opacity of the hull (0.0 to 1.0). If None, the opacity is not
                changed from the previous setting. Defaults to None.
            edgecolor (str | None): The edge color of the hull. If None, the edge color is not
                changed from the previous setting. Defaults to None.
            edgewidth (float | None): The edge width of the hull. If None, the edge width is not
                changed from the previous setting. Defaults to None.
            edgealpha (float | None): The edge opacity of the hull (0.0 to 1.0). If None, the edge
                opacity is not changed from the previous setting. Defaults to None.
        """
        if color is not None:
            self.options.hullcolor = color
        if alpha is not None:
            self.options.hullalpha = alpha
        if edgecolor is not None:
            self.options.hulledgecolor = edgecolor
        if edgewidth is not None:
            self.options.hulledgewidth = edgewidth
        if edgealpha is not None:
            self.options.hulledgealpha = edgealpha

    # =====================
    #   STRING GENERATORS
    # =====================

    def _compute_starting_ending_points_for_line_with_slope(
        self, slope: float
    ) -> tuple[float, float, float, float]:
        """Compute the starting and ending points for a line with the given slope.

        The line is drawn within the unit square from (0,0) to (1,1) and must pass through
        the center point (0.5, 0.5).

        Args:
            slope (float): The slope of the line.

        Returns:
            tuple[float, float, float, float]: The starting and ending points of the line
                in the format (starting_x, starting_y, ending_x, ending_y).
        """
        if slope == 0:
            starting_x = 0.0
            ending_x = 1.0
            starting_y = 0.5
            ending_y = 0.5
        elif slope == float("inf") or slope == float("-inf"):
            starting_x = 0.5
            ending_x = 0.5
            starting_y = 0.0
            ending_y = 1.0
        elif slope >= 1:
            starting_x = 0.5 - (0.5 / slope)
            starting_y = 0.0
            ending_x = 0.5 + (0.5 / slope)
            ending_y = 1.0
        elif 0 < slope < 1:
            starting_x = 0.0
            starting_y = 0.5 - (0.5 * slope)
            ending_x = 1.0
            ending_y = 0.5 + (0.5 * slope)
        elif -1 < slope < 0:
            starting_x = 0.0
            starting_y = 0.5 - (0.5 * slope)
            ending_x = 1.0
            ending_y = 0.5 + (0.5 * slope)
        else:
            starting_x = 0.5 - (0.5 / slope)
            starting_y = 0.0
            ending_x = 0.5 + (0.5 / slope)
            ending_y = 1.0

        starting_x = round(starting_x, 4)
        starting_y = round(starting_y, 4)
        ending_x = round(ending_x, 4)
        ending_y = round(ending_y, 4)

        return starting_x, starting_y, ending_x, ending_y

    def _paintball_points_str(self) -> str:
        """Generate the LaTeX string for the paintball points."""
        tex_string = "\\foreach \\votes/\\seats in {\n"
        for v, s in zip(self._voteshare_data, self._seatshare_data):
            tex_string += f"    {v}/{s},\n"
        tex_string = tex_string.rstrip(",\n") + "\n"  # Remove trailing comma
        tex_string += "} {\n"
        tex_string += (
            f"    \\node[transform shape=false, circle , fill={self.options.markercolor}, "
            f"fill opacity={self.options.markeralpha}, inner sep=0pt, "
            f"minimum size={self.options.markersize}pt, draw={self.options.markeredgecolor}, "
            f"line width={self.options.markeredgewidth}, "
            f"draw opacity={self.options.markeredgealpha}] \n"
            "    at (1-\\votes, 1-\\seats) {{}};\n}"
        )
        return tex_string

    def _paintball_hull_str(self) -> str:
        """Generate the LaTeX string for the horizontal hull of the paintball points."""
        decimal_places = max(
            [len(str(v).split(".")[1]) if "." in str(v) else 0 for v in self._seatshare_data]
            + [len(str(v).split(".")[1]) if "." in str(v) else 0 for v in self._voteshare_data]
        )

        points = [
            (round(1 - v, decimal_places), round(1 - s, decimal_places))
            for v, s in zip(self._voteshare_data, self._seatshare_data)
        ]
        sorted_points = sorted(points, key=lambda x: x[1])
        y_val_to_min_and_max_x = {}
        for x, y in sorted_points:
            if y not in y_val_to_min_and_max_x:
                y_val_to_min_and_max_x[y] = [x, x]
            else:
                if x < y_val_to_min_and_max_x[y][0]:
                    y_val_to_min_and_max_x[y][0] = x
                if x > y_val_to_min_and_max_x[y][1]:
                    y_val_to_min_and_max_x[y][1] = x

        sorted_y_vals = sorted(y_val_to_min_and_max_x.keys())

        # Use marker color as default hull color if none specified
        fillcolor = self.options.hullcolor or self.options.markercolor
        fillalpha = self.options.hullalpha or self.options.markeralpha
        linecolor = self.options.hulledgecolor or self.options.markeredgecolor
        linewidth = self.options.hulledgewidth or self.options.markeredgewidth
        linealpha = self.options.hulledgealpha or self.options.markeredgealpha

        draw_string = (
            f"\\draw [fill={fillcolor}, fill opacity={fillalpha}, line width={linewidth}, "
            f"color={linecolor}, draw opacity={linealpha}] "
        )

        # Draw left side
        draw_string += "\n"
        for val in sorted_y_vals:
            x = y_val_to_min_and_max_x[val][0]
            draw_string += f"  ({x},{val})--\n"
        # Draw right side
        for val in sorted_y_vals[::-1]:
            x = y_val_to_min_and_max_x[val][1]
            draw_string += f"  ({x},{val})--\n"
        draw_string = draw_string.rstrip("--\n") + ";"

        return draw_string

    def _generate_latex(self, *, hull=False) -> str:
        """Generate the complete LaTeX table string.

        Args:
            hull (bool): Whether to draw the horizontal hull of the paintball points rather
                than the individual points. Defaults to False.
        """
        tex_string = "\\begin{tikzpicture}\n\\begin{scope}"
        tex_string += f"[xscale={self.options.xscale}, yscale={self.options.yscale}]\n\n"

        # Clip the drawing area
        tex_string += (
            f"\\clip [draw] ({self.options.xlim[0]}, {self.options.ylim[0]}) "
            f"rectangle ({self.options.xlim[1]}, {self.options.ylim[1]});\n\n"
        )

        # Draw crosshairs
        tex_string += (
            f"\\draw [line width={self.options.crosshair_width}pt, "
            f"color={self.options.crosshair_color}] (0.5, 0) -- (0.5, 1);\n"
            f"\\draw [line width={self.options.crosshair_width}pt, "
            f"color={self.options.crosshair_color}] (0, 0.5) -- (1, 0.5);\n\n"
        )

        # Draw lines
        for line in self._nammed_lines.values():
            starting_x, starting_y, ending_x, ending_y = (
                self._compute_starting_ending_points_for_line_with_slope(line.slope)
            )
            tex_string += (
                f"\\draw [color={line.linecolor}, line width={line.linewidth}pt, "
                f"{line.linestyle}] "
                f"({starting_x}, {starting_y}) -- ({ending_x}, {ending_y});\n"
            )
        for slope, lines in self._lines.items():
            for line in lines:
                starting_x, starting_y, ending_x, ending_y = (
                    self._compute_starting_ending_points_for_line_with_slope(line.slope)
                )
                tex_string += (
                    f"\\draw [color={line.linecolor}, line width={line.linewidth}pt, "
                    f"{line.linestyle}] "
                    f"({starting_x}, {starting_y}) -- ({ending_x}, {ending_y});\n"
                )

        # Add paintballs or hull
        tex_string += "\n"
        if hull:
            tex_string += self._paintball_hull_str()
        else:
            tex_string += self._paintball_points_str()

        tex_string += "\n\n"
        tex_string += "\\end{scope}\n\\end{tikzpicture}\n"

        return tex_string
