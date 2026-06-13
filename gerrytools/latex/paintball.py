from dataclasses import dataclass
from typing import Iterable, get_args

from gerrytools._geometry import line_segment_through_unit_square
from gerrytools.latex._colors import classify_tikz_color
from gerrytools.latex.document import TexDocument
from gerrytools.logging import get_logger
from gerrytools.typing import Color, TikzLineStyle

logger = get_logger(__name__)


@dataclass(frozen=True)
class PaintBallLine:
    """Dataclass for storing paintball line properties.

    Deliberately parallel to :class:`gerrytools.plotting.data.paintball.PaintBallLine` (the
    Matplotlib backend); the two stay separate because their style vocabularies differ (TikZ tokens
    here, Matplotlib linestyle strings there). Keep field changes in sync where the concepts
    overlap.

    Attributes:
        slope (float): The slope of the line.
        linecolor (Color): The color of the line.
        linewidth (float): The width of the line.
        linestyle (TikzLineStyle): The style of the line.
    """

    slope: float
    linecolor: Color
    linewidth: float
    linestyle: TikzLineStyle

    def __post_init__(self):
        linestyle_str = str(self.linestyle)
        valid_linestyles = get_args(TikzLineStyle)
        if linestyle_str not in valid_linestyles:
            raise ValueError(
                f"Invalid linestyle: {linestyle_str}. Must be a valid TikZ line style "
                f"({', '.join(repr(style) for style in valid_linestyles)})."
            )


@dataclass(slots=True)
class PaintBallOptions:
    """Class for storing paintball plot options.

    Attributes:
        markersize (float): The size of the markers in points.
        markercolor (Color): The color of the markers.
        markeralpha (float): The opacity of the markers (0.0 to 1.0).
        markeredgecolor (Color): The color of the marker edges.
        markeredgewidth (float): The width of the marker edges in points.
        markeredgealpha (float): The opacity of the marker edges (0.0 to 1.0).
        hullcolor (Color | None): The fill color of the convex hull, or None for no fill.
        hullalpha (float | None): The opacity of the hull fill (0.0 to 1.0).
        hulledgecolor (Color | None): The color of the hull edge, or None for no edge.
        hulledgewidth (float | None): The width of the hull edge in points.
        hulledgealpha (float | None): The opacity of the hull edge (0.0 to 1.0).
        crosshair_color (Color): The color of the crosshair lines.
        crosshair_width (float): The width of the crosshair lines.
        xlim (tuple[float, float]): The x-axis limits.
        ylim (tuple[float, float]): The y-axis limits.
        xscale (float): The x-axis scale factor.
        yscale (float): The y-axis scale factor.
    """

    markersize: float = 8
    markercolor: Color = "cadmiumgreen"
    markeralpha: float = 0.8
    markeredgecolor: Color = "cadmiumgreen"
    markeredgewidth: float = 0.5
    markeredgealpha: float = 1.0
    hullcolor: Color | None = None
    hullalpha: float | None = None
    hulledgecolor: Color | None = None
    hulledgewidth: float | None = 2.0
    hulledgealpha: float | None = None
    crosshair_color: Color = "gray!50"
    crosshair_width: float = 5.0
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0, 1)
    xscale: float = 10
    yscale: float = 10

    def __setattr__(self, key: str, value) -> None:
        match key:
            case "markersize":
                if not (0.0 < value):
                    raise ValueError("markersize must be positive")
                object.__setattr__(self, key, round(float(value), 4))
            case "markercolor":
                object.__setattr__(self, key, value)
            case "markeralpha":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("markeralpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "markeredgewidth":
                if not (0.0 <= value):
                    raise ValueError("markeredgewidth must be non-negative")
                object.__setattr__(self, key, round(float(value), 4))
            case "markeredgecolor":
                object.__setattr__(self, key, value)
            case "markeredgealpha":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("markeredgealpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "hullcolor":
                object.__setattr__(self, key, value)
            case "hullalpha":
                if value is None:
                    object.__setattr__(self, key, None)
                    return
                if not (0.0 <= value <= 1.0):
                    raise ValueError("hullalpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "hulledgecolor":
                object.__setattr__(self, key, value)
            case "hulledgewidth":
                if value is None:
                    object.__setattr__(self, key, None)
                    return
                if not (0.0 <= value):
                    raise ValueError("hulledgewidth must be non-negative")
                object.__setattr__(self, key, round(float(value), 4))
            case "hulledgealpha":
                if value is None:
                    object.__setattr__(self, key, None)
                    return
                if not (0.0 <= value <= 1.0):
                    raise ValueError("hulledgealpha must be in [0.0, 1.0]")
                object.__setattr__(self, key, round(float(value), 4))
            case "crosshair_color":
                object.__setattr__(self, key, value)
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
    """Class for generating paintball plots in TikZ/LaTeX.

    The paintball plot is defined in vote-share / seat-share coordinates in the unit square.
    Vote shares are expected in [0, 1]. Seat data is either interpreted as shares in [0, 1]
    or normalized from seat counts using ``maximum_seats``.
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
        """Initialize a LaTeX paintball plot.

        Args:
            voteshare_data (Iterable[float]): Vote-share values for each plan outcome.
                Every value must be in [0, 1].
            seats_data (Iterable[float]): Seat-share values or seat counts for each plan outcome.
                If ``maximum_seats`` is None, values are interpreted as seat shares and must be
                in [0, 1]. If ``maximum_seats`` is provided, values are interpreted as seat counts
                and normalized by dividing by ``maximum_seats``.
            maximum_seats (int | None, optional): Maximum seat count used to normalize
                ``seats_data`` when seat counts are provided. Defaults to None.
            round_data_to (int, optional): Decimal precision used when storing plot data for
                LaTeX output. Defaults to 4.
            include_efficiency_gap_line (bool, optional): Whether to include the default
                efficiency-gap guide line. Defaults to True.
            include_proportionality_line (bool, optional): Whether to include the default
                proportionality guide line. Defaults to True.
        """
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
        """Return the LaTeX document for the point-based paintball plot.

        Returns:
            TexDocument: Document object containing the generated TikZ code.
        """
        self._document.body_string = self._generate_latex()
        return self._document

    @property
    def hull_document(self) -> TexDocument:
        """Return the LaTeX document for the hull-based paintball plot.

        Returns:
            TexDocument: Document object containing hull-rendered TikZ code.
        """
        self._document.body_string = self._generate_latex(hull=True)
        return self._document

    def print(self, *, hull: bool = False) -> None:
        """Print the generated TikZ body to stdout.

        Args:
            hull (bool, optional): If True, print the hull-rendered plot; otherwise print
                the point-rendered plot. Defaults to False.

        Returns:
            None
        """
        if hull:
            print(self._generate_latex(hull=True))
        else:
            print(self._generate_latex())

    def clear_options(self) -> None:
        """Reset paintball options to defaults.

        Returns:
            None
        """
        self.options = PaintBallOptions()

    def preview(self, hull: bool = False) -> None:  # pragma: no cover
        """Preview the rendered LaTeX plot.

        Args:
            hull (bool, optional): If True, preview the hull-rendered plot; otherwise preview
                the point-rendered plot. Defaults to False.

        Returns:
            None
        """
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
        """Validate and normalize incoming vote-share and seat-share data.

        ``voteshare_data`` values must lie in [0, 1].
        ``seats_data`` values are either interpreted directly as seat shares in [0, 1]
        (when ``maximum_seats`` is None), or as seat counts normalized by
        ``maximum_seats`` (when provided). Returned values are rounded to ``round_data_to``
        decimal places for stable LaTeX output.

        Args:
            voteshare_data (list[float]): Vote-share values in ``[0, 1]``.
            seats_data (Iterable[float]): Seat-share values in ``[0, 1]`` or raw seat counts.
            maximum_seats (int | None, optional): Total seats used to normalize raw seat counts.
                Defaults to None.
            round_data_to (int, optional): Decimal precision used for normalized outputs.
                Defaults to ``4``.

        Returns:
            tuple[list[float], list[float]]: Normalized vote-share and seat-share vectors.

        Raises:
            ValueError: If lengths mismatch, inputs are empty, or shares are out of range.
        """

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
        maximum_seats: int | None = None,
        *,
        round_data_to: int = 4,
    ) -> None:
        """Add vote-share / seat-share data points to the paintball plot.

        Args:
            voteshare_data (Iterable[float]): Vote-share values to add. Every value must be
                in [0, 1].
            seats_data (Iterable[float]): Seat-share values or seat counts to add.
                If ``maximum_seats`` is None, values are interpreted as seat shares and must be
                in [0, 1]. If ``maximum_seats`` is provided, values are interpreted as seat counts
                and normalized by dividing by ``maximum_seats``.
            maximum_seats (int | None, optional): The maximum number of seats. If provided,
                seats_data will be scaled by this value to obtain seat shares. If None,
                seats_data is assumed to already be in seat share format (i.e., in [0, 1]).
            round_data_to (int, optional): Decimal precision used when storing plot data for
                LaTeX output. Defaults to 4.
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
        linecolor: Color = "black",
        linewidth: float = 1.0,
        linestyle: TikzLineStyle = "solid",
        *,
        name: str | None = None,
    ) -> None:
        """Adds lines with specified slopes to the paintball plot.

        Args:
            slopes (Iterable[float]): The slopes of the lines to be added.
            linecolor (Color, optional): The color of the lines. Defaults to "black".
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
            xscale (float | None, optional): The x-axis scale factor. If None, the x-scale
                is left unchanged. Defaults to None.
            yscale (float | None, optional): The y-axis scale factor. If None, the y-scale
                is left unchanged. Defaults to None.
        """
        if xscale is not None:
            self.set_xscale(xscale)
        if yscale is not None:
            self.set_yscale(yscale)

    def set_crosshair_options(self, color: Color, width: float) -> None:
        """Sets the crosshair options for the paintball plot.

        Args:
            color (Color): The color of the crosshair.
            width (float): The width of the crosshair lines.
        """
        self.options.crosshair_color = color
        self.options.crosshair_width = width

    def set_marker_options(
        self,
        size: float | None = None,
        color: Color | None = None,
        alpha: float | None = None,
        edgecolor: Color | None = None,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Sets the marker options for the paintball plot.

        Args:
            size (float | None, optional): The size of the markers in points. If None, the size is
                not changed from the previous setting. Defaults to None.
            color (Color | None, optional): The color of the markers. If None, the color is not
                changed from the previous setting. Defaults to None.
            alpha (float | None, optional): The opacity of the markers (0.0 to 1.0). If None, the
                opacity is not changed from the previous setting. Defaults to None.
            edgecolor (Color | None, optional): The edge color of the markers. If None, the edge
                color is not changed from the previous setting. Defaults to None.
            edgewidth (float | None, optional): The edge width of the markers. If None, the edge
                width is not changed from the previous setting. Defaults to None.
            edgealpha (float | None, optional): The edge opacity of the markers (0.0 to 1.0). If
                None, the edge opacity is not changed from the previous setting. Defaults to None.
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
        color: Color | None = None,
        alpha: float | None = None,
        edgecolor: Color | None = None,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Sets the hull options for the paintball plot.

        Args:
            color (Color | None): The color of the hull. If None, the color is not changed from
                the previous setting. Defaults to None.
            alpha (float | None): The opacity of the hull (0.0 to 1.0). If None, the opacity is not
                changed from the previous setting. Defaults to None.
            edgecolor (Color | None): The edge color of the hull. If None, the edge color is not
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

    def _to_latex_color(self, color: Color, *, prefix: str) -> str:
        """Resolve a color to a LaTeX-safe token for TikZ commands.

        Args:
            color (Color): Input color value.
            prefix (str): Prefix for auto-generated color names.

        Returns:
            str: LaTeX color token safe to reference in TikZ commands.
        """
        color_kind, color_value = classify_tikz_color(color)
        if color_kind in ("none", "xcolor"):
            return color_value
        return self._document.resolve_color(f"#{color_value}", prefix=prefix)

    def _paintball_points_str(self) -> str:
        """Generate TikZ code for point markers.

        Returns:
            str: TikZ snippet that renders all paintball markers.
        """
        marker_fill = self._to_latex_color(self.options.markercolor, prefix="pbmarker")
        marker_edge = self._to_latex_color(self.options.markeredgecolor, prefix="pbmarker")
        tex_string = "\\foreach \\votes/\\seats in {\n"
        for v, s in zip(self._voteshare_data, self._seatshare_data):
            tex_string += f"    {v}/{s},\n"
        tex_string = tex_string.rstrip(",\n") + "\n"  # Remove trailing comma
        tex_string += "} {\n"
        tex_string += (
            f"    \\node[transform shape=false, circle , fill={marker_fill}, "
            f"fill opacity={self.options.markeralpha}, inner sep=0pt, "
            f"minimum size={self.options.markersize}pt, draw={marker_edge}, "
            f"line width={self.options.markeredgewidth}, "
            f"draw opacity={self.options.markeredgealpha}] \n"
            "    at (\\votes, \\seats) {{}};\n}"
        )
        return tex_string

    def _paintball_hull_str(self) -> str:
        """Generate TikZ code for the horizontal hull polygon.

        Returns:
            str: TikZ snippet that renders the hull polygon.
        """
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

        # Use marker settings as defaults only when hull options are unset.
        fillcolor = (
            self.options.hullcolor
            if self.options.hullcolor is not None
            else self.options.markercolor
        )
        fillalpha = (
            self.options.hullalpha
            if self.options.hullalpha is not None
            else self.options.markeralpha
        )
        linecolor = (
            self.options.hulledgecolor
            if self.options.hulledgecolor is not None
            else self.options.markeredgecolor
        )
        linewidth = (
            self.options.hulledgewidth
            if self.options.hulledgewidth is not None
            else self.options.markeredgewidth
        )
        linealpha = (
            self.options.hulledgealpha
            if self.options.hulledgealpha is not None
            else self.options.markeredgealpha
        )
        fillcolor_str = self._to_latex_color(fillcolor, prefix="pbhull")
        linecolor_str = self._to_latex_color(linecolor, prefix="pbhull")

        draw_string = (
            f"\\draw [fill={fillcolor_str}, fill opacity={fillalpha}, line width={linewidth}, "
            f"color={linecolor_str}, draw opacity={linealpha}] "
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
        """Generate complete TikZ content for the paintball plot.

        Args:
            hull (bool, optional): Whether to draw the horizontal hull of the paintball points rather
                than the individual points. Defaults to False.

        Returns:
            str: Complete TikZ picture source.
        """
        tex_string = "\\begin{tikzpicture}\n\\begin{scope}"
        tex_string += f"[xscale={self.options.xscale}, yscale={self.options.yscale}]\n\n"

        # Clip the drawing area
        tex_string += (
            f"\\clip [draw] ({self.options.xlim[0]}, {self.options.ylim[0]}) "
            f"rectangle ({self.options.xlim[1]}, {self.options.ylim[1]});\n\n"
        )

        # Draw crosshairs
        crosshair_color = self._to_latex_color(self.options.crosshair_color, prefix="pbcross")
        tex_string += (
            f"\\draw [line width={self.options.crosshair_width}pt, "
            f"color={crosshair_color}] (0.5, 0) -- (0.5, 1);\n"
            f"\\draw [line width={self.options.crosshair_width}pt, "
            f"color={crosshair_color}] (0, 0.5) -- (1, 0.5);\n\n"
        )

        # Draw lines
        for line in self._nammed_lines.values():
            starting_x, starting_y, ending_x, ending_y = line_segment_through_unit_square(
                line.slope, round_to=4
            )
            line_color = self._to_latex_color(line.linecolor, prefix="pbline")
            tex_string += (
                f"\\draw [color={line_color}, line width={line.linewidth}pt, "
                f"{line.linestyle}] "
                f"({starting_x}, {starting_y}) -- ({ending_x}, {ending_y});\n"
            )
        for _slope, lines in self._lines.items():
            for line in lines:
                starting_x, starting_y, ending_x, ending_y = line_segment_through_unit_square(
                    line.slope, round_to=4
                )
                line_color = self._to_latex_color(line.linecolor, prefix="pbline")
                tex_string += (
                    f"\\draw [color={line_color}, line width={line.linewidth}pt, "
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
