from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict, get_args

import numpy as np

from gerrytools.colors import convert_color_to_hexa_or_none
from gerrytools.latex._colors import is_latex_color_expression
from gerrytools.latex._geometry import line_segment_through_unit_square
from gerrytools.latex._text import latex_escape
from gerrytools.latex.document import TexDocument
from gerrytools.logging import get_logger
from gerrytools.typing import Color, TikzLineStyle

logger = get_logger(__name__)


def _to_tikz_linestyle(linestyle: str) -> str:
    """Map Matplotlib-style line strings to TikZ line styles.

    Args:
        linestyle (str): Matplotlib-style or TikZ-style line token.

    Returns:
        str: Equivalent TikZ line style token.

    Raises:
        ValueError: If ``linestyle`` is neither a known Matplotlib token nor a valid TikZ line
            style. Unknown tokens used to pass through silently and break the LaTeX compile instead.
    """
    style_map = {
        "-": "solid",
        "--": "dashed",
        ":": "dotted",
        "-.": "dashdotted",
        "dashdot": "dashdotted",
    }
    mapped_style = style_map.get(str(linestyle), str(linestyle))
    valid_linestyles = get_args(TikzLineStyle)
    if mapped_style not in valid_linestyles:
        raise ValueError(
            f"Invalid linestyle: {linestyle!r}. Must be a Matplotlib token "
            f"({', '.join(repr(token) for token in style_map)}) or a TikZ line style "
            f"({', '.join(repr(style) for style in valid_linestyles)})."
        )
    return mapped_style


@dataclass(slots=True, frozen=True)
class SeatsVotesData:
    """Container for one seats-votes series and its marker metadata.

    Attributes:
        pov_party_vote_counts (np.ndarray): Per-district party-of-interest vote totals.
        total_vote_counts (np.ndarray): Per-district total vote totals.
        name (str): Legend label for the seats-votes curve.
        linecolor (Color): Color for the step curve.
        markercolor (Color): Color for the election-result marker.
        markerlabel (str): Legend label for the marker.
    """

    pov_party_vote_counts: np.ndarray
    total_vote_counts: np.ndarray
    name: str
    linecolor: Color
    markercolor: Color
    markerlabel: str

    def seats_votes_curve_values(
        self,
    ) -> tuple[list[float], list[float]]:
        """Compute standard uniform-swing seats-votes step-curve positions.

        Returns:
            tuple[list[float], list[float]]: Vote-share breakpoints and seat-share breakpoints.

        Raises:
            ValueError: If vote arrays do not align or contain nonpositive totals.
        """
        if self.pov_party_vote_counts.shape != self.total_vote_counts.shape:
            raise ValueError("pov_party_vote_counts and total_vote_counts must have same shape.")
        if np.any(self.total_vote_counts <= 0):
            raise ValueError("total_vote_counts must be positive for all districts.")

        vote_shares = self.pov_party_vote_counts / self.total_vote_counts
        weights = self.total_vote_counts

        overall_percent = float(np.sum(vote_shares * weights) / np.sum(weights))
        vote_share_shift_positions = (
            [0.0] + sorted([float(overall_percent - r + 0.5) for r in vote_shares]) + [1.0]
        )

        n_seats = len(vote_shares)
        seat_shares_shift_positions = [0.0] + list(map(float, np.arange(n_seats + 1) / n_seats))
        return vote_share_shift_positions, seat_shares_shift_positions


@dataclass(frozen=True)
class SVPlotLine:
    """Dataclass for seats-votes guide-line styling.

    Attributes:
        slope (float): Line slope through ``(0.5, 0.5)``.
        linecolor (Color): Line color.
        linewidth (float): Line width in points.
        linestyle (str): Line style token.
        label (str | None): Legend label.
    """

    slope: float
    linecolor: Color
    linewidth: float
    linestyle: str
    label: str | None = None

    def __post_init__(self) -> None:
        slope = float(self.slope)
        if math.isnan(slope):
            raise ValueError("slope must not be NaN.")
        object.__setattr__(self, "slope", slope)

        line_width = float(self.linewidth)
        if not math.isfinite(line_width):
            raise ValueError("linewidth must be finite.")
        if line_width < 0:
            raise ValueError("linewidth must be nonnegative.")
        object.__setattr__(self, "linewidth", line_width)


class _CrosshairXSettings(TypedDict):
    xmin: float
    xmax: float
    color: Color
    alpha: float


class _CrosshairYSettings(TypedDict):
    ymin: float
    ymax: float
    color: Color
    alpha: float


class _CrosshairSettings(TypedDict):
    x: _CrosshairXSettings
    y: _CrosshairYSettings


@dataclass(slots=True, frozen=True)
class _TikzColorToken:
    """Internal representation of a color token for TikZ emission.

    Attributes:
        kind (Literal["xcolor", "html", "none"]): Output encoding category.
        value (str): Color payload. For ``kind="xcolor"``, this is an xcolor expression
            such as ``"denim!20!amber"``. For ``kind="html"``, this is an uppercase
            6-digit hex token such as ``"1560BD"``. For ``kind="none"``, this is ``"none"``.
    """

    kind: Literal["xcolor", "html", "none"]
    value: str


@dataclass(slots=True)
class SeatsVotesOptions:
    """Configuration for LaTeX seats-votes rendering."""

    crosshair_x_width: float = 0.02
    crosshair_y_width: float = 0.02
    crosshair_color: Color = "lightgrey"
    crosshair_alpha: float = 1.0
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    xscale: float = 10.0
    yscale: float = 10.0
    linewidth: float = 1.5
    markersize: float = 8.0
    fontsize: float = 16.0
    legend_fontsize: float = 16.0

    def __setattr__(self, key: str, value) -> None:
        match key:
            case "crosshair_x_width" | "crosshair_y_width":
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{key} must be finite.")
                if value < 0:
                    raise ValueError(f"{key} must be nonnegative.")
                object.__setattr__(self, key, value)
            case "crosshair_color":
                object.__setattr__(self, key, value)
            case "crosshair_alpha":
                value = float(value)
                if not (0.0 <= value <= 1.0):
                    raise ValueError("crosshair_alpha must be in [0, 1].")
                object.__setattr__(self, key, value)
            case "xlim" | "ylim":
                lower = float(value[0])
                upper = float(value[1])
                if not (lower < upper):
                    raise ValueError(f"{key}[0] must be less than {key}[1].")
                object.__setattr__(self, key, (lower, upper))
            case "xscale" | "yscale":
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{key} must be finite.")
                if value <= 0:
                    raise ValueError(f"{key} must be positive.")
                object.__setattr__(self, key, value)
            case "linewidth" | "markersize" | "fontsize" | "legend_fontsize":
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{key} must be finite.")
                if value < 0:
                    raise ValueError(f"{key} must be nonnegative.")
                object.__setattr__(self, key, value)
            case _:
                raise AttributeError(f"Unknown SeatsVotesOptions attribute: {key}")


class SeatsVotes:
    """Generate seats-votes plots as TikZ/LaTeX."""

    def __init__(
        self,
        figure_size: tuple[float, float] = (10, 10),
        dpi: int = 300,
        *,
        include_legend: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a LaTeX seats-votes plot.

        Args:
            figure_size (tuple[float, float], optional): Target plot size interpreted as
                ``(xscale, yscale)`` in TikZ units. Defaults to ``(10, 10)``.
            dpi (int, optional): Stored dpi metadata for previews. Defaults to ``300``.
            include_legend (bool, optional): Whether to include a legend. Defaults to True.
            xlabel (str | None, optional): X-axis label text. Defaults to None.
            ylabel (str | None, optional): Y-axis label text. Defaults to None.
            title (str | None, optional): Plot title text. Defaults to None.
        """
        self._document = TexDocument()
        self._document.add_packages("tikz")

        self.figure_size = figure_size
        self.dpi = dpi
        self.include_legend = include_legend
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self.options = SeatsVotesOptions()
        self.set_scale(xscale=figure_size[0], yscale=figure_size[1])

        self._sv_data_list: list[SeatsVotesData] = []
        self._line_data_list: list[SVPlotLine] = []

        self._crosshair_settings: _CrosshairSettings | None = None
        self.update_crosshair_settings()

        self._display_election_markers = True
        self.standard_marker_color: Color = "#daa520"
        self.standard_election_color: Color = "#006400"
        self._display_line_legend = True

    def __repr__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    def __str__(self) -> str:  # pragma: no cover
        return self._generate_latex()

    @property
    def document(self) -> TexDocument:
        """Return the LaTeX document associated with this seats-votes plot.

        Returns:
            TexDocument: Document object containing generated TikZ source.
        """
        self._document.body_string = self._generate_latex()
        return self._document

    def print(self) -> None:
        """Print the raw TikZ body for this seats-votes plot."""
        print(self._generate_latex())

    def preview(self) -> None:  # pragma: no cover
        """Preview the seats-votes plot via TexDocument."""
        self.document.preview()

    def clear_options(self) -> None:
        """Reset seats-votes options to defaults.

        Returns:
            None
        """
        self.options = SeatsVotesOptions()
        self.set_scale(xscale=self.figure_size[0], yscale=self.figure_size[1])
        self.update_crosshair_settings()

    # ====================
    #   FEATURE ADDITION
    # ====================
    def add_seat_votes_data(
        self,
        pov_party_vote_shares: Sequence[int | float],
        total_vote_shares: Sequence[int | float] | None = None,
        *,
        name: str | None = None,
        linecolor: Color | None = None,
        markercolor: Color | None = None,
        markerlabel: str | None = None,
    ) -> None:
        """Add a seats-votes curve to the plot.

        Args:
            pov_party_vote_shares (Sequence[int | float]): Per-district vote totals or vote shares
                for the party of interest. If ``total_vote_shares`` is None, these are interpreted
                as vote shares and must be in [0, 1].
            total_vote_shares (Sequence[int | float] | None, optional): Per-district total vote
                totals. If None, all totals are treated as 1.0 and
                ``pov_party_vote_shares`` is interpreted as vote shares.
                Defaults to None.
            name (str | None, optional): Legend label for the seats-votes curve.
                Defaults to None.
            linecolor (Color | None, optional): Curve color. Defaults to None, which uses
                ``self.standard_election_color``.
            markercolor (Color | None, optional): Election-result marker color. Defaults to None,
                which uses ``self.standard_marker_color``.
            markerlabel (str | None, optional): Legend label for election-result markers.
                Defaults to None.
        """
        if total_vote_shares is None:
            if any(v < 0 or v > 1 for v in pov_party_vote_shares):
                raise ValueError(
                    "If total_vote_shares is not provided, then pov_party_vote_shares must be "
                    "vote shares in [0, 1]."
                )
            total_vote_shares = [1.0] * len(pov_party_vote_shares)

        self._sv_data_list.append(
            SeatsVotesData(
                pov_party_vote_counts=np.array(pov_party_vote_shares, dtype=float),
                total_vote_counts=np.array(total_vote_shares, dtype=float),
                name=name if name is not None else "Election Seats-Votes Curve",
                linecolor=linecolor if linecolor is not None else self.standard_election_color,
                markercolor=markercolor if markercolor is not None else self.standard_marker_color,
                markerlabel=markerlabel if markerlabel is not None else "Election Result",
            )
        )

    # ========================
    # ==  Cosmetic helpers  ==
    # ========================
    def update_crosshair_settings(
        self,
        *,
        x_width: float = 0.02,
        y_width: float = 0.02,
        color: Color = "lightgrey",
        alpha: float = 1.0,
    ) -> None:
        """Configure centered crosshair bands.

        Args:
            x_width (float, optional): Horizontal band width around ``x=0.5``.
                Defaults to ``0.02``.
            y_width (float, optional): Vertical band width around ``y=0.5``.
                Defaults to ``0.02``.
            color (Color, optional): Crosshair fill color. Defaults to ``"lightgrey"``.
            alpha (float, optional): Crosshair fill opacity in ``[0, 1]``.
                Defaults to ``1.0``.

        Returns:
            None
        """
        self.options.crosshair_x_width = x_width
        self.options.crosshair_y_width = y_width
        self.options.crosshair_color = color
        self.options.crosshair_alpha = alpha

        dx = self.options.crosshair_x_width / 2
        dy = self.options.crosshair_y_width / 2
        self._crosshair_settings = _CrosshairSettings(
            x=_CrosshairXSettings(
                xmin=0.5 - dx,
                xmax=0.5 + dx,
                color=self.options.crosshair_color,
                alpha=self.options.crosshair_alpha,
            ),
            y=_CrosshairYSettings(
                ymin=0.5 - dy,
                ymax=0.5 + dy,
                color=self.options.crosshair_color,
                alpha=self.options.crosshair_alpha,
            ),
        )

    def remove_crosshairs(self) -> None:
        """Remove crosshairs from the plot."""
        self._crosshair_settings = None

    def show_election_markers(self) -> None:
        """Show overall election-result markers."""
        self._display_election_markers = True

    def hide_election_markers(self) -> None:
        """Hide overall election-result markers."""
        self._display_election_markers = False

    def show_additional_lines_in_legend(self) -> None:
        """Include additional guide lines in the legend."""
        self._display_line_legend = True

    def hide_additional_lines_in_legend(self) -> None:
        """Hide additional guide lines from the legend."""
        self._display_line_legend = False

    def add_proportionality_line(
        self,
        *,
        color: Color = "grey",
        linestyle: str = "--",
        linewidth: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Add a proportionality line (y=x) to the plot.

        Args:
            color (Color, optional): Line color. Defaults to "grey".
            linestyle (str, optional): Line style. Defaults to "--".
            linewidth (float, optional): Line width. Defaults to 1.0.
            name (str | None, optional): Legend label. Defaults to "Proportionality".
        """
        self._line_data_list.append(
            SVPlotLine(
                slope=1.0,
                linecolor=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name if name is not None else "Proportionality",
            )
        )

    def add_efficiency_gap_line(
        self,
        *,
        color: Color = "grey",
        linestyle: str = "-",
        linewidth: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Add an efficiency-gap line (y=2x-0.5) to the plot.

        Args:
            color (Color, optional): Line color. Defaults to "grey".
            linestyle (str, optional): Line style. Defaults to "-".
            linewidth (float, optional): Line width. Defaults to 1.0.
            name (str | None, optional): Legend label. Defaults to "Efficiency Gap".
        """
        self._line_data_list.append(
            SVPlotLine(
                slope=2.0,
                linecolor=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name if name is not None else "Efficiency Gap",
            )
        )

    def add_custom_line(
        self,
        slope: float,
        *,
        linecolor: Color,
        linestyle: str,
        linewidth: float,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        """Add a custom slope-constrained line passing through (0.5, 0.5).

        Args:
            slope (float): Line slope.
            linecolor (Color): Line color.
            linestyle (str): Line style.
            linewidth (float): Line width.
            label (str | None, optional): Legend label. Defaults to None.
            name (str | None, optional): Alias for ``label`` for consistency with the
                Matplotlib version. Defaults to None.
        """
        if label is not None and name is not None and label != name:
            raise ValueError("name and label must match if both are provided.")
        legend_label = name if name is not None else label
        self._line_data_list.append(
            SVPlotLine(
                slope=slope,
                linecolor=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
                label=legend_label,
            )
        )

    def set_tick_fontsize(self, fontsize: float) -> None:
        """Set font size used for axis labels and tick labels.

        Args:
            fontsize (float): Font size in points.

        Returns:
            None
        """
        self.options.fontsize = fontsize

    def set_fontsize(self, fontsize: float) -> None:
        """Set a unified font size for axis labels/ticks and legend text.

        Args:
            fontsize (float): Font size in points.

        Returns:
            None
        """
        self.options.fontsize = fontsize
        self.options.legend_fontsize = fontsize

    def set_markersize(self, markersize: float) -> None:
        """Set election-result marker size.

        Args:
            markersize (float): Marker size in points.

        Returns:
            None
        """
        self.options.markersize = markersize

    def set_linewidth(self, linewidth: float) -> None:
        """Set seats-votes curve line width.

        Args:
            linewidth (float): Line width in points.

        Returns:
            None
        """
        self.options.linewidth = linewidth

    def set_xlim(self, xmin: float, xmax: float, rescale: bool = False) -> None:
        """Set x-axis limits.

        Args:
            xmin (float): Lower x-axis limit.
            xmax (float): Upper x-axis limit.
            rescale (bool, optional): If True, adjust xscale to preserve visual span.
                Defaults to False.

        Returns:
            None
        """
        self.options.xlim = (xmin, xmax)
        if rescale:
            self.set_xscale(self.options.xscale * (1.0 / (float(xmax) - float(xmin))))

    def set_ylim(self, ymin: float, ymax: float, rescale: bool = False) -> None:
        """Set y-axis limits.

        Args:
            ymin (float): Lower y-axis limit.
            ymax (float): Upper y-axis limit.
            rescale (bool, optional): If True, adjust yscale to preserve visual span.
                Defaults to False.

        Returns:
            None
        """
        self.options.ylim = (ymin, ymax)
        if rescale:
            self.set_yscale(self.options.yscale * (1.0 / (float(ymax) - float(ymin))))

    def set_xscale(self, xscale: float) -> None:
        """Set TikZ xscale factor.

        Args:
            xscale (float): X-axis TikZ scale factor.

        Returns:
            None
        """
        self.options.xscale = xscale

    def set_yscale(self, yscale: float) -> None:
        """Set TikZ yscale factor.

        Args:
            yscale (float): Y-axis TikZ scale factor.

        Returns:
            None
        """
        self.options.yscale = yscale

    def set_scale(self, xscale: float | None = None, yscale: float | None = None) -> None:
        """Set TikZ xscale/yscale factors.

        Args:
            xscale (float | None, optional): X-axis scale factor. Defaults to None.
            yscale (float | None, optional): Y-axis scale factor. Defaults to None.

        Returns:
            None
        """
        if xscale is not None:
            self.set_xscale(xscale)
        if yscale is not None:
            self.set_yscale(yscale)

    # =====================
    #   STRING GENERATORS
    # =====================
    @staticmethod
    def _fontsize_command(fontsize: float) -> str:
        """Build a LaTeX fontsize command.

        Args:
            fontsize (float): Font size in points.

        Returns:
            str: LaTeX command string that sets font size and baseline skip.
        """
        baseline_skip = fontsize + 2.0
        return rf"\fontsize{{{fontsize:0.2f}}}{{{baseline_skip:0.2f}}}\selectfont "

    @staticmethod
    def _compute_starting_ending_points_for_line_with_slope(
        slope: float,
    ) -> tuple[float, float, float, float]:
        """Compute line endpoints inside the unit square for a slope through ``(0.5, 0.5)``.

        Args:
            slope (float): Line slope.

        Returns:
            tuple[float, float, float, float]: ``(x_start, y_start, x_end, y_end)`` inside
                the unit square.
        """
        return line_segment_through_unit_square(slope, round_to=4)

    @staticmethod
    def _step_path(vote_shares: list[float], seat_shares: list[float]) -> str:
        """Convert step-curve vectors into a TikZ path string.

        Args:
            vote_shares (list[float]): Vote-share breakpoints.
            seat_shares (list[float]): Seat-share breakpoints.

        Returns:
            str: TikZ path expression.

        Raises:
            ValueError: If vectors are empty or lengths differ.
        """
        if len(vote_shares) != len(seat_shares):
            raise ValueError("vote_shares and seat_shares must have same length.")
        if len(vote_shares) == 0:
            raise ValueError("vote_shares and seat_shares must not be empty.")

        path_points: list[tuple[float, float]] = [(vote_shares[0], seat_shares[0])]
        for i in range(1, len(vote_shares)):
            path_points.append((vote_shares[i - 1], seat_shares[i]))
            path_points.append((vote_shares[i], seat_shares[i]))

        return " -- ".join(f"({x:0.4f}, {y:0.4f})" for x, y in path_points)

    def _curve_legend_entries(self) -> list[tuple[Color, str]]:
        """Collect unique seats-votes curve legend entries.

        Returns:
            list[tuple[Color, str]]: Unique ``(linecolor, name)`` pairs in insertion order.
        """
        unique_pairs = dict.fromkeys((sdata.linecolor, sdata.name) for sdata in self._sv_data_list)
        return list(unique_pairs.keys())

    def _marker_legend_entries(self) -> list[tuple[Color, str]]:
        """Collect unique election-marker legend entries.

        Returns:
            list[tuple[Color, str]]: Unique ``(markercolor, markerlabel)`` pairs in insertion
                order.
        """
        unique_pairs = dict.fromkeys(
            (sdata.markercolor, sdata.markerlabel) for sdata in self._sv_data_list
        )
        return list(unique_pairs.keys())

    def _line_legend_entries(self) -> list[tuple[Color, str, str]]:
        """Collect legend entries for additional guide lines.

        Returns:
            list[tuple[Color, str, str]]: ``(linecolor, linestyle, label)`` entries for
                lines with non-None labels.
        """
        entries: list[tuple[Color, str, str]] = []
        for line in self._line_data_list:
            if line.label is None:
                continue
            entries.append((line.linecolor, line.linestyle, line.label))
        return entries

    def _to_latex_color(self, color: Color) -> _TikzColorToken:
        """Convert a color token into an internal TikZ color representation.

        Unlike ``PaintBall._to_latex_color``, no auto-color name is registered on the document and
        HTML hex tokens are emitted inline via ``\\color[HTML]{...}``, so no name prefix is needed.

        Args:
            color (Color): Input color token.

        Returns:
            _TikzColorToken: Color token encoded as one of:
                ``kind="xcolor"`` for valid xcolor expressions,
                ``kind="html"`` for HTML hex colors used with ``\\color[HTML]{...}``,
                ``kind="none"`` for transparent/no-color tokens.
        """
        if isinstance(color, str):
            color_expr = color.strip()
            if color_expr.lower() == "none":
                return _TikzColorToken(kind="none", value="none")
            if is_latex_color_expression(color_expr):
                return _TikzColorToken(kind="xcolor", value=color_expr)

        hex8_or_none = convert_color_to_hexa_or_none(color)
        if hex8_or_none.lower() == "none":
            return _TikzColorToken(kind="none", value="none")

        hex6 = hex8_or_none.lstrip("#")[:6]
        return _TikzColorToken(kind="html", value=hex6.upper())

    @staticmethod
    def _color_prefix(color: _TikzColorToken) -> str:
        """Build a color prefix command for a TikZ command scope.

        Args:
            color (_TikzColorToken): Internal color token.

        Returns:
            str: Color-setting prefix command or empty string for ``none``.
        """
        if color.kind == "html":
            return rf"\color[HTML]{{{color.value}}}"
        if color.kind == "xcolor":
            return rf"\color{{{color.value}}}"
        return ""

    @staticmethod
    def _wrap_with_color_scope(command: str, color: _TikzColorToken) -> str:
        """Wrap a TikZ command in a local color scope when needed.

        Args:
            command (str): TikZ command ending in ``;``.
            color (_TikzColorToken): Internal color token.

        Returns:
            str: Scoped TikZ command with color prefix, or ``command`` unchanged.
        """
        color_prefix = SeatsVotes._color_prefix(color)
        if len(color_prefix) == 0:
            return command
        return "{" + color_prefix + command + "}"

    def _draw_path_command(
        self,
        *,
        path: str,
        color: _TikzColorToken,
        linewidth: float,
        linestyle: str | None = None,
    ) -> str:
        """Build a ``\\draw`` command with the provided styling and color token.

        Args:
            path (str): TikZ path expression without trailing semicolon.
            color (_TikzColorToken): Internal color token.
            linewidth (float): Line width in points.
            linestyle (str | None, optional): TikZ line-style token. Defaults to None.

        Returns:
            str: Fully formed TikZ ``\\draw`` command.
        """
        options = [f"line width={linewidth:0.2f}pt"]
        if linestyle is not None:
            options.append(linestyle)
        if color.kind == "none":
            options.append("draw=none")

        command = rf"\draw [{', '.join(options)}] {path};"
        return self._wrap_with_color_scope(command, color)

    def _fill_rectangle_command(
        self,
        *,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        color: _TikzColorToken,
        fill_opacity: float,
    ) -> str:
        """Build a ``\\fill`` rectangle command with color and opacity.

        Args:
            xmin (float): Left x-coordinate.
            ymin (float): Bottom y-coordinate.
            xmax (float): Right x-coordinate.
            ymax (float): Top y-coordinate.
            color (_TikzColorToken): Internal color token.
            fill_opacity (float): Fill opacity in ``[0, 1]``.

        Returns:
            str: Fully formed TikZ ``\\fill`` command.
        """
        options = [f"fill opacity={fill_opacity:0.4f}"]
        if color.kind == "none":
            options.append("fill=none")

        command = (
            rf"\fill [{', '.join(options)}] ({xmin:0.4f}, {ymin:0.4f}) rectangle "
            rf"({xmax:0.4f}, {ymax:0.4f});"
        )
        return self._wrap_with_color_scope(command, color)

    def _marker_node_command(
        self,
        *,
        x: float,
        y: float,
        color: _TikzColorToken,
        size_pt: float,
    ) -> str:
        """Build a circular marker node command.

        Args:
            x (float): Marker x-coordinate.
            y (float): Marker y-coordinate.
            color (_TikzColorToken): Internal color token.
            size_pt (float): Marker diameter in points.

        Returns:
            str: Fully formed TikZ ``\\node`` command.
        """
        options = ["circle", "inner sep=0pt", f"minimum size={size_pt:0.2f}pt"]
        if color.kind == "none":
            options.extend(["fill=none", "draw=none"])
        else:
            options.extend(["fill", "draw"])

        command = rf"\node [{', '.join(options)}] at ({x:0.4f}, {y:0.4f}) {{}};"
        return self._wrap_with_color_scope(command, color)

    def _add_labels(self, lines: list[str]) -> None:
        """Append title/xlabel/ylabel TikZ nodes to the output line list.

        Args:
            lines (list[str]): Mutable list of TikZ lines under construction.

        Returns:
            None
        """
        if self.title is None and self.xlabel is None and self.ylabel is None:
            return

        font_cmd = self._fontsize_command(self.options.fontsize)
        xmid = (self.options.xlim[0] + self.options.xlim[1]) / 2
        ymid = (self.options.ylim[0] + self.options.ylim[1]) / 2
        width = self.options.xlim[1] - self.options.xlim[0]
        height = self.options.ylim[1] - self.options.ylim[0]

        lines.append(
            f"\\begin{{scope}}[xscale={self.options.xscale}, yscale={self.options.yscale}]"
        )
        if self.title is not None:
            lines.append(
                rf"\node [anchor=south] at ({xmid:0.4f}, {self.options.ylim[1] + 0.03 * height:0.4f}) "
                rf"{{{font_cmd}{latex_escape(self.title)}}};"
            )
        if self.xlabel is not None:
            lines.append(
                rf"\node [anchor=north] at ({xmid:0.4f}, {self.options.ylim[0] - 0.03 * height:0.4f}) "
                rf"{{{font_cmd}{latex_escape(self.xlabel)}}};"
            )
        if self.ylabel is not None:
            lines.append(
                rf"\node [anchor=south, rotate=90] at ({self.options.xlim[0] - 0.03 * width:0.4f}, {ymid:0.4f}) "
                rf"{{{font_cmd}{latex_escape(self.ylabel)}}};"
            )
        lines.append(r"\end{scope}")

    def _add_legend(self, lines: list[str]) -> None:
        """Append legend glyph and text nodes to the output line list.

        Args:
            lines (list[str]): Mutable list of TikZ lines under construction.

        Returns:
            None
        """
        if not self.include_legend:
            return

        legend_rows: list[tuple[Literal["line", "marker"], Color, str, str]] = []
        legend_rows.extend(
            ("line", color, "solid", label) for color, label in self._curve_legend_entries()
        )
        if self._display_election_markers:
            legend_rows.extend(
                ("marker", color, "solid", label) for color, label in self._marker_legend_entries()
            )
        if self._display_line_legend:
            legend_rows.extend(
                ("line", color, linestyle, label)
                for color, linestyle, label in self._line_legend_entries()
            )

        if len(legend_rows) == 0:
            return

        legend_font_cmd = self._fontsize_command(self.options.legend_fontsize)

        x_start = self.options.xlim[1] + 0.03
        y_step = 0.06
        y_center = (self.options.ylim[0] + self.options.ylim[1]) / 2
        y_start = y_center + 0.5 * (len(legend_rows) - 1) * y_step
        line_length = 0.06
        label_offset = 0.08

        lines.append(
            f"\\begin{{scope}}[xscale={self.options.xscale}, yscale={self.options.yscale}]"
        )
        for idx, (row_type, color, linestyle, label) in enumerate(legend_rows):
            y_pos = y_start - idx * y_step
            color_token = self._to_latex_color(color)

            if row_type == "line":
                tikz_style = _to_tikz_linestyle(linestyle)
                lines.append(
                    self._draw_path_command(
                        path=(
                            rf"({x_start:0.4f}, {y_pos:0.4f}) -- "
                            rf"({x_start + line_length:0.4f}, {y_pos:0.4f})"
                        ),
                        color=color_token,
                        linewidth=1.2,
                        linestyle=tikz_style,
                    )
                )
            else:
                lines.append(
                    self._marker_node_command(
                        x=x_start + line_length / 2,
                        y=y_pos,
                        color=color_token,
                        size_pt=self.options.markersize,
                    )
                )

            lines.append(
                rf"\node [anchor=west] at ({x_start + label_offset:0.4f}, {y_pos:0.4f}) "
                rf"{{{legend_font_cmd}{latex_escape(label)}}};"
            )

        lines.append(r"\end{scope}")

    def _generate_latex(self) -> str:
        """Generate complete LaTeX/TikZ seats-votes plot content.

        Returns:
            str: Complete TikZ picture source for the configured plot.
        """
        lines = [r"\begin{tikzpicture}"]
        lines.append(
            f"\\begin{{scope}}[xscale={self.options.xscale}, yscale={self.options.yscale}]"
        )
        lines.append(
            rf"\clip [draw] ({self.options.xlim[0]:0.4f}, {self.options.ylim[0]:0.4f}) "
            rf"rectangle ({self.options.xlim[1]:0.4f}, {self.options.ylim[1]:0.4f});"
        )
        lines.append("")

        if self._crosshair_settings is not None:
            x_settings = self._crosshair_settings["x"]
            y_settings = self._crosshair_settings["y"]

            x_color = self._to_latex_color(x_settings["color"])
            y_color = self._to_latex_color(y_settings["color"])

            lines.append(
                self._fill_rectangle_command(
                    xmin=float(x_settings["xmin"]),
                    ymin=self.options.ylim[0],
                    xmax=float(x_settings["xmax"]),
                    ymax=self.options.ylim[1],
                    color=x_color,
                    fill_opacity=float(x_settings["alpha"]),
                )
            )
            lines.append(
                self._fill_rectangle_command(
                    xmin=self.options.xlim[0],
                    ymin=float(y_settings["ymin"]),
                    xmax=self.options.xlim[1],
                    ymax=float(y_settings["ymax"]),
                    color=y_color,
                    fill_opacity=float(y_settings["alpha"]),
                )
            )
            lines.append("")

        for line in self._line_data_list:
            x_start, y_start, x_end, y_end = (
                self._compute_starting_ending_points_for_line_with_slope(line.slope)
            )
            line_color = self._to_latex_color(line.linecolor)
            tikz_style = _to_tikz_linestyle(line.linestyle)
            lines.append(
                self._draw_path_command(
                    path=rf"({x_start:0.4f}, {y_start:0.4f}) -- ({x_end:0.4f}, {y_end:0.4f})",
                    color=line_color,
                    linewidth=line.linewidth,
                    linestyle=tikz_style,
                )
            )

        if len(self._line_data_list) > 0:
            lines.append("")

        for sv_series in self._sv_data_list:
            vote_shares, seat_shares = sv_series.seats_votes_curve_values()
            curve_color = self._to_latex_color(sv_series.linecolor)
            curve_path = self._step_path(vote_shares, seat_shares)
            lines.append(
                self._draw_path_command(
                    path=curve_path,
                    color=curve_color,
                    linewidth=self.options.linewidth,
                )
            )

        if len(self._sv_data_list) > 0:
            lines.append("")

        if self._display_election_markers:
            for sv_series in self._sv_data_list:
                marker_color = self._to_latex_color(sv_series.markercolor)
                total_vote_share = float(
                    sv_series.pov_party_vote_counts.sum() / sv_series.total_vote_counts.sum()
                )
                district_vote_shares = sv_series.pov_party_vote_counts / sv_series.total_vote_counts
                total_seat_share = float(np.mean(district_vote_shares > 0.5))

                lines.append(
                    self._marker_node_command(
                        x=total_vote_share,
                        y=total_seat_share,
                        color=marker_color,
                        size_pt=self.options.markersize,
                    )
                )

        lines.append(r"\end{scope}")

        self._add_labels(lines)
        self._add_legend(lines)

        lines.append(r"\end{tikzpicture}")
        lines.append("")
        return "\n".join(lines)
