import math
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Real
from typing import Literal, Sequence, cast
from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting._axes_state import (
    UNIT_ASPECT,
    UNIT_FRAME,
    UNIT_LEGEND,
    UNIT_TITLE,
    UNIT_X_LABEL,
    UNIT_X_LIMITS,
    UNIT_X_SCALE,
    UNIT_X_TICK_STYLE,
    UNIT_X_TICKS,
    UNIT_Y_LABEL,
    UNIT_Y_LIMITS,
    UNIT_Y_SCALE,
    UNIT_Y_TICK_STYLE,
    UNIT_Y_TICKS,
    _label_snapshot,
    _LabelSnapshot,
    _ManagedAxesState,
    _tick_style_snapshot,
    _title_snapshot,
    _TitleSnapshot,
)
from gerrytools.plotting._figure_io import save_figure, show_figure
from gerrytools.plotting._legend_utils import build_legend_options, save_legend_handles
from gerrytools.plotting.data._annotations import _Annotations
from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowData,
    ArrowPlacement,
    ArrowTextStyle,
    BandData,
    LabelArrowStyle,
    LineData,
    TextArrowStyle,
)
from gerrytools.plotting.data.options import BandOptions, LineOptions
from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.tick_style import TickStyle
from gerrytools.plotting.utils import _coerce_real_iter
from gerrytools.typing import Color, LegendHandle, TickType

logger = get_logger(__name__)

# Internal sentinel marking "no explicit label/title opinion set". Distinct
# from ``None``, which is a real public API value (``self.title = None`` is
# the explicit-clear path). Never appears in any public signature, default,
# getter, or ``__repr__``.
_UNSET_TEXT: str = "__gerryplot_unset_text__"


class GerryPlotBase(ABC):
    """Abstract base class for GerryPlot plotting classes."""

    fig: Figure

    def __init__(
        self,
        figure_size: tuple[float, float] | None = None,
        dpi: int | None = None,
        *,
        ax: Axes | None = None,
        include_legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a GerryPlotBase instance.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the figure
                in inches. Defaults to ``(10, 6)`` when ``ax`` is not provided. Ignored
                (with a warning) when ``ax`` is provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure. Defaults
                to ``300`` when ``ax`` is not provided. Ignored (with a warning) when
                ``ax`` is provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Useful for
                callers familiar with matplotlib / seaborn idioms who want to compose
                this plot into a larger figure they control. When provided, the
                plot's lazy build draws onto this axes; content already on the
                axes is left in place, and on rebuilds only the artists this
                plot created are removed. Defaults to None.
            include_legend (bool, optional): Whether to include a legend in the plot.
                Defaults to True.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        # --- Pass 1: resolve self._ax and self.fig ---
        # Remembered so ``bind_to_ax(None)`` can recreate a fresh figure with the same geometry as
        # construction.
        self._figure_size: tuple[float, float] = figure_size if figure_size is not None else (10, 6)
        self._figure_dpi: int = dpi if dpi is not None else 300
        if ax is not None:
            if figure_size is not None or dpi is not None:
                warn(
                    "figure_size and dpi are ignored when ax is provided; "
                    "the plot will use the existing figure's size and dpi.",
                    UserWarning,
                    stacklevel=2,
                )
            self.fig = cast(Figure, ax.figure)
            self._ax = ax
            # User owns the figure; do not register a finalizer to close it.
            self._finalizer: weakref.finalize | None = None
        else:
            self.fig, self._ax = plt.subplots(
                figsize=self._figure_size,
                dpi=self._figure_dpi,
            )

            # IMPORTANT: prevent implicit display in notebooks
            # Only close in Jupyter so init doesn't display
            try:
                from IPython import get_ipython

                ip = get_ipython()
                if (
                    ip is not None and getattr(ip, "kernel", None) is not None
                ):  # pragma: no cover - only reachable inside a live Jupyter kernel
                    plt.close(self.fig)  # pragma: no cover
            except Exception:  # pragma: no cover - IPython import failure is suppressed
                pass  # pragma: no cover

            self._finalizer = weakref.finalize(self, plt.close, self.fig)

        # --- Pass 2: initialize backing fields to internal "no opinion" values ---
        # Property-wrapped fields use leading-underscore storage; text fields
        # use the _UNSET_TEXT sentinel so the property setter's "explicit
        # clear via None" path is distinguishable from constructor-omitted.
        self._title: str = _UNSET_TEXT
        self._xlabel: str = _UNSET_TEXT
        self._ylabel: str = _UNSET_TEXT
        self._include_legend: bool = True

        self._legend_options = build_legend_options()

        self._xlabel_style: AxisLabelStyle | None = None
        self._ylabel_style: AxisLabelStyle | None = None
        self._title_style: TitleStyle | None = None

        self._x_tick_locations: list[float] | None = None
        self._x_tick_labels: list[str] | None = None
        self._x_limits: tuple[float, float] | None = None
        self._x_tick_style: TickStyle | None = None

        self._y_tick_locations: list[float] | None = None
        self._y_tick_labels: list[str] | None = None
        self._y_limits: tuple[float, float] | None = None
        self._y_tick_style: TickStyle | None = None

        self._annotations = _Annotations()

        self._frame_visibility: dict[str, bool] = {
            "top": True,
            "right": True,
            "bottom": True,
            "left": True,
        }

        # --- Pass 3: artist registry + managed-axes state ---
        # Created and initialized BEFORE re-applying constructor args so any
        # pre-existing state on a user-supplied axes is classified first.
        self._artists = _ArtistRegistry()
        self._axes_state = _ManagedAxesState()
        self._axes_state_initialized: bool = False
        self._axes_state.initialize_from_ax(self._ax)
        self._axes_state_initialized = True

        # --- Pass 4: re-apply non-default constructor args via property setters ---
        # Setters reclaim their managed unit; omitted args stay "no opinion".
        if title is not None:
            self.title = title
        if xlabel is not None:
            self.xlabel = xlabel
        if ylabel is not None:
            self.ylabel = ylabel
        if include_legend is not True:
            self.include_legend = include_legend

    # ------------------------------------------------------------------
    # Property wrappers for label/title/legend (atomic managed-axes units)
    # ------------------------------------------------------------------

    @property
    def title(self) -> str | None:
        """Plot title, or None if unset."""
        return None if self._title is _UNSET_TEXT else self._title

    @title.setter
    def title(self, value: str | None) -> None:
        self._title = _UNSET_TEXT if value is _UNSET_TEXT else value  # type: ignore[assignment]
        # Sentinel re-assignment only happens from internal code; user values
        # are str | None. Reclaim only when the state machine is initialized
        # (suppressed during the two-pass __init__).
        if self._axes_state_initialized and value is not _UNSET_TEXT:
            self._apply_title_now()
            self._axes_state.reclaim_and_mark(UNIT_TITLE, self._snapshot_title())

    @property
    def xlabel(self) -> str | None:
        """X-axis label text, or None if unset."""
        return None if self._xlabel is _UNSET_TEXT else self._xlabel

    @xlabel.setter
    def xlabel(self, value: str | None) -> None:
        self._xlabel = _UNSET_TEXT if value is _UNSET_TEXT else value  # type: ignore[assignment]
        if self._axes_state_initialized and value is not _UNSET_TEXT:
            self._apply_xlabel_now()
            self._axes_state.reclaim_and_mark(UNIT_X_LABEL, self._snapshot_xlabel())

    @property
    def ylabel(self) -> str | None:
        """Y-axis label text, or None if unset."""
        return None if self._ylabel is _UNSET_TEXT else self._ylabel

    @ylabel.setter
    def ylabel(self, value: str | None) -> None:
        self._ylabel = _UNSET_TEXT if value is _UNSET_TEXT else value  # type: ignore[assignment]
        if self._axes_state_initialized and value is not _UNSET_TEXT:
            self._apply_ylabel_now()
            self._axes_state.reclaim_and_mark(UNIT_Y_LABEL, self._snapshot_ylabel())

    @property
    def include_legend(self) -> bool:
        """Whether to render a legend on rebuild."""
        return self._include_legend

    @include_legend.setter
    def include_legend(self, value: bool) -> None:
        self._include_legend = bool(value)
        # Legend identity is recorded by ``_apply_legend`` at the next
        # rebuild. The store-and-claim path keeps last-applied history clean
        # of placeholder values.
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_LEGEND)

    def _claim_legend_if_named(self, name: str | None) -> None:
        """Reclaim the legend unit when an ``add_*`` is given a user-supplied name.

        Any ``add_*`` call with a non-None ``name`` reclaims the legend unit,
        because the resulting legend content is derived from named plot
        elements. ``add_*`` calls with ``name=None`` do not reclaim — they
        may still contribute to the legend handles (subclasses often fall
        back to an auto-generated label), but they should not displace an
        externally-placed legend.

        Subclasses' named-add methods (``add_histogram``, ``add_boxplot_datasets``,
        etc.) call this helper immediately after appending to their data list.
        """
        if name is not None and self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_LEGEND)

    # ------------------------------------------------------------------
    # Snapshot helpers used by reclaim paths
    # ------------------------------------------------------------------

    def _snapshot_xlabel(self) -> _LabelSnapshot:
        return _label_snapshot(self._ax, "x")

    def _snapshot_ylabel(self) -> _LabelSnapshot:
        return _label_snapshot(self._ax, "y")

    def _snapshot_title(self) -> _TitleSnapshot:
        # Title pad is unobservable via public matplotlib getters in 3.10.6,
        # so we round-trip the last-applied pad rather than reading from mpl.
        pad: float | None = None
        if self._title_style is not None:
            pad = self._title_style.pad
        return _title_snapshot(self._ax, pad=pad)

    def bind_to_ax(self, ax: Axes | None) -> None:
        """Retarget this plot to render onto a different matplotlib ``Axes``.

        The plot's accumulated state (added series, lines, bands, arrows,
        labels, styles, etc.) is preserved and re-applied to the new axes on
        the next access to :attr:`ax` (or call to :meth:`show` / :meth:`save`).
        Any prior rendered output on the *old* axes is left alone; this plot
        simply stops managing it.

        Pass ``ax=None`` to unbind — the plot creates a fresh figure on the
        next render, just as it did on construction.

        Args:
            ax (matplotlib.axes.Axes | None): The matplotlib axes to render onto,
                or ``None`` to revert to a fresh-figure render.
        """
        # Suppress reclaim during the re-classification step. Mirrors the
        # two-pass init contract so initialize_from_ax sees a clean state.
        self._axes_state_initialized = False

        if ax is None:
            self.fig, self._ax = plt.subplots(
                figsize=self._figure_size,
                dpi=self._figure_dpi,
            )
            try:
                from IPython import get_ipython

                ip = get_ipython()
                if (
                    ip is not None and getattr(ip, "kernel", None) is not None
                ):  # pragma: no cover - only reachable inside a live Jupyter kernel
                    plt.close(self.fig)  # pragma: no cover
            except Exception:  # pragma: no cover - IPython import failure is suppressed
                pass  # pragma: no cover
            self._finalizer = weakref.finalize(self, plt.close, self.fig)
        else:
            self.fig = cast(Figure, ax.figure)
            self._ax = ax
            # User owns the figure now; clear any finalizer we registered earlier.
            if self._finalizer is not None:
                self._finalizer.detach()
            self._finalizer = None

        # Detach the artist registry from the old axes without removing its
        # artists — rebind is non-destructive.
        self._artists = _ArtistRegistry()

        # Reset per-axes last-applied history; reclaim flags survive because
        # they describe plot configuration, not the previous axes.
        self._axes_state.reset_history()
        self._axes_state.initialize_from_ax(self._ax)
        self._axes_state_initialized = True

    def add_vertical_lines(
        self,
        x_values: float | Iterable[float],
        *,
        line_options: LineOptions | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a vertical line to the figure.

        Args:
            x_values (float | Iterable[float]): The x-value(s) where the vertical line(s) should be
                drawn.
            line_options (LineOptions | None, optional): Optional pre-built styling. Any styling
                kwarg passed explicitly overrides the corresponding field on ``line_options``.
                Defaults to None.
            linecolor (Color, optional): The color of the vertical line. Defaults to "#cccccc".
            linealpha (float | None, optional): The alpha transparency of the vertical line.
                Defaults to None in which case the alpha from linecolor is used if specified.
            linestyle (str, optional): The linestyle of the vertical line. Defaults to "-".
            linewidth (float, optional): The width of the vertical line. Defaults to 1.0.
            zorder (int, optional): The z-order of the vertical line. Defaults to 3.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        if isinstance(x_values, (str, bytes)):
            raise TypeError("x_values must be a number or an iterable of numbers, not a string.")
        if isinstance(x_values, bool):
            raise TypeError("x_values must be a number or an iterable of numbers, not a bool.")
        # Safe to shadow here because we pass ints and floats by value not object reference
        if isinstance(x_values, Real):
            x_values = [float(x_values)]

        base = line_options if line_options is not None else LineOptions()
        resolved_linecolor = linecolor if linecolor is not None else base.linecolor
        resolved_linealpha = linealpha if linealpha is not None else base.linealpha
        resolved_linestyle = linestyle if linestyle is not None else base.linestyle
        resolved_linewidth = linewidth if linewidth is not None else base.linewidth
        resolved_zorder = zorder if zorder is not None else base.zorder

        xs = _coerce_real_iter(x_values, field="x_values")
        self._annotations.vertical_lines.append(
            LineData(
                values=xs,
                linecolor=resolved_linecolor,
                linealpha=resolved_linealpha,
                linestyle=resolved_linestyle,
                linewidth=float(resolved_linewidth),
                zorder=resolved_zorder,
                name=name,
            )
        )

    def add_vertical_band(
        self,
        x_low: float,
        x_high: float,
        *,
        band_options: BandOptions | None = None,
        bandcolor: Color | None = None,
        bandalpha: float | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a vertical band to the figure.

        Args:
            x_low (float): The lower x-value of the vertical band.
            x_high (float): The upper x-value of the vertical band.
            bandcolor (Color, optional): The fill color of the band. Defaults to "#cccccc".
            bandalpha (float | None, optional): The alpha transparency of the band. Defaults to None.
            linecolor (Color | None, optional): The color of the bounding lines of the band.
                If set to None and bandcolor is also None, defaults to "#cccccc".
                If set to None and bandcolor is not None, defaults to bandcolor.
                Defaults to None.
            linealpha (float | None, optional): The alpha transparency of the bounding lines.
                Defaults to None which uses the alpha from linecolor if specified.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to 3.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        base = band_options if band_options is not None else BandOptions()
        resolved_bandcolor = bandcolor if bandcolor is not None else base.bandcolor
        resolved_bandalpha = bandalpha if bandalpha is not None else base.bandalpha
        resolved_linecolor = linecolor if linecolor is not None else base.linecolor
        resolved_linealpha = linealpha if linealpha is not None else base.linealpha
        resolved_linestyle = linestyle if linestyle is not None else base.linestyle
        resolved_linewidth = linewidth if linewidth is not None else base.linewidth
        resolved_zorder = zorder if zorder is not None else base.zorder

        self._annotations.vertical_bands.append(
            BandData(
                lower_bound=min(x_low, x_high),
                upper_bound=max(x_low, x_high),
                bandcolor=resolved_bandcolor,
                bandalpha=resolved_bandalpha,
                linecolor=resolved_linecolor,
                linealpha=resolved_linealpha,
                linestyle=resolved_linestyle,
                linewidth=resolved_linewidth,
                zorder=resolved_zorder,
                name=name,
            )
        )

    def add_horizontal_lines(
        self,
        y_values: float | Iterable[float],
        *,
        line_options: LineOptions | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a horizontal line to the figure.

        Args:
            y_values (float | Iterable[float]): The y-value(s) where the horizontal line(s) should
                be drawn.
            linecolor (Color, optional): The color of the horizontal line. Defaults to "#cccccc".
            linealpha (float | None, optional): The alpha transparency of the horizontal line.
                Defaults to None in which case the alpha from linecolor is used if specified.
            linestyle (str, optional): The linestyle of the horizontal line. Defaults to "-".
            linewidth (float, optional): The width of the horizontal line. Defaults to 1.0.
            zorder (int, optional): The z-order of the horizontal line. Defaults to 4.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        if isinstance(y_values, (str, bytes)):
            raise TypeError("y_values must be a number or an iterable of numbers, not a string.")
        if isinstance(y_values, bool):
            raise TypeError("y_values must be a number or an iterable of numbers, not a bool.")
        # Safe to shadow here because we pass ints and floats by value not object reference
        if isinstance(y_values, Real):
            y_values = [float(y_values)]

        ys = _coerce_real_iter(y_values, field="y_values")

        base = line_options if line_options is not None else LineOptions(zorder=4)
        resolved_linecolor = linecolor if linecolor is not None else base.linecolor
        resolved_linealpha = linealpha if linealpha is not None else base.linealpha
        resolved_linestyle = linestyle if linestyle is not None else base.linestyle
        resolved_linewidth = linewidth if linewidth is not None else base.linewidth
        resolved_zorder = zorder if zorder is not None else base.zorder

        self._annotations.horizontal_lines.append(
            LineData(
                values=ys,
                linecolor=resolved_linecolor,
                linealpha=resolved_linealpha,
                linestyle=resolved_linestyle,
                linewidth=float(resolved_linewidth),
                zorder=resolved_zorder,
                name=name,
            )
        )
        return

    def add_horizontal_band(
        self,
        y_low: float,
        y_high: float,
        *,
        band_options: BandOptions | None = None,
        bandcolor: Color | None = None,
        bandalpha: float | None = None,
        linecolor: Color | None = None,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a horizontal band to the figure.

        Args:
            y_low (float): The lower y-value of the horizontal band.
            y_high (float): The upper y-value of the horizontal band.
            bandcolor (Color | None, optional): The fill color of the band. Defaults to "#cccccc".
            bandalpha (float | None, optional): The alpha transparency of the band. Defaults to None
            linecolor (Color | None, optional): The color of the bounding lines of the band.
                If set to None and bandcolor is also None, defaults to "#cccccc".
                If set to None and bandcolor is not None, defaults to bandcolor.
                Defaults to None.
            linealpha (float | None, optional): The alpha transparency of the bounding lines.
                Defaults to None which uses the alpha from linecolor if specified.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to 4.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        base = band_options if band_options is not None else BandOptions(zorder=4)
        resolved_bandcolor = bandcolor if bandcolor is not None else base.bandcolor
        resolved_bandalpha = bandalpha if bandalpha is not None else base.bandalpha
        resolved_linecolor = linecolor if linecolor is not None else base.linecolor
        resolved_linealpha = linealpha if linealpha is not None else base.linealpha
        resolved_linestyle = linestyle if linestyle is not None else base.linestyle
        resolved_linewidth = linewidth if linewidth is not None else base.linewidth
        resolved_zorder = zorder if zorder is not None else base.zorder

        self._annotations.horizontal_bands.append(
            BandData(
                lower_bound=min(y_low, y_high),
                upper_bound=max(y_low, y_high),
                bandcolor=resolved_bandcolor,
                bandalpha=resolved_bandalpha,
                linecolor=resolved_linecolor,
                linealpha=resolved_linealpha,
                linestyle=resolved_linestyle,
                linewidth=resolved_linewidth,
                zorder=resolved_zorder,
                name=name,
            )
        )

    def add_text_arrow(
        self,
        arrowtip: tuple[float, float],
        direction: Literal["right", "left", "up", "down"],
        text: str = "   ",
        *,
        textrotation: float | None = None,
        arrowfacecolor: Color | None = None,
        arrowfacealpha: float | None = None,
        arrowedgecolor: Color | None = None,
        arrowedgealpha: float | None = None,
        arrowedgewidth: float | None = None,
        arrowtextstyle: ArrowTextStyle | None = None,
        arrowplacement: ArrowPlacement | None = None,
        arrowstyle: TextArrowStyle | None = None,
        name: str | None = None,
    ) -> None:
        """Add a deferred text-style arrow to the plot.

        This renders via ``Axes.text(..., bbox=...)`` and stores the arrow so it is redrawn
        whenever the plot is rebuilt. The arrow tip is aligned to ``arrowtip`` during rendering.

        Args:
            arrowtip (tuple[float, float]): Arrow-tip coordinate in the selected placement
                coordinate system.
            direction (Literal["right", "left", "up", "down"]): Arrow direction.
            text (str, optional): Text drawn inside the arrow box. Empty strings are normalized
                to ``"   "`` so the arrow still renders. Defaults to ``"   "``.
            textrotation (float | None, optional): Top-level text rotation override in degrees.
                When set, this overrides ``arrowtextstyle.rotation``. Defaults to None.
            arrowfacecolor (Color | None, optional): Optional override for
                ``arrowstyle.arrowfacecolor``. Defaults to None.
            arrowfacealpha (float | None, optional): Optional override for
                ``arrowstyle.arrowfacealpha``. Defaults to None.
            arrowedgecolor (Color | None, optional): Optional override for
                ``arrowstyle.arrowedgecolor``. Defaults to None.
            arrowedgealpha (float | None, optional): Optional override for
                ``arrowstyle.arrowedgealpha``. Defaults to None.
            arrowedgewidth (float | None, optional): Optional override for
                ``arrowstyle.arrowedgewidth``. Defaults to None.
            arrowtextstyle (AnnotationArrowTextStyle | None, optional): Text styling options
                (font, alignment, outline, and rotation). Defaults to None.
            arrowplacement (AnnotationArrowPlacement | None, optional): Placement options
                (coordinate system, offsets, clipping, and z-order). Defaults to None.
            arrowstyle (TextAnnotationArrowStyle | None, optional): Text-arrow box styling
                options. Defaults to None.
            name (str | None, optional): Optional identifier for callers. Defaults to None.

        Returns:
            None
        """
        base_text_style = arrowtextstyle if arrowtextstyle is not None else ArrowTextStyle()
        if textrotation is None:
            arrow_text_style = base_text_style
        else:
            arrow_text_style = ArrowTextStyle(
                fontsize=base_text_style.fontsize,
                fontcolor=base_text_style.fontcolor,
                fontalpha=base_text_style.fontalpha,
                fontoutlinecolor=base_text_style.fontoutlinecolor,
                fontoutlinealpha=base_text_style.fontoutlinealpha,
                fontoutlinewidth=base_text_style.fontoutlinewidth,
                fontweight=base_text_style.fontweight,
                fontstyle=base_text_style.fontstyle,
                fontfamily=base_text_style.fontfamily,
                rotation=float(textrotation),
                horizontalalignment=base_text_style.horizontalalignment,
                verticalalignment=base_text_style.verticalalignment,
            )
        arrow_placement = arrowplacement if arrowplacement is not None else ArrowPlacement()
        style = arrowstyle if arrowstyle is not None else TextArrowStyle()
        merged_textarrowstyle = TextArrowStyle(
            arrowfacecolor=arrowfacecolor if arrowfacecolor is not None else style.arrowfacecolor,
            arrowfacealpha=arrowfacealpha if arrowfacealpha is not None else style.arrowfacealpha,
            arrowedgecolor=(arrowedgecolor if arrowedgecolor is not None else style.arrowedgecolor),
            arrowedgealpha=(arrowedgealpha if arrowedgealpha is not None else style.arrowedgealpha),
            arrowedgewidth=(arrowedgewidth if arrowedgewidth is not None else style.arrowedgewidth),
            boxpad=style.boxpad,
            boxstyle=style.boxstyle,
        )

        text_value = text if text != "" else "   "
        self._annotations.annotation_arrows.append(
            ArrowData(
                arrowtip=arrowtip,
                direction=direction,
                arrowtype="text",
                text=text_value,
                textstyle=arrow_text_style,
                placement=arrow_placement,
                textarrowstyle=merged_textarrowstyle,
                labelarrowstyle=None,
                name=name,
            )
        )

    def add_label_arrow(
        self,
        arrowtip: tuple[float, float],
        direction: Literal["right", "left", "up", "down"],
        text: str | None = None,
        *,
        label_position: tuple[float, float] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        arrow_length: float | None = None,
        arrowfacecolor: Color | None = None,
        arrowfacealpha: float | None = None,
        arrowedgecolor: Color | None = None,
        arrowedgealpha: float | None = None,
        arrowedgewidth: float | None = None,
        arrowtextstyle: ArrowTextStyle | None = None,
        arrowplacement: ArrowPlacement | None = None,
        arrowstyle: LabelArrowStyle | None = None,
        name: str | None = None,
    ) -> None:
        """Add a deferred label-style arrow to the plot.

        This renders a true annotation arrow and an optional separate text label, so the
        arrow length is controlled by tail placement rather than text size.

        Args:
            arrowtip (tuple[float, float]): Arrow-tip coordinate in the selected placement
                coordinate system.
            direction (Literal["right", "left", "up", "down"]): Arrow direction.
            text (str | None, optional): Optional label text near the arrow tail.
                Defaults to None.
            label_position (tuple[float, float] | None, optional): Optional explicit text-anchor
                position in ``arrowplacement.coordinate_system``. If None, uses the arrow tail
                plus ``arrowplacement.label_padding`` and ``arrowplacement.text_offset``.
                Defaults to None.
            labelfont_options (LabelFontOptions | None, optional): Optional geoplot-style label
                font settings. Defaults to None.
            labelbox_options (LabelBoxOptions | None, optional): Optional geoplot-style text-box
                settings. Defaults to None.
            arrow_length (float | None, optional): Optional label-arrow length as a percent of
                axes span in the arrow direction. ``0`` means zero length, and ``100`` means one
                full axes width (horizontal) or height (vertical). Cannot be combined with
                ``arrowplacement.arrowtail``. Defaults to None.
            arrowfacecolor (Color | None, optional): Optional override for
                ``arrowstyle.arrowfacecolor``. Defaults to None.
            arrowfacealpha (float | None, optional): Optional override for
                ``arrowstyle.arrowfacealpha``. Defaults to None.
            arrowedgecolor (Color | None, optional): Optional override for
                ``arrowstyle.arrowedgecolor``. Defaults to None.
            arrowedgealpha (float | None, optional): Optional override for
                ``arrowstyle.arrowedgealpha``. Defaults to None.
            arrowedgewidth (float | None, optional): Optional override for
                ``arrowstyle.arrowedgewidth``. Defaults to None.
            arrowtextstyle (AnnotationArrowTextStyle | None, optional): Text style settings used
                for alignment/rotation and as a fallback when ``labelfont_options`` is None.
                Defaults to None.
            arrowplacement (AnnotationArrowPlacement | None, optional): Placement settings.
                Defaults to None. When not provided, this method uses
                ``AnnotationArrowPlacement(tail_length=0.04)``.
            arrowstyle (LabelAnnotationArrowStyle | None, optional): Base label-arrow styling
                options. Defaults to None.
            name (str | None, optional): Optional identifier for callers. Defaults to None.

        Returns:
            None
        """
        arrow_text_style = arrowtextstyle if arrowtextstyle is not None else ArrowTextStyle()
        arrow_placement = (
            arrowplacement if arrowplacement is not None else ArrowPlacement(tail_length=0.04)
        )
        arrow_length_percentage: float | None = None
        if arrow_length is not None:
            arrow_length_value = float(arrow_length)
            if not math.isfinite(arrow_length_value):
                raise ValueError("arrow_length must be finite.")
            if not (0.0 <= arrow_length_value <= 100.0):
                raise ValueError("arrow_length must be in [0, 100].")
            if arrow_placement.arrowtail is not None:
                raise ValueError("arrow_length cannot be set when placement.arrowtail is set.")
            arrow_length_percentage = arrow_length_value
        style = arrowstyle if arrowstyle is not None else LabelArrowStyle()
        merged_labelarrowstyle = LabelArrowStyle(
            arrowstyle=style.arrowstyle,
            connectionstyle=style.connectionstyle,
            arrowhead_scale=style.arrowhead_scale,
            shrink_a=style.shrink_a,
            shrink_b=style.shrink_b,
            arrowfacecolor=arrowfacecolor if arrowfacecolor is not None else style.arrowfacecolor,
            arrowfacealpha=arrowfacealpha if arrowfacealpha is not None else style.arrowfacealpha,
            arrowedgecolor=(arrowedgecolor if arrowedgecolor is not None else style.arrowedgecolor),
            arrowedgealpha=(arrowedgealpha if arrowedgealpha is not None else style.arrowedgealpha),
            arrowedgewidth=(arrowedgewidth if arrowedgewidth is not None else style.arrowedgewidth),
            linestyle=style.linestyle,
        )

        self._annotations.annotation_arrows.append(
            ArrowData(
                arrowtip=arrowtip,
                direction=direction,
                arrowtype="label",
                text=text,
                textstyle=arrow_text_style,
                arrow_length_percentage=arrow_length_percentage,
                label_position=label_position,
                labelfont_options=labelfont_options,
                labelbox_options=labelbox_options,
                placement=arrow_placement,
                textarrowstyle=None,
                labelarrowstyle=merged_labelarrowstyle,
                name=name,
            )
        )

    def clear_annotation_arrows(self) -> None:
        """Clear all annotation arrows from the figure."""
        self._annotations.clear_annotation_arrows()

    def clear_verticals(self) -> None:
        """Clear all vertical lines and bands from the figure."""
        self._annotations.clear_verticals()

    def clear_horizontals(self) -> None:
        """Clear all horizontal lines and bands from the figure."""
        self._annotations.clear_horizontals()

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get subclass-provided default x-tick locations.

        Returns:
            list[float] | None: Default x-tick locations, or None to keep Matplotlib defaults.
        """
        return None

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get subclass-provided default x-tick labels.

        Args:
            tick_locations (list[float]): Final x-tick locations selected for the axes.

        Returns:
            list[str] | None: Tick labels aligned to ``tick_locations``, or None to keep current
                labels.
        """
        return None

    # ------------------------------------------------------------------
    # Per-managed-unit apply helpers (used by the rebuild flow). Each:
    #   - skips entirely when its unit is in the external set;
    #   - otherwise applies gerrytools state to ``self._ax`` and records
    #     either explicit-reclaim (via reclaim_and_mark) or default ownership
    #     (via record_default) so the next rebuild's external detection works.
    # ------------------------------------------------------------------

    def _apply_x_limits(self, external: set[str]) -> None:
        if UNIT_X_LIMITS in external:
            return
        if self._x_limits is not None:
            self._ax.set_xlim(*self._x_limits)
            self._axes_state.reclaim_and_mark(
                UNIT_X_LIMITS, tuple(float(v) for v in self._ax.get_xlim())
            )
        else:
            # Implicit default: matplotlib's autoscale has set the data range.
            self._axes_state.record_default(
                UNIT_X_LIMITS, tuple(float(v) for v in self._ax.get_xlim())
            )

    def _apply_y_limits(self, external: set[str]) -> None:
        if UNIT_Y_LIMITS in external:
            return
        if self._y_limits is not None:
            self._ax.set_ylim(*self._y_limits)
            self._axes_state.reclaim_and_mark(
                UNIT_Y_LIMITS, tuple(float(v) for v in self._ax.get_ylim())
            )
        else:
            self._axes_state.record_default(
                UNIT_Y_LIMITS, tuple(float(v) for v in self._ax.get_ylim())
            )

    def _apply_x_ticks(self, external: set[str]) -> None:
        if UNIT_X_TICKS in external:
            return
        # Determine locations: explicit user set, subclass default, or
        # leave matplotlib's locator alone. Materializing get_xticks() into
        # a set_xticks() call would convert matplotlib's dynamic locator
        # into a FixedLocator that persists across rebuilds — and because
        # we no longer ax.clear() between rebuilds, that stale locator
        # would expand xlim to cover its stored ticks on later rebuilds.
        if self._x_tick_locations is not None:
            x_tick_locations: list[float] | None = list(self._x_tick_locations)
        else:
            default_locs = self._default_x_tick_locations()
            x_tick_locations = list(default_locs) if default_locs is not None else None
        if x_tick_locations is not None:
            self._ax.set_xticks(x_tick_locations)
        if self._x_tick_labels == []:
            self._ax.tick_params(axis="x", labelbottom=False)
            self._record_x_ticks()
            return
        if self._x_tick_labels is None:
            if x_tick_locations is None:
                # Matplotlib will compute labels from the locator at draw
                # time; nothing to apply.
                self._record_x_ticks()
                return
            labels = self._default_x_tick_labels(x_tick_locations)
            if labels is not None:
                self._ax.set_xticklabels(list(labels))
            self._record_x_ticks()
            return
        if x_tick_locations is None:
            # Labels supplied without locations: materialize current ticks
            # via set_xticks so set_xticklabels matches them and matplotlib
            # does not warn about labels-without-fixed-locator.
            x_tick_locations = self._ax.get_xticks().tolist()
            self._ax.set_xticks(x_tick_locations)
        if len(self._x_tick_labels) != len(x_tick_locations):
            raise ValueError(
                f"Expected {len(x_tick_locations)} x tick labels, got {len(self._x_tick_labels)}."
            )
        self._ax.set_xticklabels(list(self._x_tick_labels))
        self._record_x_ticks()

    def _record_x_ticks(self) -> None:
        """Record what we just applied for the x_ticks unit.

        Reclaim-vs-default depends on whether a public setter has already
        claimed the unit: store-and-claim (update_xtick_labels, set_xticks,
        clear_xticks, clear_xtick_labels) calls ``reclaim_without_value``,
        leaving the unit in the gerrytools_explicit ownership state. In that
        case we keep ownership and record the concrete snapshot. Otherwise
        this is an implicit default applied via subclass default hooks.
        """
        from gerrytools.plotting._axes_state import _tick_snapshot

        snapshot = _tick_snapshot(self._ax, "x")
        if self._axes_state.is_reclaimed(UNIT_X_TICKS):
            self._axes_state.reclaim_and_mark(UNIT_X_TICKS, snapshot)
        else:
            self._axes_state.record_default(UNIT_X_TICKS, snapshot)

    def _apply_y_ticks(self, external: set[str]) -> None:
        if UNIT_Y_TICKS in external:
            return
        # See _apply_x_ticks for the "don't materialize matplotlib's
        # default locator into a FixedLocator" rationale.
        if self._y_tick_locations is not None:
            y_tick_locations: list[float] | None = list(self._y_tick_locations)
            self._ax.set_yticks(y_tick_locations)
        else:
            y_tick_locations = None

        if self._y_tick_labels is None:
            self._record_y_ticks()
            return
        if self._y_tick_labels == []:
            self._ax.tick_params(axis="y", labelleft=False)
            self._record_y_ticks()
            return
        if y_tick_locations is None:
            # Labels supplied without locations: materialize current ticks
            # via set_yticks so set_yticklabels matches them and matplotlib
            # does not warn about labels-without-fixed-locator.
            y_tick_locations = self._ax.get_yticks().tolist()
            self._ax.set_yticks(y_tick_locations)
        if len(self._y_tick_labels) != len(y_tick_locations):
            raise ValueError(
                f"Expected {len(y_tick_locations)} y tick labels, got {len(self._y_tick_labels)}."
            )
        self._ax.set_yticklabels(list(self._y_tick_labels))
        self._record_y_ticks()

    def _record_y_ticks(self) -> None:
        from gerrytools.plotting._axes_state import _tick_snapshot

        snapshot = _tick_snapshot(self._ax, "y")
        if self._axes_state.is_reclaimed(UNIT_Y_TICKS):
            self._axes_state.reclaim_and_mark(UNIT_Y_TICKS, snapshot)
        else:
            self._axes_state.record_default(UNIT_Y_TICKS, snapshot)

    def _apply_x_tick_style(self, external: set[str]) -> None:
        if UNIT_X_TICK_STYLE in external:
            return
        if self._x_tick_style is None:
            return
        self._apply_tick_style("x", self._x_tick_style)
        self._axes_state.reclaim_and_mark(UNIT_X_TICK_STYLE, _tick_style_snapshot(self._ax, "x"))

    def _apply_y_tick_style(self, external: set[str]) -> None:
        if UNIT_Y_TICK_STYLE in external:
            return
        if self._y_tick_style is None:
            return
        self._apply_tick_style("y", self._y_tick_style)
        self._axes_state.reclaim_and_mark(UNIT_Y_TICK_STYLE, _tick_style_snapshot(self._ax, "y"))

    def _apply_xlabel(self, external: set[str]) -> None:
        if UNIT_X_LABEL in external:
            return
        if self._xlabel is _UNSET_TEXT:
            # No gerrytools opinion. Leave any pre-set text alone.
            return
        self._apply_xlabel_now()
        self._axes_state.reclaim_and_mark(UNIT_X_LABEL, self._snapshot_xlabel())

    def _apply_ylabel(self, external: set[str]) -> None:
        if UNIT_Y_LABEL in external:
            return
        if self._ylabel is _UNSET_TEXT:
            return
        self._apply_ylabel_now()
        self._axes_state.reclaim_and_mark(UNIT_Y_LABEL, self._snapshot_ylabel())

    def _apply_title(self, external: set[str]) -> None:
        if UNIT_TITLE in external:
            return
        if self._title is _UNSET_TEXT:
            return
        self._apply_title_now()
        self._axes_state.reclaim_and_mark(UNIT_TITLE, self._snapshot_title())

    def _apply_x_scale(self, external: set[str]) -> None:
        """Reconcile the ``x_scale`` managed unit.

        Gerrytools data plots do not currently set a matplotlib axis scale
        (``"linear" / "log" / "symlog"``); PaintBall's ``set_xscale`` is a
        data-transform factor, not a call to ``ax.set_xscale``. So the apply
        helper records the current scale as a gerrytools default so the next
        rebuild's external-detection has an anchor: if the user runs
        ``ax.set_xscale("log")`` between rebuilds, the snapshot will differ
        from this recorded default and the unit will yield to external state.
        """
        if UNIT_X_SCALE in external:
            return
        self._axes_state.record_default(UNIT_X_SCALE, self._ax.get_xscale())

    def _apply_y_scale(self, external: set[str]) -> None:
        """Reconcile the ``y_scale`` managed unit. See ``_apply_x_scale``."""
        if UNIT_Y_SCALE in external:
            return
        self._axes_state.record_default(UNIT_Y_SCALE, self._ax.get_yscale())

    def _apply_aspect_now(self) -> None:
        """Subclass hook: apply the desired aspect to ``self._ax``.

        Base implementation is a no-op (gerrytools defaults to matplotlib's
        ``"auto"``). Subclasses that need a fixed or computed aspect (e.g.
        ``SeatsVotes``, ``PaintBall``) override this and call
        ``self._ax.set_aspect(...)``. The base ``_apply_aspect`` helper then
        records the resulting aspect as a gerrytools default so external
        ``ax.set_aspect(...)`` changes between rebuilds are detected and
        respected on the next pass.
        """

    def _apply_aspect(self, external: set[str]) -> None:
        """Reconcile the ``aspect`` managed unit.

        Subclass hook ``_apply_aspect_now`` controls what gerrytools applies.
        External changes (a user ``ax.set_aspect("auto")`` between rebuilds)
        win until a subclass override drives a new value through this path.
        """
        if UNIT_ASPECT in external:
            return
        self._apply_aspect_now()
        self._axes_state.record_default(UNIT_ASPECT, self._ax.get_aspect())

    def update_xtick_labels(
        self, *, locations: list[float] | None = None, labels: list[str] | None = None
    ) -> None:
        """Update x-tick locations and/or labels.

        Overrides existing values if provided.

        Args:
            locations (list[float] | None, optional): New x-tick locations. Defaults
                to None.
            labels (list[str] | None, optional): New x-tick labels. Defaults to
                None.

        Raises:
            ValueError: If the lengths of provided locations and labels do not match
                existing values.

        Returns:
            None
        """
        if locations is None and labels is None:
            return

        if locations is not None and labels is not None:
            if (locations == [] and labels not in (None, [])) or (
                labels == [] and locations not in (None, [])
            ):
                raise ValueError(
                    "If clearing ticks/labels, clear both (locations=[] and labels=[])."
                )

            if labels != [] and locations != [] and len(locations) != len(labels):
                raise ValueError(
                    f"Locations length {len(locations)} does not match labels length {len(labels)}."
                )
            self._x_tick_locations = list(locations)
            self._x_tick_labels = list(labels)
            self._claim_x_ticks()
            return

        if locations is not None:
            if (
                self._x_tick_labels is not None
                and self._x_tick_labels != []
                and locations != []
                and len(locations) != len(self._x_tick_labels)
            ):
                raise ValueError(
                    f"Locations length {len(locations)} does not match existing labels length "
                    f"{len(self._x_tick_labels)}."
                )
            if locations == [] and labels is None:
                self._x_tick_locations = []
                self._x_tick_labels = []
                self._claim_x_ticks()
                return
            self._x_tick_locations = list(locations)
            self._claim_x_ticks()
            return

        if labels is not None:
            if labels == []:
                self._x_tick_labels = []
                self._claim_x_ticks()
                return

            if (
                self._x_tick_locations is not None
                and self._x_tick_locations != []
                and len(labels) != len(self._x_tick_locations)
            ):
                raise ValueError(
                    f"Labels length {len(labels)} does not match existing locations length "
                    f"{len(self._x_tick_locations)}."
                )
            self._x_tick_labels = list(labels)
            self._claim_x_ticks()
            return

    def _claim_x_ticks(self) -> None:
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_X_TICKS)

    def _claim_y_ticks(self) -> None:
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_Y_TICKS)

    def update_ytick_labels(
        self, *, locations: list[float] | None = None, labels: list[str] | None = None
    ) -> None:
        """Update y-tick locations and/or labels.

        Overrides existing values if provided.

        Args:
            locations (list[float] | None, optional): New y-tick locations. Defaults
                to None.
            labels (list[str] | None, optional): New y-tick labels. Defaults to
                None.

        Raises:
            ValueError: If the lengths of provided locations and labels do not match
                existing values.

        Returns:
            None
        """
        if locations is None and labels is None:
            return

        if locations is not None and labels is not None:
            if (locations == [] and labels not in (None, [])) or (
                labels == [] and locations not in (None, [])
            ):
                raise ValueError(
                    "If clearing ticks/labels, clear both (locations=[] and labels=[])."
                )

            if labels != [] and locations != [] and len(locations) != len(labels):
                raise ValueError(
                    f"Locations length {len(locations)} does not match labels length {len(labels)}."
                )
            self._y_tick_locations = list(locations)
            self._y_tick_labels = list(labels)
            self._claim_y_ticks()
            return

        if locations is not None:
            if (
                self._y_tick_labels is not None
                and self._y_tick_labels != []
                and locations != []
                and len(locations) != len(self._y_tick_labels)
            ):
                raise ValueError(
                    f"Locations length {len(locations)} does not match existing labels length "
                    f"{len(self._y_tick_labels)}."
                )
            if locations == [] and labels is None:
                self._y_tick_locations = []
                self._y_tick_labels = []
                self._claim_y_ticks()
                return
            self._y_tick_locations = list(locations)
            self._claim_y_ticks()
            return

        if labels is not None:
            if labels == []:
                self._y_tick_labels = []
                self._claim_y_ticks()
                return
            if (
                self._y_tick_locations is not None
                and self._y_tick_locations != []
                and len(labels) != len(self._y_tick_locations)
            ):
                raise ValueError(
                    f"Labels length {len(labels)} does not match existing locations length "
                    f"{len(self._y_tick_locations)}."
                )
            self._y_tick_labels = list(labels)
            self._claim_y_ticks()
            return

    @staticmethod
    def _apply_ticklabel_textprops(
        labels: Iterable[Text],
        *,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
    ) -> None:
        """Apply text properties to tick labels.

        Args:
            labels (Iterable[Text]): Iterable of Matplotlib tick-label ``Text`` objects.
            fontweight (str | None, optional): Font weight to apply. Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style to apply. Defaults to None.
            fontfamily (str | None, optional): Font family to apply. Defaults to None.

        Returns:
            None
        """
        # These are matplotlib.text.Text objects.
        for text in labels:
            if fontweight is not None:
                text.set_fontweight(fontweight)
            if fontstyle is not None:
                text.set_fontstyle(fontstyle)
            if fontfamily is not None:
                text.set_fontfamily(fontfamily)

    def _resolved_rgba(
        self,
        color: Color,
        alpha: float | None = None,
        *,
        field: str = "color",
    ) -> tuple[float, float, float, float]:
        """Resolve a ``Color`` plus optional alpha override to an RGBA tuple.

        Args:
            color (Color): GerryTools color input.
            alpha (float | None, optional): Optional alpha override. Defaults to None.
            field (str, optional): Field name used in validation and warning messages.
                Defaults to ``"color"``.

        Returns:
            tuple[float, float, float, float]: Resolved RGBA values in ``[0, 1]``.
        """
        resolved_color, resolved_alpha = resolve_color_and_alpha(
            color,
            alpha=alpha,
            allow_none=True,
            field=field,
            owner=self.__class__.__name__,
            logger=logger,
        )
        rgba = mcolors.to_rgba(resolved_color, alpha=resolved_alpha)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))

    def _apply_tick_style(self, axis: Literal["x", "y", "both"], style: TickStyle) -> None:
        """Apply tick style to the specified axis.

        Args:
            axis (Literal["x", "y", "both"]): The axis to apply the style to.
            style (TickStyle): The tick style to apply.

        Returns:
            None
        """
        # Tick marks + tick label basics
        label_color_resolved = self._resolved_rgba(
            style.fontcolor,
            style.fontalpha,
            field="fontcolor",
        )
        tick_color_resolved = self._resolved_rgba(
            style.tickcolor,
            style.tickalpha,
            field="tickcolor",
        )
        self._ax.tick_params(
            axis=axis,
            which=style.ticktype,
            labelsize=style.size,
            rotation=style.rotation,
            labelcolor=label_color_resolved,
            color=tick_color_resolved,
        )

        # Tick label text styling (weight/style/family)
        if axis in ("x", "both"):
            if style.ticktype in ("major", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_xticklabels(minor=False),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )
            if style.ticktype in ("minor", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_xticklabels(minor=True),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )

        if axis in ("y", "both"):
            if style.ticktype in ("major", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_yticklabels(minor=False),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )
            if style.ticktype in ("minor", "both"):
                self._apply_ticklabel_textprops(
                    self._ax.get_yticklabels(minor=True),
                    fontweight=style.fontweight,
                    fontstyle=style.fontstyle,
                    fontfamily=style.fontfamily,
                )

    # ------------------------------------------------------------------
    # Atomic label/title writers (shared by property setters and the rebuild
    # flow). ``_apply_<unit>_now`` writes immediately; ``_apply_<unit>``
    # (defined below) adds the rebuild-time external-skip guard.
    # ------------------------------------------------------------------

    def _apply_xlabel_now(self) -> None:
        # ``_xlabel is _UNSET_TEXT`` means "no opinion" — leave any pre-set
        # external xlabel alone. A real ``None`` is the explicit-clear path
        # that user code reached via ``plot.xlabel = None`` after
        # construction; that path applies ``set_xlabel("")`` to clear.
        if self._xlabel is _UNSET_TEXT:
            return
        text = "" if self._xlabel is None else self._xlabel
        if self._xlabel_style is None:
            self._ax.set_xlabel(text)
        else:
            self._ax.set_xlabel(text, **self._xlabel_style.to_mpl_settings_dict())

    def _apply_ylabel_now(self) -> None:
        if self._ylabel is _UNSET_TEXT:
            return
        text = "" if self._ylabel is None else self._ylabel
        if self._ylabel_style is None:
            self._ax.set_ylabel(text)
        else:
            self._ax.set_ylabel(text, **self._ylabel_style.to_mpl_settings_dict())

    def _apply_title_now(self) -> None:
        if self._title is _UNSET_TEXT:
            return
        text = "" if self._title is None else self._title
        if self._title_style is None:
            self._ax.set_title(text)
        else:
            self._ax.set_title(text, **self._title_style.to_mpl_settings_dict())

    def set_xaxis_label_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        labelpad: float | None = None,
    ) -> None:
        """Sets the styling for the x-axis label.

        Args:
            fontsize (float | int | None, optional): Font size for the x-axis label.
                Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color, optional): Color of the x-axis label. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the x-axis label color.
                If None, uses alpha from color if specified. Defaults to None.
            labelpad (float | None, optional): Padding between the x-axis label and the axis
                in points. Defaults to None.

        Returns:
            None
        """
        self._xlabel_style = AxisLabelStyle(
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            labelpad=labelpad,
        )
        # x_label is an atomic text+style managed unit. Reclaim and apply
        # immediately so future ax-level edits are detected as external.
        if self._axes_state_initialized and self.xlabel is not None:
            self._apply_xlabel_now()
            self._axes_state.reclaim_and_mark(UNIT_X_LABEL, self._snapshot_xlabel())
        elif self._axes_state_initialized:
            # Style without text yet: claim the unit so we re-apply on next
            # text assignment; concrete value recorded then.
            self._axes_state.reclaim_without_value(UNIT_X_LABEL)

    def set_yaxis_label_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        labelpad: float | None = None,
    ) -> None:
        """Sets the styling for the y-axis label.

        Args:
            fontsize (float | int | None, optional): Font size for the y-axis label.
                Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color, optional): Color of the y-axis label. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the y-axis label color.
                If None, uses alpha from color if specified. Defaults to None.
            labelpad (float | None, optional): Padding between the y-axis label and the axis
                in points. Defaults to None.

        Returns:
            None
        """
        self._ylabel_style = AxisLabelStyle(
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            labelpad=labelpad,
        )
        if self._axes_state_initialized and self.ylabel is not None:
            self._apply_ylabel_now()
            self._axes_state.reclaim_and_mark(UNIT_Y_LABEL, self._snapshot_ylabel())
        elif self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_Y_LABEL)

    def set_title_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        loc: Literal["left", "center", "right"] | None = None,
        pad: float | None = None,
    ) -> None:
        """Sets the styling for the axes title.

        Args:
            fontsize (float | int | None, optional): Font size for the title. Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color, optional): Color of the title. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the title color.
                If None, uses alpha from color if specified. Defaults to None.
            loc (Literal["left", "center", "right"] | None, optional): Title location.
                Defaults to None.
            pad (float | None, optional): Padding between the title and the axes in points.
                Defaults to None.

        Returns:
            None
        """
        self._title_style = TitleStyle(
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            loc=loc,
            pad=pad,
        )
        if self._axes_state_initialized and self.title is not None:
            self._apply_title_now()
            self._axes_state.reclaim_and_mark(UNIT_TITLE, self._snapshot_title())
        elif self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_TITLE)

    def set_xlabel(self, text: str | None) -> None:
        """Set deferred x-axis label text.

        Args:
            text (str | None): Label text, or None to clear.

        Returns:
            None
        """
        self.xlabel = text

    def set_ylabel(self, text: str | None) -> None:
        """Set deferred y-axis label text.

        Args:
            text (str | None): Label text, or None to clear.

        Returns:
            None
        """
        self.ylabel = text

    def set_title(self, text: str | None) -> None:
        """Set deferred axes title text.

        Args:
            text (str | None): Title text, or None to clear.

        Returns:
            None
        """
        self.title = text

    def clear_xlabel_style(self) -> None:
        """Clear the x-axis label styling, reverting to matplotlib defaults."""
        self._xlabel_style = None
        if self._axes_state_initialized and self.xlabel is not None:
            self._apply_xlabel_now()
            self._axes_state.reclaim_and_mark(UNIT_X_LABEL, self._snapshot_xlabel())

    def clear_ylabel_style(self) -> None:
        """Clear the y-axis label styling, reverting to matplotlib defaults."""
        self._ylabel_style = None
        if self._axes_state_initialized and self.ylabel is not None:
            self._apply_ylabel_now()
            self._axes_state.reclaim_and_mark(UNIT_Y_LABEL, self._snapshot_ylabel())

    def clear_title_style(self) -> None:
        """Clear the plot title styling, reverting to matplotlib defaults."""
        self._title_style = None
        if self._axes_state_initialized and self.title is not None:
            self._apply_title_now()
            self._axes_state.reclaim_and_mark(UNIT_TITLE, self._snapshot_title())

    def set_xaxis_tick_style(
        self,
        *,
        size: float | int = 10,
        rotation: float | int = 0,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        tickcolor: Color = "black",
        tickalpha: float | None = None,
        fontweight: str = "normal",
        fontstyle: Literal["normal", "italic", "oblique"] = "normal",
        fontfamily: str = "sans-serif",
        ticktype: TickType = "major",
    ) -> None:
        """Set x-axis tick style.

        Args:
            size (float, optional): Font size of tick labels. Defaults to 10.
            rotation (float | int, optional): Rotation angle of tick labels in degrees.
                Defaults to 0.
            fontcolor (str, optional): Color of tick labels. Defaults to "black".
            fontalpha (float, optional): Alpha transparency of tick label color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            tickcolor (str, optional): Color of tick marks. Defaults to "black".
            tickalpha (float, optional): Alpha transparency of tick mark color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            fontweight (str, optional): Font weight of tick labels (e.g., 'normal 'bold').
                Defaults to "normal".
            fontstyle (Literal["normal", "italic", "oblique"], optional): Font style of tick
                labels (e.g., 'normal', 'italic'). Defaults to "normal".
            fontfamily (str, optional): Font family of tick labels (e.g., 'serif', 'sans-serif').
                Defaults to "sans-serif".
            ticktype (TickType, optional): Type of ticks to style ('major', 'minor', 'both').
                Defaults to 'major'.

        Returns:
            None
        """
        self._x_tick_style = TickStyle(
            size=size,
            rotation=rotation,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            tickcolor=tickcolor,
            tickalpha=tickalpha,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            ticktype=ticktype,
        )
        # Store-and-claim: tick style depends on the rebuild flow having
        # already laid out ticks (so label-text artists exist).
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_X_TICK_STYLE)

    def set_yaxis_tick_style(
        self,
        *,
        size: float | int = 10,
        rotation: float | int = 0,
        fontcolor: Color = "black",
        fontalpha: float | None = None,
        tickcolor: Color = "black",
        tickalpha: float | None = None,
        fontweight: str = "normal",
        fontstyle: Literal["normal", "italic", "oblique"] = "normal",
        fontfamily: str = "sans-serif",
        ticktype: TickType = "major",
    ) -> None:
        """Set y-axis tick style.

        Args:
            size (float, optional): Font size of tick labels. Defaults to 10.
            rotation (float | int, optional): Rotation angle of tick labels in degrees.
                Defaults to 0.
            fontcolor (str, optional): Color of tick labels. Defaults to "black".
            fontalpha (float, optional): Alpha transparency of tick label color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            tickcolor (str, optional): Color of tick marks. Defaults to "black".
            tickalpha (float, optional): Alpha transparency of tick mark color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            fontweight (str, optional): Font weight of tick labels (e.g., 'normal 'bold').
                Defaults to "normal".
            fontstyle (Literal["normal", "italic", "oblique"], optional): Font style of tick
                labels (e.g., 'normal', 'italic'). Defaults to "normal".
            fontfamily (str, optional): Font family of tick labels (e.g., 'serif', 'sans-serif').
                Defaults to "sans-serif".
            ticktype (TickType, optional): Type of ticks to style ('major', 'minor', 'both').
                Defaults to 'major'.

        Returns:
            None
        """
        self._y_tick_style = TickStyle(
            size=size,
            rotation=rotation,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            tickcolor=tickcolor,
            tickalpha=tickalpha,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            ticktype=ticktype,
        )
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_Y_TICK_STYLE)

    def clear_xtick_labels(self) -> None:
        """Clear x-tick labels."""
        self._x_tick_labels = []
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_X_TICKS)

    def clear_ytick_labels(self) -> None:
        """Clear y-tick labels."""
        self._y_tick_labels = []
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_Y_TICKS)

    def clear_xticks(self) -> None:
        """Clear x-tick locations and labels."""
        self._x_tick_locations = []
        self._x_tick_labels = []
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_X_TICKS)

    def clear_yticks(self) -> None:
        """Clear y-tick locations and labels."""
        self._y_tick_locations = []
        self._y_tick_labels = []
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_Y_TICKS)

    def set_xticks(
        self,
        locations: Sequence[float] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Set x-axis tick locations and optionally labels.

        Args:
            locations (Sequence[float] | None, optional): X-axis tick locations. Defaults to None.
            labels (Sequence[str] | None, optional): X-axis tick labels. Defaults to None.

        Returns:
            None
        """
        self.update_xtick_labels(
            locations=None if locations is None else list(locations),
            labels=None if labels is None else list(labels),
        )

    def set_yticks(
        self,
        locations: Sequence[float] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Set y-axis tick locations and optionally labels.

        Args:
            locations (Sequence[float] | None, optional): Y-axis tick locations. Defaults to None.
            labels (Sequence[str] | None, optional): Y-axis tick labels. Defaults to None.

        Returns:
            None
        """
        self.update_ytick_labels(
            locations=None if locations is None else list(locations),
            labels=None if labels is None else list(labels),
        )

    def set_xlim(self, left: float, right: float) -> None:
        """Set x-axis limits.

        Matches the matplotlib convention ``Axes.set_xlim(left, right)``.

        Args:
            left (float): Left x-axis limit.
            right (float): Right x-axis limit.
        """
        self._x_limits = (float(left), float(right))
        # Apply-now: matplotlib resolves a (left, right) pair against any
        # axes immediately. Record the getter-read value so a later external
        # set_xlim is detected as a difference.
        if self._axes_state_initialized:
            self._ax.set_xlim(*self._x_limits)
            self._axes_state.reclaim_and_mark(
                UNIT_X_LIMITS, tuple(float(v) for v in self._ax.get_xlim())
            )

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set y-axis limits.

        Matches the matplotlib convention ``Axes.set_ylim(bottom, top)``.

        Args:
            bottom (float): Bottom y-axis limit.
            top (float): Top y-axis limit.
        """
        self._y_limits = (float(bottom), float(top))
        if self._axes_state_initialized:
            self._ax.set_ylim(*self._y_limits)
            self._axes_state.reclaim_and_mark(
                UNIT_Y_LIMITS, tuple(float(v) for v in self._ax.get_ylim())
            )

    def show_or_hide_frame(
        self,
        show_top: bool = True,
        show_right: bool = True,
        show_bottom: bool = True,
        show_left: bool = True,
    ) -> None:
        """Set the visibility of each spine of the plot frame.

        The parameter order (top, right, bottom, left) matches the frame
        snapshot tuple used by the managed-axes state.

        Args:
            show_top (bool, optional): Whether to show the top spine. Defaults to True.
            show_right (bool, optional): Whether to show the right spine. Defaults to True.
            show_bottom (bool, optional): Whether to show the bottom spine. Defaults to True.
            show_left (bool, optional): Whether to show the left spine. Defaults to True.

        Returns:
            None
        """
        self._frame_visibility = {
            "top": show_top,
            "right": show_right,
            "bottom": show_bottom,
            "left": show_left,
        }
        if self._axes_state_initialized:
            self._apply_frame_visibility_now()
            self._axes_state.reclaim_and_mark(UNIT_FRAME, self._snapshot_frame())

    def _get_named_line_legend_handles(self) -> list[LegendHandle]:
        """Get legend handles for all named lines.

        Returns:
            list[LegendHandle]: A list of legend handles.
        """
        handles: list[LegendHandle] = []
        for line in self._annotations.vertical_lines + self._annotations.horizontal_lines:
            if line.name is not None:
                handle = Line2D(
                    [0],
                    [0],
                    color=self._resolved_rgba(
                        line.linecolor,
                        line.linealpha,
                        field="linecolor",
                    ),
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                    label=line.name,
                )
                handles.append(handle)

        return handles

    def _get_named_band_legend_handles(self) -> list[LegendHandle]:
        """Get legend handles for all named bands.

        Returns:
            list[LegendHandle]: A list of legend handles.
        """
        handles: list[LegendHandle] = []
        for band in self._annotations.vertical_bands + self._annotations.horizontal_bands:
            if band.name is None:
                continue

            if band.linecolor is None or band.linewidth == 0.0:
                edgecolor = "none"
            else:
                edgecolor = self._resolved_rgba(
                    band.linecolor,
                    band.linealpha,
                    field="linecolor",
                )
            handle = Patch(
                facecolor=self._resolved_rgba(
                    band.bandcolor,
                    band.bandalpha,
                    field="bandcolor",
                ),
                edgecolor=edgecolor,
                linestyle=band.linestyle,
                linewidth=band.linewidth,
                label=band.name,
            )
            handles.append(handle)

        return handles

    def set_legend_options(
        self,
        *,
        loc: str | int = "center left",
        bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = (
            1.01,
            0.5,
        ),
        ncols: int = 1,
        fontsize: float | str | None = None,
        frameon: bool = True,
        fancybox: bool = False,
        shadow: bool = False,
        framealpha: float | None = None,
        facecolor: Color | None = None,
        edgecolor: Color | None = None,
        title: str | None = None,
        alignment: Literal["center", "left", "right"] = "center",
        labelspacing: float = 0.5,
        columnspacing: float = 2.0,
    ) -> None:
        """Set legend options used by ``Axes.legend`` during plot build.

        Args:
            loc (str | int, optional): Matplotlib legend location. Defaults to
                ``"center left"``.
            bbox_to_anchor (tuple[float, float] | tuple[float, float, float, float] | None,
                optional): Legend anchor box. Defaults to ``(1.01, 0.5)``.
            ncols (int, optional): Number of legend columns. Defaults to ``1``.
            fontsize (float | str | None, optional): Legend text size. Defaults to None.
            frameon (bool, optional): Whether to draw the legend frame. Defaults to True.
            fancybox (bool, optional): Whether to use a rounded frame. Defaults to False.
            shadow (bool, optional): Whether to draw a shadow. Defaults to False.
            framealpha (float | None, optional): Frame alpha override. Defaults to None.
            facecolor (Color | None, optional): Frame face color. Defaults to None.
            edgecolor (Color | None, optional): Frame edge color. Defaults to None.
            title (str | None, optional): Legend title. Defaults to None.
            alignment (Literal["center", "left", "right"], optional): Legend content
                alignment. Defaults to ``"center"``.
            labelspacing (float, optional): Vertical spacing between entries.
                Defaults to ``0.5``.
            columnspacing (float, optional): Horizontal spacing between columns.
                Defaults to ``2.0``.

        Returns:
            None
        """
        self._legend_options = build_legend_options(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncols=ncols,
            fontsize=fontsize,
            frameon=frameon,
            fancybox=fancybox,
            shadow=shadow,
            framealpha=framealpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            title=title,
            alignment=alignment,
            labelspacing=labelspacing,
            columnspacing=columnspacing,
        )
        # Legend identity is recorded by ``_apply_legend`` at the next
        # rebuild after the legend is re-placed with these new options.
        if self._axes_state_initialized:
            self._axes_state.reclaim_without_value(UNIT_LEGEND)

    def save_legend(
        self,
        filepath: str,
        *,
        outer_padding: float = 0.07,
        dpi: int | None = None,
        **legend_kwargs: object,
    ) -> None:
        """Save legend handles to a standalone image.

        Args:
            filepath (str): Output file path.
            outer_padding (float, optional): Fractional padding around the legend bounding box.
                Defaults to ``0.07``.
            dpi (int | None, optional): Output DPI. If None, uses figure DPI.
                Defaults to None.
            **legend_kwargs (object): Additional keyword arguments passed to
                ``matplotlib.axes.Axes.legend``.

        Returns:
            None
        """
        save_legend_handles(
            handles=self._legend_handles,
            legend_options=self._legend_options,
            filepath=filepath,
            outer_padding=outer_padding,
            dpi=dpi or self.fig.dpi,
            **legend_kwargs,
        )

    # ------------------------------------------------------------------
    # Frame, legend, and the top-level rebuild flow
    # ------------------------------------------------------------------

    def _snapshot_frame(self) -> tuple[bool, bool, bool, bool]:
        return (
            self._ax.spines["top"].get_visible(),
            self._ax.spines["right"].get_visible(),
            self._ax.spines["bottom"].get_visible(),
            self._ax.spines["left"].get_visible(),
        )

    def _apply_frame_visibility_now(self) -> None:
        """Write frame visibility to the axes without ownership bookkeeping.

        Shared by the public setter and the rebuild flow's external-aware
        applier below.
        """
        for spine, visible in self._frame_visibility.items():
            self._ax.spines[spine].set_visible(visible)

    def _apply_frame_visibility(self, external: set[str]) -> None:
        if UNIT_FRAME in external:
            return
        self._apply_frame_visibility_now()
        snapshot = self._snapshot_frame()
        if self._axes_state.is_reclaimed(UNIT_FRAME):
            self._axes_state.reclaim_and_mark(UNIT_FRAME, snapshot)
        else:
            self._axes_state.record_default(UNIT_FRAME, snapshot)

    def _apply_legend(self, external: set[str]) -> None:
        """Place, update, or remove the legend per the managed-unit contract.

        Decision tree:
        - External legend present → skip entirely.
        - User wants no legend AND gerrytools owns the unit → remove and record.
        - No handles AND gerrytools owns the unit → remove and record.
        - include_legend is True and handles exist → place and record identity.
        """
        if UNIT_LEGEND in external:
            return
        is_reclaimed = self._axes_state.is_reclaimed(UNIT_LEGEND)
        if not self._include_legend:
            if is_reclaimed:
                current = self._ax.get_legend()
                if current is not None:
                    current.remove()
                self._axes_state.reclaim_and_mark(UNIT_LEGEND, None)
            return
        handles = self._legend_handles
        if not handles:
            if is_reclaimed:
                current = self._ax.get_legend()
                if current is not None:
                    current.remove()
                self._axes_state.reclaim_and_mark(UNIT_LEGEND, None)
            return
        # Remove any prior gerrytools legend on this axes so the new one
        # supersedes it (matplotlib only renders one legend slot per axes).
        prior = self._ax.get_legend()
        if prior is not None and is_reclaimed:
            prior.remove()
        self._ax.legend(handles=handles, **(self._legend_options.to_dict()))
        new_legend = self._ax.get_legend()
        if new_legend is not None:
            self._artists.track(new_legend)
        self._axes_state.reclaim_and_mark(UNIT_LEGEND, new_legend)

    def _build_and_apply_settings(self) -> None:
        """Rebuild the plot: snapshot → remove gerrytools artists → redraw → apply units.

        Ordering highlights:
        - The pre-redraw snapshot is taken *before* removing artists so we can
          detect direct matplotlib changes since the last apply.
        - Only gerrytools-tracked artists are removed; external artists
          survive.
        - Autoscale-protected units are restored after artist drawing so
          matplotlib's autoscale cannot clobber externally set limits.
        - Ticks are applied after the legend so positions are consistent
          with any side effects from the legend draw.
        """
        before = self._axes_state.snapshot(self._ax)
        external = self._axes_state.detect_external_changes(before)

        self._artists.remove_all()
        self._build_plot()

        self._axes_state.restore_autoscale_protected(self._ax, before, external)
        self._apply_x_limits(external)
        self._apply_y_limits(external)
        self._apply_frame_visibility(external)
        self._apply_xlabel(external)
        self._apply_ylabel(external)
        self._apply_title(external)
        self._apply_x_scale(external)
        self._apply_y_scale(external)
        self._apply_aspect(external)
        self._apply_legend(external)
        # Apply ticks after the legend so positions reflect any legend-draw
        # side effects on layout.
        self._apply_x_ticks(external)
        self._apply_y_ticks(external)
        self._apply_x_tick_style(external)
        self._apply_y_tick_style(external)

        # Annotations (arrows / vlines / hlines / bands) render last so
        # their canvas-draw-and-measure alignment passes (used for arrow
        # tip placement) see the final ticks, limits, and label layout.
        self._annotations.apply(
            self._ax,
            fig=self.fig,
            color_resolver=self._resolved_rgba,
            registry=self._artists,
        )

    @property
    def ax(self) -> Axes:
        """Build the plot and return the matplotlib ``Axes``.

        Access to this property triggers a **lazy render**: every accumulated
        setting (added data, lines, bands, arrows, styles, etc.) is reapplied
        to the underlying axes. This is the canonical hook for embedding the
        plot into a larger matplotlib workflow.

        Why lazy? In a Jupyter notebook, instantiating a plot class (e.g.
        ``Histogram()``) without lazy rendering would auto-display an empty
        figure as the cell output. Deferring the render until ``.ax`` (or
        :meth:`show` / :meth:`save`) is accessed keeps the notebook clean.

        Calling ``.ax`` multiple times is safe; each call rebuilds from the
        current accumulated state. Use :meth:`bind_to_ax` to retarget the
        plot to a different ``Axes`` (e.g. one inside your own figure).

        Returns:
            Axes: The matplotlib ``Axes`` object with every setting applied.
        """
        self._build_and_apply_settings()
        return self._ax

    def show(self, **kwargs: object) -> None:
        """Display the figure.

        Args:
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
                Defaults: ``bbox_inches="tight"``, ``dpi=fig.dpi``.
        """
        self._build_and_apply_settings()
        show_figure(
            self.fig,
            non_gui_filename="gerrytools_plot.png",
            non_gui_prefix="GerryTools Plotting",
            **kwargs,
        )

    def save(self, filepath: str, **kwargs: object) -> None:
        """Save the figure to a file.

        Args:
            filepath (str): The file path to save the figure to.
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.

        Returns:
            None
        """
        self._build_and_apply_settings()
        save_figure(self.fig, filepath, **kwargs)

    @abstractmethod
    def _build_plot(self) -> None:
        """Build the plot by applying all settings and drawing elements."""
        pass  # pragma: no cover - abstract stub; every concrete subclass overrides this

    @property
    @abstractmethod
    def _legend_handles(self) -> list[LegendHandle]:
        """Get legend handles for all named elements in the plot.

        Returns:
            list[LegendHandle]: A list of legend handles.
        """
        return []  # pragma: no cover - abstract stub; every concrete subclass overrides this
