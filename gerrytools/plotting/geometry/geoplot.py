from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
from warnings import warn

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes import Axes
from shapely.geometry import Point, box

from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting._axes_state import (
    UNIT_AXIS_VISIBILITY,
    UNIT_X_LIMITS,
    UNIT_Y_LIMITS,
    _ManagedAxesState,
)
from gerrytools.plotting._figure_io import save_figure, show_figure
from gerrytools.plotting.geometry._layers import (
    _as_geoseries,
    _CategoricalColorLayer,
    _GeoLayer,
    _MarkerLayer,
)
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import (
    CategoryKey,
    Color,
    CRSLike,
    GeoSource,
)

# Re-exported so existing imports like
#   `from gerrytools.plotting.geometry.geoplot import _CategoricalColorLayer`
# keep working from the test suite without forcing test changes.
__all__ = [
    "GeoPlot",
    "_GeoLayer",
    "_CategoricalColorLayer",
    "_MarkerLayer",
    "_LabelRequest",
    "_as_geoseries",
]


@dataclass(frozen=True, slots=True)
class _LabelRequest:
    gdf: GeoDataFrame
    label_column: str
    labelfont_options: LabelFontOptions | None
    labelbox_options: LabelBoxOptions | None
    label_format_fn: Callable[[CategoryKey], str] | None = None
    zorder: int = 100


class GeoPlot(ABC):
    """A class for creating geographic plots with multiple layers.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        target_crs: The target CRS for reprojecting geometries.
        silent (bool): Whether to suppress informational output throughout
            the rendering process.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        dpi: int | None = None,
        ax: Axes | None = None,
        show_axis: bool = False,
        target_crs: CRSLike | None = None,
        default_outline: bool = True,
        silent: bool = False,
    ) -> None:
        """Initialize a GeoPlot.

        Args:
            gdf (GeoDataFrame): The base GeoDataFrame for the plot.
            dpi (int | None): The DPI for the Matplotlib Figure. Defaults to 300 when
                ``ax`` is not provided. Ignored (with a warning) when ``ax`` is provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Useful for
                callers familiar with matplotlib / seaborn idioms who want to compose
                this plot into a larger figure they control. Defaults to None.
            show_axis (bool): Whether to show axis ticks and labels. Default is False.
            target_crs (CRSLike | None): The target CRS for reprojecting geometries.
                If None, uses the CRS of `gdf`. Default is None.
            default_outline (bool): Whether to include a default outline layer around
                the geometries in `gdf`. Default is True.
            silent (bool): Whether to suppress informational output throughout the rendering
                process. Default is False.
        """
        self.gdf = gdf

        # --- Pass 1: resolve self._ax + self.fig + _figure_is_shared ---
        if ax is not None:
            if dpi is not None:
                warn(
                    "dpi is ignored when ax is provided; the plot will use the "
                    "existing figure's dpi.",
                    UserWarning,
                    stacklevel=2,
                )
            self.fig = ax.figure
            self._ax = ax
            # The user owns this figure; gerrytools must not mutate
            # figure-level layout (e.g. ``subplots_adjust``) on it.
            self._figure_is_shared: bool = True
        else:
            self.fig, self._ax = plt.subplots(dpi=dpi if dpi is not None else 300)
            self._figure_is_shared = False

            # IMPORTANT: prevent implicit display in notebooks
            # Only close in Jupyter so init doesn't display
            try:
                from IPython import get_ipython

                ip = get_ipython()
                if ip is not None and getattr(ip, "kernel", None) is not None:  # pragma: no cover
                    plt.close(self.fig)  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover

        self._canvas = self.fig.canvas  # renderer/manager handled by backend

        # --- Pass 2: backing fields (no opinion until step 4 reapplies args) ---
        self._show_axis: bool = False
        self.target_crs: CRSLike | None = (
            target_crs if target_crs is not None else getattr(gdf, "crs", None)
        )

        self._xlim: tuple[float, float] | None = None
        self._ylim: tuple[float, float] | None = None

        self._outline_layers: list[_CategoricalColorLayer] = []
        self._highlight_layers: list[_CategoricalColorLayer] = []
        self._marker_layers: list[_MarkerLayer] = []

        self._label_requests: list[_LabelRequest] = []
        self.silent = silent

        # --- Pass 3: artist registry + managed-axes state ---
        self._artists = _ArtistRegistry()
        self._axes_state = _ManagedAxesState()
        self._axes_state_initialized: bool = False
        self._axes_state.initialize_from_ax(self._ax)
        self._axes_state_initialized = True

        # --- Pass 4: re-apply non-default constructor args via reclaim path ---
        if show_axis is not False:
            self.show_axis = show_axis

        if default_outline:
            fully_dissolved_geos = GeoSeries(gdf.geometry.union_all())
            self.add_outline_layer(
                geosource=fully_dissolved_geos,
                edgecolor="black",
                edgewidth=0.5,
            )

        # NOTE: Do we want to focus axes here? So if you add more layers later it keeps the same
        # view?

    def add_outline_layer(
        self,
        geosource: GeoDataFrame | GeoSeries | None = None,
        *,
        geometry_mask: pd.Series | None = None,
        dissolve_column: str | None = None,
        edgecolor: Color = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        show_labels: bool = False,
        exclude_labels: Sequence[CategoryKey] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 3,
    ) -> None:
        """Add an outline layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source
                for the layer. If None, uses the base gdf of the GeoPlot. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
                Default is None.
            dissolve_column (str | None): Optional column to dissolve geometries by
                before outlining. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "black".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            show_labels (bool): Whether to show labels on the outlined geometries. Default is False.
            exclude_labels (Sequence[CategoryKey] | None): Labels to exclude from labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfont_options (LabelFontOptions | None): Font options for labels.
                If None, uses the following defaults:
                    - fontcolor="black",
                    - fontsize=4,
                    - fontweight="roman",
                    - outlinecolor="grey",
                    - outlinewidth=0.2.
                Default is None.
            labelbox_options (LabelBoxOptions | None): Box options for labels. If None the box
                is disabled. Default is None.
            zorder (int): Z-order for rendering. Default is 3.
        """
        if geosource is None:
            geosource = self.gdf

        if dissolve_column is not None:
            if not isinstance(geosource, GeoDataFrame):
                raise TypeError(
                    "Tried to dissolve geosource of type "
                    f"{type(geosource).__name__!r}; geosource must be a GeoDataFrame",
                )
            geosource = GeoDataFrame(geosource.dissolve(by=dissolve_column).reset_index())

        layer = _CategoricalColorLayer(
            geometry_source=geosource,
            geometry_mask=geometry_mask,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )
        self._outline_layers.append(layer)

        if show_labels:
            processed_geosource = layer.geosource
            if not isinstance(processed_geosource, GeoDataFrame):
                raise TypeError(
                    "Tried to add labels to geosource of type "
                    f"{type(geosource).__name__!r}; geosource must be a GeoDataFrame",
                )

            if dissolve_column is None:
                raise ValueError(
                    "'dissolve_column' must be set to add labels to an outline layer",
                )

            new_exclude_labels = list(exclude_labels) if exclude_labels is not None else []
            labeled_gdf = GeoDataFrame(
                processed_geosource.query(f"`{dissolve_column}` not in {new_exclude_labels}")
            )

            if labelfont_options is None:
                labelfont_options = LabelFontOptions(
                    fontcolor="black",
                    fontsize=4,
                    fontweight="roman",
                    outlinecolor="grey",
                    outlinewidth=0.2,
                )

            self._label_requests.append(
                _LabelRequest(
                    gdf=labeled_gdf,
                    label_column=dissolve_column,
                    labelfont_options=labelfont_options,
                    labelbox_options=labelbox_options,
                    zorder=zorder + 1,
                )
            )

    def add_highlight_layer(
        self,
        geosource: GeoDataFrame | GeoSeries | None = None,
        label_column: str | None = None,
        *,
        geometry_mask: pd.Series | None = None,
        facecolor: Color = "gray",
        facealpha: float | None = 0.5,
        show_labels: bool = False,
        exclude_labels: Sequence[CategoryKey] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 10,
    ) -> None:
        """Add a highlight layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source
                for the layer. If None, uses the base gdf of the GeoPlot. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
                Default is None.
            label_column (str | None): Optional column to label geometries by
                before highlighting. Default is None.
            facecolor (Color): Color for geometry faces. Default is "gray".
            facealpha (float | None): Alpha transparency for face colors. Default is 0.5.
            show_labels (bool): Whether to show labels on the highlighted geometries. Default is
                False.
            exclude_labels (Sequence[CategoryKey] | None): Labels to exclude from labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfont_options (LabelFontOptions | None): Font options for labels.
                When None, defaults to fontcolor="black", fontsize=4, fontweight="roman",
                outlinecolor="grey", and outlinewidth=0.2.
            labelbox_options (LabelBoxOptions | None): Box options for labels. If None the box
                is disabled. Default is None.
            zorder (int): Z-order for rendering. Default is 10.
        """
        if show_labels:
            if label_column is None:
                raise ValueError(
                    "add_highlight_layer(show_labels=True) requires label_column=... to know "
                    "what to label. Example: dissolve_column='COUNTYFP10'."
                )
            if geosource is None:
                raise ValueError(
                    "add_highlight_layer(show_labels=True) requires geosource=... (a GeoDataFrame) "
                    "so the dissolve_column exists."
                )
            if not isinstance(geosource, GeoDataFrame):
                raise TypeError(
                    "add_highlight_layer(show_labels=True) requires geosource to be a GeoDataFrame "
                    f"so it has the label_column {label_column!r}. "
                    f"You passed {type(geosource).__name__!r}. "
                    "Either pass a GeoDataFrame, or set show_labels=False."
                )

        if geosource is None:
            geometries = self.gdf.geometry
        else:
            geometries = _as_geoseries(geosource)

        if geometry_mask is not None:
            geometries = geometries[geometry_mask]

        geometries = GeoSeries(geometries.union_all())

        layer = _CategoricalColorLayer(
            geometry_source=geometries,
            colormap=facecolor,
            missing_color="none",
            facealpha=facealpha,
            edgecolor="none",
            edgealpha=None,
            edgewidth=0.0,
            zorder=zorder,
        )
        self._highlight_layers.append(layer)

        if show_labels:
            label_gdf = geosource
            if (
                label_gdf is None
            ):  # pragma: no cover - defensive guard; geosource=None + show_labels=True already raised above
                raise RuntimeError(  # pragma: no cover
                    "An unexpected error occured in add_highlight_layer. "
                    "The geosource was None when trying to add labels."
                )  # pragma: no cover

            if isinstance(
                label_gdf, GeoSeries
            ):  # pragma: no cover - defensive guard; GeoSeries geosource + show_labels=True already raised above
                raise TypeError(  # pragma: no cover
                    "add_highlight_layer(show_labels=True) requires geosource to be a GeoDataFrame "
                    f"so it has the label_column {label_column!r}. "
                    f"You passed a GeoSeries. Either pass a GeoDataFrame, or set show_labels=False."
                )  # pragma: no cover

            if geometry_mask is not None:
                label_gdf = GeoDataFrame(label_gdf.loc[geometry_mask])

            new_exclude_labels = list(exclude_labels) if exclude_labels is not None else []
            labeled_gdf = GeoDataFrame(
                label_gdf.query(f"`{label_column}` not in {new_exclude_labels}")
            )

            if labelfont_options is None:
                labelfont_options = LabelFontOptions(
                    fontcolor="black",
                    fontsize=4,
                    fontweight="roman",
                    outlinecolor="grey",
                    outlinewidth=0.2,
                )

            if (
                label_column is None
            ):  # pragma: no cover - defensive guard; label_column=None + show_labels=True already raised above
                raise RuntimeError(  # pragma: no cover
                    "An unexpected error occured in add_highlight_layer. "
                    "The dissolve_column was None when trying to add labels."
                )  # pragma: no cover

            self._label_requests.append(
                _LabelRequest(
                    gdf=labeled_gdf,
                    label_column=label_column,
                    labelfont_options=labelfont_options,
                    labelbox_options=labelbox_options,
                    zorder=zorder + 1,
                )
            )

        return None

    def add_marker_layer(
        self,
        points_geoseries: gpd.GeoSeries | None = None,
        *,
        latlon_list: Sequence[tuple[float, float]] | None = None,
        input_crs: CRSLike | None = None,
        marker_options: PointMarkerOptions | None = None,
        show_labels: bool = True,
        labels: Sequence[str] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of markers (points) to the GeoPlot.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latlon_list` must be provided. Default is None.
            latlon_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            input_crs (CRSLike | None, optional): The CRS of the input points if using
                ``latlon_list``.
                If None, assumes EPSG:4326 (lat/lon). Default is None.
            marker_options (PointMarkerOptions | None): Marker style settings.
                If None, uses the following defaults:
                    - markerfacecolor="white",
                    - markerfacealpha=1.0,
                    - marker="o",
                    - markersize=3.0,
                    - markeredgecolor="black",
                    - markeredgealpha=1.0,
                    - markeredgewidth=0.5.
                Default is None.
            show_labels (bool): Whether to show labels on the markers. Default is True.
            labels (Sequence[str] | None): Optional labels for each marker. Default is None.
            labelfont_options (LabelFontOptions | None): Font options for the labels If None, uses
                default LabelFontOptions().
            labelbox_options (LabelBoxOptions | None): Box options for the labels. If None the
                box is disabled. Default is None.
            zorder (int, optional): Z-order for rendering. Defaults to ``2``.
        """
        if marker_options is None:
            marker_options = PointMarkerOptions(
                markerfacecolor="white",
                markerfacealpha=1.0,
                marker="o",
                markersize=3.0,
                markeredgecolor="black",
                markeredgealpha=1.0,
                markeredgewidth=0.5,
            )
        if labelfont_options is None:
            labelfont_options = LabelFontOptions()
        if labelbox_options is None:
            labelbox_options = LabelBoxOptions(enabled=False)

        if points_geoseries is None and latlon_list is None:
            raise ValueError("Either `points_geoseries` or `latlon_list` must be set.")
        if points_geoseries is not None and latlon_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latlon_list` may be set at a time.",
            )

        if latlon_list is not None:
            # crs EPSG:4326 corresponds to lat/lon
            point_geometries = gpd.GeoSeries(
                [Point(float(longitude), float(latitude)) for latitude, longitude in latlon_list],
                crs="EPSG:4326",
            )
            point_geometries = point_geometries.to_crs(
                input_crs if input_crs is not None else self.gdf.crs
            )
        elif points_geoseries is not None:
            point_geometries = points_geoseries
            if getattr(point_geometries, "crs", None) is None and input_crs is not None:
                point_geometries = point_geometries.set_crs(input_crs)
        else:  # pragma: no cover - defensive guard; the preceding if/elif already covers all valid states
            raise RuntimeError(  # pragma: no cover
                "An unexpected error occured in add_marker_layer. One of the argurments "
                "'points_geoseries' or 'latlon_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latlon_list': {type(latlon_list).__name__!r}",
            )  # pragma: no cover

        marker_layer = _MarkerLayer(
            point_geometries=point_geometries,
            labels=labels,
            marker_options=marker_options,
            show_labels=show_labels,
            labelfont_options=labelfont_options,
            labelbox_options=labelbox_options,
            zorder=zorder,
        )
        self._marker_layers.append(marker_layer)

    def add_label_layer(
        self,
        points_geoseries: gpd.GeoSeries | None = None,
        labels: Sequence[str] | None = None,
        *,
        latlon_list: Sequence[tuple[float, float]] | None = None,
        input_crs: CRSLike | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of markers (points) to the GeoPlot.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latlon_list` must be provided. Default is None.
            latlon_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            input_crs (CRSLike | None, optional): The CRS of the input points if using
                ``latlon_list``.
                If None, assumes EPSG:4326 (lat/lon). Default is None.
            labels (Sequence[str] | None): Optional labels for each marker. Default is None which
                results numerical labels.
            labelfont_options (LabelFontOptions | None): Font options for the labels If None, uses
                the following defaults:
                    - fontcolor="black",
                    - fontsize=4,
                    - fontweight="roman",
                    - outlinecolor="grey",
                    - outlinewidth=0.2.
            labelbox_options (LabelBoxOptions | None): Box options for the labels. If None the
                box is disabled. Default is None.
            zorder (int, optional): Z-order for rendering. Defaults to ``2``.
        """
        if points_geoseries is None and latlon_list is None:
            raise ValueError("Either `points_geoseries` or `latlon_list` must be set.")
        if points_geoseries is not None and latlon_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latlon_list` may be set at a time.",
            )
        if points_geoseries is None and latlon_list is not None:
            n_labels = len(list(latlon_list))
        elif points_geoseries is not None:
            n_labels = len(points_geoseries)
        else:  # pragma: no cover - defensive guard; the preceding if/elif already covers all valid states
            raise RuntimeError(  # pragma: no cover
                "An unexpected error occured in add_label_layer. One of the argurments "
                "'points_geoseries' or 'latlon_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latlon_list': {type(latlon_list).__name__!r}",
            )  # pragma: no cover

        if labels is None:
            labels = [str(i) for i in range(n_labels)]

        if labelfont_options is None:
            labelfont_options = LabelFontOptions(
                fontcolor="black",
                fontsize=4,
                fontweight="roman",
                outlinecolor="grey",
                outlinewidth=0.2,
            )

        self.add_marker_layer(
            points_geoseries=points_geoseries,
            latlon_list=latlon_list,
            input_crs=input_crs,
            marker_options=PointMarkerOptions(
                markerfacecolor="none",
                markerfacealpha=0.0,
                marker="o",
                markersize=0.0,
                markeredgecolor="none",
                markeredgealpha=0.0,
                markeredgewidth=0.0,
            ),
            show_labels=True,
            labels=labels,
            labelfont_options=labelfont_options,
            labelbox_options=labelbox_options,
            zorder=zorder,
        )

    # ------------------------------------------------------------------
    # show_axis property — axis_visibility managed unit
    # ------------------------------------------------------------------

    @property
    def show_axis(self) -> bool:
        """Whether to render axis ticks/labels (False hides via ``ax.set_axis_off``)."""
        return self._show_axis

    @show_axis.setter
    def show_axis(self, value: bool) -> None:
        self._show_axis = bool(value)
        if self._axes_state_initialized:
            if self._show_axis:
                self._ax.set_axis_on()
            else:
                self._ax.set_axis_off()
            self._axes_state.reclaim_and_mark(UNIT_AXIS_VISIBILITY, bool(self._ax.axison))

    def set_xlim(self, left: float, right: float) -> None:
        """Set x-axis limits to apply when the plot is built.

        Matches the matplotlib convention ``Axes.set_xlim(left, right)``.

        Args:
            left (float): The left x-axis limit.
            right (float): The right x-axis limit.
        """
        self._xlim = (float(left), float(right))
        if self._axes_state_initialized:
            self._ax.set_xlim(*self._xlim)
            self._axes_state.reclaim_and_mark(
                UNIT_X_LIMITS, tuple(float(v) for v in self._ax.get_xlim())
            )

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set y-axis limits to apply when the plot is built.

        Matches the matplotlib convention ``Axes.set_ylim(bottom, top)``.

        Args:
            bottom (float): The bottom y-axis limit.
            top (float): The top y-axis limit.
        """
        self._ylim = (float(bottom), float(top))
        if self._axes_state_initialized:
            self._ax.set_ylim(*self._ylim)
            self._axes_state.reclaim_and_mark(
                UNIT_Y_LIMITS, tuple(float(v) for v in self._ax.get_ylim())
            )

    def clear_limits(self) -> None:
        """Clear any stored x/y limits so autoscaling can occur."""
        self._xlim = None
        self._ylim = None

    def focus_axes(
        self,
        *,
        geosource: GeoSource | None = None,
        geometry_mask: pd.Series | None = None,
        pad: float | tuple[float, float] = 0.02,
        pad_mode: Literal["fraction", "data"] = "fraction",
    ) -> None:
        """Set x/y limits to the (padded) bounding box of a geosource.

        Args:
            geosource (GeoSource | None, optional): GeoDataFrame or GeoSeries to focus on.
                Defaults to this plot's gdf.
                If None, will use the base gdf used to initialize GeoPlot. Defaults to None.
            geometry_mask (pd.Series | None): Optional boolean mask aligned to geosource index.
                If None, will use all geometries in geosouce. Defaults to None.
            pad (float | tuple[float, float]): Padding around bounds.
                 - If pad_mode="fraction": fraction of width/height (e.g., 0.02 = 2%)
                 - If pad_mode="data": absolute units in data coords.
                 You can pass a single float or (pad_x, pad_y).
                 Defaults to 0.02 (2%).
            pad_mode (Literal): "fraction" or "data". Defaults to "fraction".
        """
        if geosource is None:
            geosource = self.gdf

        geoseries = _as_geoseries(geosource)

        if geometry_mask is not None:
            geoseries = geoseries[geometry_mask]

        geoseries = geoseries[geoseries.notna()]
        try:
            geoseries = geoseries[~geoseries.is_empty]
        except (
            Exception
        ):  # pragma: no cover - older shapely/geopandas combos may not have is_empty reliably
            # older shapely/geopandas combos may not have is_empty reliably; ignore
            pass  # pragma: no cover

        if geoseries.empty:
            raise ValueError(
                "focus_on(): no geometries after applying mask / dropping empties. "
                "Double check your geometry_mask to make sure that it is a valid filter "
                "for the provided geosource.",
            )

        gs_crs = getattr(geoseries, "crs", None)
        if gs_crs is not None and self.target_crs is not None and gs_crs != self.target_crs:
            geoseries = geoseries.to_crs(self.target_crs)

        minx, miny, maxx, maxy = map(float, geoseries.total_bounds)

        width = maxx - minx
        height = maxy - miny

        if isinstance(pad, tuple):
            pad_x, pad_y = float(pad[0]), float(pad[1])
        else:
            pad_x = pad_y = float(pad)

        if pad_mode == "fraction":
            # If width/height are 0 (single point/line), give a small default pad
            dx = (width * pad_x) if width > 0 else pad_x
            dy = (height * pad_y) if height > 0 else pad_y
        elif pad_mode == "data":
            dx, dy = pad_x, pad_y
        else:
            raise ValueError("pad_mode must be 'fraction' or 'data'.")

        self.set_xlim(minx - dx, maxx + dx)
        self.set_ylim(miny - dy, maxy + dy)

    def _list_layers_in_draw_order(self) -> list[_GeoLayer | _MarkerLayer]:
        """Iterate over all layers in the order they should be drawn."""

        def _sorted(layers: Sequence[_GeoLayer | _MarkerLayer]) -> list[_GeoLayer | _MarkerLayer]:
            """Sort layers by integer z-order.

            Args:
                layers (Sequence[_GeoLayer | _MarkerLayer]): Layers to sort.

            Returns:
                list[_GeoLayer | _MarkerLayer]: Layers sorted ascending by ``zorder``.
            """
            return sorted(layers, key=lambda L: int(L.zorder))

        return (
            _sorted(self._marker_layers)
            + _sorted(self._outline_layers)
            + _sorted(self._highlight_layers)
        )

    @abstractmethod
    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and tracking their artists.

        Each layer's ``render()`` returns the matplotlib artists it created
        on the axes; we track them via ``self._artists`` so the next rebuild
        can remove gerrytools-managed artists without disturbing external
        content.
        """
        start_idx_to_layer_type: dict[int, tuple[str, int]] = {}
        if not self.silent:
            layer_order = [
                ("marker", len(self._marker_layers)),
                ("outline", len(self._outline_layers)),
                ("highlight", len(self._highlight_layers)),
            ]
            prev_idx = 0
            for layer_type, count in layer_order:
                if count > 0:
                    start_idx_to_layer_type[prev_idx] = (layer_type, count)
                    prev_idx += count

        all_layers = self._list_layers_in_draw_order()
        for idx, layer in enumerate(all_layers):
            if idx in start_idx_to_layer_type:
                layer_type, count = start_idx_to_layer_type[idx]
                print(f"Rendering {count} {layer_type} layer{'s' if count > 1 else ''}...")
            layer_artists = layer.render(self._ax, target_crs=self.target_crs)
            if layer_artists:
                self._artists.track(layer_artists)

    def _apply_limits(self, external: set[str] | None = None) -> None:
        """Apply stored x/y limits to the axes, respecting external changes."""
        external = external or set()
        if self._xlim is not None and UNIT_X_LIMITS not in external:
            self._ax.set_xlim(*self._xlim)
            self._axes_state.reclaim_and_mark(
                UNIT_X_LIMITS, tuple(float(v) for v in self._ax.get_xlim())
            )
        elif self._xlim is None and UNIT_X_LIMITS not in external:
            self._axes_state.record_default(
                UNIT_X_LIMITS, tuple(float(v) for v in self._ax.get_xlim())
            )
        if self._ylim is not None and UNIT_Y_LIMITS not in external:
            self._ax.set_ylim(*self._ylim)
            self._axes_state.reclaim_and_mark(
                UNIT_Y_LIMITS, tuple(float(v) for v in self._ax.get_ylim())
            )
        elif self._ylim is None and UNIT_Y_LIMITS not in external:
            self._axes_state.record_default(
                UNIT_Y_LIMITS, tuple(float(v) for v in self._ax.get_ylim())
            )

    def _apply_axis_visibility(self, external: set[str]) -> None:
        if UNIT_AXIS_VISIBILITY in external:
            return
        if self._show_axis:
            self._ax.set_axis_on()
        else:
            self._ax.set_axis_off()
        current = bool(self._ax.axison)
        if self._axes_state.is_reclaimed(UNIT_AXIS_VISIBILITY):
            self._axes_state.reclaim_and_mark(UNIT_AXIS_VISIBILITY, current)
        else:
            self._axes_state.record_default(UNIT_AXIS_VISIBILITY, current)

    def _draw_deferred_labels(self) -> dict[str, Point]:
        """Draw all deferred labels and return their positions.

        Returns:
            dict[str, Point]: A dictionary mapping label text to Point objects.
        """
        label_positions: dict[str, Point] = {}
        if not self._label_requests:
            return label_positions

        ax = self._ax

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        clip_geom = box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))

        for req in self._label_requests:
            # One label per dissolved part
            dissolved = GeoDataFrame(req.gdf.dissolve(by=req.label_column).reset_index())

            # Match plot CRS
            if getattr(dissolved, "crs", None) is not None and self.target_crs is not None:
                if dissolved.crs != self.target_crs:
                    dissolved = dissolved.to_crs(self.target_crs)

            # Clip to current view
            clipped = dissolved.geometry.intersection(clip_geom)
            keep = (~clipped.isna()) & (~clipped.is_empty)
            if not keep.any():
                continue

            dissolved = dissolved.loc[keep].copy()
            dissolved["geometry"] = clipped.loc[keep]

            # Representative points inside the clipped geometry
            pts = dissolved.representative_point()

            labels: list[str] = []
            for raw in dissolved[req.label_column].tolist():
                txt = str(raw)
                if req.label_format_fn is not None:
                    try:
                        txt = str(req.label_format_fn(raw))
                    except Exception:
                        pass
                labels.append(txt)

            # Defaults
            font = (
                req.labelfont_options if req.labelfont_options is not None else LabelFontOptions()
            )
            boxopt = (
                req.labelbox_options
                if req.labelbox_options is not None
                else LabelBoxOptions(enabled=False)
            )

            # Ephemeral label-only marker options (no visible marker)
            label_marker_opts = PointMarkerOptions(
                markerfacecolor="none",
                markerfacealpha=0.0,
                marker="o",
                markersize=0.0,
                markeredgecolor="none",
                markeredgealpha=0.0,
                markeredgewidth=0.0,
            )

            # Create an ephemeral marker layer and render immediately
            tmp = _MarkerLayer(
                point_geometries=pts,
                labels=labels,
                marker_options=label_marker_opts,
                show_labels=True,
                labelfont_options=font,
                labelbox_options=boxopt,
                zorder=req.zorder,
            )
            label_artists = tmp.render(ax, target_crs=self.target_crs)
            if label_artists:
                self._artists.track(label_artists)
            label_positions.update(
                {label: Point(pt.x, pt.y) for label, pt in zip(labels, pts.geometry.tolist())}
            )
        return label_positions

    @abstractmethod
    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Snapshot → remove gerrytools artists → rebuild → apply settings.

        Concrete subclasses fully override without super(); this stub
        documents the canonical sequence and exists only for the abstract
        contract.
        """
        before = self._axes_state.snapshot(self._ax)  # pragma: no cover
        external = self._axes_state.detect_external_changes(before)  # pragma: no cover
        self._artists.remove_all()  # pragma: no cover
        self._build_plot()  # pragma: no cover
        self._axes_state.restore_autoscale_protected(self._ax, before, external)  # pragma: no cover
        self._apply_axis_visibility(external)  # pragma: no cover
        self._apply_limits(external)  # pragma: no cover
        label_points = self._draw_deferred_labels()  # pragma: no cover
        return label_points  # pragma: no cover

    @property
    def ax(self) -> Axes:
        """Build the plot and return the matplotlib ``Axes``.

        Access to this property triggers a **lazy render**: every accumulated
        setting (layers, labels, colorbar requests, etc.) is reapplied. This
        is the canonical hook for embedding the plot into a larger workflow.

        Why lazy? In a Jupyter notebook, instantiating ``GeoPlot(gdf)`` without
        lazy rendering would auto-display an empty figure. Deferring the build
        until ``.ax`` (or :meth:`show` / :meth:`save`) is accessed keeps
        notebook output clean.

        Use :meth:`bind_to_ax` to retarget the plot to a different ``Axes``
        (e.g. one inside your own figure).

        Returns:
            Axes: The matplotlib ``Axes`` with every setting applied.
        """
        self._build_and_apply_settings()
        return self._ax

    def bind_to_ax(self, ax: Axes | None) -> None:
        """Retarget this plot to render onto a different matplotlib ``Axes``.

        The plot's accumulated layers, labels, and style settings are preserved
        and re-applied to the new axes on the next access to :attr:`ax` (or
        call to :meth:`show` / :meth:`save`). Any prior rendered output on the
        *old* axes is left alone; this plot simply stops managing it.

        Pass ``ax=None`` to unbind — the plot creates a fresh figure on the
        next render, just as it did on construction.

        Args:
            ax (matplotlib.axes.Axes | None): The matplotlib axes to render
                onto, or ``None`` to revert to a fresh-figure render.
        """
        # Suppress reclaim during the re-classification step. Mirrors the
        # two-pass init contract.
        self._axes_state_initialized = False

        if ax is None:
            self.fig, self._ax = plt.subplots()
            self._figure_is_shared = False
            try:
                from IPython import get_ipython

                ip = get_ipython()
                if ip is not None and getattr(ip, "kernel", None) is not None:  # pragma: no cover
                    plt.close(self.fig)  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        else:
            self.fig = ax.figure
            self._ax = ax
            self._figure_is_shared = True
        self._canvas = self.fig.canvas

        # Detach registry from old axes (non-destructive rebind); reset
        # per-axes history, classify new axes, re-enable reclaim.
        self._artists = _ArtistRegistry()
        self._axes_state.reset_history()
        self._axes_state.initialize_from_ax(self._ax)
        self._axes_state_initialized = True

    def get_label_positions(self, *, as_lat_long: bool = False) -> tuple[str, dict[str, Point]]:
        """Get computed label positions from the current plot build.

        Args:
            as_lat_long (bool, optional): Whether to convert points to ``EPSG:4326``.
                Defaults to False.

        Returns:
            tuple[str, dict[str, Point]]: CRS string and label-to-point mapping.
        """
        label_points = GeoSeries(self._build_and_apply_settings(), crs=self.target_crs)
        if as_lat_long:
            label_points = label_points.to_crs("EPSG:4326")
        return (
            str(label_points.crs.to_string() if label_points.crs is not None else "undefined"),
            {str(label): Point(pt.x, pt.y) for label, pt in label_points.items()},
        )

    def show(self, **kwargs: object) -> None:
        """Display inline in notebooks, or open a GUI window in scripts.

        Args:
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
                Defaults: ``bbox_inches="tight"``, ``dpi=fig.dpi``.
        """
        self._build_and_apply_settings()
        show_figure(self.fig, non_gui_filename="geoplot.png", non_gui_prefix="GeoPlot", **kwargs)

    def save(self, filepath: str, **kwargs: object) -> None:
        """Save the plot to a file.

        Args:
            filepath (str): Output file path.
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.

        Returns:
            None
        """
        self._build_and_apply_settings()
        save_figure(self.fig, filepath, **kwargs)
