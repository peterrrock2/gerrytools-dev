from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_hex
from matplotlib.pyplot import get_cmap
from shapely.geometry import Point, box

from gerrytools.colors import districtr, resolve_color_and_alpha
from gerrytools.plotting._figure_io import save_figure, show_figure
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import (
    CategoryColorMap,
    CategoryKey,
    Color,
    CRSLike,
    GeoColorMap,
    GeoSource,
    MplCompatibleColor,
    ResolvedColor,
)


def _as_geoseries(source: GeoSource) -> gpd.GeoSeries:
    """Return geometry column as a ``GeoSeries`` for a GeoDataFrame/GeoSeries input.

    Args:
        source (GeoSource): GeoDataFrame or GeoSeries source object.

    Returns:
        gpd.GeoSeries: Geometry series extracted from ``source``.
    """
    return source.geometry if isinstance(source, gpd.GeoDataFrame) else source


@dataclass(frozen=True, slots=True)
class _GeoLayer(ABC):
    """Abstract base class for a geographic layer to be rendered on a GeoPlot.

    Attributes:
        geosource (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        datacolumn (str | None): Optional data column for color mapping. Default is None.
        colormap (GeoColorMap | None): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (MplCompatibleColor | None): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color): Color for geometry edges. Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
    """

    # Try to keep the GeoSource as a reference so that users don't copy the polygons all the time.
    geometry_source: GeoSource
    geometry_mask: pd.Series | None = None
    datacolumn: str | None = None
    colormap: GeoColorMap | None = "Purples"
    missing_color: MplCompatibleColor | None = "lightgrey"
    facealpha: float | None = None
    edgecolor: Color = "none"
    edgealpha: float | None = None
    edgewidth: float = 0.5
    zorder: int = 1

    def _geometries_in_crs(self, target_crs: CRSLike | None) -> gpd.GeoSeries:
        """Return this layer's geometries (respecting mask) reprojected to target_crs.

        Args:
            target_crs (CRSLike | None): Target CRS to reproject to.

        Returns:
            gpd.GeoSeries: The geometries in the target CRS.
        """
        geoseries = self.geometries

        # If either side has no CRS, don't try to reproject; let GeoPandas/Matplotlib handle.
        if getattr(geoseries, "crs", None) is None or target_crs is None:
            return geoseries

        if geoseries.crs != target_crs:
            return geoseries.to_crs(target_crs)

        return geoseries

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_color",
            resolve_color_and_alpha(
                self.missing_color, alpha=self.facealpha, field="missing_color"
            ),
        )

    @property
    @abstractmethod
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries."""
        raise NotImplementedError  # pragma: no cover - abstract stub

    @property
    def geometries(self) -> gpd.GeoSeries:
        """Get this layer's geometries, applying any geometry mask."""
        gs = _as_geoseries(self.geometry_source)
        if self.geometry_mask is not None:
            gs = gs[self.geometry_mask]
        return gs

    @property
    def geosource(self) -> GeoSource:
        """Get this layer's geosource, applying any geometry mask."""
        if self.geometry_mask is not None:
            if isinstance(self.geometry_source, GeoDataFrame):
                return self.geometry_source[self.geometry_mask]
            else:
                return self.geometry_source[self.geometry_mask]
        return self.geometry_source

    @abstractmethod
    def render(self, ax: Axes, **kwargs: object) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): Target axes.
            **kwargs (object): Layer-specific keyword arguments.

        Returns:
            Axes: Axes with the layer rendered.
        """
        raise NotImplementedError  # pragma: no cover - abstract stub


@dataclass(frozen=True, slots=True)
class _CategoricalColorLayer(_GeoLayer):
    """A geographic layer with categorical color mapping based on a data column.

    Attributes:
        geosource (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        datacolumn (str | None): Optional data column for color mapping. Default is None.
        colormap (GeoColorMap | None): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (MplCompatibleColor | None): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color): Color for geometry edges. Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
        colormap (GeoColorMap | None): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "districtr".
    """

    colormap: GeoColorMap | None = "districtr"

    def __post_init__(self) -> None:
        super(_CategoricalColorLayer, self).__post_init__()

        if isinstance(self.geometry_source, GeoSeries) and self.colormap == "districtr":
            object.__setattr__(self, "colormap", "none")

        needs_datacolumn = (
            self.colormap == "districtr"
            or isinstance(self.colormap, (dict, pd.Series, Colormap))
            or (isinstance(self.colormap, str) and self.colormap in plt.colormaps())
        )

        if (
            isinstance(self.geometry_source, GeoDataFrame)
            and needs_datacolumn
            and self.datacolumn is None
        ):
            raise TypeError("'datacolumn' must be set for color-mapped layers")

        if self.colormap == "districtr" and isinstance(self.geometry_source, GeoDataFrame):
            unique_values = self.geometry_source[self.datacolumn].unique()
            districtr_colors = districtr(len(unique_values))
            object.__setattr__(
                self,
                "colormap",
                self.__map_unique_values_to_colors(unique_values, districtr_colors),
            )

    @staticmethod
    def __map_unique_values_to_colors(
        unique_values: pd.Index,
        color_list: list[Color],
    ) -> CategoryColorMap:
        """Map unique values in the data to colors from the provided list. Filters out NaN values.

        Args:
            unique_values (pd.Index): The unique values to map.
            color_list (list[Color]): The list of colors to use for mapping.
        """
        n_colors = len(color_list)
        non_na_values: list[CategoryKey] = []
        for value in unique_values:
            if pd.notna(value):
                non_na_values.append(value)

        if len(non_na_values) > n_colors:
            raise ValueError(
                "Not enough colors provided to map all unique values; "
                f"received {n_colors} colors for {len(unique_values)} unique values",
            )

        # Try to convert to integers and sort by those if possible
        # Just in case the values are something like ["1", "2", "10"]
        # which would incorrectly sort to ["1", "10", "2"] as strings
        try:
            key_int_pairs = [(key, int(str(key))) for key in non_na_values]
            sorted_keys = sorted(key_int_pairs, key=lambda x: x[1])
            keys_in_order = [k for (k, _) in sorted_keys]
        except (ValueError, TypeError):
            keys_in_order = sorted(non_na_values, key=lambda value: str(value))

        return {k: color_list[i] for i, k in enumerate(keys_in_order)}

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        ret_colors_series: pd.Series

        if self.colormap is None:
            ret_colors_series = pd.Series(
                ["none"] * len(self.geometry_source), index=self.geometry_source.index
            )

        elif isinstance(self.colormap, str) and (
            self.colormap not in plt.colormaps() or self.datacolumn is None
        ):
            color = resolve_color_and_alpha(self.colormap, alpha=self.facealpha)
            ret_colors_series = pd.Series(
                [color] * len(self.geometry_source), index=self.geometry_source.index
            )

        elif isinstance(
            self.colormap, pd.Series
        ):  # pragma: no cover - __post_init__ raises ValueError when colormap is pd.Series (ambiguous truth value); this branch is unreachable
            new_entries = [
                resolve_color_and_alpha(c, alpha=self.facealpha) for c in self.colormap
            ]  # pragma: no cover
            ret_colors_series = pd.Series(
                new_entries, index=self.colormap.index
            )  # pragma: no cover
        elif isinstance(self.colormap, Colormap) or (
            isinstance(self.colormap, str) and self.colormap in plt.colormaps()
        ):
            cmap: Colormap = (
                get_cmap(self.colormap) if isinstance(self.colormap, str) else self.colormap
            )

            # Almost all color maps have at most 256 discrete colors (even the "continuous" ones).
            # This is just a safeguard to avoid indexing errors
            n_colors = int(getattr(cmap, "N", 256))

            value_to_color_dict = self.__map_unique_values_to_colors(
                self.geometry_source[self.datacolumn].unique(),
                [to_hex(cmap(i), keep_alpha=True) for i in range(n_colors)],
            )

            new_entries = []
            for val in self.geometry_source[self.datacolumn]:
                new_color = self.missing_color
                if pd.notna(val):
                    # Try to convert to integer index
                    new_color = resolve_color_and_alpha(
                        value_to_color_dict[val], alpha=self.facealpha
                    )
                new_entries.append(new_color)
            ret_colors_series = pd.Series(new_entries, index=self.geometry_source.index)

        elif isinstance(self.colormap, dict):
            new_entries: list[ResolvedColor] = []
            for val in self.geometry_source[self.datacolumn]:
                color = self.colormap.get(val, self.missing_color)
                color_tup = resolve_color_and_alpha(color, alpha=self.facealpha)
                new_entries.append(color_tup)
            ret_colors_series = pd.Series(new_entries, index=self.geometry_source.index)
        else:
            raise TypeError(
                "'colormap' must be one of: None, str (named colormap or color), "
                "Colormap, dict, or pd.Series; got "
                f"{type(self.colormap).__name__!r}",
            )

        return ret_colors_series.reindex(self.geometries.index)

    def render(
        self,
        ax: Axes,
        *,
        target_crs: CRSLike | None = None,
        **kwargs: object,
    ) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.
            **kwargs (object): Additional keyword arguments (not used but included to satisfy
                render function signature contract).

        Returns:
            Axes: The Axes with the layer rendered.
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        if (
            not isinstance(self.geometry_source, GeoSeries)
            and self.datacolumn is not None
            and self.datacolumn not in self.geometry_source.columns
        ):
            raise KeyError(
                f"Column {self.datacolumn!r} not found in GeoDataFrame."
                f" Available columns: {list(self.geometry_source.columns)}"
            )

        edge_color_tup = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
        )

        geoseries = self._geometries_in_crs(target_crs)
        _ = geoseries.plot(
            ax=ax,
            color=self.color_series,
            edgecolor=edge_color_tup,
            linewidth=self.edgewidth,
            zorder=self.zorder,
        )

        return ax


@dataclass(frozen=True, slots=True)
class _MarkerLayer:
    """A layer of point markers with optional labels.

    Attributes:
        point_geometries (GeoSeries): A GeoSeries of Point geometries for the markers.
        labels (Sequence[str] | None): Optional labels for each marker.
        marker_options (PointMarkerOptions): Marker style settings. Uses default constructor if
            not provided.
        show_labels (bool): Whether to show labels on the markers. Default is True.
        font_options (LabelFontOptions): Font options for the labels. Uses default constructor if
            not provided.
        zorder (int): Z-order for rendering. Default is 2.
    """

    point_geometries: GeoSeries

    # Optional labels (same length as point_geometries)
    labels: Sequence[str] | None = None

    # Marker style (shared across the layer)
    marker_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)

    # Label style (centered in marker)
    show_labels: bool = True
    labelfont_options: LabelFontOptions = field(default_factory=LabelFontOptions)
    labelbox_options: LabelBoxOptions = field(default_factory=LabelBoxOptions)
    zorder: int = 2

    def __post_init__(self) -> None:
        if self.point_geometries is None:
            raise TypeError("MarkerLayer requires `point_geometries` (a GeoSeries of Points).")

        if self.labels is not None and len(self.labels) != len(self.point_geometries):
            raise ValueError("`labels` must have the same length as `point_geometries`.")

        if self.marker_options is None:
            object.__setattr__(self, "marker_options", PointMarkerOptions())

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        # required by _GeoLayer, unused for markers
        return pd.Series(
            dtype=object
        )  # pragma: no cover - implemented only to satisfy the abstract interface

    def render(
        self,
        ax: Axes,
        *,
        target_crs: CRSLike | None = None,
        **kwargs: object,
    ) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.
            **kwargs (object): Additional keyword arguments (not used).

        Returns:
            Axes: The Axes with the layer rendered.
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        point_geometries = self.point_geometries

        # Reproject points if needed
        if getattr(point_geometries, "crs", None) is not None and target_crs is not None:
            if point_geometries.crs != target_crs:
                point_geometries = point_geometries.to_crs(target_crs)

        x_coordinates = point_geometries.x.to_numpy()
        y_coordinates = point_geometries.y.to_numpy()

        # PointMarkerOptions already returns RGBA colors with alpha baked in.
        marker_kwargs = dict(self.marker_options.to_mpl_settings_dict())
        marker_kwargs.pop("zorder", None)

        if not self.show_labels or self.labels is None:
            ax.plot(
                x_coordinates,
                y_coordinates,
                linestyle="None",
                zorder=int(self.zorder),
                **marker_kwargs,
            )
        else:
            outline_color, _ = resolve_color_and_alpha(
                self.labelfont_options.outlinecolor,
                alpha=1.0,
            )
            text_effects: list[patheffects.AbstractPathEffect] = [
                patheffects.Stroke(
                    linewidth=float(self.labelfont_options.outlinewidth),
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]

            text_color, text_alpha = resolve_color_and_alpha(
                self.labelfont_options.fontcolor,
                alpha=self.labelfont_options.fontalpha,
            )

            for x_value, y_value, label_text in zip(x_coordinates, y_coordinates, self.labels):
                ax.plot(
                    x_value,
                    y_value,
                    linestyle="None",
                    zorder=int(self.zorder),
                    **marker_kwargs,
                )

                text_artist = ax.text(
                    float(x_value),
                    float(y_value),
                    str(label_text),
                    ha="center",
                    va="center",
                    zorder=int(self.zorder),
                    bbox=self.labelbox_options.to_mpl_bbox(),
                    clip_on=True,
                    **self.labelfont_options.to_mpl_text_kwargs(),
                )
                text_artist.set_clip_path(ax.patch)
                text_artist.set_path_effects(text_effects)

        return ax


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
        dpi: int = 300,
        show_axis: bool = False,
        target_crs: CRSLike | None = None,
        include_default_outline: bool = True,
        silent: bool = False,
    ) -> None:
        """Initialize a GeoPlot.

        Args:
            gdf (GeoDataFrame): The base GeoDataFrame for the plot.
            dpi (int): The DPI for the Matplotlib Figure. Default is 300.
            show_axis (bool): Whether to show axis ticks and labels. Default is False.
            target_crs (CRSLike | None): The target CRS for reprojecting geometries.
                If None, uses the CRS of `gdf`. Default is None.
            include_default_outline (bool): Whether to include a default outline layer around
                the geometries in `gdf`. Default is True.
            silent (bool): Whether to suppress informational output throughout the rendering
                process. Default is False.
        """
        self.gdf = gdf

        self.fig, self._ax = plt.subplots(dpi=dpi)

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

        self.show_axis = show_axis
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

        if include_default_outline:
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
        *,
        geosource: GeoDataFrame | GeoSeries | None = None,
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
        *,
        geosource: GeoDataFrame | GeoSeries | None = None,
        geometry_mask: pd.Series | None = None,
        label_column: str | None = None,
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
        *,
        points_geoseries: gpd.GeoSeries | None = None,
        latitude_longitude_list: Sequence[tuple[float, float]] | None = None,
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
                markers. If None, `latitude_longitude_list` must be provided. Default is None.
            latitude_longitude_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            input_crs (CRSLike | None, optional): The CRS of the input points if using
                ``latitude_longitude_list``.
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

        if points_geoseries is None and latitude_longitude_list is None:
            raise ValueError("Either `points_geoseries` or `latitude_longitude_list` must be set.")
        if points_geoseries is not None and latitude_longitude_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latitude_longitude_list` may be set at a time.",
            )

        if latitude_longitude_list is not None:
            # crs EPSG:4326 corresponds to lat/lon
            point_geometries = gpd.GeoSeries(
                [
                    Point(float(longitude), float(latitude))
                    for latitude, longitude in latitude_longitude_list
                ],
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
                "'points_geoseries' or 'latitude_longitude_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latitude_longitude_list': {type(latitude_longitude_list).__name__!r}",
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
        *,
        points_geoseries: gpd.GeoSeries | None = None,
        latitude_longitude_list: Sequence[tuple[float, float]] | None = None,
        input_crs: CRSLike | None = None,
        labels: Sequence[str] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of markers (points) to the GeoPlot.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latitude_longitude_list` must be provided. Default is None.
            latitude_longitude_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            input_crs (CRSLike | None, optional): The CRS of the input points if using
                ``latitude_longitude_list``.
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
        if points_geoseries is None and latitude_longitude_list is None:
            raise ValueError("Either `points_geoseries` or `latitude_longitude_list` must be set.")
        if points_geoseries is not None and latitude_longitude_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latitude_longitude_list` may be set at a time.",
            )
        if points_geoseries is None and latitude_longitude_list is not None:
            n_labels = len(list(latitude_longitude_list))
        elif points_geoseries is not None:
            n_labels = len(points_geoseries)
        else:  # pragma: no cover - defensive guard; the preceding if/elif already covers all valid states
            raise RuntimeError(  # pragma: no cover
                "An unexpected error occured in add_label_layer. One of the argurments "
                "'points_geoseries' or 'latitude_longitude_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latitude_longitude_list': {type(latitude_longitude_list).__name__!r}",
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
            latitude_longitude_list=latitude_longitude_list,
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

    def set_xlimits(self, lower: float, upper: float) -> None:
        """Set x-axis limits to apply when the plot is built.

        Args:
            lower (float): The left x-axis limit.
            upper (float): The right x-axis limit.
        """
        self._xlim = (float(lower), float(upper))

    def set_ylimits(self, lower: float, upper: float) -> None:
        """Set y-axis limits to apply when the plot is built.

        Args:
            lower (float): The bottom y-axis limit.
            upper (float): The top y-axis limit.
        """
        self._ylim = (float(lower), float(upper))

    def set_xlim(self, left: float, right: float) -> None:
        """Alias for :meth:`set_xlimits`.

        Args:
            left (float): Left x-axis limit.
            right (float): Right x-axis limit.

        Returns:
            None
        """
        self.set_xlimits(lower=left, upper=right)

    def set_ylim(self, bottom: float, top: float) -> None:
        """Alias for :meth:`set_ylimits`.

        Args:
            bottom (float): Bottom y-axis limit.
            top (float): Top y-axis limit.

        Returns:
            None
        """
        self.set_ylimits(lower=bottom, upper=top)

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
        """Build the plot by rendering all layers and applying settings."""
        self._ax.clear()

        if not self.show_axis:
            self._ax.set_axis_off()

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
            layer.render(self._ax, target_crs=self.target_crs)

    def _apply_limits(self) -> None:
        """Apply stored x/y limits to the axes."""
        if self._xlim is not None:
            self._ax.set_xlim(*self._xlim)
        if self._ylim is not None:
            self._ax.set_ylim(*self._ylim)

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
            tmp.render(ax, target_crs=self.target_crs)
            label_positions.update(
                {label: Point(pt.x, pt.y) for label, pt in zip(labels, pts.geometry.tolist())}
            )
        return label_positions

    @abstractmethod
    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Build the plot and apply stored settings like limits."""
        self._build_plot()  # pragma: no cover - abstract stub body; concrete subclasses fully override without super()
        self._apply_limits()  # pragma: no cover
        label_points = self._draw_deferred_labels()  # pragma: no cover
        return label_points  # pragma: no cover

    @property
    def ax(self) -> Axes:
        """The Matplotlib Axes object for the plot."""
        self._build_and_apply_settings()
        return self._ax

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

    def show(self) -> None:
        """Display inline in notebooks, or open a GUI window in scripts."""
        self._build_and_apply_settings()
        show_figure(self.fig, non_gui_filename="geoplot.png", non_gui_prefix="GeoPlot")

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
