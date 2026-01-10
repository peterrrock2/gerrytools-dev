from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

import geopandas as gpd
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from shapely.geometry import Point

from gerrytools.colors import districtr, resolve_color_and_alpha
from gerrytools.plotting.gerryplot import PointMarkerSettings
from gerrytools.typing import Color

GeoSource = GeoDataFrame | GeoSeries


def _as_geoseries(source: GeoSource) -> gpd.GeoSeries:
    return source.geometry if isinstance(source, gpd.GeoDataFrame) else source


@dataclass(frozen=True, slots=True)
class _GeoLayer(ABC):
    """Abstract base class for a geographic layer to be rendered on a GeoPlot.

    Attributes:
        geosource (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        datacolumn (str | None): Optional data column for color mapping. Default is None.
        colormap (str | Color | Colormap | dict[Any, Color] | pd.Series): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (Any): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color): Color for geometry edges. Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
    """

    # Try to keep the GeoSource as a reference so that users don't copy the polygons all the time.
    geosource: GeoSource
    geometry_mask: pd.Series | None = None
    datacolumn: str | None = None
    colormap: str | Color | Colormap | dict[Any, Color] | pd.Series = "Purples"
    missing_color: Any = "lightgrey"
    facealpha: float | None = None
    edgecolor: Color = "none"
    edgealpha: float | None = None
    edgewidth: float = 0.5
    zorder: int = 1

    def _geometries_in_crs(self, target_crs) -> gpd.GeoSeries:
        """Return this layer's geometries (respecting mask) reprojected to target_crs.

        Args:
            target_crs: The target CRS to reproject to.

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
        raise NotImplementedError

    @property
    def geometries(self) -> gpd.GeoSeries:
        """Get this layer's geometries, applying any geometry mask."""
        gs = _as_geoseries(self.geosource)
        if self.geometry_mask is not None:
            gs = gs[self.geometry_mask]
        return gs

    @abstractmethod
    def render(self, ax: Axes, **kwargs) -> Axes:
        """Render this layer onto the given Axes."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _ContinuousColorLayer(_GeoLayer):
    """A geographic layer with continuous color mapping based on a data column.

    Attributes:
        colormap (str | Colormap): The colormap to use for continuous color mapping.
        vmin (float | None): Minimum data value for color mapping.
        vmax (float | None): Maximum data value for color mapping.
        norm (Normalize | None): Custom normalization for color mapping.
        bins (int | list[float] | None): Optional binning specification for discrete intervals.
    """

    colormap: str | Colormap = "Purples"
    vmin: float | None = None
    vmax: float | None = None
    norm: Normalize | None = None
    bins: int | list[float] | None = None

    def __post_init__(self) -> None:
        super(_ContinuousColorLayer, self).__post_init__()
        if not isinstance(self.geosource, GeoDataFrame):
            raise TypeError(
                "Tried to create a continuous color layer using geosource of type "
                f"{type(self.geosource).__name__!r}; geosource must be a GeoDataFrame",
            )

        if not isinstance(self.colormap, (str, Colormap)):
            raise TypeError(
                "'colormap' must be a str or Colormap for continuous color layers; got "
                f"{type(self.colormap).__name__!r}",
            )
        if isinstance(self.colormap, str) and self.colormap not in plt.colormaps():
            raise ValueError(
                f"Colormap name {self.colormap!r} not found in matplotlib colormaps. "
                f"Available colormaps are: {plt.colormaps()}",
            )
        if self.datacolumn is None:
            raise TypeError("'datacolumn' must be set for color-mapped layers")

    def _data_series(self) -> pd.Series:
        """Get the data series (used in color mapping)."""
        return self.geosource[self.datacolumn]

    def _effective_bounds(self, dataseries: pd.Series) -> tuple[float, float]:
        """Determine the effective data bounds for color mapping.

        Args:
            dataseries (pd.Series): The data series to analyze.
        """
        non_na = dataseries.dropna()
        if non_na.empty:
            lo = float(self.vmin if self.vmin is not None else 0.0)
            hi = float(self.vmax if self.vmax is not None else 1.0)
            return lo, hi

        lo = float(non_na.min() if self.vmin is None else self.vmin)
        hi = float(non_na.max() if self.vmax is None else self.vmax)
        return lo, hi

    def _bin_boundaries(self, lower: float, upper: float) -> pd.IntervalIndex:
        """Get the bin boundaries as an IntervalIndex.

        Args:
            lower (float): The lower bound of the data.
            upper (float): The upper bound of the data.
        """
        if self.bins is None:
            raise RuntimeError("Called _bin_boundaries but bins is None")

        if isinstance(self.bins, int):
            return pd.interval_range(
                start=lower,
                end=upper,
                periods=self.bins,
                closed="left",
            )

        return pd.IntervalIndex.from_breaks(self.bins, closed="left")

    def _color_mapping_for_bins(
        self, boundaries: pd.IntervalIndex
    ) -> tuple[list[float], list[str]]:
        """Get the color mapping for the given bin boundaries.

        Args:
            boundaries (pd.IntervalIndex): The bin boundaries.

        Returns:
            tuple[list[float], list[str]]: The edges and corresponding hex colors for the bins.
        """
        cmap = get_cmap(self.colormap, lut=len(boundaries))
        colors = [to_hex(cmap(i), keep_alpha=True) for i in range(len(boundaries))]
        edges = boundaries.left.tolist() + [boundaries.right[-1]]
        return edges, colors

    def _mappable(self) -> tuple[ScalarMappable, dict[str, Any]]:
        """Get a ScalarMappable and the colorbar kwargs for this layer.

        Returns:
            tuple[ScalarMappable, dict[str, Any]]: The ScalarMappable and colorbar kwargs.
        """
        s = self._data_series()
        lower, upper = self._effective_bounds(s)

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower, upper)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            listed = ListedColormap(interval_hex_colors)

            norm = BoundaryNorm(edges, ncolors=listed.N, clip=False)

            m = ScalarMappable(norm=norm, cmap=listed)
            m.set_array([])

            cbar_kwargs = {
                "ticks": edges,
                "spacing": "uniform",
                "boundaries": edges,
            }
            return m, cbar_kwargs

        if self.norm is not None:
            norm = self.norm
        else:
            norm = Normalize(vmin=lower, vmax=upper)

        cmap = get_cmap(self.colormap)
        m = ScalarMappable(norm=norm, cmap=cmap)
        m.set_array([])

        return m, {}

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        data_series = self._data_series()
        lower_bound, upper_bound = self._effective_bounds(data_series)

        colors: dict[Any, Any] = {}

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower_bound, upper_bound)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            interval_to_hex = {
                interval: interval_hex_colors[i] for i, interval in enumerate(boundaries)
            }

            for idx, value in data_series.items():
                if pd.isna(value):
                    colors[idx] = self.missing_color
                    continue

                # ensure upper bound gets last bin
                if value == upper_bound:
                    interval_i = len(boundaries) - 1
                else:
                    try:
                        loc = boundaries.get_loc(value)
                        interval_i = int(loc) if not isinstance(loc, slice) else loc.start
                    except KeyError:
                        interval_i = -1

                if interval_i == -1:
                    colors[idx] = self.missing_color
                else:
                    colors[idx] = resolve_color_and_alpha(
                        interval_to_hex[boundaries[interval_i]],
                        alpha=self.facealpha,
                    )

            return pd.Series(colors).reindex(self.geometries.index)
        else:
            if self.norm is not None:
                norm = self.norm
            else:
                norm = Normalize(vmin=lower_bound, vmax=upper_bound)

            cmap = get_cmap(self.colormap)

            for idx, value in data_series.items():
                if pd.isna(value):
                    color = self.missing_color
                else:
                    normalized_value = norm(value)
                    color: tuple[str, int | float] = resolve_color_and_alpha(
                        to_hex(cmap(normalized_value), keep_alpha=True),
                        alpha=self.facealpha,
                    )
                colors[idx] = color

        return pd.Series(colors).reindex(self.geometries.index)

    def render(self, ax: Axes, *, target_crs=None, **kwargs) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs: The target CRS to reproject geometries to.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Axes: The Axes with the layer rendered.
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        if self.datacolumn not in self.geosource.columns:
            raise KeyError(
                f"Column {self.datacolumn!r} not found in GeoDataFrame."
                f" Available columns: {list(self.geosource.columns)}"
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
class _CategoricalColorLayer(_GeoLayer):
    """A geographic layer with categorical color mapping based on a data column.

    Attributes:
        colormap (str | Color | Colormap | dict[Any, Color] | pd.Series): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "districtr".
    """

    colormap: str | Color | Colormap | dict[Any, Color] | pd.Series = "districtr"

    def __post_init__(self) -> None:
        super(_CategoricalColorLayer, self).__post_init__()

        if isinstance(self.geosource, GeoSeries) and self.colormap == "districtr":
            object.__setattr__(self, "colormap", "none")

        needs_datacolumn = (
            self.colormap == "districtr"
            or isinstance(self.colormap, (dict, pd.Series, Colormap))
            or (isinstance(self.colormap, str) and self.colormap in plt.colormaps())
        )

        if (
            isinstance(self.geosource, GeoDataFrame)
            and needs_datacolumn
            and self.datacolumn is None
        ):
            raise TypeError("'datacolumn' must be set for color-mapped layers")

        if self.colormap == "districtr" and isinstance(self.geosource, GeoDataFrame):
            unique_values = self.geosource[self.datacolumn].unique()
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
    ) -> dict[Any, Color]:
        """Map unique values in the data to colors from the provided list. Filters out NaN values.

        Args:
            unique_values (pd.Index): The unique values to map.
            color_list (list[Color]): The list of colors to use for mapping.
        """
        n_colors = len(color_list)
        non_na_values = list(filter(pd.notna, unique_values))
        if len(non_na_values) > n_colors:
            raise ValueError(
                "Not enough colors provided to map all unique values; "
                f"received {n_colors} colors for {len(unique_values)} unique values",
            )

        # Try to convert to integers and sort by those if possible
        # Just in case the values are something like ["1", "2", "10"]
        # which would incorrectly sort to ["1", "10", "2"] as strings
        try:
            key_int_pairs = [(key, int(key)) for key in non_na_values]
            sorted_keys = sorted(key_int_pairs, key=lambda x: x[1])
            keys_in_order = [k for (k, _) in sorted_keys]
        except (ValueError, TypeError):
            keys_in_order = sorted(non_na_values)

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
                ["none"] * len(self.geosource), index=self.geosource.index
            )

        elif isinstance(self.colormap, str) and (
            self.colormap not in plt.colormaps() or self.datacolumn is None
        ):
            color = resolve_color_and_alpha(self.colormap, alpha=self.facealpha)
            ret_colors_series = pd.Series([color] * len(self.geosource), index=self.geosource.index)

        elif isinstance(self.colormap, pd.Series):
            new_entries = [resolve_color_and_alpha(c, alpha=self.facealpha) for c in self.colormap]
            ret_colors_series = pd.Series(new_entries, index=self.colormap.index)
        elif isinstance(self.colormap, Colormap) or (
            isinstance(self.colormap, str) and self.colormap in plt.colormaps()
        ):
            cmap = self.colormap
            if isinstance(self.colormap, str):
                cmap = get_cmap(self.colormap)

            # Almost all color maps have at most 256 discrete colors even the "continuous" ones.
            # This is just a safeguard to avoid indexing errors
            if hasattr(cmap, "N") and isinstance(cmap.N, int):
                n_colors = cmap.N
            else:
                n_colors = 256

            value_to_color_dict = self.__map_unique_values_to_colors(
                self.geosource[self.datacolumn].unique(),
                [to_hex(cmap(i), keep_alpha=True) for i in range(n_colors)],
            )

            new_entries = []
            for val in self.geosource[self.datacolumn]:
                new_color = self.missing_color
                if pd.notna(val):
                    # Try to convert to integer index
                    new_color = resolve_color_and_alpha(
                        value_to_color_dict[val], alpha=self.facealpha
                    )
                new_entries.append(new_color)
            ret_colors_series = pd.Series(new_entries, index=self.geosource.index)

        elif isinstance(self.colormap, dict):
            new_entries = []
            for val in self.geosource[self.datacolumn]:
                color = self.colormap.get(val, self.missing_color)
                color_tup = resolve_color_and_alpha(color, alpha=self.facealpha)
                new_entries.append(color_tup)
            ret_colors_series = pd.Series(new_entries, index=self.geosource.index)
        else:
            raise TypeError(
                "'colormap' must be one of: None, str (named colormap or color), "
                "Colormap, dict, or pd.Series; got "
                f"{type(self.colormap).__name__!r}",
            )

        return ret_colors_series.reindex(self.geometries.index)

    def render(self, ax: Axes, *, target_crs=None, **kwargs) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs: The target CRS to reproject geometries to.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Axes: The Axes with the layer rendered.
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        if (
            not isinstance(self.geosource, GeoSeries)
            and self.datacolumn is not None
            and self.datacolumn not in self.geosource.columns
        ):
            raise KeyError(
                f"Column {self.datacolumn!r} not found in GeoDataFrame."
                f" Available columns: {list(self.geosource.columns)}"
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
class LabelFontOptions:
    """Font options for labels on marker layers.

    Attributes:
        fontcolor (Color): The color of the font. Default is "white".
        fontalpha (float | None): The alpha transparency of the font. Default is 1.0.
        fontsize (float): The size of the font. Default is 6.0.
        fontweight (str): The weight of the font. Default is "bold".
        outlinecolor (Color): The color of the text outline. Default is "black".
        outlinewidth (float): The width of the text outline. Default is 0.75.
    """

    fontcolor: Color = "white"
    fontalpha: float | None = 1.0
    fontsize: float = 6.0
    fontweight: str = "bold"
    outlinecolor: Color = "black"
    outlinewidth: float = 0.75


@dataclass(frozen=True, slots=True)
class _MarkerLayer:
    """A layer of point markers with optional labels.

    Attributes:
        point_geometries (GeoSeries): A GeoSeries of Point geometries for the markers.
        labels (Sequence[str] | None): Optional labels for each marker.
        marker_options (PointMarkerSettings): Marker style settings. Uses default constructor if
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
    marker_options: PointMarkerSettings = PointMarkerSettings()

    # Label style (centered in marker)
    show_labels: bool = True
    font_options: LabelFontOptions = LabelFontOptions()
    zorder: int = 2

    def __post_init__(self) -> None:
        if self.point_geometries is None:
            raise TypeError("MarkerLayer requires `point_geometries` (a GeoSeries of Points).")

        if self.labels is not None and len(self.labels) != len(self.point_geometries):
            raise ValueError("`labels` must have the same length as `point_geometries`.")

        if self.marker_options is None:
            object.__setattr__(self, "marker_options", PointMarkerSettings())

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        # required by _GeoLayer, unused for markers
        return pd.Series(dtype=object)

    def render(self, ax: Axes, *, target_crs=None, **kwargs) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs: The target CRS to reproject geometries to.
            **kwargs: Additional keyword arguments (not used).
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

        # PointMarkerSettings already returns RGBA colors with alpha baked in.
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
                self.font_options.outlinecolor,
                alpha=1.0,
            )
            text_effects = [
                patheffects.Stroke(
                    linewidth=float(self.font_options.outlinewidth),
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]

            text_color, text_alpha = resolve_color_and_alpha(
                self.font_options.fontcolor,
                alpha=self.font_options.fontalpha,
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
                    fontsize=float(self.font_options.fontsize),
                    fontweight=str(self.font_options.fontweight),
                    color=text_color,
                    alpha=float(text_alpha) if text_alpha is not None else None,
                    zorder=int(self.zorder),
                )
                text_artist.set_path_effects(text_effects)

        return ax


@dataclass(slots=True)
class ColorbarOptions:
    """Options for configuring colorbars in GeoPlot.

    Attributes:
        outer_pad (float): Padding between the colorbar and the plot edges (figure-relative).
        inner_pad (float): Padding between the colorbar and the main plot area (figure-relative).
        width (float): Width of the colorbar (figure-relative).
        right_margin (float): Margin to the right of the colorbar (figure-relative).
        tick_fontsize (float): Font size for colorbar ticks.
        tick_pad (float): Padding for colorbar ticks.
        label_fontsize (float | None): Font size for colorbar label.
        label_rotation (float | None): Rotation angle for colorbar label.
        label_pad (float | None): Padding for colorbar label.
        orientation (Literal["vertical", "horizontal"]): Orientation of the colorbar.
        extend (Literal["neither", "both", "min", "max"]): Extension style for the colorbar.
        format (str | None): Format string for colorbar tick labels.
        fraction (float | None): Fraction of original size for colorbar.
        shrink (float | None): Shrink factor for colorbar.
        aspect (float | None): Aspect ratio for colorbar.
        force_ticks (list[float] | None): Explicit tick locations for the colorbar.
        force_ticklabels (list[str] | None): Explicit tick labels for the colorbar.
        max_n_ticks (int | None): Maximum number of ticks on the colorbar.
    """

    # --- layout (figure-relative coords) ---
    outer_pad: float = 0.03
    inner_pad: float = 0.06
    width: float = 0.02
    right_margin: float = 0.02

    # --- tick appearance (axes.tick_params) ---
    tick_fontsize: float = 8.0
    tick_pad: float = 2.0

    # --- label appearance (cb.set_label) ---
    label_fontsize: float | None = None
    label_rotation: float | None = None
    label_pad: float | None = None

    # --- fig.colorbar behavior ---
    orientation: Literal["vertical", "horizontal"] = "vertical"
    extend: Literal["neither", "both", "min", "max"] = "neither"
    format: str | None = None  # e.g. ".2f"
    fraction: float | None = None  # rarely needed when using cax
    shrink: float | None = None  # rarely needed when using cax
    aspect: float | None = None  # rarely needed when using cax

    # --- explicit overrides (optional) ---
    force_ticks: list[float] | None = None
    force_ticklabels: list[str] | None = None
    max_n_ticks: int | None = None


class GeoPlot:
    """A class for creating geographic plots with multiple layers.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        show_axis (bool): Whether to show axis lines and labels.
        target_crs: The target CRS for reprojecting geometries.
        show_colorbars (bool): Whether to display colorbars for layers.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        dpi: int = 300,
        show_axis: bool = False,
        target_crs=None,
        show_colorbars: bool = False,
    ) -> None:
        self.gdf = gdf

        self.fig = Figure(dpi=dpi)
        self._canvas = FigureCanvas(self.fig)  # gives the Figure a renderer
        self._ax = self.fig.add_subplot(111)

        self.show_axis = show_axis
        self.target_crs = target_crs if target_crs is not None else getattr(gdf, "crs", None)

        self.show_colorbars = show_colorbars

        self._xlim: tuple[float, float] | None = None
        self._ylim: tuple[float, float] | None = None

        self._colorbar_options: ColorbarOptions = ColorbarOptions()

        self._choropleth_layers: list[_ContinuousColorLayer] = []
        self._districting_plan_layers: list[_CategoricalColorLayer] = []
        self._outline_layers: list[_CategoricalColorLayer] = []
        self._highlight_layers: list[_CategoricalColorLayer] = []
        self._marker_layers: list[_MarkerLayer] = []

        self._colorbar_axes: list[Axes] = []

    def add_choropleth_layer(
        self,
        *,
        geosource: GeoDataFrame | None = None,
        datacolumn: str,
        colormap: str | Colormap = "Purples",
        missing_color: Any = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: Normalize | None = None,
        bins: int | list[float] | None = None,
        zorder: int = 0,
    ) -> None:
        """Add a choropleth layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | None): The GeoDataFrame source for the layer.
                If None, uses the base gdf of the GeoPlot. Default is None.
            datacolumn (str): The data column to use for color mapping.
            colormap (str | Colormap): The colormap to use for color mapping. Default is "Purples".
            missing_color (Any): Color to use for missing data. Default is "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            vmin (float | None): Minimum data value for color mapping. Default is None.
            vmax (float | None): Maximum data value for color mapping. Default is None.
            norm (Normalize | None): Custom normalization for color mapping. Default is None.
            bins (int | list[float] | None): Optional binning specification for discrete intervals.
                Default is None.
            zorder (int): Z-order for rendering. Default is 0.
        """
        if geosource is None:
            geosource = self.gdf
        layer = _ContinuousColorLayer(
            geosource=geosource,
            datacolumn=datacolumn,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            bins=bins,
            zorder=zorder,
        )
        self._choropleth_layers.append(layer)

    def add_districting_plan_layer(
        self,
        *,
        geosource: GeoDataFrame | None = None,
        plancolumn: str,
        dissolve: bool = False,
        show_labels: bool = False,
        labelmarkeroptions: PointMarkerSettings | None = None,
        labelfontoptions: LabelFontOptions | None = None,
        colormap: str | Colormap | dict[Any, Color] | pd.Series = "districtr",
        missing_color: Any = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        zorder: int = 2,
    ) -> None:
        """Add a districting plan layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | None): The GeoDataFrame source for the layer.
                If None, uses the base gdf of the GeoPlot. Default is None.
            plancolumn (str): The column containing district identifiers.
            dissolve (bool): Whether to dissolve geometries by district. Default is False.
            show_labels (bool): Whether to show district labels. Default is False.
            labelmarkeroptions (PointMarkerSettings | None): Marker settings for district labels.
                If None, uses default settings. Default is None.
            labelfontoptions (LabelFontOptions | None): Font options for district labels.
                If None, uses default settings. Default is None.
            colormap (str | Colormap | dict[Any, Color] | pd.Series): Color mapping specification.
                Can be a single color, a named colormap, a Colormap object, or a mapping from district
                identifiers to colors. Default is "districtr".
            missing_color (Any): Color to use for missing data. Default is "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            zorder (int): Z-order for rendering. Default is 2.
        """
        if geosource is None:
            plan_gdf = self.gdf
        else:
            plan_gdf = geosource

        if dissolve:
            plan_gdf = GeoDataFrame(plan_gdf.dissolve(by=plancolumn).reset_index())

        layer = _CategoricalColorLayer(
            geosource=plan_gdf,
            datacolumn=plancolumn,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )
        self._districting_plan_layers.append(layer)

        if show_labels:
            # Use dissolved geometry for label placement so it's one marker per district
            label_geometry_gdf = plan_gdf
            if not dissolve:
                label_geometry_gdf = GeoDataFrame(plan_gdf.dissolve(by=plancolumn).reset_index())

            # Compute interior points in the plot CRS (so placement matches what you see)
            label_geometries = label_geometry_gdf.geometry
            if getattr(label_geometries, "crs", None) is not None and self.target_crs is not None:
                if label_geometries.crs != self.target_crs:
                    label_geometries = label_geometries.to_crs(self.target_crs)

            label_points = label_geometries.representative_point()
            label_text = []

            for label in label_geometry_gdf[plancolumn].astype(str).tolist():
                # try to convert to ints for labelling
                try:
                    label_text.append(str(int(label)))
                except (ValueError, TypeError):
                    label_text.append(label)

            self.add_marker_layer(
                points_geoseries=label_points,
                labels=label_text,
                labelmarker_options=(
                    labelmarkeroptions
                    if labelmarkeroptions is not None
                    else PointMarkerSettings(
                        markerfacecolor="none",
                        markerfacealpha=0.0,
                        marker="o",
                        markersize=10.0,
                        markeredgecolor="none",
                        markeredgealpha=0.0,
                        markeredgewidth=0.0,
                    )
                ),
                labelfont_options=(
                    labelfontoptions if labelfontoptions is not None else LabelFontOptions()
                ),
            )

    def add_outline_layer(
        self,
        *,
        geosource: GeoDataFrame | GeoSeries | None = None,
        dissolve_column: str | None = None,
        geometry_mask: pd.Series | None = None,
        edgecolor: Color = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        zorder: int = 3,
    ) -> None:
        """Add an outline layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source
                for the layer. If None, uses the base gdf of the GeoPlot. Default is None.
            dissolve_column (str | None): Optional column to dissolve geometries by
                before outlining. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
                Default is None.
            edgecolor (Color): Color for geometry edges. Default is "black".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
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
            geosource=geosource,
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

    def add_highlight_layer(
        self,
        *,
        geosource: GeoDataFrame | GeoSeries | None = None,
        geometry_mask: pd.Series | None = None,
        filter_pairs: tuple[Any, Any] | tuple[tuple[Any, Any], ...] | None = None,
        facecolor: Color = "gray",
        facealpha: float | None = 0.5,
        zorder: int = 10,
    ) -> None:
        """Add a highlight layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source
                for the layer. If None, uses the base gdf of the GeoPlot. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
                Default is None.
            filter_pairs (tuple[Any, Any] | tuple[tuple[Any, Any], ...] | None): Optional pairs of
                (column name, value) to filter geometries by. Default is None.
            facecolor (Color): Color for geometry faces. Default is "gray".
            facealpha (float | None): Alpha transparency for face colors. Default is 0.5.
            zorder (int): Z-order for rendering. Default is 10.
        """
        if geosource is None:
            geometries = self.gdf.geometry
        else:
            geometries = _as_geoseries(geosource)
        if geometry_mask is not None:
            geometries = geometries[geometry_mask]

        geometries = GeoSeries(geometries.union_all())

        layer = _CategoricalColorLayer(
            geosource=geometries,
            colormap=facecolor,
            missing_color="none",
            facealpha=facealpha,
            edgecolor="none",
            edgealpha=None,
            edgewidth=0.0,
            zorder=zorder,
        )
        self._highlight_layers.append(layer)

    def add_marker_layer(
        self,
        *,
        points_geoseries: gpd.GeoSeries | None = None,
        latitude_longitude_list: Sequence[tuple[float, float]] | None = None,
        labels: Sequence[str] | None = None,
        input_crs=None,
        zorder: int = 2,
        labelmarker_options: PointMarkerSettings = PointMarkerSettings(
            markerfacecolor="white",
            markerfacealpha=1.0,
            marker="o",
            markersize=3.0,
            markeredgecolor="black",
            markeredgealpha=1.0,
            markeredgewidth=0.5,
        ),
        show_labels: bool = True,
        labelfont_options: LabelFontOptions = LabelFontOptions(),
    ) -> None:
        """Add a layer of markers (points) to the GeoPlot.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latitude_longitude_list` must be provided. Default is None.
            latitude_longitude_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            labels (Sequence[str] | None): Optional labels for each marker. Default is None.
            input_crs: The CRS of the input points if using `latitude_longitude_list`.
                If None, assumes EPSG:4326 (lat/lon). Default is None.
            zorder (int) Z-order for rendering. Default is 2.
            labelmarker_options (PointMarkerSettings): Marker style settings. Uses the following
                defaults:
                    - markerfacecolor="white",
                    - markerfacealpha=1.0,
                    - marker="o",
                    - markersize=3.0,
                    - markeredgecolor="black",
                    - markeredgealpha=1.0,
                    - markeredgewidth=0.5.
            show_labels (bool): Whether to show labels on the markers. Default is True.
            labelfont_options (LabelFontOptions): Font options for the labels. Uses default
                constructor if not provided. Uses the default constructor if not provided.
        """
        if points_geoseries is None and latitude_longitude_list is None:
            raise ValueError("Either `points_geoseries` or `latitude_longitude_list` must be set.")
        if points_geoseries is not None and latitude_longitude_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latitude_longitude_list` "
                "may be set at a time.",
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
        else:
            raise RuntimeError(
                "An unexpected error occured in add_marker_layer. One of the argurments "
                "'points_geoseries' or 'latitude_longitude_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latitude_longitude_list': {type(latitude_longitude_list).__name__!r}",
            )

        marker_layer = _MarkerLayer(
            point_geometries=point_geometries,
            labels=labels,
            marker_options=labelmarker_options,
            show_labels=show_labels,
            font_options=labelfont_options,
            zorder=zorder,
        )
        self._marker_layers.append(marker_layer)

    def set_colorbar_options(
        self,
        *,
        outer_pad: float | None = None,
        inner_pad: float | None = None,
        width: float | None = None,
        right_margin: float | None = None,
        tick_fontsize: float | None = None,
        tick_pad: float | None = None,
        label_fontsize: float | None = None,
        label_rotation: float | None = None,
        label_pad: float | None = None,
        orientation: Literal["vertical", "horizontal"] | None = None,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        format: str | None = None,
        fraction: float | None = None,
        shrink: float | None = None,
        aspect: float | None = None,
        force_ticks: list[float] | None = None,
        force_ticklabels: list[str] | None = None,
        max_n_ticks: int | None = None,
    ) -> None:
        """Set options for colorbars in the GeoPlot.

        All arguments are optional; only those provided will be updated.

        Args:
            outer_pad (float | None): Padding between the colorbar and the plot edges
                (figure-relative). Default is None.
            inner_pad (float | None): Padding between the colorbar and the main plot area
                (figure-relative). Default is None.
            width (float | None): Width of the colorbar (figure-relative). Default is None.
            right_margin (float | None): Margin to the right of the colorbar (figure-relative).
                Default is None.
            tick_fontsize (float | None): Font size for colorbar ticks. Default is None.
            tick_pad (float | None): Padding for colorbar ticks. Default is None.
            label_fontsize (float | None): Font size for colorbar label. Default is None.
            label_rotation (float | None): Rotation angle for colorbar label. Default is None.
            label_pad (float | None): Padding for colorbar label. Default is None.
            orientation (Literal["vertical", "horizontal"] | None): Orientation of the colorbar.
                Default is None.
            extend (Literal["neither", "both", "min", "max"] | None): Extension style for the
                colorbar. Default is None.
            format (str | None): Format string for colorbar tick labels. Default is None.
            fraction (float | None): Fraction of original size for colorbar. Default is None.
            shrink (float | None): Shrink factor for colorbar. Default is None.
            aspect (float | None): Aspect ratio for colorbar. Default is None.
            force_ticks (list[float] | None): Explicit tick locations for the colorbar.
                Default is None.
            force_ticklabels (list[str] | None): Explicit tick labels for the colorbar.
                Default is None.
            max_n_ticks (int | None): Maximum number of ticks on the colorbar. Default is None.
        """
        cb_options = self._colorbar_options

        if outer_pad is not None:
            cb_options.outer_pad = float(outer_pad)
        if inner_pad is not None:
            cb_options.inner_pad = float(inner_pad)
        if width is not None:
            cb_options.width = float(width)
        if right_margin is not None:
            cb_options.right_margin = float(right_margin)

        if tick_fontsize is not None:
            cb_options.tick_fontsize = float(tick_fontsize)
        if tick_pad is not None:
            cb_options.tick_pad = float(tick_pad)

        if label_fontsize is not None:
            cb_options.label_fontsize = float(label_fontsize)
        if label_rotation is not None:
            cb_options.label_rotation = float(label_rotation)
        if label_pad is not None:
            cb_options.label_pad = float(label_pad)

        if orientation is not None:
            cb_options.orientation = orientation
        if extend is not None:
            cb_options.extend = extend
        if format is not None:
            cb_options.format = format

        if fraction is not None:
            cb_options.fraction = float(fraction)
        if shrink is not None:
            cb_options.shrink = float(shrink)
        if aspect is not None:
            cb_options.aspect = float(aspect)

        if force_ticks is not None:
            cb_options.force_ticks = list(force_ticks)
        if force_ticklabels is not None:
            cb_options.force_ticklabels = list(force_ticklabels)
        if max_n_ticks is not None:
            cb_options.max_n_ticks = int(max_n_ticks)

    def set_show_colorbars(self, show: bool = True) -> None:
        """Set whether to show colorbars when the plot is built.

        Args:
            show (bool): Whether to show colorbars. Default is True.
        """
        self.show_colorbars = bool(show)

    def set_xlim(self, left: float, right: float) -> None:
        """Set the x-axis limits to apply when the plot is built.

        Args:
            left (float): The left x-axis limit.
            right (float): The right x-axis limit.
        """
        self._xlim = (float(left), float(right))

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set the y-axis limits to apply when the plot is built.

        Args:
            bottom (float): The bottom y-axis limit.
            top (float): The top y-axis limit.
        """
        self._ylim = (float(bottom), float(top))

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
            geosource: GeoDataFrame or GeoSeries to focus on. Defaults to this plot's gdf.
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
        except Exception:
            # older shapely/geopandas combos may not have is_empty reliably; ignore
            pass

        if geoseries.empty:
            raise ValueError("focus_on(): no geometries after applying mask / dropping empties.")

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

    def _iter_layers_in_draw_order(self) -> list[_GeoLayer | _MarkerLayer]:
        """Iterate over all layers in the order they should be drawn."""

        def _sorted(layers: Sequence[_GeoLayer | _MarkerLayer]) -> list[_GeoLayer | _MarkerLayer]:
            return sorted(layers, key=lambda L: int(L.zorder))

        return (
            _sorted(self._choropleth_layers)
            + _sorted(self._districting_plan_layers)
            + _sorted(self._marker_layers)
            + _sorted(self._outline_layers)
            + _sorted(self._highlight_layers)
        )

    def _draw_colorbars(self) -> None:
        """Draw colorbars for all choropleth layers in the plot."""
        if not self._choropleth_layers:
            return

        # remove old colorbar axes in case they exist
        for cax in list(self._colorbar_axes):
            try:
                cax.remove()
            except Exception:
                pass
        self._colorbar_axes = []

        layers = sorted(self._choropleth_layers, key=lambda L: int(L.zorder))
        n_layers = len(layers)

        cb_options = self._colorbar_options
        outer_pad = float(cb_options.outer_pad)
        inner_pad = float(cb_options.inner_pad)
        width = float(cb_options.width)
        right_margin = float(cb_options.right_margin)

        # Reserve space on the right for the number of colorbars we have
        total_width = n_layers * width + (n_layers - 1) * inner_pad + outer_pad + right_margin
        right = max(0.05, 1.0 - total_width)

        self.fig.subplots_adjust(right=right)
        self.fig.canvas.draw_idle()

        main_pos = self._ax.get_position()
        x0 = float(main_pos.x1 + outer_pad)
        y0 = float(main_pos.y0)
        h = float(main_pos.height)

        for i, layer in enumerate(layers):
            xi = x0 + i * (width + inner_pad)
            rect = (float(xi), float(y0), float(width), float(h))
            cax = self.fig.add_axes(rect)
            self._colorbar_axes.append(cax)

            mappable, layer_defaults = layer._mappable()

            cbar_kwargs: dict[str, Any] = dict(layer_defaults)
            cbar_kwargs["orientation"] = cb_options.orientation
            cbar_kwargs["extend"] = cb_options.extend

            if cb_options.format is not None:
                cbar_kwargs["format"] = cb_options.format
            if cb_options.fraction is not None:
                cbar_kwargs["fraction"] = cb_options.fraction
            if cb_options.shrink is not None:
                cbar_kwargs["shrink"] = cb_options.shrink
            if cb_options.aspect is not None:
                cbar_kwargs["aspect"] = cb_options.aspect

            colorbar = self.fig.colorbar(mappable, cax=cax, **cbar_kwargs)

            if layer.datacolumn is not None:
                label_kwargs: dict[str, Any] = {}
                if cb_options.label_fontsize is not None:
                    label_kwargs["fontsize"] = cb_options.label_fontsize
                if cb_options.label_rotation is not None:
                    label_kwargs["rotation"] = cb_options.label_rotation
                if cb_options.label_pad is not None:
                    label_kwargs["labelpad"] = cb_options.label_pad
                colorbar.set_label(layer.datacolumn, **label_kwargs)

            cax.tick_params(labelsize=cb_options.tick_fontsize, pad=cb_options.tick_pad)

            if cb_options.force_ticks is not None:
                colorbar.set_ticks(cb_options.force_ticks)
            if cb_options.force_ticklabels is not None:
                colorbar.set_ticklabels(cb_options.force_ticklabels)

            if cb_options.max_n_ticks is not None and cb_options.force_ticks is None:
                try:
                    ticks = list(colorbar.get_ticks())
                    if len(ticks) > cb_options.max_n_ticks:
                        step = max(1, len(ticks) // cb_options.max_n_ticks)
                        colorbar.set_ticks(ticks[::step])
                except Exception:
                    pass

    def _apply_limits(self) -> None:
        """Apply stored x/y limits to the axes."""
        if self._xlim is not None:
            self._ax.set_xlim(*self._xlim)
        if self._ylim is not None:
            self._ax.set_ylim(*self._ylim)

    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and applying settings."""
        self._ax.clear()

        if not self.show_axis:
            self._ax.set_axis_off()

        for layer in self._iter_layers_in_draw_order():
            layer.render(self._ax, target_crs=self.target_crs)

        if self.show_colorbars:
            self._draw_colorbars()

    def _build_and_apply_settings(self) -> None:
        """Build the plot and apply stored settings like limits."""
        self._build_plot()
        self._apply_limits()

    @property
    def ax(self) -> Axes:
        """The Matplotlib Axes object for the plot."""
        self._build_and_apply_settings()
        return self._ax

    def show(self) -> None:
        """Display the plot inline (e.g., in a Jupyter notebook) or in a window."""
        self._build_and_apply_settings()

        try:
            from IPython.display import Image, display

            # Render to PNG in memory and display inline. We have to do this because we are
            # building the Figure directly.
            buf = BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.fig.dpi)
            buf.seek(0)
            display(Image(data=buf.getvalue()))
        except Exception:
            self.fig.show()

        self._build_and_apply_settings()

    def save(self, filepath: str, **kwargs: Any) -> None:
        """Save the plot to a file."""
        self._build_and_apply_settings()
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("dpi", self.fig.dpi)
        self.fig.savefig(filepath, **kwargs)
