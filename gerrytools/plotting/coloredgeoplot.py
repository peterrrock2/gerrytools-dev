from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize, to_hex
from matplotlib.pyplot import get_cmap
from numpy import linspace
from shapely.geometry import Point

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting._geoplot_option_classes import (
    ColorbarOptions,
    _ColorbarLayoutOptions,
)
from gerrytools.plotting._gerryplot_option_classes import (
    LabelBoxOptions,
    LabelFontOptions,
)
from gerrytools.plotting.geoplot import (
    GeoPlot,
    _CategoricalColorLayer,
    _GeoLayer,
    _LabelRequest,
    _MarkerLayer,
)
from gerrytools.typing import Color


@dataclass(frozen=True, slots=True)
class _ContinuousColorLayer(_GeoLayer):
    """A geographic layer with continuous color mapping based on a data column.

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
        colormap (str | Colormap): The colormap to use for continuous color mapping.
        vmin (float | None): Lower bound value for color mapping.
        vmax (float | None): Upper bound value for color mapping.
        bins (int | list[float] | None): Optional binning specification for discrete intervals.
    """

    colormap: str | Colormap = "Purples"
    vmin: float | None = None
    vmax: float | None = None
    bins: int | list[float] | None = None

    def __post_init__(self) -> None:
        super(_ContinuousColorLayer, self).__post_init__()
        if not isinstance(self.geometry_source, GeoDataFrame):
            raise TypeError(
                "Tried to create a continuous color layer using geosource of type "
                f"{type(self.geometry_source).__name__!r}; geosource must be a GeoDataFrame",
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
        return self.geometry_source[self.datacolumn]

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
        colors = []
        for i in range(len(boundaries)):
            rgba = cmap(i)
            if self.facealpha is not None:
                rgba = (rgba[0], rgba[1], rgba[2], float(self.facealpha))
            colors.append(to_hex(rgba, keep_alpha=True))
        edges = boundaries.left.tolist() + [boundaries.right[-1]]
        return edges, colors

    @staticmethod
    def _with_alpha(cmap: Colormap, alpha: float) -> Colormap:
        """Return a copy of the given colormap with the specified alpha applied.

        Args:
            cmap (Colormap): The original colormap.
            alpha (float): The alpha value to apply (0.0 to 1.0).

        Returns:
            Colormap: A new colormap with the specified alpha applied.
        """
        n = getattr(cmap, "N", 256)
        rgba = cmap(linspace(0, 1, n))
        rgba[:, 3] = float(alpha)
        return ListedColormap(rgba, name=f"{cmap.name}_a{alpha:g}")

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
        else:
            norm = Normalize(vmin=lower, vmax=upper)

            cmap = get_cmap(self.colormap)
            if self.facealpha is not None:
                cmap = self._with_alpha(cmap, self.facealpha)
            m = ScalarMappable(norm=norm, cmap=cmap)
            m.set_array([])
            cbar_kwargs = {}

        return m, cbar_kwargs

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
                        if value < boundaries.left[0]:
                            interval_i = 0
                        elif value > boundaries.right[-1]:
                            interval_i = len(boundaries) - 1
                        else:
                            interval_i = -1

                if interval_i == -1:
                    colors[idx] = self.missing_color
                else:
                    colors[idx] = resolve_color_and_alpha(
                        interval_to_hex[boundaries[interval_i]],
                        alpha=self.facealpha,
                    )

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

        if self.datacolumn not in self.geometry_source.columns:
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
class _ColorbarRequest:
    layer: _ContinuousColorLayer
    label: str | None = None  # override label shown on the bar
    zorder: int = 0  # used only for ordering colorbars
    options: ColorbarOptions | None = None  # optional per-bar overrides (optional feature)


class ColoredGeoPlot(GeoPlot):
    """A class for creating geographic plots with multiple colored layers.

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
        target_crs=None,
        include_default_outline: bool = True,
        silent: bool = False,
    ) -> None:
        """Initialize a ColoredGeoPlot.

        Args:
            gdf (GeoDataFrame): The base GeoDataFrame for the plot.
            dpi (int): The DPI for the Matplotlib Figure. Default is 300.
            show_axis (bool): Whether to show axis ticks and labels. Default is False.
            target_crs: The target CRS for reprojecting geometries. If None, uses the CRS of
                `gdf`. Default is None.
            include_default_outline (bool): Whether to include a default outline layer around
                the geometries in `gdf`. Default is True.
            silent (bool): Whether to suppress informational output throughout the rendering
                process. Default is False.
        """
        super().__init__(
            gdf,
            dpi=dpi,
            show_axis=show_axis,
            target_crs=target_crs,
            include_default_outline=include_default_outline,
            silent=silent,
        )

        self._colorbar_layout_options: _ColorbarLayoutOptions = _ColorbarLayoutOptions()
        self._colorbar_requests: list[_ColorbarRequest] = []

        self._choropleth_layers: list[_ContinuousColorLayer] = []
        self._districting_plan_layers: list[_CategoricalColorLayer] = []

        self._label_requests: list[_LabelRequest] = []

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
        bins: int | list[float] | None = None,
        show_colorbar: bool = False,
        colorbar_label: str | None = None,
        colorbar_options: ColorbarOptions | None = None,
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
            vmin (float | None): Lower bound for color mapping range. Default is None which then
                uses the minimum value in the data.
            vmax (float | None): Upper bound for color mapping range. Default is None which then
                uses the maximum value in the data.
            show_colorbar (bool): Whether to show a colorbar for this layer. Default is False.
            colorbar_label (str | None): Label for the colorbar. Default is None which then
                uses the datacolumn name.
            colorbar_options (ColorbarOptions | None): Options for customizing the colorbar.
                Default is None.
            bins (int | list[float] | None): Optional binning specification for discrete intervals.
                Default is None.
            zorder (int): Z-order for rendering. Default is 0.
        """
        if geosource is None:
            geosource = self.gdf
        layer = _ContinuousColorLayer(
            geometry_source=geosource,
            datacolumn=datacolumn,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            vmin=vmin,
            vmax=vmax,
            bins=bins,
            zorder=zorder,
        )
        self._choropleth_layers.append(layer)

        if show_colorbar:
            self._colorbar_requests.append(
                _ColorbarRequest(
                    layer=layer,
                    label=colorbar_label,
                    zorder=zorder,
                    options=colorbar_options,
                )
            )

    def add_districting_plan_layer(
        self,
        *,
        geosource: GeoDataFrame | None = None,
        plancolumn: str,
        dissolve: bool = False,
        show_labels: bool = False,
        exclude_labels: list[Any] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
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
            exclude_labels (list[Any] | None): List of district labels to exclude from labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfontoptions (LabelFontOptions | None): Font options for district labels.
                If None, uses default settings. Default is None.
            colormap (str | Colormap | dict[Any, Color] | pd.Series): Color mapping specification.
                Can be a single color, a named colormap, a Colormap object, or a mapping from
                district identifiers to colors. Default is "districtr".
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
            geometry_source=plan_gdf,
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
            dissolved_plan_gdf = plan_gdf.dissolve(by=plancolumn).reset_index()

            def coerce_labels(x: Any) -> str:
                try:
                    return str(int(x))
                except Exception:
                    return str(x)

            dissolved_plan_gdf[plancolumn] = dissolved_plan_gdf[plancolumn].apply(coerce_labels)
            new_exclude_labels = (
                list(map(coerce_labels, exclude_labels)) if exclude_labels is not None else []
            )
            dissolved_plan_gdf = GeoDataFrame(
                dissolved_plan_gdf.query(f"`{plancolumn}` not in {new_exclude_labels}")
            )

            self._label_requests.append(
                _LabelRequest(
                    gdf=dissolved_plan_gdf,
                    label_column=plancolumn,
                    labelfont_options=labelfont_options,
                    labelbox_options=labelbox_options,
                    label_format_fn=lambda x: str(int(x)),
                    zorder=zorder + 1,
                )
            )

    def set_colorbar_layout(
        self,
        *,
        outer_pad: float | None = None,
        inner_pad: float | None = None,
        width: float | None = None,
        right_margin: float | None = None,
    ) -> None:
        """Set the spacing between colorbars in GeoPlot.

        All arguments are optional; only those provided will be updated.

        Args:
            outer_pad (float | None): Padding between the colorbar and the plot edges
                (figure-relative). Default is None.
            inner_pad (float | None): Padding between the colorbar and the main plot area
                (figure-relative). Default is None.
            width (float | None): Width of the colorbar (figure-relative). Default is None.
            right_margin (float | None): Margin to the right of the colorbar (figure-relative).
                Default is None.
        """
        cb_options = self._colorbar_layout_options

        if outer_pad is not None:
            cb_options.outer_pad = float(outer_pad)
        if inner_pad is not None:
            cb_options.inner_pad = float(inner_pad)
        if width is not None:
            cb_options.width = float(width)
        if right_margin is not None:
            cb_options.right_margin = float(right_margin)

    def _list_layers_in_draw_order(self) -> list[_GeoLayer | _MarkerLayer]:
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

    def _clear_colorbars_and_reset_layout(self) -> None:
        """Clear any existing colorbars and reset layout to default."""
        for cax in list(self._colorbar_axes):
            try:
                cax.remove()
            except Exception:
                pass
        self._colorbar_axes = []

        # reset layout so we don't keep a shrunken main axes
        self.fig.subplots_adjust(right=0.98)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _draw_colorbars(self) -> None:
        """Draw colorbars for all requested layers."""
        self._clear_colorbars_and_reset_layout()

        if not self._colorbar_requests:
            return

        # Sort requests by their requested zorder (usually layer.zorder)
        sorted_cb_requests = sorted(self._colorbar_requests, key=lambda r: int(r.zorder))
        n_layers = len(sorted_cb_requests)

        cb_global_options = self._colorbar_layout_options
        outer_pad = float(cb_global_options.outer_pad)
        inner_pad = float(cb_global_options.inner_pad)
        width = float(cb_global_options.width)
        right_margin = float(cb_global_options.right_margin)

        total_width = n_layers * width + (n_layers - 1) * inner_pad + outer_pad + right_margin
        right = max(0.05, 1.0 - total_width)

        self.fig.subplots_adjust(right=right)
        self.fig.canvas.draw_idle()

        main_pos = self._ax.get_position()
        x0 = float(main_pos.x1 + outer_pad)
        y0 = float(main_pos.y0)
        h = float(main_pos.height)

        for i, cb_request in enumerate(sorted_cb_requests):
            layer = cb_request.layer
            cb_options = cb_request.options if cb_request.options is not None else ColorbarOptions()

            xi = x0 + i * (width + inner_pad)
            rect = (float(xi), float(y0), float(width), float(h))
            cb_ax = self.fig.add_axes(rect)
            self._colorbar_axes.append(cb_ax)

            mappable, layer_defaults = layer._mappable()

            cb_kwargs: dict[str, Any] = dict(layer_defaults)
            cb_kwargs["orientation"] = cb_options.orientation
            cb_kwargs["extend"] = cb_options.extend

            if cb_options.format is not None:
                cb_kwargs["format"] = cb_options.format
            if cb_options.fraction is not None:
                cb_kwargs["fraction"] = cb_options.fraction
            if cb_options.shrink is not None:
                cb_kwargs["shrink"] = cb_options.shrink
            if cb_options.aspect is not None:
                cb_kwargs["aspect"] = cb_options.aspect

            colorbar = self.fig.colorbar(mappable, cax=cb_ax, **cb_kwargs)

            # label: request override > datacolumn > none
            label_text = cb_request.label if cb_request.label is not None else layer.datacolumn
            if label_text is not None:
                label_kwargs: dict[str, Any] = {}
                if cb_options.label_fontsize is not None:
                    label_kwargs["fontsize"] = cb_options.label_fontsize
                if cb_options.label_rotation is not None:
                    label_kwargs["rotation"] = cb_options.label_rotation
                if cb_options.label_pad is not None:
                    label_kwargs["labelpad"] = cb_options.label_pad
                colorbar.set_label(str(label_text), **label_kwargs)

            cb_ax.tick_params(labelsize=cb_options.tick_fontsize, pad=cb_options.tick_pad)

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

    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and applying settings."""
        self._ax.clear()

        if not self.show_axis:
            self._ax.set_axis_off()

        start_idx_to_layer_type: dict[int, tuple[str, int]] = {}
        if not self.silent:

            layer_order = [
                ("choropleth", len(self._choropleth_layers)),
                ("districting plan", len(self._districting_plan_layers)),
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

        self._draw_colorbars()

    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Build the plot and apply stored settings like limits."""
        self._build_plot()
        self._apply_limits()
        label_points = self._draw_deferred_labels()
        return label_points
