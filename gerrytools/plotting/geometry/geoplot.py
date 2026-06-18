from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from shapely.geometry import Point

from gerrytools.plotting._figure_io import save_figure
from gerrytools.plotting.geometry._layers import (
    ColormapLayer,
    _CategoricalColorLayer,
    _ContinuousColorLayer,
    _GeoLayer,
    _MarkerLayer,
)
from gerrytools.plotting.geometry.geoplotbase import GeoPlotBase, _LabelRequest
from gerrytools.plotting.mpl.geoplot_options import (
    ColorbarOptions,
    _ColorbarLayoutOptions,
)
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.typing import (
    CategoryKey,
    Color,
    CRSLike,
    GeoColorMap,
    MplCompatibleColor,
)

# Re-export `_ContinuousColorLayer` so existing imports like
#   `from gerrytools.plotting.geometry.geoplot import _ContinuousColorLayer`
# keep working without forcing test changes.
__all__ = [
    "GeoPlot",
    "_ContinuousColorLayer",
]


@dataclass(frozen=True, slots=True)
class _ColorbarRequest:
    layer: _ContinuousColorLayer
    label: str | None = None  # override label shown on the bar
    zorder: int = 0  # used only for ordering colorbars
    options: ColorbarOptions | None = None  # optional per-bar overrides (optional feature)


class GeoPlot(GeoPlotBase):
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
                ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None): Render onto an existing matplotlib
                ``Axes`` instead of creating a fresh figure. Defaults to None.
            show_axis (bool): Whether to show axis ticks and labels. Default is False.
            target_crs (CRSLike | None): The target CRS for reprojecting geometries.
                If None, uses the CRS of `gdf`. Default is None.
            default_outline (bool): Whether to include a default outline layer around
                the geometries in `gdf`. Default is True.
            silent (bool): Whether to suppress informational output throughout the rendering
                process. Default is False.
        """
        super().__init__(
            gdf,
            dpi=dpi,
            ax=ax,
            show_axis=show_axis,
            target_crs=target_crs,
            default_outline=default_outline,
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
        geosource: GeoDataFrame | None = None,
        datacolumn: str | None = None,
        *,
        colormap: str | Colormap = "Purples",
        missing_color: MplCompatibleColor | None = "lightgrey",
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
    ) -> ColormapLayer:
        """Add a choropleth layer to the GeoPlotBase.

        Args:
            geosource (GeoDataFrame | None): The GeoDataFrame source for the layer.
                If None, uses the base gdf of the GeoPlotBase. Default is None.
            datacolumn (str): The data column to use for color mapping.
            colormap (str | Colormap): The colormap to use for color mapping. Default is "Purples".
            missing_color (MplCompatibleColor | None): Color to use for missing data.
                Default is "lightgrey".
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

        Returns:
            ColormapLayer: The layer object, which can be passed to ``save_colorbar()``.
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

        return layer

    def add_districting_plan_layer(
        self,
        geosource: GeoDataFrame | None = None,
        plancolumn: str | None = None,
        *,
        dissolve: bool = False,
        show_labels: bool = False,
        exclude_labels: Sequence[CategoryKey] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        colormap: GeoColorMap | None = "districtr",
        missing_color: MplCompatibleColor | None = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        zorder: int = 2,
    ) -> None:
        """Add a districting plan layer to the GeoPlotBase.

        Args:
            geosource (GeoDataFrame | None): The GeoDataFrame source for the layer.
                If None, uses the base gdf of the GeoPlotBase. Default is None.
            plancolumn (str): The column containing district identifiers.
            dissolve (bool): Whether to dissolve geometries by district. Default is False.
            show_labels (bool): Whether to show district labels. Default is False.
            exclude_labels (Sequence[CategoryKey] | None): District labels to exclude from
                labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfont_options (LabelFontOptions | None): Font options for district labels.
                If None, uses default settings. Default is None.
            labelbox_options (LabelBoxOptions | None): Optional label box styling.
                If None, label boxes are disabled. Defaults to None.
            colormap (GeoColorMap | None): Color mapping specification.
                Can be a single color, a named colormap, a Colormap object, or a mapping from
                district identifiers to colors. Default is "districtr".
            missing_color (MplCompatibleColor | None): Color to use for missing data.
                Default is "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            zorder (int): Z-order for rendering. Default is 2.
        """
        if plancolumn is None:
            raise ValueError("'plancolumn' must be provided for a districting plan layer")

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

            def coerce_labels(x: CategoryKey) -> str:
                """Normalize a district label value to a string.

                Args:
                    x (CategoryKey): Raw label value.

                Returns:
                    str: Normalized label text.
                """
                try:
                    return str(int(str(x)))
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
                    label_format_fn=lambda x: str(int(str(x))),
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
        """Set the spacing between colorbars in GeoPlotBase.

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
            """Sort layers by integer z-order.

            Args:
                layers (Sequence[_GeoLayer | _MarkerLayer]): Layers to sort.

            Returns:
                list[_GeoLayer | _MarkerLayer]: Layers sorted ascending by ``zorder``.
            """
            return sorted(layers, key=lambda L: int(L.zorder))

        return (
            _sorted(self._choropleth_layers)
            + _sorted(self._districting_plan_layers)
            + _sorted(self._marker_layers)
            + _sorted(self._outline_layers)
            + _sorted(self._highlight_layers)
        )

    def _clear_colorbars_and_reset_layout(self) -> None:
        """Clear any existing colorbars and reset layout to default.

        ``subplots_adjust`` is only safe to call on a figure gerrytools
        owns. When the user supplied their own ``ax=``, we share their
        figure and must not mutate its global layout.
        """
        for cax in list(self._colorbar_axes):
            cax.remove()
        self._colorbar_axes = []

        # reset layout so we don't keep a shrunken main axes
        if not self._figure_is_shared:
            self.fig.subplots_adjust(right=0.98)
            self.fig.canvas.draw_idle()

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

        if not self._figure_is_shared:
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
            colorbar = self.fig.colorbar(
                mappable=mappable,
                cax=cb_ax,
                orientation=cb_options.orientation,
                extend=cb_options.extend,
                format=cb_options.format,
                fraction=cb_options.fraction,
                shrink=cb_options.shrink,
                aspect=cb_options.aspect,
            )

            layer_ticks = layer_defaults.get("ticks")
            if isinstance(layer_ticks, list):
                colorbar.set_ticks(layer_ticks)

            # label: request override > datacolumn > none
            label_text = cb_request.label if cb_request.label is not None else layer.datacolumn
            if label_text is not None:
                if (
                    cb_options.label_fontsize is None
                    and cb_options.label_rotation is None
                    and cb_options.label_pad is None
                ):
                    colorbar.set_label(str(label_text))
                else:
                    colorbar.set_label(
                        str(label_text),
                        fontsize=cb_options.label_fontsize,
                        rotation=cb_options.label_rotation,
                        labelpad=cb_options.label_pad,
                    )

            cb_ax.tick_params(labelsize=cb_options.tick_fontsize, pad=cb_options.tick_pad)

            if cb_options.force_ticks is not None:
                colorbar.set_ticks(cb_options.force_ticks)
            if cb_options.force_ticklabels is not None:
                colorbar.set_ticklabels(cb_options.force_ticklabels)

            if cb_options.max_n_ticks is not None and cb_options.force_ticks is None:
                ticks = list(colorbar.get_ticks())
                if len(ticks) > cb_options.max_n_ticks:
                    step = max(1, len(ticks) // cb_options.max_n_ticks)
                    colorbar.set_ticks(ticks[::step])

    def _build_plot(self) -> None:
        """Render all layers and colorbars, tracking every artist created."""
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
            layer_artists = layer.render(self._ax, target_crs=self.target_crs)
            if layer_artists:
                self._artists.track(layer_artists)

        self._draw_colorbars()

    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Snapshot → remove gerrytools artists → rebuild → apply settings."""
        from gerrytools.plotting._axes_state import _ManagedAxesState  # noqa: F401

        before = self._axes_state.snapshot(self._ax)
        external = self._axes_state.detect_external_changes(before)
        self._artists.remove_all()
        self._build_plot()
        self._axes_state.restore_autoscale_protected(self._ax, before, external)
        self._apply_axis_visibility(external)
        self._apply_limits(external)
        label_points = self._draw_deferred_labels()
        return label_points

    def save_colorbar(
        self,
        layer: ColormapLayer,
        filepath: str,
        *,
        figsize: tuple[float, float] = (1.0, 4.0),
        orientation: str = "vertical",
        **kwargs: object,
    ) -> None:
        """Save a standalone colorbar image for a choropleth layer.

        Args:
            layer (ColormapLayer): The layer whose colorbar to save. This is the value
                returned by ``add_choropleth_layer()``.
            filepath (str): Output file path.
            figsize (tuple[float, float]): Figure size (width, height) in inches.
                Defaults to (1.0, 4.0) for a vertical bar.
            orientation (str): ``"vertical"`` or ``"horizontal"``. Defaults to ``"vertical"``.
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
        """
        mappable, layer_defaults = layer.mappable()

        cb_fig, cb_ax = plt.subplots(figsize=figsize, dpi=self.fig.dpi)
        colorbar = cb_fig.colorbar(mappable=mappable, cax=cb_ax, orientation=orientation)

        layer_ticks = layer_defaults.get("ticks")
        if isinstance(layer_ticks, list):
            colorbar.set_ticks(cast("Sequence[float]", layer_ticks))

        if layer.datacolumn is not None:
            colorbar.set_label(str(layer.datacolumn))

        try:
            save_figure(cb_fig, filepath, **kwargs)
        finally:
            plt.close(cb_fig)
