"""`_ContinuousColorLayer` — continuous color mapping over a data column."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize, to_hex
from matplotlib.pyplot import get_cmap
from numpy import linspace

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.geometry._layers._base import _GeoLayer
from gerrytools.typing import CategoryKey, CRSLike, MplKwargs, ResolvedColor


@runtime_checkable
class ColormapLayer(Protocol):
    """Public protocol for layers that can produce a standalone colorbar.

    Any object implementing ``datacolumn`` and ``mappable()`` satisfies this
    protocol and can be passed to ``ColoredGeoPlot.save_colorbar()``.
    """

    @property
    def datacolumn(self) -> str | None: ...

    def mappable(self) -> tuple[ScalarMappable, MplKwargs]: ...


@dataclass(frozen=True, slots=True)
class _ContinuousColorLayer(_GeoLayer):
    """A geographic layer with continuous color mapping based on a data column.

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

    def _mappable(self) -> tuple[ScalarMappable, MplKwargs]:
        """Get a ScalarMappable and the colorbar kwargs for this layer.

        Returns:
            tuple[ScalarMappable, MplKwargs]: The ScalarMappable and colorbar kwargs.
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

            cbar_kwargs: MplKwargs = {
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
            cbar_kwargs: MplKwargs = {}

        return m, cbar_kwargs

    def mappable(self) -> tuple[ScalarMappable, MplKwargs]:
        """Public wrapper around ``_mappable()`` for use via ``ColormapLayer``."""
        return self._mappable()

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        data_series = self._data_series()
        lower_bound, upper_bound = self._effective_bounds(data_series)
        missing_color = resolve_color_and_alpha(self.missing_color, alpha=self.facealpha)

        colors: dict[CategoryKey, ResolvedColor] = {}

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower_bound, upper_bound)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            interval_to_hex = {
                interval: interval_hex_colors[i] for i, interval in enumerate(boundaries)
            }

            for idx, value in data_series.items():
                if pd.isna(value):
                    colors[idx] = missing_color
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

                colors[idx] = resolve_color_and_alpha(
                    interval_to_hex[boundaries[interval_i]],
                    alpha=self.facealpha,
                )

        else:
            norm = Normalize(vmin=lower_bound, vmax=upper_bound)

            cmap = get_cmap(self.colormap)

            for idx, value in data_series.items():
                if pd.isna(value):
                    color = missing_color
                else:
                    normalized_value = norm(value)
                    color = resolve_color_and_alpha(
                        to_hex(cmap(normalized_value), keep_alpha=True),
                        alpha=self.facealpha,
                    )
                colors[idx] = color

        return pd.Series(colors).reindex(self.geometries.index)

    def render(
        self,
        ax: Axes,
        *,
        target_crs: CRSLike | None = None,
        **kwargs: object,
    ) -> list:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.
            **kwargs (object): Additional keyword arguments (not used).

        Returns:
            list[Artist]: The matplotlib artists added to ``ax`` by this layer.
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
        from gerrytools.plotting.geometry._layers._base import _capture_geopandas_artists

        return _capture_geopandas_artists(
            ax,
            plot_call=lambda: geoseries.plot(
                ax=ax,
                color=self.color_series,
                edgecolor=edge_color_tup,
                linewidth=self.edgewidth,
                zorder=self.zorder,
            ),
        )
