"""`_CategoricalColorLayer` — categorical color mapping over a data column."""

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_hex
from matplotlib.pyplot import get_cmap

from gerrytools.colors import districtr, resolve_color_and_alpha
from gerrytools.plotting.geometry._layers._base import _GeoLayer
from gerrytools.typing import (
    CategoryColorMap,
    CategoryKey,
    Color,
    CRSLike,
    GeoColorMap,
    ResolvedColor,
)


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
        color_list: Sequence[Color],
    ) -> CategoryColorMap:
        """Map unique values in the data to colors from the provided list. Filters out NaN values.

        Args:
            unique_values (pd.Index): The unique values to map.
            color_list (Sequence[Color]): The colors to use for mapping.
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
    ) -> list:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.
            **kwargs (object): Additional keyword arguments (not used but included to satisfy
                render function signature contract).

        Returns:
            list[Artist]: The matplotlib artists added to ``ax`` by this layer.
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
