"""`_GeoLayer` ABC — the contract every geometry layer satisfies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.axes import Axes

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.typing import (
    Color,
    CRSLike,
    GeoColorMap,
    GeoSource,
    MplCompatibleColor,
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
