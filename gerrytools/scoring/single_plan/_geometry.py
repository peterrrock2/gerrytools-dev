"""Compactness and fixed-region functions for one plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import networkx as nx
from geopandas import GeoDataFrame
from gerrychain import Partition
from pandas import DataFrame, Series
from shapely.geometry.base import BaseGeometry

from ..evaluator import Assignment
from ..metrics import (
    ConvexHullRatio,
    CutEdges,
    PolsbyPopper,
    PopulationPolygon,
    RegionParts,
    RegionPieces,
    RegionSplits,
    Reock,
    Schwartzberg,
    StateClippedConvexHullRatio,
    TallyByRegion,
)
from ._base import GeoAssignment, SinglePlanSource, _columns, _evaluate


def _resolve_compactness_source(
    source: SinglePlanSource,
    assignment_or_geometry: GeoAssignment | GeoDataFrame | None,
    geometry: GeoDataFrame | None,
) -> tuple[GeoAssignment | None, GeoDataFrame | None]:
    """Resolve the overloaded compactness assignment/geometry argument."""
    if isinstance(source, Partition):
        if assignment_or_geometry is not None:
            if not isinstance(assignment_or_geometry, GeoDataFrame):
                raise TypeError("the second Partition argument must be a GeoDataFrame")
            if geometry is not None:
                raise TypeError("supply Partition geometry positionally or by keyword, not both")
            geometry = assignment_or_geometry
        return None, geometry

    if isinstance(assignment_or_geometry, GeoDataFrame):
        if isinstance(source, nx.Graph):
            raise TypeError("a graph source requires district labels as its second argument")
        raise TypeError("a GeoDataFrame source requires district labels as its second argument")

    return assignment_or_geometry, geometry


def cut_edges(
    source: Partition | nx.Graph,
    assignment: Assignment | None = None,
    *,
    weight: str | None = None,
) -> int | float:
    """Count or weight cut edges in one partition or graph assignment."""
    return cast(
        "int | float",
        _evaluate(source, assignment, CutEdges(weight), topology_required=True),
    )


def polsby_popper(
    source: SinglePlanSource,
    assignment_or_geometry: GeoAssignment | GeoDataFrame | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
    area: str | None = None,
    perimeter: str | None = None,
    boundary_perimeter: str | None = None,
    shared_perimeter: str | None = None,
) -> Series:
    """Calculate Polsby-Popper from partition graph measurements or geometry.

    ``polsby_popper(partition)`` uses graph measurements.
    ``polsby_popper(graph, assignment)`` uses graph measurements.
    ``polsby_popper(partition, geometry)`` and ``polsby_popper(frame, assignment)`` use geometry.
    """
    assignment, geometry = _resolve_compactness_source(source, assignment_or_geometry, geometry)

    metric = PolsbyPopper(
        area=area,
        perimeter=perimeter,
        boundary_perimeter=boundary_perimeter,
        shared_perimeter=shared_perimeter,
    )
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            metric,
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        ),
    )


def schwartzberg(
    source: SinglePlanSource,
    assignment_or_geometry: GeoAssignment | GeoDataFrame | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
    area: str | None = None,
    perimeter: str | None = None,
    boundary_perimeter: str | None = None,
    shared_perimeter: str | None = None,
) -> Series:
    """Calculate Schwartzberg from partition graph measurements or geometry."""
    assignment, geometry = _resolve_compactness_source(source, assignment_or_geometry, geometry)

    metric = Schwartzberg(
        area=area,
        perimeter=perimeter,
        boundary_perimeter=boundary_perimeter,
        shared_perimeter=shared_perimeter,
    )
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            metric,
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        ),
    )


def reock(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
) -> Series:
    """Calculate Reock compactness for one plan using geometry."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            Reock(),
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        ),
    )


def convex_hull_ratio(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
) -> Series:
    """Calculate convex-hull compactness for one plan using geometry."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            ConvexHullRatio(),
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        ),
    )


def state_clipped_convex_hull_ratio(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    state_geometry: BaseGeometry,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
) -> Series:
    """Calculate state-clipped convex-hull compactness for one plan."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            StateClippedConvexHullRatio(state_geometry),
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        ),
    )


def population_polygon(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population_col: str,
    alternative_pop_gdf: GeoDataFrame | None = None,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
) -> Series:
    """Calculate population-polygon compactness for one plan."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            PopulationPolygon(population_col, alternative_pop_gdf=alternative_pop_gdf),
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        ),
    )


def region_splits(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    regions: str | Iterable[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int | Series:
    """Count split fixed regions for one plan."""
    return cast(
        "int | Series",
        _evaluate(
            source,
            assignment,
            RegionSplits(*_columns(regions)),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def region_pieces(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    regions: str | Iterable[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int | Series:
    """Count occupied fixed-region and district pairs for one plan."""
    return cast(
        "int | Series",
        _evaluate(
            source,
            assignment,
            RegionPieces(*_columns(regions)),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def region_parts(
    source: Partition | nx.Graph,
    assignment: Assignment | None = None,
    *,
    regions: str | Iterable[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int | Series:
    """Count connected fixed-region and district parts in one partition or graph assignment."""
    return cast(
        "int | Series",
        _evaluate(
            source,
            assignment,
            RegionParts(*_columns(regions)),
            geometry=geometry,
            node_column=node_column,
            topology_required=True,
        ),
    )


def tally_by_region(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    region: str,
    columns: str | Iterable[str] | Mapping[str, str] | None = None,
    include_count: bool = False,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> DataFrame:
    """Sum named unit columns by fixed region and proposed district for one plan."""
    return cast(
        DataFrame,
        _evaluate(
            source,
            assignment,
            TallyByRegion(region, columns, include_count=include_count),
            geometry=geometry,
            node_column=node_column,
        ),
    )
