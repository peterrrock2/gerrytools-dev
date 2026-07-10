"""Compactness and fixed-region metric descriptions."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias, cast

from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from ..result import _Dtype
from ._base import (
    _KeyedMetric,
    _merged_keys,
    _MetricBase,
    _OutputSpec,
    _ResourceSpec,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from gerrytools._scoring_engine import ScoringEngine

    from ..evaluator import PlanEvaluator

_GeometryAdder: TypeAlias = Callable[[Sequence[bytes], list[tuple[int, int]]], None]
_GraphAdder: TypeAlias = Callable[
    [list[float], list[float], list[tuple[int, int]], list[float]], None
]


@dataclass(frozen=True, slots=True)
class ConvexHullRatio(_MetricBase):
    r"""Score district area divided by the area of its convex hull.

    For district geometry :math:`D`, the score is

    .. math::
        \operatorname{ConvexHullRatio}(D)
        = \frac{\operatorname{area}(D)}{\operatorname{area}(H(D))},

    where :math:`H(D)` is the smallest convex set containing :math:`D`. Values lie in
    :math:`(0, 1]`, with one indicating a convex district. Geometry must use a projected CRS
    appropriate for area calculations.

    This is the standard, unclipped statistic. See Moon Duchin,
    `Political Geometry
    <https://data-democracy.org/publications/political-geometry/01-Duchin.pdf>`_, and Duchin and
    Tenner, `Discrete Geometry for Electoral Geography
    <https://doi.org/10.1016/j.polgeo.2023.103040>`_.
    """

    _kind: ClassVar[str] = "convex_hull_ratio"

    def _validate(self, evaluator: PlanEvaluator) -> None:
        evaluator._require_geometry("ConvexHullRatio")

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        del evaluator
        return _ResourceSpec(geometry=True)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        backend.add_convex_hull_ratio(evaluator._require_geometry("ConvexHullRatio").wkb)
        return _OutputSpec("district", ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {"source": "geometry"}


@dataclass(frozen=True, slots=True)
class StateClippedConvexHullRatio(_MetricBase):
    r"""Score district area divided by its state-clipped convex-hull area.

    For district geometry :math:`D` and explicit state geometry :math:`S`, the score is

    .. math::
        \operatorname{StateClippedConvexHullRatio}(D; S)
        = \frac{\operatorname{area}(D)}
        {\operatorname{area}\left(H(D) \cap S\right)}.

    Clipping removes the ordinary convex hull's area outside the state, so this score is at least
    the standard :class:`ConvexHullRatio`. Values lie in :math:`(0, 1]`. Unit and state geometry
    must use the same projected CRS appropriate for area calculations, and the state must cover
    every unit geometry.

    This boundary-adjusted variant follows the clipped convex-hull construction discussed by Moon
    Duchin in `Political Geometry
    <https://data-democracy.org/publications/political-geometry/01-Duchin.pdf>`_. See also Duchin
    and Tenner, `Discrete Geometry for Electoral Geography
    <https://doi.org/10.1016/j.polgeo.2023.103040>`_.

    Args:
        state_geometry: Nonempty, valid Polygon or MultiPolygon covering every scoring unit.
    """

    _kind: ClassVar[str] = "state_clipped_convex_hull_ratio"
    state_geometry: BaseGeometry

    def __post_init__(self) -> None:
        state = self.state_geometry
        if not isinstance(state, BaseGeometry):
            raise TypeError("state_geometry must be a Shapely Polygon or MultiPolygon")
        if state.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("state_geometry must be a Polygon or MultiPolygon")
        if state.is_empty or not state.is_valid or state.area <= 0:
            raise ValueError("state_geometry must be nonempty, valid, and have positive area")

    def _validate(self, evaluator: PlanEvaluator) -> None:
        evaluator._require_geometry("StateClippedConvexHullRatio")

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        del evaluator
        return _ResourceSpec(geometry=True)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        backend.add_state_clipped_convex_hull_ratio(
            evaluator._require_geometry("StateClippedConvexHullRatio").wkb,
            bytes(self.state_geometry.wkb),
        )
        return _OutputSpec("district", ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {"source": "geometry", "state_geometry": "explicit"}


@dataclass(frozen=True, slots=True)
class PopulationPolygon(_MetricBase):
    r"""Score district population against full-weight polygons intersecting its convex hull.

    By default, the evaluator's aligned GeoDataFrame supplies both population geometry and the
    ``population_col`` values. Each row already corresponds to one graph node. Supplying
    ``alternative_pop_gdf`` instead uses its finer population polygons and infers the unique scorer
    geometry containing each one.

    For district :math:`D`, population polygons :math:`G_i`, weights :math:`w_i`, their uniquely
    containing graph nodes :math:`c(i)`, and assignment :math:`a`, the score is

    .. math::
        \operatorname{PopulationPolygon}(D)
        =
        \frac{\sum_{i: a(c(i))=D} w_i}
        {\sum_{i: G_i \cap H(D) \ne \varnothing} w_i}.

    Any nonempty intersection contributes the polygon's complete weight. The scorer does not
    apportion weight by intersection area, so a boundary touch and a polygon lying wholly inside
    the hull both contribute equally. This deliberately reproduces the historical GerryTools
    population-polygon convention.

    Every alternative population polygon must be covered by exactly one evaluator geometry. Weights
    must be finite and nonnegative, with positive total weight overall and within every observed
    district. Alternative and scorer geometries must use the same projected CRS. Duplicate
    observations contribute independently.

    This statistic is resolution-dependent. A coarse polygon contributes its full population even
    if only a small sliver intersects the hull. It is not areal interpolation and must not be
    interpreted as the population estimated to physically lie inside the hull. See Moon Duchin,
    `Political Geometry
    <https://data-democracy.org/publications/political-geometry/01-Duchin.pdf>`_, and Duchin and
    Tenner, `Discrete Geometry for Electoral Geography
    <https://doi.org/10.1016/j.polgeo.2023.103040>`_.

    Args:
        population_col: Column containing nonnegative population values.
        alternative_pop_gdf: Optional projected GeoDataFrame containing a finer population surface.
            When omitted, ``population_col`` is read from the evaluator's aligned geometry.
    """

    _kind: ClassVar[str] = "population_polygon"
    population_col: str
    alternative_geometries: tuple[bytes, ...] | None
    alternative_weights: tuple[float, ...] | None
    alternative_crs: str | None

    def __init__(
        self,
        population_col: str,
        *,
        alternative_pop_gdf: GeoDataFrame | None = None,
        name: str | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        if not isinstance(population_col, str) or not population_col:
            raise ValueError("PopulationPolygon population_col must be a nonempty column name")
        object.__setattr__(self, "population_col", population_col)

        if alternative_pop_gdf is None:
            object.__setattr__(self, "alternative_geometries", None)
            object.__setattr__(self, "alternative_weights", None)
            object.__setattr__(self, "alternative_crs", None)
            return
        if not isinstance(alternative_pop_gdf, GeoDataFrame):
            raise TypeError("alternative_pop_gdf must be a GeoDataFrame or None")
        if population_col not in alternative_pop_gdf.columns:
            raise ValueError(
                f"alternative_pop_gdf does not contain population column {population_col!r}"
            )
        if alternative_pop_gdf.empty:
            raise ValueError("PopulationPolygon requires at least one observation")
        if alternative_pop_gdf.crs is None:
            raise ValueError("PopulationPolygon alternative_pop_gdf must have a CRS")
        population_crs = CRS.from_user_input(alternative_pop_gdf.crs)
        if not population_crs.is_projected:
            raise ValueError("PopulationPolygon alternative_pop_gdf must use a projected CRS")

        geometry = alternative_pop_gdf.geometry
        if geometry.isna().any():
            raise ValueError("PopulationPolygon geometries cannot contain missing values")
        if geometry.is_empty.any():
            raise ValueError("PopulationPolygon geometries cannot contain empty values")
        if not geometry.is_valid.all():
            raise ValueError("PopulationPolygon geometries must be topologically valid")
        if not geometry.geom_type.isin(("Polygon", "MultiPolygon")).all():
            raise ValueError(
                "PopulationPolygon geometries must contain only Polygon or MultiPolygon values"
            )
        if (geometry.area <= 0).any():
            raise ValueError("PopulationPolygon geometries must have positive area")

        weight_values = tuple(alternative_pop_gdf[population_col])
        checked_weights = []
        for index, value in enumerate(weight_values):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"PopulationPolygon weight {index} must be finite and nonnegative")
            checked_weights.append(float(value))
        if not any(weight > 0 for weight in checked_weights):
            raise ValueError("PopulationPolygon requires positive total weight")

        object.__setattr__(
            self,
            "alternative_geometries",
            tuple(bytes(value) for value in geometry.to_wkb(hex=False)),
        )
        object.__setattr__(self, "alternative_weights", tuple(checked_weights))
        object.__setattr__(self, "alternative_crs", population_crs.to_wkt())

    def _resolve(
        self, evaluator: PlanEvaluator
    ) -> tuple[tuple[bytes, ...], list[float], list[int] | None]:
        """Resolve validated ``(aligned rows, weights, covering positions)`` scorer inputs.

        Positions are ``None`` on the aligned path, where each weight already belongs to the graph
        node at its own row.
        """
        geometry = evaluator._require_geometry("PopulationPolygon")
        if self.alternative_geometries is None:
            weights = evaluator._nonnegative_node_column(self.population_col, "PopulationPolygon")
            if sum(weights) <= 0:
                raise ValueError("PopulationPolygon population must have a positive total")
            return geometry.wkb, weights, None
        assert self.alternative_crs is not None and self.alternative_weights is not None
        if CRS.from_user_input(self.alternative_crs) != geometry.crs:
            raise ValueError(
                "PopulationPolygon alternative_pop_gdf and evaluator geometry must use the same CRS"
            )
        positions = evaluator._population_positions(self.alternative_geometries)
        return geometry.wkb, list(self.alternative_weights), positions

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._resolve(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        columns = (
            frozenset()
            if self.alternative_geometries is not None
            else frozenset((evaluator._ordinary_column_resource(self.population_col),))
        )
        surfaces = (
            frozenset()
            if self.alternative_geometries is None
            else frozenset((self.alternative_geometries,))
        )
        return _ResourceSpec(
            node_columns=columns,
            geometry=True,
            population_surfaces=surfaces,
        )

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        rows, weights, positions = self._resolve(evaluator)
        if positions is None:
            backend.add_population_polygon_aligned(rows, weights)
        else:
            assert self.alternative_geometries is not None
            backend.add_population_polygon(
                rows,
                list(self.alternative_geometries),
                weights,
                positions,
            )
        return _OutputSpec("district", ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "model": "full_weight_intersection",
            "population_col": self.population_col,
            "surface": (
                "scorer_geometry" if self.alternative_geometries is None else "alternative_pop_gdf"
            ),
        }
        if self.alternative_geometries is not None:
            options["observations"] = len(self.alternative_geometries)
        return options


@dataclass(frozen=True, slots=True)
class Reock(_MetricBase):
    """Score district area divided by its minimum enclosing-circle area."""

    _kind: ClassVar[str] = "reock"

    def _validate(self, evaluator: PlanEvaluator) -> None:
        evaluator._require_geometry("Reock")

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        del evaluator
        return _ResourceSpec(geometry=True)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        backend.add_reock(evaluator._require_geometry("Reock").wkb)
        return _OutputSpec("district", ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {"source": "geometry"}


@dataclass(frozen=True, slots=True)
class PolsbyPopper(_MetricBase):
    """Score district compactness from graph measurements or aligned geometry.

    The default ``source="auto"`` uses aligned geometry when the evaluator has it and graph
    measurements otherwise. Supplying any graph-column option also selects graph measurements.
    Set ``source`` explicitly to override that choice.

    Graph-backed scoring uses ``area`` and ``shared_perimeter`` node/edge columns. Supply either a
    total ``perimeter`` node column or a ``boundary_perimeter`` column containing only the portion
    on the exterior boundary. Geometry-backed scoring derives all measurements from aligned WKB.
    """

    _kind: ClassVar[str] = "polsby_popper"
    _geometry_label: ClassVar[str] = "PolsbyPopper"
    _output_columns: ClassVar[tuple[str, ...]] = ("score",)
    source: Literal["auto", "graph", "geometry"] = "auto"
    area: str | None = None
    perimeter: str | None = None
    boundary_perimeter: str | None = None
    shared_perimeter: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"auto", "graph", "geometry"}:
            raise ValueError("PolsbyPopper source must be 'auto', 'graph', or 'geometry'")
        graph_options = (
            self.area,
            self.perimeter,
            self.boundary_perimeter,
            self.shared_perimeter,
        )
        if self.source == "geometry":
            if any(option is not None for option in graph_options):
                raise ValueError(
                    "graph column options cannot be used with geometry-backed scoring; "
                    "omit geometry to use graph columns"
                )
            return
        if self.perimeter is not None and self.boundary_perimeter is not None:
            raise ValueError("provide perimeter or boundary_perimeter, not both")

    def _resolved_source(self, evaluator: PlanEvaluator) -> Literal["graph", "geometry"]:
        if self.source != "auto":
            return self.source
        if any(
            option is not None
            for option in (
                self.area,
                self.perimeter,
                self.boundary_perimeter,
                self.shared_perimeter,
            )
        ):
            return "graph"
        return "geometry" if evaluator._has_geometry else "graph"

    def _graph_keys(self) -> tuple[str, str | None, str | None, str]:
        perimeter = self.perimeter
        boundary = self.boundary_perimeter
        if perimeter is None:
            boundary = boundary or "boundary_perim"
        return (
            self.area or "area",
            perimeter,
            boundary,
            self.shared_perimeter or "shared_perim",
        )

    def _graph_columns(
        self, evaluator: PlanEvaluator
    ) -> tuple[list[float], list[float] | None, list[float] | None, list[float]]:
        """Resolve ``(areas, total perimeters, boundary perimeters, shared perimeters)``."""
        area, perimeter, boundary_perimeter, shared_perimeter = self._graph_keys()
        total = None if perimeter is None else evaluator._numeric_graph_node_column(perimeter)
        boundary = (
            None
            if boundary_perimeter is None
            else evaluator._numeric_graph_node_column(boundary_perimeter)
        )
        return (
            evaluator._numeric_graph_node_column(area),
            total,
            boundary,
            evaluator._numeric_edge_column(shared_perimeter),
        )

    def _same_measurements(self, other: "PolsbyPopper") -> bool:
        return (
            self.source,
            self.area,
            self.perimeter,
            self.boundary_perimeter,
            self.shared_perimeter,
        ) == (
            other.source,
            other.area,
            other.perimeter,
            other.boundary_perimeter,
            other.shared_perimeter,
        )

    def _merge(self, other: _MetricBase) -> _MetricBase | None:
        if not isinstance(other, PolsbyPopper) or not self._same_measurements(other):
            return None
        if type(other) is type(self):
            return self
        return _AreaPerimeterMetrics(
            source=self.source,
            area=self.area,
            perimeter=self.perimeter,
            boundary_perimeter=self.boundary_perimeter,
            shared_perimeter=self.shared_perimeter,
        )

    def _column_indices(self, available: tuple[Hashable, ...]) -> tuple[int, ...]:
        return (available.index(self._kind),) if self._kind in available else (0,)

    def _result_columns(
        self,
        available: tuple[Hashable, ...],
        indices: tuple[int, ...],
    ) -> tuple[Hashable, ...]:
        del available, indices
        return ("score",)

    def _engine_adders(
        self, backend: ScoringEngine
    ) -> tuple[_GeometryAdder, _GraphAdder, _GraphAdder]:
        """Return the ``(geometry, graph-total, graph-boundary)`` engine registration hooks."""
        return (
            backend.add_polsby_popper_geometry,
            backend.add_polsby_popper_graph_total,
            backend.add_polsby_popper_graph_boundary,
        )

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        add_geometry, add_graph_total, add_graph_boundary = self._engine_adders(backend)
        if self._resolved_source(evaluator) == "geometry":
            geometry = evaluator._require_geometry(self._geometry_label)
            add_geometry(geometry.wkb, evaluator._geometry_rook_edges())
        else:
            areas, total, boundary, shared = self._graph_columns(evaluator)
            if total is not None:
                add_graph_total(areas, total, evaluator._edges, shared)
            else:
                assert boundary is not None
                add_graph_boundary(areas, boundary, evaluator._edges, shared)
        dtypes: tuple[_Dtype, ...] = ("float",) * len(self._output_columns)
        return _OutputSpec("district", self._output_columns, dtypes)

    def _validate(self, evaluator: PlanEvaluator) -> None:
        if self._resolved_source(evaluator) == "geometry":
            evaluator._require_geometry(self._geometry_label)
            return
        self._graph_columns(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        if self._resolved_source(evaluator) == "geometry":
            return _ResourceSpec(geometry=True, rook=True)
        area, perimeter, boundary, shared = self._graph_keys()
        columns = (area,) + ((perimeter,) if perimeter is not None else (boundary,))
        return _ResourceSpec(
            topology=True,
            node_columns=frozenset(("graph", column) for column in columns if column is not None),
            edge_columns=frozenset((shared,)),
        )

    def _stream_options(self, evaluator: PlanEvaluator) -> dict[str, object]:
        options = self._options()
        if options.get("source") != "auto":
            return options
        if self._resolved_source(evaluator) == "geometry":
            return {"source": "geometry"}

        area, perimeter, boundary_perimeter, shared_perimeter = self._graph_keys()
        options: dict[str, object] = {
            "source": "graph",
            "area": area,
            "shared_perimeter": shared_perimeter,
        }
        if perimeter is not None:
            options["perimeter"] = perimeter
        else:
            assert boundary_perimeter is not None
            options["boundary_perimeter"] = boundary_perimeter
        return options

    def _options(self) -> dict[str, object]:
        if self.source == "geometry":
            return {"source": "geometry"}
        if self.source == "auto" and not any(
            option is not None
            for option in (
                self.area,
                self.perimeter,
                self.boundary_perimeter,
                self.shared_perimeter,
            )
        ):
            return {"source": "auto"}
        area, perimeter, boundary_perimeter, shared_perimeter = self._graph_keys()
        options: dict[str, object] = {
            "source": self.source,
            "area": area,
            "shared_perimeter": shared_perimeter,
        }
        if perimeter is not None:
            options["perimeter"] = perimeter
        else:
            options["boundary_perimeter"] = boundary_perimeter
        return options


@dataclass(frozen=True, slots=True)
class Schwartzberg(PolsbyPopper):
    r"""Score perimeter compactness as the reciprocal square root of Polsby-Popper.

    For Polsby-Popper score :math:`PP`, Schwartzberg is :math:`1/\sqrt{PP}`. A circle has value
    one and larger values indicate less compactness. It accepts the same geometry and graph
    measurement sources as :class:`PolsbyPopper`. When both metrics use the same source and column
    options in one evaluator, they share one engine area-and-perimeter state.

    References:
        - Schwartzberg, "Reapportionment, Gerrymanders, and the Notion of Compactness."
          https://doi.org/10.24926/265535.2601
        - Polsby and Popper, "The Third Criterion: Compactness as a Procedural Safeguard Against
          Partisan Gerrymandering."
          https://openyls.law.yale.edu/handle/20.500.13051/17448
        - Duchin and Tenner, "Discrete Geometry for Electoral Geography."
          https://doi.org/10.1016/j.polgeo.2023.103040
    """

    _kind: ClassVar[str] = "schwartzberg"
    _geometry_label: ClassVar[str] = "Schwartzberg"

    def _engine_adders(
        self, backend: ScoringEngine
    ) -> tuple[_GeometryAdder, _GraphAdder, _GraphAdder]:
        return (
            backend.add_schwartzberg_geometry,
            backend.add_schwartzberg_graph_total,
            backend.add_schwartzberg_graph_boundary,
        )


@dataclass(frozen=True, slots=True)
class _AreaPerimeterMetrics(PolsbyPopper):
    _kind: ClassVar[str] = "area_perimeter_metrics"
    _geometry_label: ClassVar[str] = "PolsbyPopper and Schwartzberg"
    _output_columns: ClassVar[tuple[str, ...]] = ("polsby_popper", "schwartzberg")

    def _engine_adders(
        self, backend: ScoringEngine
    ) -> tuple[_GeometryAdder, _GraphAdder, _GraphAdder]:
        return (
            backend.add_area_perimeter_metrics_geometry,
            backend.add_area_perimeter_metrics_graph_total,
            backend.add_area_perimeter_metrics_graph_boundary,
        )


@dataclass(frozen=True, slots=True)
class CutEdges(_MetricBase):
    """Count cut graph edges or sum a numeric edge weight over them."""

    _kind: ClassVar[str] = "cut_edges"
    weight: str | None = None

    def __post_init__(self) -> None:
        if self.weight is not None and (not isinstance(self.weight, str) or not self.weight):
            raise ValueError("CutEdges weight must be a nonempty string or None")

    def _weights(self, evaluator: PlanEvaluator) -> list[float] | None:
        return None if self.weight is None else evaluator._numeric_edge_column(self.weight)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        backend.add_cut_edges(evaluator._node_count, evaluator._edges, self._weights(evaluator))
        dtype = "int" if self.weight is None else "float"
        return _OutputSpec("plan", (self.weight or "count",), (dtype,))

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._weights(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        del evaluator
        return _ResourceSpec(
            topology=True,
            edge_columns=frozenset() if self.weight is None else frozenset((self.weight,)),
        )

    def _options(self) -> dict[str, object]:
        return {"weight": self.weight}


@dataclass(frozen=True, slots=True, init=False)
class _RegionMetric(_KeyedMetric):
    def _columns(self, evaluator: PlanEvaluator) -> list[list[int | None]]:
        return [evaluator._region_column(key)[0] for key in self.keys]

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._columns(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        return _ResourceSpec(
            region_columns=frozenset(evaluator._ordinary_column_resource(key) for key in self.keys)
        )

    def _merge(self, other: _MetricBase) -> _MetricBase | None:
        if type(other) is not type(self):
            return None
        return type(self)(*_merged_keys(self.keys, other.keys))

    def _register(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> None:
        raise NotImplementedError

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        self._register(backend, evaluator)
        return _OutputSpec("plan", self.keys, ("int",) * len(self.keys))


@dataclass(frozen=True, slots=True, init=False)
class RegionSplits(_RegionMetric):
    """Count fixed regions assigned to more than one district."""

    _kind: ClassVar[str] = "region_splits"

    def _register(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> None:
        backend.add_region_splits(self._columns(evaluator))


@dataclass(frozen=True, slots=True, init=False)
class RegionPieces(_RegionMetric):
    """Count occupied ``(fixed region, proposed district)`` pairs."""

    _kind: ClassVar[str] = "region_pieces"

    def _register(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> None:
        backend.add_region_pieces(self._columns(evaluator))


@dataclass(frozen=True, slots=True, init=False)
class RegionParts(_RegionMetric):
    """Count connected fixed-region-by-district parts in the induced region subgraphs."""

    _kind: ClassVar[str] = "region_parts"

    def _register(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> None:
        backend.add_region_parts(self._columns(evaluator), evaluator._edges)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        return _RegionMetric._resources(self, evaluator) | _ResourceSpec(topology=True)


@dataclass(frozen=True, slots=True, init=False)
class TallyByRegion(_MetricBase):
    """Sum one or more node columns by fixed region and proposed district.

    ``columns`` accepts column names or maps display names to source columns. Mapping insertion
    order is preserved.
    ``include_count=True`` places a unit-count value named ``"count"`` first. Nodes with a
    missing region label contribute to no region.
    """

    _kind: ClassVar[str] = "tally_by_region"
    region: str
    columns: tuple[tuple[str, str], ...]
    include_count: bool

    def __init__(
        self,
        region: str,
        columns: str | Iterable[str] | Mapping[str, str] | None = None,
        *,
        include_count: bool = False,
        name: str | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        if not isinstance(region, str) or not region:
            raise ValueError("TallyByRegion region must be a nonempty string")
        if not isinstance(include_count, bool):
            raise TypeError("TallyByRegion include_count must be a bool")

        if columns is None:
            items: tuple[tuple[object, object], ...] = ()
        elif isinstance(columns, Mapping):
            items = tuple(columns.items())
        elif isinstance(columns, str):
            items = ((columns, columns),)
        else:
            try:
                items = tuple((column, column) for column in columns)
            except TypeError:
                raise TypeError(
                    "TallyByRegion columns must be a string, iterable, mapping, or None"
                ) from None
        if any(not isinstance(name, str) or not name for name, _ in items):
            raise ValueError("TallyByRegion column names must be nonempty strings")
        if len({name for name, _ in items}) != len(items):
            raise ValueError("TallyByRegion column names cannot repeat")
        if any(not isinstance(column, str) or not column for _, column in items):
            raise ValueError("TallyByRegion source columns must be nonempty strings")
        if include_count and any(name == "count" for name, _ in items):
            raise ValueError("TallyByRegion columns cannot contain 'count' when include_count=True")
        if not include_count and not items:
            raise ValueError("TallyByRegion requires at least one column or include_count=True")

        object.__setattr__(self, "region", region)
        object.__setattr__(self, "columns", cast("tuple[tuple[str, str], ...]", items))
        object.__setattr__(self, "include_count", include_count)

    def _inputs(
        self, evaluator: PlanEvaluator
    ) -> tuple[list[int | None], tuple[Hashable, ...], list[list[float]]]:
        regions, labels = evaluator._region_column(self.region)
        columns = [evaluator._numeric_node_column(column) for _, column in self.columns]
        return regions, labels, columns

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        regions, labels, columns = self._inputs(evaluator)
        backend.add_tally_by_region(regions, self.include_count, columns)
        names = (("count",) if self.include_count else ()) + tuple(name for name, _ in self.columns)
        dtypes: tuple[_Dtype, ...] = (("int",) if self.include_count else ()) + ("float",) * len(
            self.columns
        )
        return _OutputSpec("region", names, dtypes, labels, self.region)

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._inputs(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        return _ResourceSpec(
            node_columns=frozenset(
                evaluator._ordinary_column_resource(column) for _, column in self.columns
            ),
            region_columns=frozenset((evaluator._ordinary_column_resource(self.region),)),
        )

    def _options(self) -> dict[str, object]:
        return {
            "region": self.region,
            "columns": dict(self.columns),
            "include_count": self.include_count,
        }
