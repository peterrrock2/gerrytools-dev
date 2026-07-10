"""Shared implementation support for metric descriptions."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from ..result import _Dtype

if TYPE_CHECKING:
    from gerrytools._scoring_engine import ScoringEngine

    from ..evaluator import PlanEvaluator

_ColumnResource: TypeAlias = tuple[Literal["graph", "geometry"], str]


@dataclass(frozen=True, slots=True)
class _ResourceSpec:
    """Immutable set of source resources required by a metric collection."""

    topology: bool = False
    node_columns: frozenset[_ColumnResource] = frozenset()
    edge_columns: frozenset[str] = frozenset()
    region_columns: frozenset[_ColumnResource] = frozenset()
    alignment: bool = False
    geometry: bool = False
    rook: bool = False
    population_surfaces: frozenset[tuple[bytes, ...]] = frozenset()
    fixed_values: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        geometry_column = any(
            source == "geometry" for source, _ in self.node_columns | self.region_columns
        )
        if geometry_column or self.geometry or self.rook or self.population_surfaces:
            object.__setattr__(self, "alignment", True)
        if self.rook or self.population_surfaces:
            object.__setattr__(self, "geometry", True)
        if self.edge_columns:
            object.__setattr__(self, "topology", True)

    def __or__(self, other: "_ResourceSpec") -> "_ResourceSpec":
        return _ResourceSpec(
            topology=self.topology or other.topology,
            node_columns=self.node_columns | other.node_columns,
            edge_columns=self.edge_columns | other.edge_columns,
            region_columns=self.region_columns | other.region_columns,
            alignment=self.alignment or other.alignment,
            geometry=self.geometry or other.geometry,
            rook=self.rook or other.rook,
            population_surfaces=self.population_surfaces | other.population_surfaces,
            fixed_values=self.fixed_values | other.fixed_values,
        )

    def contains(self, other: "_ResourceSpec") -> bool:
        """Return whether every resource in ``other`` is present."""
        return (
            (self.topology or not other.topology)
            and self.node_columns.issuperset(other.node_columns)
            and self.edge_columns.issuperset(other.edge_columns)
            and self.region_columns.issuperset(other.region_columns)
            and (self.alignment or not other.alignment)
            and (self.geometry or not other.geometry)
            and (self.rook or not other.rook)
            and self.population_surfaces.issuperset(other.population_surfaces)
            and self.fixed_values.issuperset(other.fixed_values)
        )


@dataclass(frozen=True, slots=True)
class _OutputSpec:
    shape: Literal["district", "plan", "region"]
    columns: tuple[Hashable, ...]
    dtypes: tuple[_Dtype, ...]
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.dtypes):
            raise ValueError("metric output columns and dtypes must have equal length")
        if self.shape == "region":
            if self.region_name is None:
                raise ValueError("region output requires a region axis name")
        elif self.regions or self.region_name is not None:
            raise ValueError("only region output can define a region axis")

    @property
    def value_count(self) -> int:
        """Number of flat engine columns represented by this output."""
        if self.shape == "region":
            return len(self.columns) * len(self.regions)
        return len(self.columns)


@dataclass(frozen=True, slots=True, kw_only=True)
class _MetricBase:
    """Implementation-sharing base for the concrete metric descriptors in this module.

    ``name`` changes only the public result key. It does not affect equality so differently named
    registrations can still share prepared engine work.
    """

    _kind: ClassVar[str]
    name: str | None = field(default=None, compare=False)

    def _default_name(self) -> str:
        return self._kind

    def _validate(self, evaluator: PlanEvaluator) -> None:
        pass

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        return _ResourceSpec(
            node_columns=frozenset(
                evaluator._ordinary_column_resource(key) for key in self._tally_keys()
            )
        )

    def _merge(self, other: _MetricBase) -> _MetricBase | None:
        return None

    def _options(self) -> dict[str, object]:
        return {}

    def _stream_options(self, evaluator: PlanEvaluator) -> dict[str, object]:
        """Options recorded in streamed manifests, with evaluator-dependent defaults resolved."""
        del evaluator
        return self._options()

    def _tally_keys(self) -> tuple[str, ...]:
        return ()

    def _column_indices(self, available: tuple[Hashable, ...]) -> tuple[int, ...]:
        return tuple(range(len(available)))

    def _result_columns(
        self,
        available: tuple[Hashable, ...],
        indices: tuple[int, ...],
    ) -> tuple[Hashable, ...]:
        return tuple(available[index] for index in indices)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        raise NotImplementedError


def _keys(values: tuple[str, ...], metric: str) -> tuple[str, ...]:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{metric} requires at least one nonempty string key")
    if len(set(values)) != len(values):
        raise ValueError(f"{metric} keys cannot repeat")
    return values


def _merged_keys(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return left + tuple(key for key in right if key not in left)


@dataclass(frozen=True, slots=True, init=False)
class _KeyedMetric(_MetricBase):
    keys: tuple[str, ...]

    def __init__(self, *keys: str, name: str | None = None) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "keys", _keys(keys, type(self).__name__))

    def _column_indices(self, available: tuple[Hashable, ...]) -> tuple[int, ...]:
        return tuple(available.index(key) for key in self.keys)
