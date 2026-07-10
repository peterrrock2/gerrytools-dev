"""Shared source normalization for single-plan functions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, TypeAlias

import networkx as nx
from geopandas import GeoDataFrame
from gerrychain import Partition
from pandas import DataFrame, Series

from ..evaluator import (
    Assignment,
    PlanEvaluator,
    _assignment_mapping,
    _is_missing,
    _partition_graph,
    _result_name,
)
from ..metrics import Metric

SinglePlanSource: TypeAlias = Partition | nx.Graph | GeoDataFrame


SinglePlanResult: TypeAlias = float | int | Series | DataFrame


GeoAssignment: TypeAlias = Assignment | str


def _columns(values: str | Iterable[str]) -> tuple[str, ...]:
    return (values,) if isinstance(values, str) else tuple(values)


def _geodataframe_assignment(
    frame: GeoDataFrame,
    assignment: GeoAssignment | None,
) -> list[Hashable]:
    if assignment is None:
        raise TypeError("a GeoDataFrame source requires an assignment")
    if isinstance(assignment, str):
        if assignment not in frame.columns:
            raise ValueError(f"GeoDataFrame does not contain assignment column {assignment!r}")
        values = list(frame[assignment])
    elif (mapping := _assignment_mapping(assignment)) is not None:
        missing = [node for node in frame.index if node not in mapping]
        unexpected = [node for node in mapping if node not in frame.index]
        if missing or unexpected:
            raise ValueError(
                "assignment keys must exactly match GeoDataFrame index; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        values = [mapping[node] for node in frame.index]
    else:
        values = list(assignment)
        if len(values) != len(frame):
            raise ValueError(f"assignment has {len(values)} values; expected {len(frame)}")
    if any(_is_missing(value) for value in values):
        raise ValueError("assignment cannot contain missing district labels")
    return values


def _evaluate(
    source: SinglePlanSource,
    assignment: GeoAssignment | None,
    metric: Metric,
    *,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
    crs: Any | None = None,
    topology_required: bool = False,
) -> SinglePlanResult:
    if isinstance(source, Partition):
        if assignment is not None:
            raise TypeError("a Partition supplies its own assignment")
        if geometry is None and (node_column is not None or crs is not None):
            raise ValueError("node_column and crs require geometry")
        evaluator = PlanEvaluator(
            _partition_graph(source),
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        )
        plan: Assignment | Partition = source
    elif isinstance(source, nx.Graph):
        if assignment is None:
            raise TypeError("a graph source requires an assignment")
        if isinstance(assignment, str):
            raise TypeError("a graph assignment cannot be a GeoDataFrame column name")
        evaluator = PlanEvaluator(
            source,
            geometry=geometry,
            node_column=node_column,
            crs=crs,
        )
        plan = assignment
    elif isinstance(source, GeoDataFrame):
        if topology_required:
            raise TypeError(f"{metric._kind} requires a GerryChain Partition or graph")
        if geometry is not None:
            raise TypeError("do not supply geometry when the source is already a GeoDataFrame")
        if node_column is not None:
            raise TypeError("node_column applies only to Partition geometry alignment")
        nodes = tuple(source.index)
        evaluator = PlanEvaluator(nx.empty_graph(nodes), geometry=source, crs=crs)
        plan = _geodataframe_assignment(source, assignment)
    else:
        raise TypeError("source must be a GerryChain Partition, graph, or GeoDataFrame")

    # Match evaluator registration, including explicit names.
    name = _result_name(metric)
    return evaluator.add_metric(metric).evaluate(plan)[name]
