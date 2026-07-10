import math
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pyarrow.parquet as pq
import pytest
from binary_ensemble import BenEncoder
from gerrychain import Partition
from pandas import Series
from shapely.geometry import box

import gerrytools.scoring as scoring
from gerrytools.scoring import formulas

Variant = Literal["standard", "mkv_chain", "twodelta"]


def resources() -> tuple[nx.Graph, gpd.GeoDataFrame, Partition]:
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {node: 1.0 for node in graph}, "area")
    nx.set_node_attributes(graph, {node: 4.0 for node in graph}, "perimeter")
    nx.set_node_attributes(
        graph,
        {node: 4.0 - graph.degree[node] for node in graph},
        "boundary_perim",
    )
    nx.set_edge_attributes(graph, {edge: 1.0 for edge in graph.edges}, "shared_perim")
    frame = gpd.GeoDataFrame(
        {"district": [0, 0, 1, 1]},
        geometry=[box(index, 0, index + 1, 1) for index in range(4)],
        crs="EPSG:3857",
    )
    return graph, frame, Partition(graph, dict(enumerate(frame["district"])))


def test_single_plan_and_prepared_schwartzberg_match_the_array_transform() -> None:
    graph, frame, partition = resources()
    evaluator = (
        scoring.PlanEvaluator(graph, geometry=frame)
        .add_metric(scoring.PolsbyPopper(name="polsby"))
        .add_metric(scoring.Schwartzberg(name="schwartzberg"))
    )

    result = evaluator.evaluate(frame["district"])
    expected = formulas.schwartzberg(cast(Series, result["polsby"]).to_numpy())

    assert len(evaluator._engine_prepared) == 1
    assert evaluator._engine_prepared[0].value_count == 2
    np.testing.assert_allclose(result["schwartzberg"], expected)
    np.testing.assert_allclose(scoring.schwartzberg(frame, "district"), expected)
    np.testing.assert_allclose(scoring.schwartzberg(partition, frame), expected)


def test_standalone_schwartzberg_resolves_auto_to_geometry_without_graph_columns() -> None:
    _, frame, _ = resources()
    bare_graph = nx.path_graph(4)  # no measurement columns, so "auto" must resolve to geometry

    schwartzberg = (
        scoring.PlanEvaluator(bare_graph, geometry=frame)
        .add_metric(scoring.Schwartzberg())
        .evaluate(frame["district"])["schwartzberg"]
    )
    polsby = (
        scoring.PlanEvaluator(bare_graph, geometry=frame)
        .add_metric(scoring.PolsbyPopper())
        .evaluate(frame["district"])["polsby_popper"]
    )

    np.testing.assert_allclose(schwartzberg, formulas.schwartzberg(polsby))


def test_standalone_schwartzberg_resolves_auto_like_standalone_polsby_popper() -> None:
    graph, frame, _ = resources()
    # Make graph-derived scores differ from geometry-derived ones so the resolution is pinned.
    nx.set_node_attributes(graph, {node: 0.5 for node in graph}, "area")

    def standalone(metric: scoring.PolsbyPopper) -> Series:
        evaluator = scoring.PlanEvaluator(graph, geometry=frame).add_metric(metric)
        return cast(Series, evaluator.evaluate(frame["district"])[evaluator.metrics[0]])

    schwartzberg = standalone(scoring.Schwartzberg())
    polsby = standalone(scoring.PolsbyPopper())

    np.testing.assert_allclose(schwartzberg, formulas.schwartzberg(polsby))
    assert not np.allclose(schwartzberg, standalone(scoring.Schwartzberg(source="graph")))


def test_standalone_graph_schwartzberg_registers_without_polsby_popper() -> None:
    graph, _, partition = resources()

    schwartzberg = (
        scoring.PlanEvaluator(graph)
        .add_metric(scoring.Schwartzberg(source="graph"))
        .evaluate(partition)["schwartzberg"]
    )
    polsby = (
        scoring.PlanEvaluator(graph)
        .add_metric(scoring.PolsbyPopper(source="graph"))
        .evaluate(partition)["polsby_popper"]
    )

    np.testing.assert_allclose(schwartzberg, formulas.schwartzberg(polsby))


def test_graph_schwartzberg_uses_the_same_measurement_contract_as_polsby_popper() -> None:
    graph, _, partition = resources()
    polsby = scoring.polsby_popper(
        partition,
        area="area",
        perimeter="perimeter",
        shared_perimeter="shared_perim",
    )
    actual = scoring.schwartzberg(
        partition,
        area="area",
        perimeter="perimeter",
        shared_perimeter="shared_perim",
    )

    np.testing.assert_allclose(actual, formulas.schwartzberg(polsby))


@pytest.mark.parametrize("perimeter", [None, "perimeter"])
def test_graph_polsby_and_schwartzberg_share_boundary_or_total_measurements(
    perimeter: str | None,
) -> None:
    graph, _, partition = resources()
    evaluator = (
        scoring.PlanEvaluator(graph)
        .add_metric(scoring.PolsbyPopper(perimeter=perimeter))
        .add_metric(scoring.Schwartzberg(perimeter=perimeter))
    )

    result = evaluator.evaluate(partition)

    assert len(evaluator._engine_prepared) == 1
    np.testing.assert_allclose(
        result["schwartzberg"],
        formulas.schwartzberg(result["polsby_popper"]),
    )


@pytest.mark.parametrize("metric_type", [scoring.PolsbyPopper, scoring.Schwartzberg])
def test_repeated_compactness_metrics_share_one_native_registration(
    metric_type: type[scoring.PolsbyPopper],
) -> None:
    graph, _, partition = resources()
    evaluator = (
        scoring.PlanEvaluator(graph)
        .add_metric(metric_type(name="first"))
        .add_metric(metric_type(name="second"))
    )

    result = evaluator.evaluate(partition)

    assert len(evaluator._engine_prepared) == 1
    np.testing.assert_allclose(result["first"], result["second"])


def test_graph_total_mode_agrees_with_boundary_mode_and_hand_computed_values() -> None:
    graph, _, _ = resources()

    def graph_scores(perimeter: str | None) -> tuple[Series, Series]:
        result = (
            scoring.PlanEvaluator(graph)
            .add_metric(scoring.PolsbyPopper(perimeter=perimeter))
            .add_metric(scoring.Schwartzberg(perimeter=perimeter))
            .evaluate([0, 1, 1, 1])
        )
        return cast(Series, result["polsby_popper"]), cast(Series, result["schwartzberg"])

    total_polsby, total_schwartzberg = graph_scores("perimeter")
    boundary_polsby, boundary_schwartzberg = graph_scores(None)

    # The fixture satisfies perimeter = boundary_perim + degree * shared_perim, so the total
    # and boundary graph modes must produce identical district scores.
    np.testing.assert_allclose(total_polsby, boundary_polsby, rtol=1e-13)
    np.testing.assert_allclose(total_schwartzberg, boundary_schwartzberg, rtol=1e-13)

    # Hand-computed total mode for plan [0, 1, 1, 1]: district 0 keeps node 0's full perimeter
    # (area 1, perimeter 4); district 1 subtracts both internal shared perimeters twice
    # (area 3, perimeter 3 * 4 - 2 * 2 = 8).
    np.testing.assert_allclose(
        total_polsby,
        [4 * math.pi * 1 / 4**2, 4 * math.pi * 3 / 8**2],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        total_schwartzberg,
        [math.sqrt(4**2 / (4 * math.pi * 1)), math.sqrt(8**2 / (4 * math.pi * 3))],
        rtol=1e-13,
    )


@pytest.mark.parametrize("variant", ["standard", "mkv_chain", "twodelta"])
def test_schwartzberg_streams_like_evaluate_many(tmp_path: Path, variant: Variant) -> None:
    source = tmp_path / f"plans-{variant}.ben"
    output = tmp_path / "scores"
    plans = [[0, 0, 1, 1], [0, 1, 0, 1]]
    with BenEncoder(source, variant=variant) as stream:
        for plan in plans:
            stream.write(plan)

    graph, frame, _ = resources()
    evaluator = (
        scoring.PlanEvaluator(graph, geometry=frame)
        .add_metric(scoring.PolsbyPopper())
        .add_metric(scoring.Schwartzberg())
    )
    expected = evaluator.evaluate_many(plans).array("schwartzberg")

    evaluator.evaluate_stream(source, output, batch_size=1)

    schwartzberg = pq.read_table(output / "schwartzberg" / "scores.parquet").to_pydict()
    polsby = pq.read_table(output / "polsby_popper" / "scores.parquet").to_pydict()
    for district in range(2):
        np.testing.assert_allclose(
            schwartzberg[f"score__district_{district}"],
            expected[:, 0, district],
        )
        np.testing.assert_allclose(
            schwartzberg[f"score__district_{district}"],
            formulas.schwartzberg(polsby[f"score__district_{district}"]),
        )


def test_top_level_schwartzberg_is_plan_oriented() -> None:
    assert scoring.schwartzberg is not formulas.schwartzberg
