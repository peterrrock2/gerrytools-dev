use super::*;
use crate::{Coordinate, PreparedUnitHulls, UnitHull};
use geo::{polygon, MultiPolygon};

fn point(x: f64, y: f64) -> Coordinate {
    Coordinate { x, y }
}

fn square(x: f64, y: f64) -> UnitHull {
    UnitHull::new(1.0, square_points(x, y))
}

fn square_points(x: f64, y: f64) -> Vec<Coordinate> {
    vec![
        point(x, y),
        point(x + 1.0, y),
        point(x + 1.0, y + 1.0),
        point(x, y + 1.0),
    ]
}

fn rectangle_state(width: f64, height: f64) -> MultiPolygon<f64> {
    MultiPolygon(vec![polygon![
        (x: 0.0, y: 0.0),
        (x: width, y: 0.0),
        (x: width, y: height),
        (x: 0.0, y: height),
        (x: 0.0, y: 0.0),
    ]])
}

fn metric(units: Vec<UnitHull>, state: MultiPolygon<f64>) -> PreparedStateClippedConvexHullRatio {
    PreparedStateClippedConvexHullRatio::from_validated_parts(
        Arc::new(PreparedUnitHulls::new(units).unwrap()),
        state,
    )
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

fn assert_same_scores(actual: &DistrictTable, expected: &DistrictTable) {
    assert_eq!(actual.district_ids(), expected.district_ids());
    for (&actual, &expected) in actual
        .column(0)
        .unwrap()
        .iter()
        .zip(expected.column(0).unwrap())
    {
        assert_close(actual, expected);
    }
}

#[test]
fn clipping_removes_hull_area_outside_a_concave_state() {
    let state = MultiPolygon(vec![polygon![
        (x: 0.0, y: 0.0),
        (x: 2.0, y: 0.0),
        (x: 2.0, y: 1.0),
        (x: 1.0, y: 1.0),
        (x: 1.0, y: 2.0),
        (x: 0.0, y: 2.0),
        (x: 0.0, y: 0.0),
    ]]);
    let metric = metric(
        vec![square(0.0, 0.0), square(1.0, 0.0), square(0.0, 1.0)],
        state,
    );

    let result = metric.score(&[0, 0, 0]).unwrap();

    assert_close(result.column(0).unwrap()[0], 1.0);
}

#[test]
fn clipping_handles_a_state_with_disconnected_islands() {
    let state = MultiPolygon(vec![
        rectangle_state(1.0, 1.0).0.remove(0),
        polygon![
            (x: 2.0, y: 0.0),
            (x: 3.0, y: 0.0),
            (x: 3.0, y: 1.0),
            (x: 2.0, y: 1.0),
            (x: 2.0, y: 0.0),
        ],
    ]);
    let metric = metric(vec![square(0.0, 0.0), square(2.0, 0.0)], state);

    let result = metric.score(&[0, 0]).unwrap();

    assert_close(result.column(0).unwrap()[0], 1.0);
}

#[test]
fn incremental_updates_match_fresh_scores_across_generated_moves() {
    let units = (0..5)
        .flat_map(|row| (0..6).map(move |column| square(column as f64, row as f64)))
        .collect();
    let metric = metric(units, rectangle_state(6.0, 5.0));
    let mut assignment = (0..metric.node_count())
        .map(|node| (node % 4) as u16)
        .collect::<Vec<_>>();
    let mut incremental = metric.incremental(&assignment).unwrap();
    let mut seed = 0x2d79_5eed_cafe_u64;

    for _ in 0..500 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let node = (seed as usize) % assignment.len();
        let old = assignment[node];
        let mut new = ((seed >> 32) % 4) as u16;
        if new == old {
            new = (new + 1) % 4;
        }
        let change = DeltaChange { node, old, new };

        incremental.update(&[change]).unwrap();
        assignment[node] = new;

        assert_same_scores(&incremental.result(), &metric.score(&assignment).unwrap());
    }
}

#[test]
fn rejects_impossible_score_and_invalid_assignment() {
    let impossible = metric(
        vec![UnitHull::new(2.0, square_points(0.0, 0.0))],
        rectangle_state(1.0, 1.0),
    );
    assert!(matches!(
        impossible.score(&[4]),
        Err(Error::ImpossibleScore {
            metric: "state-clipped convex-hull ratio",
            district: 4,
            ..
        })
    ));

    let metric = metric(vec![square(0.0, 0.0)], rectangle_state(1.0, 1.0));
    assert_eq!(
        metric.score(&[]).unwrap_err(),
        Error::AssignmentLength {
            actual: 0,
            expected: 1,
        }
    );
}
