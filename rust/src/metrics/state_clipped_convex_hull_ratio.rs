use super::hull_metric::{
    full_scan_score, HullScorer, IncrementalHullMetric, CLIPPED_RATIO_SCORE_EPS,
};
use crate::scoring::delta::DeltaChange;
use crate::{DistrictTable, Error, PreparedUnitHulls, Result};
use geo::algorithm::convex_hull::quick_hull;
use geo::{Area, BooleanOps, Coord, MultiPolygon, Polygon};
use std::sync::Arc;

#[derive(Debug)]
/// Prepared unit hulls and state geometry for clipped convex-hull ratios.
pub struct PreparedStateClippedConvexHullRatio {
    unit_hulls: Arc<PreparedUnitHulls>,
    state: MultiPolygon<f64>,
}

impl PreparedStateClippedConvexHullRatio {
    pub(crate) fn from_validated_parts(
        unit_hulls: Arc<PreparedUnitHulls>,
        state: MultiPolygon<f64>,
    ) -> Self {
        Self { unit_hulls, state }
    }

    /// Return a shared handle to the prepared unit hulls.
    pub fn unit_hulls(&self) -> Arc<PreparedUnitHulls> {
        Arc::clone(&self.unit_hulls)
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.unit_hulls.node_count()
    }

    /// Create incremental state-clipped convex-hull state for an initial assignment.
    pub fn incremental(
        &self,
        assignment: &[u16],
    ) -> Result<IncrementalStateClippedConvexHullRatio<'_>> {
        Ok(IncrementalStateClippedConvexHullRatio(
            IncrementalHullMetric::new(self, assignment)?,
        ))
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        full_scan_score(self, assignment)
    }
}

impl HullScorer for PreparedStateClippedConvexHullRatio {
    type Scratch = Vec<Coord<f64>>;

    fn scratch(&self, _secondary: bool) -> Self::Scratch {
        Vec::new()
    }

    fn node_count(&self) -> usize {
        self.unit_hulls.node_count()
    }

    fn unit_area(&self, node: usize) -> f64 {
        self.unit_hulls.unit_area(node)
    }

    fn score_district(
        &self,
        nodes: &[usize],
        area: f64,
        district: u16,
        points: &mut Self::Scratch,
    ) -> Result<f64> {
        points.clear();
        for &node in nodes {
            points.extend(
                self.unit_hulls
                    .unit_hull_points(node)
                    .iter()
                    .map(|point| Coord {
                        x: point.x,
                        y: point.y,
                    }),
            );
        }
        state_clipped_convex_hull_ratio(points, area, district, &self.state)
    }
}

/// State-clipped district convex-hull ratios maintained across assignment changes.
pub struct IncrementalStateClippedConvexHullRatio<'a>(
    IncrementalHullMetric<'a, PreparedStateClippedConvexHullRatio>,
);

impl IncrementalStateClippedConvexHullRatio<'_> {
    /// Replace the assignment and recompute every score from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        self.0.reset(assignment)
    }

    /// Apply a delta whose `old` labels are validated against this state's current assignment.
    pub fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
        self.0.update(changes)
    }

    pub(crate) fn update_trusted(
        &mut self,
        canonical_assignment: Option<&[u16]>,
        changes: &[DeltaChange],
    ) -> Result<()> {
        self.0.update_trusted(canonical_assignment, changes)
    }

    /// Return the current score for every observed district.
    pub fn result(&self) -> DistrictTable {
        self.0.result()
    }
}

fn state_clipped_convex_hull_ratio(
    points: &mut [Coord<f64>],
    district_area: f64,
    district: u16,
    state: &MultiPolygon<f64>,
) -> Result<f64> {
    if !district_area.is_finite() || district_area <= 0.0 {
        return Err(Error::InvalidDistrictArea {
            district,
            area: district_area,
        });
    }
    let hull = Polygon::new(quick_hull(points), Vec::new());
    let clipped_area = hull.intersection(state).unsigned_area();
    if !clipped_area.is_finite() || clipped_area <= 0.0 {
        return Err(Error::InvalidEnclosureArea {
            metric: "state-clipped convex-hull",
            district,
            area: clipped_area,
        });
    }

    let score = district_area / clipped_area;
    if score > 1.0 + CLIPPED_RATIO_SCORE_EPS {
        return Err(Error::ImpossibleScore {
            metric: "state-clipped convex-hull ratio",
            district,
            score,
        });
    }
    Ok(score.min(1.0))
}

#[cfg(test)]
#[path = "../tests/state_clipped_convex_hull_ratio.rs"]
mod tests;
