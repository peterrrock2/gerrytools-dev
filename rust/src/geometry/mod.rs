use crate::adjacency::validate_edge_nodes;
use crate::{Error, PreparedPolsbyPopper, PreparedStateClippedConvexHullRatio, Result};
use geo::algorithm::unary_union;
use geo::{Area, BooleanOps, ConvexHull, Coord, Geometry, MultiPolygon, Polygon, Validation};
use geo_traits::to_geo::ToGeoGeometry;
use std::sync::Arc;

mod unit_hulls;

pub use unit_hulls::{Coordinate, PreparedUnitHulls, UnitHull, UnitHullCache};

const GEOMETRY_EPS: f64 = 1e-8;
const STATE_COVERAGE_REL_EPS: f64 = 1e-12;

impl PreparedStateClippedConvexHullRatio {
    /// Test-only convenience: decode `rows` into unit hulls, then validate the state geometry.
    #[cfg(test)]
    pub(crate) fn from_wkb<W: AsRef<[u8]>>(rows: &[W], state: &[u8]) -> Result<Self> {
        Self::from_unit_hulls_and_wkb(Arc::new(PreparedUnitHulls::from_wkb(rows)?), rows, state)
    }

    /// Decode an explicit state Polygon or MultiPolygon WKB and check it covers `rows`, which
    /// must be the same unit WKB the shared hulls were decoded from.
    pub fn from_unit_hulls_and_wkb<W: AsRef<[u8]>>(
        unit_hulls: Arc<PreparedUnitHulls>,
        rows: &[W],
        state: &[u8],
    ) -> Result<Self> {
        if rows.len() != unit_hulls.node_count() {
            return Err(geometry_error(format!(
                "geometry has {} rows; expected {}",
                rows.len(),
                unit_hulls.node_count()
            )));
        }
        let state = decode_state_polygon(state)?;
        let geometries = rows
            .iter()
            .enumerate()
            .map(|(row, bytes)| decode_polygon(row, bytes.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        validate_state_coverage(&geometries, &state)?;
        Ok(Self::from_validated_parts(unit_hulls, state))
    }
}

impl PreparedPolsbyPopper {
    /// Decode ordered, projected Polygon or MultiPolygon WKB rows into a prepared metric.
    pub fn from_wkb<W: AsRef<[u8]>>(rows: &[W], edges: Vec<(u32, u32)>) -> Result<Self> {
        let units = rows
            .iter()
            .enumerate()
            .map(|(row, bytes)| polsby_popper_unit(row, bytes.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        validate_edge_nodes(units.len(), &edges)?;

        let mut shared_perimeters = Vec::with_capacity(edges.len());
        let mut shared_by_node = vec![0.0; units.len()];
        for &(u, v) in &edges {
            let left = &units[u as usize];
            let right = &units[v as usize];
            let overlap = left.geometry.intersection(&right.geometry).unsigned_area();
            validate_nonnegative("overlap area", overlap)?;
            if overlap > GEOMETRY_EPS {
                return Err(geometry_error(format!(
                    "edge ({u}, {v}) geometries overlap by area {overlap}"
                )));
            }

            let shared = shared_boundary_length(&left.geometry, &right.geometry);
            validate_nonnegative("shared boundary length", shared)?;
            if shared <= GEOMETRY_EPS {
                return Err(geometry_error(format!(
                    "edge ({u}, {v}) geometries have no shared boundary"
                )));
            }
            shared_perimeters.push(shared);
            shared_by_node[u as usize] += shared;
            shared_by_node[v as usize] += shared;
        }

        for (node, unit) in units.iter().enumerate() {
            if shared_by_node[node] > unit.perimeter + GEOMETRY_EPS {
                return Err(geometry_error(format!(
                    "shared boundaries for node {node} exceed its perimeter"
                )));
            }
        }

        Self::new(
            units.iter().map(|unit| unit.area).collect(),
            units.iter().map(|unit| unit.perimeter).collect(),
            edges,
            shared_perimeters,
        )
    }
}

pub(crate) fn decode_polygon(row: usize, bytes: &[u8]) -> Result<MultiPolygon<f64>> {
    let polygons = match parse_wkb_envelope(row, bytes)? {
        WkbEnvelope::Polygon => match decode_wkb_geometry(row, bytes)? {
            Geometry::Polygon(polygon) => vec![polygon],
            _ => {
                return Err(geometry_error(format!(
                    "WKB row {row} is not a Polygon or MultiPolygon"
                )));
            }
        },
        WkbEnvelope::MultiPolygon(ranges) => {
            let mut polygons = Vec::with_capacity(ranges.len());
            for range in ranges {
                match decode_wkb_geometry(row, &bytes[range])? {
                    Geometry::Polygon(polygon) => polygons.push(polygon),
                    _ => {
                        return Err(geometry_error(format!(
                            "WKB row {row} MultiPolygon contains a non-Polygon geometry"
                        )));
                    }
                }
            }
            polygons
        }
    };
    validate_geometry(row, MultiPolygon(polygons))
}

fn decode_state_polygon(bytes: &[u8]) -> Result<MultiPolygon<f64>> {
    let state = decode_polygon(0, bytes)
        .map_err(|error| geometry_error(format!("invalid state geometry: {error}")))?;
    validate_positive("state geometry area", state.unsigned_area())?;
    Ok(state)
}

fn validate_state_coverage(
    geometries: &[MultiPolygon<f64>],
    state: &MultiPolygon<f64>,
) -> Result<()> {
    let unit_union = unary_union(geometries);
    let area = unit_union.unsigned_area();
    validate_positive("combined geometry area", area)?;
    let uncovered_area = unit_union.difference(state).unsigned_area();
    validate_nonnegative("uncovered geometry area", uncovered_area)?;
    // Permit only the area-scaled residue expected when independent overlay engines snap edges.
    let tolerance = STATE_COVERAGE_REL_EPS * area.max(1.0);
    if uncovered_area <= tolerance {
        return Ok(());
    }

    // Unary union can add a small overlay residue around holes. Confirm a material gap against
    // individual units only on this slow error path.
    for (unit, geometry) in geometries.iter().enumerate() {
        let area = geometry.unsigned_area();
        let uncovered_area = geometry.difference(state).unsigned_area();
        validate_nonnegative("uncovered geometry area", uncovered_area)?;
        let tolerance = STATE_COVERAGE_REL_EPS * area.max(1.0);
        if uncovered_area > tolerance {
            return Err(Error::StateGeometryCoverage {
                unit,
                uncovered_area,
                tolerance,
            });
        }
    }
    Ok(())
}

fn decode_wkb_geometry(row: usize, bytes: &[u8]) -> Result<Geometry<f64>> {
    let wkb = wkb::reader::read_wkb(bytes)
        .map_err(|error| geometry_error(format!("WKB row {row} cannot be decoded: {error}")))?;
    wkb.try_to_geometry()
        .ok_or_else(|| geometry_error(format!("WKB row {row} is empty")))
}

fn validate_geometry(row: usize, geometry: MultiPolygon<f64>) -> Result<MultiPolygon<f64>> {
    geometry
        .check_validation()
        .map_err(|error| geometry_error(format!("WKB row {row} is invalid: {error}")))?;
    Ok(geometry)
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Big,
    Little,
}

/// Structural envelope of one WKB geometry, parsed and length-checked exactly once.
enum WkbEnvelope {
    /// The whole buffer is one Polygon.
    Polygon,
    /// Byte ranges of the nested polygons, each carrying its own header and byte order.
    MultiPolygon(Vec<std::ops::Range<usize>>),
}

fn parse_wkb_envelope(row: usize, bytes: &[u8]) -> Result<WkbEnvelope> {
    let (order, code) = wkb_header(bytes).ok_or_else(|| {
        geometry_error(format!("WKB row {row} cannot be decoded: truncated header"))
    })?;
    let invalid_lengths =
        || geometry_error(format!("WKB row {row} cannot be decoded: invalid lengths"));

    match code & 0x7 {
        3 => {
            if polygon_wkb_size(bytes).ok_or_else(invalid_lengths)? != bytes.len() {
                return Err(invalid_lengths());
            }
            Ok(WkbEnvelope::Polygon)
        }
        6 => {
            let count_offset = 5 + usize::from(code & 0x2000_0000 != 0) * 4;
            let polygon_count =
                read_u32(bytes, count_offset, order).ok_or_else(invalid_lengths)? as usize;
            let mut offset = count_offset + 4;

            // Every nested polygon needs a five-byte header and a four-byte ring count.
            if polygon_count > bytes.len().saturating_sub(offset) / 9 {
                return Err(invalid_lengths());
            }
            let mut polygons = Vec::with_capacity(polygon_count);
            for _ in 0..polygon_count {
                let size = bytes
                    .get(offset..)
                    .and_then(polygon_wkb_size)
                    .ok_or_else(invalid_lengths)?;
                polygons.push(offset..offset + size);
                offset += size;
            }
            if offset != bytes.len() {
                return Err(invalid_lengths());
            }
            Ok(WkbEnvelope::MultiPolygon(polygons))
        }
        _ => Err(geometry_error(format!(
            "WKB row {row} is not a Polygon or MultiPolygon"
        ))),
    }
}

fn polygon_wkb_size(bytes: &[u8]) -> Option<usize> {
    let (order, code) = wkb_header(bytes)?;
    let dimension = match (code & 0x8000_0000 != 0, code & 0x4000_0000 != 0) {
        (true, true) => 4,
        (true, false) | (false, true) => 3,
        (false, false) => match code / 1_000 {
            1 | 2 => 3,
            3 => 4,
            _ => 2,
        },
    };
    let count_offset = 5 + usize::from(code & 0x2000_0000 != 0) * 4;
    let ring_count = read_u32(bytes, count_offset, order)? as usize;
    let mut offset = count_offset.checked_add(4)?;

    // Every ring needs at least its four-byte point count.
    if ring_count > bytes.len().saturating_sub(offset) / 4 {
        return None;
    }
    for _ in 0..ring_count {
        let point_count = read_u32(bytes, offset, order)? as usize;
        let coordinate_bytes = point_count.checked_mul(dimension)?.checked_mul(8)?;
        offset = offset.checked_add(4)?.checked_add(coordinate_bytes)?;
        if offset > bytes.len() {
            return None;
        }
    }
    Some(offset)
}

fn wkb_header(bytes: &[u8]) -> Option<(ByteOrder, u32)> {
    let order = match *bytes.first()? {
        0 => ByteOrder::Big,
        1 => ByteOrder::Little,
        _ => return None,
    };
    Some((order, read_u32(bytes, 1, order)?))
}

fn read_u32(bytes: &[u8], offset: usize, order: ByteOrder) -> Option<u32> {
    let value: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(match order {
        ByteOrder::Big => u32::from_be_bytes(value),
        ByteOrder::Little => u32::from_le_bytes(value),
    })
}

pub(crate) fn unit_hull(row: usize, bytes: &[u8]) -> Result<UnitHull> {
    let geometry = decode_polygon(row, bytes)?;
    unit_hull_from_geometry(row, &geometry)
}

pub(crate) fn unit_hull_from_geometry(
    row: usize,
    geometry: &MultiPolygon<f64>,
) -> Result<UnitHull> {
    let area = geometry.unsigned_area();
    validate_positive("geometry area", area)?;
    let hull = geometry.convex_hull();
    let mut points = hull
        .exterior()
        .points()
        .map(|point| Coordinate {
            x: point.x(),
            y: point.y(),
        })
        .collect::<Vec<_>>();
    if points.first() == points.last() {
        points.pop();
    }
    if points.len() < 3 {
        return Err(geometry_error(format!(
            "WKB row {row} has a convex hull with fewer than 3 distinct points"
        )));
    }
    Ok(UnitHull::new(area, points))
}

struct PolsbyPopperUnit {
    geometry: MultiPolygon<f64>,
    area: f64,
    perimeter: f64,
}

fn polsby_popper_unit(row: usize, bytes: &[u8]) -> Result<PolsbyPopperUnit> {
    let geometry = decode_polygon(row, bytes)?;
    let area = geometry.unsigned_area();
    let perimeter = multipolygon_perimeter(&geometry);
    validate_positive("geometry area", area)?;
    validate_positive("geometry perimeter", perimeter)?;
    Ok(PolsbyPopperUnit {
        geometry,
        area,
        perimeter,
    })
}

fn multipolygon_perimeter(multipolygon: &MultiPolygon<f64>) -> f64 {
    multipolygon.0.iter().map(polygon_perimeter).sum()
}

fn polygon_perimeter(polygon: &Polygon<f64>) -> f64 {
    ring_perimeter(&polygon.exterior().0)
        + polygon
            .interiors()
            .iter()
            .map(|ring| ring_perimeter(&ring.0))
            .sum::<f64>()
}

fn ring_perimeter(coords: &[Coord<f64>]) -> f64 {
    coords
        .windows(2)
        .map(|pair| coordinate_distance(pair[0], pair[1]))
        .sum()
}

fn coordinate_distance(left: Coord<f64>, right: Coord<f64>) -> f64 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn boundary_segments(multipolygon: &MultiPolygon<f64>) -> Vec<(Coord<f64>, Coord<f64>)> {
    let mut segments = Vec::new();
    for polygon in &multipolygon.0 {
        push_ring_segments(&polygon.exterior().0, &mut segments);
        for ring in polygon.interiors() {
            push_ring_segments(&ring.0, &mut segments);
        }
    }
    segments
}

fn push_ring_segments(coords: &[Coord<f64>], segments: &mut Vec<(Coord<f64>, Coord<f64>)>) {
    segments.extend(
        coords
            .windows(2)
            .filter(|pair| coordinate_distance(pair[0], pair[1]) > 0.0)
            .map(|pair| (pair[0], pair[1])),
    );
}

fn shared_boundary_length(left: &MultiPolygon<f64>, right: &MultiPolygon<f64>) -> f64 {
    let left = boundary_segments(left);
    let right = boundary_segments(right);

    left.iter()
        .flat_map(|&(a0, a1)| {
            right
                .iter()
                .map(move |&(b0, b1)| shared_segment_length(a0, a1, b0, b1))
        })
        .sum()
}

fn shared_segment_length(a0: Coord<f64>, a1: Coord<f64>, b0: Coord<f64>, b1: Coord<f64>) -> f64 {
    let axis = Coord {
        x: a1.x - a0.x,
        y: a1.y - a0.y,
    };
    let length = axis.x.hypot(axis.y);
    if length == 0.0 {
        return 0.0;
    }
    let cross0 = axis.x * (b0.y - a0.y) - axis.y * (b0.x - a0.x);
    let cross1 = axis.x * (b1.y - a0.y) - axis.y * (b1.x - a0.x);
    if cross0.abs() > GEOMETRY_EPS * length || cross1.abs() > GEOMETRY_EPS * length {
        return 0.0;
    }

    let unit = Coord {
        x: axis.x / length,
        y: axis.y / length,
    };
    let position = |point: Coord<f64>| (point.x - a0.x) * unit.x + (point.y - a0.y) * unit.y;
    let b0 = position(b0);
    let b1 = position(b1);
    (length.min(b0.max(b1)) - 0.0_f64.max(b0.min(b1))).max(0.0)
}

fn validate_positive(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(geometry_error(format!("computed invalid {label} {value}")));
    }
    Ok(())
}

fn validate_nonnegative(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(geometry_error(format!("computed invalid {label} {value}")));
    }
    Ok(())
}

fn geometry_error(message: impl Into<String>) -> Error {
    Error::Geometry(message.into())
}

#[cfg(test)]
#[path = "../tests/geometry.rs"]
mod tests;
