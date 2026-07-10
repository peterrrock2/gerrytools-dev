from collections.abc import Callable, Sequence
from pathlib import Path

DEBUG_ASSERTIONS: bool
MAX_DISTRICTS: int

class ScoringEngine:
    """Prepared metric scorer backed by the Rust extension.

    District occupancy uses dynamic storage. Assignments support up to ``MAX_DISTRICTS`` distinct
    labels, the number representable by the backend's ``u16`` encoding.
    """

    def __init__(self) -> None: ...
    def set_tally_bank(self, columns: list[list[float]]) -> None: ...
    def add_tally_projection(self, columns: list[int]) -> None: ...
    def add_eguia(self, party: int, opposition: int, benchmark: float) -> None: ...
    def add_paired_derived(
        self,
        kind: str,
        party: int,
        opposition: int,
        turnout_model: str,
    ) -> None: ...
    def add_population_derived(
        self,
        kind: str,
        population: int,
        relative: bool,
    ) -> None: ...
    def add_demographic_derived(
        self,
        kind: str,
        subgroup: int,
        total: int,
        threshold: float,
    ) -> None: ...
    def add_cross_election_derived(
        self,
        kind: str,
        party: list[int],
        opposition: list[int],
        points_within: float,
    ) -> None: ...
    def add_reock(self, rows: Sequence[bytes]) -> None: ...
    def add_population_polygon(
        self,
        rows: Sequence[bytes],
        population_rows: Sequence[bytes],
        weights: list[float],
        owners: list[int],
    ) -> None: ...
    def add_population_polygon_aligned(
        self,
        rows: Sequence[bytes],
        weights: list[float],
    ) -> None: ...
    def add_convex_hull_ratio(self, rows: Sequence[bytes]) -> None: ...
    def add_state_clipped_convex_hull_ratio(self, rows: Sequence[bytes], state: bytes) -> None: ...
    def add_polsby_popper_geometry(
        self, rows: Sequence[bytes], edges: list[tuple[int, int]]
    ) -> None: ...
    def add_polsby_popper_graph_total(
        self,
        areas: list[float],
        total_perimeters: list[float],
        edges: list[tuple[int, int]],
        shared_perimeters: list[float],
    ) -> None: ...
    def add_polsby_popper_graph_boundary(
        self,
        areas: list[float],
        boundary_perimeters: list[float],
        edges: list[tuple[int, int]],
        shared_perimeters: list[float],
    ) -> None: ...
    def add_schwartzberg_geometry(
        self, rows: Sequence[bytes], edges: list[tuple[int, int]]
    ) -> None: ...
    def add_schwartzberg_graph_total(
        self,
        areas: list[float],
        total_perimeters: list[float],
        edges: list[tuple[int, int]],
        shared_perimeters: list[float],
    ) -> None: ...
    def add_schwartzberg_graph_boundary(
        self,
        areas: list[float],
        boundary_perimeters: list[float],
        edges: list[tuple[int, int]],
        shared_perimeters: list[float],
    ) -> None: ...
    def add_area_perimeter_metrics_geometry(
        self, rows: Sequence[bytes], edges: list[tuple[int, int]]
    ) -> None: ...
    def add_area_perimeter_metrics_graph_total(
        self,
        areas: list[float],
        total_perimeters: list[float],
        edges: list[tuple[int, int]],
        shared_perimeters: list[float],
    ) -> None: ...
    def add_area_perimeter_metrics_graph_boundary(
        self,
        areas: list[float],
        boundary_perimeters: list[float],
        edges: list[tuple[int, int]],
        shared_perimeters: list[float],
    ) -> None: ...
    def add_cut_edges(
        self,
        node_count: int,
        edges: list[tuple[int, int]],
        weights: list[float] | None,
    ) -> None: ...
    def add_region_splits(self, columns: list[list[int | None]]) -> None: ...
    def add_region_pieces(self, columns: list[list[int | None]]) -> None: ...
    def add_region_parts(
        self,
        columns: list[list[int | None]],
        edges: list[tuple[int, int]],
    ) -> None: ...
    def add_tally_by_region(
        self,
        regions: list[int | None],
        include_count: bool,
        values: list[list[float]],
    ) -> None: ...
    def score_many(
        self,
        assignments: list[list[int]],
        track_uniqueness: bool = False,
        progress: Callable[[int], object] | None = None,
    ) -> tuple[list[int], list[list[list[float]]], tuple[int, int] | None]: ...
    def score_run(
        self,
        source_path: Path,
        output_path: Path,
        metadata_json: str,
        stream_options: tuple[int | None, int, bool, Callable[[int], object] | None],
        projections: list[tuple[int, list[int]]],
    ) -> None: ...
