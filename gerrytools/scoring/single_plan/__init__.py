"""Single-plan functions for scoring-engine metric descriptions.

These functions prepare a fresh :class:`PlanEvaluator` for each call. Use an evaluator directly
when evaluating several metrics or plans so graph, geometry, and engine resources are prepared
once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, cast

from geopandas import GeoDataFrame
from pandas import DataFrame, Series

from ..metrics import (
    AggregateSeats,
    CompetitiveContests,
    DemographicShares,
    DistrictsAboveThreshold,
    DistrictVoteShares,
    DistrictWins,
    EfficiencyGap,
    Eguia,
    MaxAbsolutePopulationDeviation,
    MaxPopulationDeviation,
    MeanAbsoluteSeatVoteGap,
    MeanMedian,
    MeanSignedSeatVoteGap,
    OppositionPartyDistricts,
    OverallVoteShare,
    PartisanBias,
    PartisanGini,
    PartyDistricts,
    PartyWinsByDistrict,
    PopulationDeviations,
    Seats,
    SimplifiedEfficiencyGap,
    SwingDistricts,
    Tally,
)
from ._base import (
    GeoAssignment,
    SinglePlanSource,
    _columns,
    _evaluate,
)
from ._geometry import (
    convex_hull_ratio as convex_hull_ratio,
)
from ._geometry import (
    cut_edges as cut_edges,
)
from ._geometry import (
    polsby_popper as polsby_popper,
)
from ._geometry import (
    population_polygon as population_polygon,
)
from ._geometry import (
    region_parts as region_parts,
)
from ._geometry import (
    region_pieces as region_pieces,
)
from ._geometry import (
    region_splits as region_splits,
)
from ._geometry import (
    reock as reock,
)
from ._geometry import (
    schwartzberg as schwartzberg,
)
from ._geometry import (
    state_clipped_convex_hull_ratio as state_clipped_convex_hull_ratio,
)
from ._geometry import (
    tally_by_region as tally_by_region,
)


def _election_columns(values: Sequence[str]) -> tuple[str, ...]:
    # Preserve a mistaken bare string so the metric description can reject it clearly.
    return cast("tuple[str, ...]", values) if isinstance(values, str) else tuple(values)


def tally(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    columns: str | Iterable[str],
    name: str | None = None,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> Series | DataFrame:
    """Sum one or more unit columns by district for a single plan.

    Args:
        source: GerryChain Partition, NetworkX graph, or authoritative GeoDataFrame.
        assignment: Graph assignment mapping or node-ordered sequence, or a GeoDataFrame
            assignment column, mapping, or row-ordered sequence. A Partition supplies its own
            assignment.
        columns: One column name, or several to tally in a single pass.
        name: Optional result name. A single-column tally is named after its column when that
            name is usable as a path component, and ``"tally"`` otherwise.
        geometry: Authoritative GeoDataFrame for a Partition source.
        node_column: Optional geometry column containing the Partition graph's original node
            labels.
    """
    return cast(
        "Series | DataFrame",
        _evaluate(
            source,
            assignment,
            Tally(*_columns(columns), name=name),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def eguia(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    region: str,
    population: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    r"""Calculate Eguia's regional-benchmark partisan-advantage score for one plan.

    The score is the party's district seat share minus the share of the population living in
    fixed regions where the party strictly wins the two-party vote. District and region ties are
    wins for neither party. See :class:`Eguia` for the full definition and input contract.

    Args:
        source: GerryChain Partition, NetworkX graph, or authoritative GeoDataFrame.
        assignment: Graph assignment mapping or node-ordered sequence, or a GeoDataFrame
            assignment column, mapping, or row-ordered sequence. A Partition supplies its own
            assignment.
        party_votes: Column containing nonnegative party votes.
        opposition_votes: Column containing nonnegative opposition votes.
        region: Column containing complete fixed-region labels.
        population: Column containing nonnegative population with positive total.
        geometry: Authoritative GeoDataFrame for a Partition source.
        node_column: Optional geometry column containing the Partition graph's original node labels.

    References:
        - Eguia, "A Measure of Partisan Advantage in Redistricting," Election Law Journal 21
          (2022), 84-103. https://doi.org/10.1089/elj.2020.0691
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
          Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            Eguia(
                party_votes=party_votes,
                opposition_votes=opposition_votes,
                region=region,
                population=population,
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def district_vote_shares(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> Series:
    """Return the party's two-party vote share in each district of one plan."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            DistrictVoteShares(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def district_wins(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> Series:
    """Identify districts strictly won by the party in one plan."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            DistrictWins(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def seats(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count districts strictly won by the party in one plan."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            Seats(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def overall_vote_share(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return the party's aggregate two-party vote share in one plan."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            OverallVoteShare(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def efficiency_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return the wasted-vote efficiency gap for one plan."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            EfficiencyGap(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def simplified_efficiency_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return the equal-turnout seat-vote efficiency-gap formula for one plan."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            SimplifiedEfficiencyGap(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def mean_median(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return the mean-median partisan-symmetry score for one plan."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            MeanMedian(party_votes, opposition_votes),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def partisan_bias(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    turnout_model: Literal["equal", "observed"] = "equal",
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return partisan bias at 50 percent under uniform partisan swing."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            PartisanBias(party_votes, opposition_votes, turnout_model),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def partisan_gini(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: str,
    opposition_votes: str,
    turnout_model: Literal["equal", "observed"] = "equal",
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return unsigned partisan Gini under uniform partisan swing."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            PartisanGini(party_votes, opposition_votes, turnout_model),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def population_deviations(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> Series:
    """Return each district's signed proportional deviation from ideal population."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            PopulationDeviations(population),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def max_absolute_population_deviation(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population: str,
    relative: bool = False,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return the largest one-district absolute departure from ideal population."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            MaxAbsolutePopulationDeviation(population, relative),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def max_population_deviation(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population: str,
    relative: bool = False,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Return the top-to-bottom population deviation of one plan."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            MaxPopulationDeviation(population, relative),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def demographic_shares(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    subgroup: str,
    total: str,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> Series:
    """Return a subgroup's share of the specified total in each district."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            DemographicShares(subgroup, total),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def districts_above_threshold(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    subgroup: str,
    total: str,
    threshold: float = 0.5,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count districts whose subgroup share is strictly above ``threshold``."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            DistrictsAboveThreshold(subgroup, total, threshold),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def competitive_contests(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    points_within: float = 0.03,
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count supplied election-district contests in an open interval around 50 percent."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            CompetitiveContests(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
                points_within,
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def party_wins_by_district(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> Series:
    """Count strict party wins in each district across supplied elections."""
    return cast(
        Series,
        _evaluate(
            source,
            assignment,
            PartyWinsByDistrict(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def swing_districts(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count districts that are not strict wins for one side in every election."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            SwingDistricts(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def party_districts(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count districts strictly won by the party in every supplied election."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            PartyDistricts(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def opposition_party_districts(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count districts strictly won by the opposition in every supplied election."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            OppositionPartyDistricts(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def aggregate_seats(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> int:
    """Count strict party wins across every supplied election and district."""
    return cast(
        int,
        _evaluate(
            source,
            assignment,
            AggregateSeats(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def mean_signed_seat_vote_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Average signed seat-share minus vote-share gaps over supplied elections."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            MeanSignedSeatVoteGap(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )


def mean_absolute_seat_vote_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_votes: Sequence[str],
    opposition_votes: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_column: str | None = None,
) -> float:
    """Average absolute seat-share minus vote-share gaps over supplied elections."""
    return cast(
        float,
        _evaluate(
            source,
            assignment,
            MeanAbsoluteSeatVoteGap(
                _election_columns(party_votes),
                _election_columns(opposition_votes),
            ),
            geometry=geometry,
            node_column=node_column,
        ),
    )
