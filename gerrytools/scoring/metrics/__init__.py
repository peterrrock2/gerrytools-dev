"""Metric descriptions registered with :class:`PlanEvaluator`."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

import numpy as np

from ..result import _Dtype, is_valid_metric_name
from ._base import (
    _KeyedMetric,
    _merged_keys,
    _MetricBase,
    _OutputSpec,
    _ResourceSpec,
)
from ._geometry import (
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

if TYPE_CHECKING:
    from gerrytools._scoring_engine import ScoringEngine

    from ..evaluator import PlanEvaluator


@dataclass(frozen=True, slots=True, init=False)
class Tally(_KeyedMetric):
    """Sum one or more numeric graph columns by district.

    Multiple registrations merge so the scoring engine reads every requested graph column in one
    pass over each assignment.
    """

    _kind: ClassVar[str] = "tally"

    def _default_name(self) -> str:
        """Name a single-column tally after its column when path-safe."""
        if len(self.keys) == 1 and is_valid_metric_name(self.keys[0]):
            return self.keys[0]
        return self._kind

    def _merge(self, other: _MetricBase) -> _MetricBase | None:
        if not isinstance(other, Tally):
            return None
        return Tally(*_merged_keys(self.keys, other.keys))

    def _columns(self, evaluator: PlanEvaluator) -> list[list[float]]:
        return [evaluator._numeric_node_column(key) for key in self.keys]

    def _tally_keys(self) -> tuple[str, ...]:
        return self.keys

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._columns(evaluator)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        backend.add_tally_projection(evaluator._tally_column_indices(self.keys))
        return _OutputSpec("district", self.keys, ("float",) * len(self.keys))


@dataclass(frozen=True, slots=True)
class Eguia(_MetricBase):
    r"""Score district seat share against a population-weighted regional benchmark.

    The benchmark is the share of the population living in fixed regions where ``party_votes``
    strictly exceeds ``opposition_votes``. If :math:`S` is the party's district seat share and
    :math:`J` is that regional share, the result is :math:`S-J`. District and region ties are wins
    for neither party.

    Regional vote and population totals are fixed by the input geography, so they are computed
    once during evaluator preparation. Only the two district vote tallies are updated as plans
    change, and those columns share the evaluator's engine tally bank with other metrics.

    Args:
        party_votes: Column containing nonnegative party votes.
        opposition_votes: Column containing nonnegative opposition votes.
        region: Column containing complete, nonmissing fixed-region labels.
        population: Column containing nonnegative population with positive total.

    References:
        - Eguia, "A Measure of Partisan Advantage in Redistricting," Election Law Journal 21
          (2022), 84-103. https://doi.org/10.1089/elj.2020.0691
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
          Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """

    _kind: ClassVar[str] = "eguia"
    party_votes: str
    opposition_votes: str
    region: str
    population: str

    def __post_init__(self) -> None:
        for name, value in (
            ("party_votes", self.party_votes),
            ("opposition_votes", self.opposition_votes),
            ("region", self.region),
            ("population", self.population),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Eguia {name} must be a nonempty column name")

    def _tally_keys(self) -> tuple[str, ...]:
        return self.party_votes, self.opposition_votes

    def _benchmark(self, evaluator: PlanEvaluator) -> float:
        return evaluator._fixed_value(("eguia", self), lambda: self._compute_benchmark(evaluator))

    def _compute_benchmark(self, evaluator: PlanEvaluator) -> float:
        party = evaluator._nonnegative_node_column(self.party_votes, "Eguia")
        opposition = evaluator._nonnegative_node_column(self.opposition_votes, "Eguia")
        population = evaluator._nonnegative_node_column(self.population, "Eguia")
        for key, values in (
            (self.party_votes, party),
            (self.opposition_votes, opposition),
            (self.population, population),
        ):
            if not math.isfinite(sum(values)):
                raise ValueError(f"Eguia column {key!r} must have a finite total")
        regions, labels = evaluator._region_column(self.region)
        if any(region is None for region in regions):
            raise ValueError("Eguia region labels cannot be missing")

        region_party = [0.0] * len(labels)
        region_opposition = [0.0] * len(labels)
        region_population = [0.0] * len(labels)
        for region, party_value, opposition_value, population_value in zip(
            regions,
            party,
            opposition,
            population,
            strict=True,
        ):
            assert region is not None
            region_party[region] += party_value
            region_opposition[region] += opposition_value
            region_population[region] += population_value

        total_population = sum(region_population)
        if total_population <= 0:
            raise ValueError("Eguia population must have positive total")
        winning_population = sum(
            population_value
            for party_value, opposition_value, population_value in zip(
                region_party,
                region_opposition,
                region_population,
                strict=True,
            )
            if party_value > opposition_value
        )
        return winning_population / total_population

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._benchmark(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        columns = (self.party_votes, self.opposition_votes, self.population)
        return _ResourceSpec(
            node_columns=frozenset(
                evaluator._ordinary_column_resource(column) for column in columns
            ),
            region_columns=frozenset((evaluator._ordinary_column_resource(self.region),)),
            fixed_values=frozenset((("eguia", self),)),
        )

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        party, opposition = evaluator._tally_column_indices(self._tally_keys())
        backend.add_eguia(party, opposition, self._benchmark(evaluator))
        return _OutputSpec("plan", ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {
            "party_votes": self.party_votes,
            "opposition_votes": self.opposition_votes,
            "region": self.region,
            "population": self.population,
        }


def _column_name(value: object, metric: str, parameter: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{metric} {parameter} must be a nonempty column name")
    return value


def _column_sequence(value: object, metric: str, parameter: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{metric} {parameter} must be a sequence of column names")
    try:
        columns = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{metric} {parameter} must be a sequence of column names") from None
    if not columns:
        raise ValueError(f"{metric} {parameter} cannot be empty")
    for column in columns:
        _column_name(column, metric, parameter)
    return columns


@dataclass(frozen=True, slots=True)
class _PairedTallyMetric(_MetricBase):
    party_votes: str
    opposition_votes: str
    _shape: ClassVar[Literal["district", "plan"]]
    _dtype: ClassVar[_Dtype]

    def _resolved_turnout_model(self) -> Literal["equal", "observed"]:
        return getattr(self, "turnout_model", "equal")

    def __post_init__(self) -> None:
        metric = type(self).__name__
        _column_name(self.party_votes, metric, "party_votes")
        _column_name(self.opposition_votes, metric, "opposition_votes")
        if self._resolved_turnout_model() not in {"equal", "observed"}:
            raise ValueError(f"{metric} turnout_model must be 'equal' or 'observed'")

    def _tally_keys(self) -> tuple[str, ...]:
        return self.party_votes, self.opposition_votes

    def _validate(self, evaluator: PlanEvaluator) -> None:
        metric = type(self).__name__
        evaluator._nonnegative_node_column(self.party_votes, metric)
        evaluator._nonnegative_node_column(self.opposition_votes, metric)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        party, opposition = evaluator._tally_column_indices(self._tally_keys())
        backend.add_paired_derived(
            self._kind,
            party,
            opposition,
            self._resolved_turnout_model(),
        )
        return _OutputSpec(self._shape, ("score",), (self._dtype,))

    def _options(self) -> dict[str, object]:
        return {
            "party_votes": self.party_votes,
            "opposition_votes": self.opposition_votes,
        }


@dataclass(frozen=True, slots=True)
class _TurnoutPairedTallyMetric(_PairedTallyMetric):
    """Paired-tally base for metrics that expose ``turnout_model`` as a public option."""

    turnout_model: Literal["equal", "observed"] = "equal"

    def _options(self) -> dict[str, object]:
        return {**_PairedTallyMetric._options(self), "turnout_model": self.turnout_model}


@dataclass(frozen=True, slots=True)
class DistrictVoteShares(_PairedTallyMetric):
    r"""Return the party's two-party vote share in each district.

    A zero-turnout district is reported as ``NaN``. See
    :func:`gerrytools.scoring.formulas.district_vote_shares` for the exact formula.
    """

    _kind: ClassVar[str] = "district_vote_shares"
    _shape: ClassVar[Literal["district", "plan"]] = "district"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class DistrictWins(_PairedTallyMetric):
    """Identify districts strictly won by the party; ties are not wins."""

    _kind: ClassVar[str] = "district_wins"
    _shape: ClassVar[Literal["district", "plan"]] = "district"
    _dtype: ClassVar[_Dtype] = "bool"


@dataclass(frozen=True, slots=True)
class Seats(_PairedTallyMetric):
    """Count districts strictly won by the party under the two-party vote."""

    _kind: ClassVar[str] = "seats"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class OverallVoteShare(_PairedTallyMetric):
    """Return the party's turnout-weighted aggregate two-party vote share."""

    _kind: ClassVar[str] = "overall_vote_share"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class EfficiencyGap(_PairedTallyMetric):
    r"""Return the wasted-vote efficiency gap from the party's point of view.

    Positive values favor the party. The winning threshold and provisional tie convention exactly
    match :func:`gerrytools.scoring.formulas.efficiency_gap`.

    References:
        - Bernstein and Duchin, "A Formula Goes to Court: Partisan Gerrymandering and the
          Efficiency Gap." https://arxiv.org/abs/1705.10812
        - Stephanopoulos and McGhee, "Partisan Gerrymandering and the Efficiency Gap."
          https://lawreview.uchicago.edu/online-archive/partisan-gerrymandering-and-efficiency-gap
    """

    _kind: ClassVar[str] = "efficiency_gap"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class SimplifiedEfficiencyGap(_PairedTallyMetric):
    r"""Return :math:`S-2V+1/2`, the equal-turnout seat-vote efficiency-gap formula.

    See :class:`EfficiencyGap` when the original wasted-vote definition is required.
    """

    _kind: ClassVar[str] = "simplified_efficiency_gap"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class MeanMedian(_PairedTallyMetric):
    r"""Return median district vote share minus mean district vote share.

    The mean weights districts equally. A zero-turnout district makes the score ``NaN``.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes."
          https://doi.org/10.1017/pan.2021.49
        - Grofman, "Measures of Bias and Proportionality in Seats-Votes Relationships."
          https://www.jstor.org/stable/25791195
    """

    _kind: ClassVar[str] = "mean_median"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class PartisanBias(_TurnoutPairedTallyMetric):
    r"""Return partisan bias at 50 percent under uniform partisan swing.

    Districts within numerical tolerance of the reference share contribute half a seat.
    ``turnout_model="equal"`` gives valid districts equal weight when locating the observed
    election; ``"observed"`` uses aggregate two-party turnout.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes."
          https://doi.org/10.1017/pan.2021.49
        - Katz, King, and Rosenblatt, "Theoretical Foundations and Empirical Evaluations of
          Partisan Fairness." https://doi.org/10.1017/S000305541900056X
    """

    _kind: ClassVar[str] = "partisan_bias"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class PartisanGini(_TurnoutPairedTallyMetric):
    r"""Return unsigned partisan Gini under uniform partisan swing.

    ``turnout_model`` has the same meaning as in :class:`PartisanBias`. A zero-turnout district
    makes the score ``NaN``.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes."
          https://doi.org/10.1017/pan.2021.49
        - Grofman, "Measures of Bias and Proportionality in Seats-Votes Relationships."
          https://www.jstor.org/stable/25791195
        - Katz, King, and Rosenblatt, "Theoretical Foundations and Empirical Evaluations of
          Partisan Fairness." https://doi.org/10.1017/S000305541900056X
    """

    _kind: ClassVar[str] = "partisan_gini"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class _PopulationMetric(_MetricBase):
    population: str
    _shape: ClassVar[Literal["district", "plan"]]

    def _resolved_relative(self) -> bool:
        return getattr(self, "relative", False)

    def __post_init__(self) -> None:
        _column_name(self.population, type(self).__name__, "population")
        if not isinstance(self._resolved_relative(), bool):
            raise TypeError(f"{type(self).__name__} relative must be a bool")

    def _tally_keys(self) -> tuple[str, ...]:
        return (self.population,)

    def _validate(self, evaluator: PlanEvaluator) -> None:
        values = evaluator._nonnegative_node_column(self.population, type(self).__name__)
        if sum(values) <= 0:
            raise ValueError(f"{type(self).__name__} population must have a positive total")

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        (population,) = evaluator._tally_column_indices(self._tally_keys())
        backend.add_population_derived(self._kind, population, self._resolved_relative())
        return _OutputSpec(self._shape, ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {"population": self.population}


@dataclass(frozen=True, slots=True)
class _RelativePopulationMetric(_PopulationMetric):
    """Population base for metrics that expose ``relative`` as a public option."""

    relative: bool = False

    def _options(self) -> dict[str, object]:
        return {**_PopulationMetric._options(self), "relative": self.relative}


@dataclass(frozen=True, slots=True)
class PopulationDeviations(_PopulationMetric):
    r"""Return each district's signed proportional deviation from ideal population.

    References:
        - Evenwel v. Abbott, 578 U.S. 54 (2016).
          https://www.govinfo.gov/app/details/USREPORTS-578
        - Duchin and Walch, eds., *Political Geometry*.
          https://doi.org/10.1007/978-3-319-69161-9
    """

    _kind: ClassVar[str] = "population_deviations"
    _shape: ClassVar[Literal["district", "plan"]] = "district"


@dataclass(frozen=True, slots=True)
class MaxAbsolutePopulationDeviation(_RelativePopulationMetric):
    """Return the largest one-district absolute departure from ideal population."""

    _kind: ClassVar[str] = "max_absolute_population_deviation"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"


@dataclass(frozen=True, slots=True)
class MaxPopulationDeviation(_RelativePopulationMetric):
    r"""Return the plan's top-to-bottom population range.

    With ``relative=True``, this is the conventional maximum population deviation described in
    Evenwel v. Abbott, 578 U.S. 54, 60 n.2 (2016).
    https://www.govinfo.gov/app/details/USREPORTS-578
    """

    _kind: ClassVar[str] = "max_population_deviation"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"


@dataclass(frozen=True, slots=True)
class _DemographicMetric(_MetricBase):
    subgroup: str
    total: str
    _shape: ClassVar[Literal["district", "plan"]]
    _dtype: ClassVar[_Dtype]

    def _resolved_threshold(self) -> float:
        return getattr(self, "threshold", 0.5)

    def __post_init__(self) -> None:
        metric = type(self).__name__
        _column_name(self.subgroup, metric, "subgroup")
        _column_name(self.total, metric, "total")
        threshold = self._resolved_threshold()
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise ValueError(f"{metric} threshold must be finite and between zero and one")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
            raise ValueError(f"{metric} threshold must be finite and between zero and one")

    def _tally_keys(self) -> tuple[str, ...]:
        return self.subgroup, self.total

    def _validate(self, evaluator: PlanEvaluator) -> None:
        metric = type(self).__name__
        subgroup = evaluator._nonnegative_node_column(self.subgroup, metric)
        total = evaluator._nonnegative_node_column(self.total, metric)
        if any(
            subgroup_value > total_value for subgroup_value, total_value in zip(subgroup, total)
        ):
            raise ValueError(f"{metric} subgroup cannot exceed total")

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        subgroup, total = evaluator._tally_column_indices(self._tally_keys())
        backend.add_demographic_derived(
            self._kind,
            subgroup,
            total,
            float(self._resolved_threshold()),
        )
        return _OutputSpec(self._shape, ("score",), (self._dtype,))

    def _options(self) -> dict[str, object]:
        return {"subgroup": self.subgroup, "total": self.total}


@dataclass(frozen=True, slots=True)
class _ThresholdDemographicMetric(_DemographicMetric):
    """Demographic base for metrics that expose ``threshold`` as a public option."""

    threshold: float | np.floating = 0.5

    def _options(self) -> dict[str, object]:
        return {**_DemographicMetric._options(self), "threshold": float(self.threshold)}


@dataclass(frozen=True, slots=True)
class DemographicShares(_DemographicMetric):
    """Return a subgroup's share of the specified total in each district."""

    _kind: ClassVar[str] = "demographic_shares"
    _shape: ClassVar[Literal["district", "plan"]] = "district"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class DistrictsAboveThreshold(_ThresholdDemographicMetric):
    """Count districts whose subgroup share is strictly greater than ``threshold``."""

    _kind: ClassVar[str] = "districts_above_threshold"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class _CrossElectionMetric(_MetricBase):
    party_votes: tuple[str, ...]
    opposition_votes: tuple[str, ...]
    _shape: ClassVar[Literal["district", "plan"]]
    _dtype: ClassVar[_Dtype]

    def _resolved_points_within(self) -> float:
        return getattr(self, "points_within", 0.03)

    def __post_init__(self) -> None:
        metric = type(self).__name__
        party = _column_sequence(self.party_votes, metric, "party_votes")
        opposition = _column_sequence(self.opposition_votes, metric, "opposition_votes")
        if len(party) != len(opposition):
            raise ValueError(f"{metric} party_votes and opposition_votes must have equal length")
        object.__setattr__(self, "party_votes", party)
        object.__setattr__(self, "opposition_votes", opposition)
        points_within = self._resolved_points_within()
        if isinstance(points_within, bool) or not isinstance(points_within, numbers.Real):
            raise ValueError(f"{metric} points_within must be finite and between zero and one half")
        points_value = float(points_within)
        if not math.isfinite(points_value) or not 0 <= points_value <= 0.5:
            raise ValueError(f"{metric} points_within must be finite and between zero and one half")

    def _tally_keys(self) -> tuple[str, ...]:
        return _merged_keys(self.party_votes, self.opposition_votes)

    def _validate(self, evaluator: PlanEvaluator) -> None:
        metric = type(self).__name__
        for column in self._tally_keys():
            evaluator._nonnegative_node_column(column, metric)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        party = evaluator._tally_column_indices(self.party_votes)
        opposition = evaluator._tally_column_indices(self.opposition_votes)
        backend.add_cross_election_derived(
            self._kind,
            party,
            opposition,
            float(self._resolved_points_within()),
        )
        return _OutputSpec(self._shape, ("score",), (self._dtype,))

    def _options(self) -> dict[str, object]:
        return {
            "party_votes": self.party_votes,
            "opposition_votes": self.opposition_votes,
        }


@dataclass(frozen=True, slots=True)
class _PointsCrossElectionMetric(_CrossElectionMetric):
    """Cross-election base for metrics that expose ``points_within`` as a public option."""

    points_within: float | np.floating = 0.03

    def _options(self) -> dict[str, object]:
        return {**_CrossElectionMetric._options(self), "points_within": float(self.points_within)}


@dataclass(frozen=True, slots=True)
class CompetitiveContests(_PointsCrossElectionMetric):
    """Count election-district contests in an open interval around 50 percent."""

    _kind: ClassVar[str] = "competitive_contests"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class PartyWinsByDistrict(_CrossElectionMetric):
    """Count strict party wins in each district across the supplied elections."""

    _kind: ClassVar[str] = "party_wins_by_district"
    _shape: ClassVar[Literal["district", "plan"]] = "district"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class SwingDistricts(_CrossElectionMetric):
    """Count districts that are not strict wins for one side in every election."""

    _kind: ClassVar[str] = "swing_districts"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class PartyDistricts(_CrossElectionMetric):
    """Count districts strictly won by the party in every supplied election."""

    _kind: ClassVar[str] = "party_districts"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class OppositionPartyDistricts(_CrossElectionMetric):
    """Count districts strictly won by the opposition in every supplied election."""

    _kind: ClassVar[str] = "opposition_party_districts"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class AggregateSeats(_CrossElectionMetric):
    """Count strict party wins across every supplied election and district."""

    _kind: ClassVar[str] = "aggregate_seats"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class MeanSignedSeatVoteGap(_CrossElectionMetric):
    """Average signed seat-share minus aggregate vote-share gaps over elections."""

    _kind: ClassVar[str] = "mean_signed_seat_vote_gap"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class MeanAbsoluteSeatVoteGap(_CrossElectionMetric):
    """Average absolute seat-share minus aggregate vote-share gaps over elections."""

    _kind: ClassVar[str] = "mean_absolute_seat_vote_gap"
    _shape: ClassVar[Literal["district", "plan"]] = "plan"
    _dtype: ClassVar[_Dtype] = "float"


Metric: TypeAlias = (
    AggregateSeats
    | CompetitiveContests
    | ConvexHullRatio
    | CutEdges
    | DemographicShares
    | DistrictVoteShares
    | DistrictWins
    | DistrictsAboveThreshold
    | EfficiencyGap
    | Eguia
    | MaxAbsolutePopulationDeviation
    | MaxPopulationDeviation
    | MeanAbsoluteSeatVoteGap
    | MeanMedian
    | MeanSignedSeatVoteGap
    | OppositionPartyDistricts
    | OverallVoteShare
    | PartisanBias
    | PartisanGini
    | PartyDistricts
    | PartyWinsByDistrict
    | PolsbyPopper
    | PopulationDeviations
    | PopulationPolygon
    | RegionPieces
    | RegionParts
    | RegionSplits
    | Reock
    | Schwartzberg
    | Seats
    | SimplifiedEfficiencyGap
    | StateClippedConvexHullRatio
    | SwingDistricts
    | Tally
    | TallyByRegion
)
"""Union of the metric descriptions :meth:`PlanEvaluator.add_metric` accepts.

The set is closed because registering a metric requires matching scoring-engine support, so
user-defined classes cannot be scored.
"""
