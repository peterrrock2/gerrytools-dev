from functools import cache
from gerrychain import Partition
import numpy as np
from typing import Iterable, Tuple
from .score_types import *

@cache
def _election_results(part: Partition, election_cols: Tuple[str], party: str):
    return np.array([
                        np.array([part[e].percent(party, d) for d in sorted(part.parts.keys()) if d != -1])
                        for e in election_cols
                    ])
@cache
def _election_stability(part: Partition, election_cols: Tuple[str], party: str):
    return (_election_results(part, election_cols, party) > 0.5).sum(axis=0)


def _competitive_districts(part: Partition, election_cols: Iterable[str], party: str,
                          points_within: float = 0.03) -> PlanWideScoreValue:
    results = _election_results(part, tuple(election_cols), party)
    return int(np.logical_and(results > 0.5 - points_within, results < 0.5 + points_within).sum())

def _swing_districts(part: Partition, election_cols: Iterable[str], party: str) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int(np.logical_and(stability != 0, stability != len(election_cols)).sum())

def _party_districts(part: Partition, election_cols: Iterable[str], party: str) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int((stability == len(election_cols)).sum())

def _opp_party_districts(part: Partition, election_cols: Iterable[str], party: str) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int((stability == 0).sum())

def _party_wins_by_district(part: Partition, election_cols: Iterable[str], party: str) -> DistrictWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    districts = [d for d in sorted(part.parts.keys()) if d != -1]
    return {d: int(d_wins) for d, d_wins in zip(districts, stability)}

def _seats(part: Partition, election_cols: Iterable[str], party: str) -> ElectionWideScoreValue:
    return {part[e].election.name: sum([part[e].won(party, d) for d in part.parts.keys() if d != -1]) for e in election_cols}

def _signed_proportionality(part: Partition, election_cols: Iterable[str], party: str) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].seats(party) - (part[e].percent(party) * len(part)) for e in election_cols}

def _absolute_proportionality(part: Partition, election_cols: Iterable[str], party: str) -> ElectionWideScoreValue:
    return {part[e].election.name: abs(part[e].seats(party) - (part[e].percent(party) * len(part))) for e in election_cols}

def _efficiency_gap(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].efficiency_gap() for e in election_cols}

def _mean_median(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].mean_median() for e in election_cols}

def _partisan_bias(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].partisan_bias() for e in election_cols}

def _partisan_gini(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].partisan_gini() for e in election_cols}


def _eguia_election(part: Partition, e: str, party: str, county_part: Partition, totpop_col: str) -> Numeric:
        seat_share = part[e].seats(party) / len(part.parts)
        counties = county_part.parts
        county_results = np.array([county_part[e].won(party, c) for c in counties])
        county_pops = np.array([county_part[totpop_col][c] for c in counties])
        ideal = np.dot(county_results, county_pops) / county_pops.sum()
        return seat_share - ideal


def _eguia(part: Partition, election_cols: Iterable[str], party: str, county_part: Partition, totpop_col: str) -> ElectionWideScoreValue:
    return {part[e].election.name: _eguia_election(part, e, party, county_part, totpop_col) for e in election_cols}