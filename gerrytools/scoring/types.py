from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

"""
    Typing Definitions:

    * A Score is a named tuple of a name and function that takes a `gerrychain.Partition` instance and
    returns a ScoreValue.  The function associated with the Score should be deterministic, that is
    always return the same value given the same partition.
    * A ScoreValue is either a numeric, a mapping from districts to numerics, or a mapping from
    elections to numerics.
"""

Numeric = float | int
DistrictID = int | str
ElectionID = str

PlanWideScoreValue = Numeric
DistrictWideScoreValue = Mapping[DistrictID, Numeric]
ElectionWideScoreValue = Mapping[ElectionID, Numeric]

ScoreValue = PlanWideScoreValue | DistrictWideScoreValue | ElectionWideScoreValue


@dataclass
class Score:
    name: str
    apply: Callable[..., Any]
    dissolved: bool = False
