"""
Oft-used utilities for working with geometric data.
"""

from .JSON import JSONtoObject, jsonify
from .rename import rename
from .states import State
from .acs5_tables import (
    ACS5TableInfo,
    CVAPTableInfo,
    VAPTableInfo,
    TotPopTableInfo,
    HispByRaceTableInfo,
)

__all__ = [
    "jsonify",
    "JSONtoObject",
    "rename",
    "State",
    "ACS5TableInfo",
    "CVAPTableInfo",
    "VAPTableInfo",
    "TotPopTableInfo",
    "HispByRaceTableInfo",
]
