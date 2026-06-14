"""
Facilities for processing data and districting plans in a standardized fashion.
"""

from .geometries import dualgraphs20, geometries20, vtds20
from .uscensus import (
    ACSCVAPTableInfo,
    ACSHispByRaceTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
    CensusRateLimitError,
    PLBlockVAPTableInfo,
    PLTableInfo,
    acs,
    acs_full,
    block_cvap_estimates,
    census,
    cvap,
    pl_pop_table,
)

__all__ = [
    # Census fetch functions
    "acs",
    "acs_full",
    "cvap",
    "census",
    "block_cvap_estimates",
    # Table definitions accepted by acs()/census()
    "ACSTableInfo",
    "ACSTotPopTableInfo",
    "ACSVAPTableInfo",
    "ACSCVAPTableInfo",
    "ACSHispByRaceTableInfo",
    "PLTableInfo",
    "PLBlockVAPTableInfo",
    "pl_pop_table",
    "CensusRateLimitError",
    # Lab-processed geometry downloads
    "vtds20",
    "dualgraphs20",
    "geometries20",
]
