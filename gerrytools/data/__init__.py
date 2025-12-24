"""
Facilities for processing data and districting plans in a standardized fashion.
"""

from .census.acs import acs5, cvap
from .census.census import census
from .districtr.legacy.fetch import Submission as LegacySubmission
from .districtr.legacy.fetch import submissions as legacy_submissions
from .districtr.legacy.fetch import tabularized as legacy_tabularized
from .geometries import dualgraphs20, geometries20, vtds20

__all__ = [
    "legacy_submissions",
    "legacy_tabularized",
    "LegacySubmission",
    "cvap",
    "acs5",
    "census",
    "vtds20",
    "dualgraphs20",
]
