"""
``gerrytools`` is a light package for redistricting analysis essentials,
making geographic data operations, map visualizations, plan
evaluation, and data retrieval simple.
"""

import geopandas as gpd
from packaging.version import Version

if Version(gpd.__version__) < Version("1.0"):
    gpd.options.use_pygeos = False
