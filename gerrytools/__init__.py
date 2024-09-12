"""
``gerrytools`` is a light package for redistricting analysis essentials,
making geographic data operations, map visualizations, plan
evaluation, and data retrieval simple.
"""

import geopandas

geopandas.options.use_pygeos = False


__version__ = "1.2.0"


__all__ = [
    "__version__",
]
