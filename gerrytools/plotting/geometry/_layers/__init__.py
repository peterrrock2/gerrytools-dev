"""Internal layer adapters for the geometry plot classes.

Each layer is a frozen dataclass implementing the `_GeoLayer` ABC. They are
private (leading underscore) and are re-exported by `geoplot.py` and
`coloredgeoplot.py` to preserve existing import paths used by tests.
"""

from gerrytools.plotting.geometry._layers._base import _as_geoseries, _GeoLayer
from gerrytools.plotting.geometry._layers._categorical import _CategoricalColorLayer
from gerrytools.plotting.geometry._layers._continuous import _ContinuousColorLayer
from gerrytools.plotting.geometry._layers._marker import _MarkerLayer

__all__ = [
    "_as_geoseries",
    "_GeoLayer",
    "_CategoricalColorLayer",
    "_ContinuousColorLayer",
    "_MarkerLayer",
]
