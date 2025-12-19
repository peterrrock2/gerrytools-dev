"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.colors import districtr, flare, latex, purples, redbluecmap

from .annotation import arrow, ideal
from .bins import bins
from .scatterplot import scatterplot
from .sealevel import sealevel
from .violin import violin

__all__ = [
    "drawplan",
    "drawgraph",
    "redbluecmap",
    "flare",
    "purples",
    "districtr",
    "histogram",
    "violin",
    "scatterplot",
    "sealevel",
    "multidimensional",
    "gif_multidimensional",
    "arrow",
    "ideal",
    "bins",
    "districtnumbers",
    "latex",
    "choropleth",
]
