"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.colors import districtr, flare, latex, purples, redbluecmap

from .annotation import arrow, ideal
from .scatterplot import scatterplot
from .violin import violin

__all__ = [
    "arrow",
    "ideal",
    "districtr",
    "flare",
    "latex",
    "purples",
    "redbluecmap",
    "histogram",
    "scatterplot",
    "violin",
]
