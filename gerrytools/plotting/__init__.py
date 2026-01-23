"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.colors import districtr, flare, latex, purples, redbluecmap
from gerrytools.plotting.boxplot import BoxPlot, BoxPlotSetData
from gerrytools.plotting.geoplot import (
    ColorbarOptions,
    FontFamily,
    FontStretch,
    FontStyle,
    FontVariant,
    FontWeight,
    GeoPlot,
    GeoSource,
    LabelBoxOptions,
    LabelFontOptions,
)
from gerrytools.plotting.gerryplot import (
    AxisLabelStyle,
    BandData,
    LegendOptions,
    LineData,
    PointMarkerOptions,
    PointSetData,
    TickStyle,
    TitleStyle,
)
from gerrytools.plotting.histogram import Histogram, HistogramData, HistPointList
from gerrytools.plotting.sealevel import SeaLevel
from gerrytools.plotting.subway import SubwaySignOptions, subway_signs
from gerrytools.plotting.violin import ViolinPlot, ViolinPlotSetData

from .annotation import arrow, ideal
from .scatterplot import scatterplot

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
    "BoxPlot",
    "BoxPlotSetData",
    "GeoPlot",
    "GeoSource",
    "FontWeight",
    "FontStyle",
    "FontVariant",
    "FontStretch",
    "FontFamily",
    "LabelBoxOptions",
    "LabelFontOptions",
    "ColorbarOptions",
    "PointMarkerOptions",
    "PointSetData",
    "LineData",
    "BandData",
    "TickStyle",
    "LegendOptions",
    "AxisLabelStyle",
    "TitleStyle",
    "HistPointList",
    "HistogramData",
    "Histogram",
    "SeaLevel",
    "SubwaySignOptions",
    "subway_signs",
    "ViolinPlotSetData",
    "ViolinPlot",
]
