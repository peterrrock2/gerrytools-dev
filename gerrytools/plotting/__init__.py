"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.colors import districtr, flare, latex, purples, redbluecmap
from gerrytools.plotting._geoplot_option_classes import ColorbarOptions
from gerrytools.plotting._gerryplot_option_classes import (
    AxisLabelStyle,
    FontFamily,
    FontStretch,
    FontStyle,
    FontVariant,
    FontWeight,
    LabelBoxOptions,
    LabelFontOptions,
    LegendOptions,
    PointMarkerOptions,
    TickStyle,
    TitleStyle,
)
from gerrytools.plotting.boxplot import BoxPlot, BoxPlotSetData
from gerrytools.plotting.coloredgeoplot import ColoredGeoPlot
from gerrytools.plotting.dotdensity import DotDensityPlot
from gerrytools.plotting.geoplot import GeoPlot
from gerrytools.plotting.gerryplot import BandData, LineData, PointSetData
from gerrytools.plotting.histogram import Histogram, HistogramData, HistPointList
from gerrytools.plotting.sealevel import SeaLevel
from gerrytools.plotting.subway import SubwaySignOptions, subway_signs
from gerrytools.plotting.violin import ViolinPlot, ViolinPlotSetData

from .scatterplot import scatterplot

__all__ = [
    "districtr",
    "flare",
    "latex",
    "purples",
    "redbluecmap",
    "ColorbarOptions",
    "AxisLabelStyle",
    "FontFamily",
    "FontStretch",
    "FontStyle",
    "FontVariant",
    "FontWeight",
    "LabelBoxOptions",
    "LabelFontOptions",
    "LegendOptions",
    "PointMarkerOptions",
    "TickStyle",
    "TitleStyle",
    "BoxPlot",
    "BoxPlotSetData",
    "ColoredGeoPlot",
    "DotDensityPlot",
    "GeoPlot",
    "BandData",
    "LineData",
    "PointSetData",
    "Histogram",
    "HistogramData",
    "HistPointList",
    "SeaLevel",
    "SubwaySignOptions",
    "subway_signs",
    "ViolinPlot",
    "ViolinPlotSetData",
    "scatterplot",
]
