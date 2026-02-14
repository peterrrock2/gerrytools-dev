"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.colors import districtr, flare, latex, purples, redbluecmap
from gerrytools.plotting._geoplot_to_mpl_option_dataclasses import ColorbarOptions
from gerrytools.plotting._gerryplot_dataclasses import BandData, LineData, PointSetData
from gerrytools.plotting._gerryplot_to_mpl_option_dataclasses import (
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
from gerrytools.plotting.histogram import Histogram, HistogramData, HistPointList
from gerrytools.plotting.scatterplot import ScatterPlot
from gerrytools.plotting.sealevel import SeaLevel
from gerrytools.plotting.subway import SubwaySignOptions, subway_signs
from gerrytools.plotting.violin import ViolinPlot, ViolinPlotSetData

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
    "ScatterPlot",
    "SeaLevel",
    "SubwaySignOptions",
    "subway_signs",
    "ViolinPlot",
    "ViolinPlotSetData",
]
