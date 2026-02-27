"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.colors import districtr, flare, latex, purples, redbluecmap
from gerrytools.plotting import data, geometry, mpl
from gerrytools.plotting.data import (
    ArrowData,
    ArrowPlacement,
    ArrowTextStyle,
    BandData,
    LabelArrowStyle,
    LineData,
    PointSetData,
    TextArrowStyle,
)
from gerrytools.plotting.data.boxplot import BoxPlot, BoxPlotSetData
from gerrytools.plotting.data.histogram import Histogram, HistogramData, HistPointList
from gerrytools.plotting.data.paintball import PaintBall, PaintBallLine
from gerrytools.plotting.data.scatterplot import ScatterPlot
from gerrytools.plotting.data.sealevel import SeaLevel
from gerrytools.plotting.data.seatsvotes import SeatsVotes, SeatsVotesData, SVPlotLine
from gerrytools.plotting.data.violin import ViolinPlot, ViolinPlotSetData
from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot
from gerrytools.plotting.geometry.dotdensity import DotDensityPlot
from gerrytools.plotting.geometry.geoplot import GeoPlot
from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.geoplot_options import ColorbarOptions
from gerrytools.plotting.mpl.label_text_options import (
    FontFamily,
    FontStretch,
    FontStyle,
    FontVariant,
    FontWeight,
    LabelBoxOptions,
    LabelFontOptions,
)
from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.mpl.tick_style import TickStyle
from gerrytools.plotting.other.subway import SubwaySignOptions, subway_signs

__all__ = [
    "data",
    "geometry",
    "mpl",
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
    "ArrowData",
    "ArrowPlacement",
    "ArrowTextStyle",
    "TextArrowStyle",
    "LabelArrowStyle",
    "Histogram",
    "HistogramData",
    "HistPointList",
    "PaintBall",
    "PaintBallLine",
    "ScatterPlot",
    "SeaLevel",
    "SeatsVotes",
    "SeatsVotesData",
    "SVPlotLine",
    "SubwaySignOptions",
    "subway_signs",
    "ViolinPlot",
    "ViolinPlotSetData",
]
