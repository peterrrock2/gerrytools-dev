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
    TextArrowStyle,
)
from gerrytools.plotting.data.boxplot import BoxPlot
from gerrytools.plotting.data.histogram import Histogram
from gerrytools.plotting.data.options import (
    BandOptions,
    BoxPlotOptions,
    HistogramOptions,
    LineOptions,
    SeaLevelLineOptions,
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
    ViolinPlotOptions,
)
from gerrytools.plotting.data.paintball import PaintBall
from gerrytools.plotting.data.scatterplot import ScatterPlot
from gerrytools.plotting.data.sealevel import SeaLevel
from gerrytools.plotting.data.seatsvotes import SeatsVotes
from gerrytools.plotting.data.violin import ViolinPlot
from gerrytools.plotting.geometry._layers._continuous import ColormapLayer
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
    "ArrowData",
    "ArrowPlacement",
    "ArrowTextStyle",
    "AxisLabelStyle",
    "BandData",
    "BandOptions",
    "BoxPlot",
    "BoxPlotOptions",
    "ColormapLayer",
    "ColorbarOptions",
    "ColoredGeoPlot",
    "DotDensityPlot",
    "FontFamily",
    "FontStretch",
    "FontStyle",
    "FontVariant",
    "FontWeight",
    "GeoPlot",
    "Histogram",
    "HistogramOptions",
    "LabelArrowStyle",
    "LabelBoxOptions",
    "LabelFontOptions",
    "LegendOptions",
    "LineData",
    "LineOptions",
    "PaintBall",
    "PointMarkerOptions",
    "ScatterPlot",
    "SeaLevel",
    "SeaLevelLineOptions",
    "SeatsVotes",
    "SeatsVotesLineOptions",
    "SeatsVotesMarkerOptions",
    "SubwaySignOptions",
    "TextArrowStyle",
    "TickStyle",
    "TitleStyle",
    "ViolinPlot",
    "ViolinPlotOptions",
    "data",
    "districtr",
    "flare",
    "geometry",
    "latex",
    "mpl",
    "purples",
    "redbluecmap",
    "subway_signs",
]
