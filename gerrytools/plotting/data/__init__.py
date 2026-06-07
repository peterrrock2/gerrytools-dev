"""Data-oriented Matplotlib plot classes."""

from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowData,
    ArrowPlacement,
    ArrowTextStyle,
    BandData,
    LabelArrowStyle,
    LineData,
    TextArrowStyle,
)
from gerrytools.plotting.data.boxplot import BoxPlot, BoxPlotStats
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

__all__ = [
    "ArrowData",
    "ArrowPlacement",
    "ArrowTextStyle",
    "BandData",
    "BandOptions",
    "BoxPlot",
    "BoxPlotOptions",
    "BoxPlotStats",
    "Histogram",
    "HistogramOptions",
    "LabelArrowStyle",
    "LineData",
    "LineOptions",
    "PaintBall",
    "ScatterPlot",
    "SeaLevel",
    "SeaLevelLineOptions",
    "SeatsVotes",
    "SeatsVotesLineOptions",
    "SeatsVotesMarkerOptions",
    "TextArrowStyle",
    "ViolinPlot",
    "ViolinPlotOptions",
]
