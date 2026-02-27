"""Data-oriented Matplotlib plot classes."""

from gerrytools.plotting.data._gerryplot_dataclasses import (
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
from gerrytools.plotting.data.scatterplot import ScatterData, ScatterPlot
from gerrytools.plotting.data.sealevel import SeaLevel, SeaLevelSetData
from gerrytools.plotting.data.seatsvotes import SeatsVotes, SeatsVotesData, SVPlotLine
from gerrytools.plotting.data.violin import ViolinPlot, ViolinPlotSetData

__all__ = [
    "BandData",
    "LineData",
    "PointSetData",
    "ArrowData",
    "ArrowPlacement",
    "ArrowTextStyle",
    "TextArrowStyle",
    "LabelArrowStyle",
    "BoxPlot",
    "BoxPlotSetData",
    "HistPointList",
    "Histogram",
    "HistogramData",
    "PaintBall",
    "PaintBallLine",
    "ScatterData",
    "ScatterPlot",
    "SeaLevel",
    "SeaLevelSetData",
    "SVPlotLine",
    "SeatsVotes",
    "SeatsVotesData",
    "ViolinPlot",
    "ViolinPlotSetData",
]
