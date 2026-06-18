from gerrytools import plotting
from gerrytools.plotting import data, geometry, mpl


def test_public_plotting_subpackages_are_visible():
    assert plotting.data is data
    assert plotting.geometry is geometry
    assert plotting.mpl is mpl


def test_public_mpl_options_imports():
    _ = mpl.PointMarkerOptions
    _ = mpl.TickStyle
    _ = mpl.LegendOptions
    _ = mpl.AxisLabelStyle
    _ = mpl.TitleStyle
    _ = mpl.LabelFontOptions
    _ = mpl.LabelBoxOptions
    _ = mpl.ColorbarOptions


def test_public_data_and_geometry_plot_imports():
    _ = data.PaintBall
    _ = data.SeatsVotes
    _ = data.Histogram
    _ = data.BoxPlot
    _ = data.ViolinPlot
    _ = geometry.GeoPlot
    _ = geometry.DotDensityPlot
