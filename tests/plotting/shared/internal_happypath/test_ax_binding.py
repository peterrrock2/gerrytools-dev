"""Tests for the ``ax=`` constructor parameter and the ``bind_to_ax`` method.

These exercise the matplotlib-idiomatic embedding path added in Theme 7a:
users can pass their own matplotlib ``Axes`` at construction time, or rebind
an existing plot to a different axes later.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from gerrytools.plotting import ColoredGeoPlot, Histogram  # noqa: E402

# ----------------------------------------------------------------------
# Constructor accepts ax=
# ----------------------------------------------------------------------


class TestAxConstructorParameter:
    def test_user_provided_ax_is_returned_by_plot_ax(self):
        fig, user_ax = plt.subplots(figsize=(4, 4))
        plot = Histogram(ax=user_ax)
        plot.add_histogram([1.0, 2.0, 3.0, 2.0, 1.0])
        assert plot.ax is user_ax

    def test_user_provided_fig_is_used(self):
        fig, user_ax = plt.subplots(figsize=(4, 4))
        plot = Histogram(ax=user_ax)
        assert plot.fig is fig

    def test_no_ax_creates_fresh_figure(self):
        plot = Histogram()
        plot.add_histogram([1.0, 2.0, 3.0])
        assert plot.fig is not None
        assert plot.ax is not None

    def test_figure_size_and_ax_together_warns(self):
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Histogram(figure_size=(8, 8), ax=user_ax)
        assert any("ignored when ax is provided" in str(w.message) for w in caught)

    def test_dpi_and_ax_together_warns(self):
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Histogram(dpi=72, ax=user_ax)
        assert any("ignored when ax is provided" in str(w.message) for w in caught)

    def test_ax_alone_does_not_warn(self):
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Histogram(ax=user_ax)
        assert not any("ignored when ax is provided" in str(w.message) for w in caught)


# ----------------------------------------------------------------------
# bind_to_ax retargets
# ----------------------------------------------------------------------


class TestBindToAx:
    def test_bind_to_ax_retargets_to_new_axes(self):
        plot = Histogram()
        plot.add_histogram([1.0, 2.0, 3.0])
        _ = plot.ax  # build onto plot's own fig

        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        assert plot.ax is new_ax

    def test_bind_to_ax_none_reverts_to_fresh_figure(self):
        _, user_ax = plt.subplots()
        plot = Histogram(ax=user_ax)
        assert plot._ax is user_ax
        plot.bind_to_ax(None)
        # After unbinding, _ax points to a fresh axes
        assert plot._ax is not user_ax

    def test_rebind_preserves_added_data(self):
        plot = Histogram()
        plot.add_histogram([1.0, 2.0, 3.0, 4.0], name="series_a")
        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        # The data should still be there for the new render
        assert len(plot._hist_data_dict["overlay"]) == 1
        assert plot._hist_data_dict["overlay"][0].name == "series_a"

    def test_bind_to_ax_none_preserves_construction_figure_size_and_dpi(self):
        plot = Histogram(figure_size=(4.0, 3.0), dpi=150)
        plot.bind_to_ax(None)
        assert tuple(plot.fig.get_size_inches()) == (4.0, 3.0)
        assert plot.fig.dpi == 150

    def test_bind_to_ax_none_matches_default_construction_geometry(self):
        plot = Histogram()
        plot.bind_to_ax(None)
        assert tuple(plot.fig.get_size_inches()) == (10.0, 6.0)
        assert plot.fig.dpi == 300

    def test_bind_to_ax_none_uses_defaults_when_constructed_with_user_ax(self):
        # A plot constructed onto a user axes has no construction geometry of
        # its own; unbinding falls back to the standard defaults rather than
        # inheriting the user figure's geometry.
        _, user_ax = plt.subplots(figsize=(2.0, 2.0), dpi=72)
        plot = Histogram(ax=user_ax)
        plot.bind_to_ax(None)
        assert tuple(plot.fig.get_size_inches()) == (10.0, 6.0)
        assert plot.fig.dpi == 300


# ----------------------------------------------------------------------
# ax= on GeoPlot too
# ----------------------------------------------------------------------


class TestGeoPlotAxParameter:
    def test_geoplot_accepts_ax(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            crs="EPSG:4326",
        )
        fig, user_ax = plt.subplots()
        plot = ColoredGeoPlot(gdf, ax=user_ax)
        assert plot._ax is user_ax

    def test_geoplot_dpi_with_ax_warns(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            crs="EPSG:4326",
        )
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ColoredGeoPlot(gdf, dpi=72, ax=user_ax)
        assert any("ignored when ax is provided" in str(w.message) for w in caught)

    def test_geoplot_bind_to_ax(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            crs="EPSG:4326",
        )
        plot = ColoredGeoPlot(gdf)
        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        assert plot._ax is new_ax

    def test_geoplot_bind_to_ax_none_preserves_construction_dpi(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            crs="EPSG:4326",
        )
        plot = ColoredGeoPlot(gdf, dpi=150)
        plot.bind_to_ax(None)
        assert plot.fig.dpi == 150


# ----------------------------------------------------------------------
# Renamed methods are reachable
# ----------------------------------------------------------------------


class TestRenamedTickMethods:
    def test_update_xtick_labels_exists(self):
        plot = Histogram()
        plot.update_xtick_labels(labels=["a", "b", "c"])
        # No error means the renamed method is callable.

    def test_update_ytick_labels_exists(self):
        plot = Histogram()
        plot.update_ytick_labels(labels=["a"])

    def test_old_update_xtick_values_is_gone(self):
        plot = Histogram()
        assert not hasattr(plot, "update_xtick_values")

    def test_set_xlimits_alias_is_gone(self):
        plot = Histogram()
        assert not hasattr(plot, "set_xlimits")


# ----------------------------------------------------------------------
# Enable/disable boolean setters (Theme 5)
# ----------------------------------------------------------------------


class TestEnableDisableSetters:
    def test_histogram_enable_grid(self):
        plot = Histogram()
        assert plot.grid is False
        plot.enable_grid()
        assert plot.grid is True
        plot.disable_grid()
        assert plot.grid is False

    def test_histogram_suppress_warnings(self):
        plot = Histogram()
        assert plot.hide_warnings is False
        plot.suppress_warnings()
        assert plot.hide_warnings is True
        plot.show_warnings()
        assert plot.hide_warnings is False


# ----------------------------------------------------------------------
# PaintBall reshape (Theme 6)
# ----------------------------------------------------------------------


class TestPaintBallReshape:
    def test_empty_constructor_works(self):
        from gerrytools.plotting import PaintBall

        plot = PaintBall()
        assert plot._voteshare_data == []
        assert plot._seatshare_data == []

    def test_add_voteshare_seatshare_data_is_canonical(self):
        from gerrytools.plotting import PaintBall

        plot = PaintBall()
        plot.add_voteshare_seatshare_data([0.4, 0.5, 0.6], [0.3, 0.5, 0.7])
        assert plot._voteshare_data == [0.4, 0.5, 0.6]

    def test_no_default_guide_lines(self):
        from gerrytools.plotting import PaintBall

        plot = PaintBall()
        plot.add_voteshare_seatshare_data([0.5], [0.5])
        assert "Efficiency Gap" not in plot._named_lines
        assert "Proportionality" not in plot._named_lines

    def test_add_efficiency_gap_line(self):
        from gerrytools.plotting import PaintBall

        plot = PaintBall()
        plot.add_voteshare_seatshare_data([0.5], [0.5])
        plot.add_efficiency_gap_line()
        assert "Efficiency Gap" in plot._named_lines
        assert plot._named_lines["Efficiency Gap"].slope == 2.0

    def test_add_proportionality_line(self):
        from gerrytools.plotting import PaintBall

        plot = PaintBall()
        plot.add_voteshare_seatshare_data([0.5], [0.5])
        plot.add_proportionality_line()
        assert "Proportionality" in plot._named_lines
        assert plot._named_lines["Proportionality"].slope == 1.0

    def test_constructor_rejects_legacy_voteshare_kwarg(self):
        from gerrytools.plotting import PaintBall

        with pytest.raises(TypeError):
            PaintBall(voteshare_data=[0.5], seats_data=[0.5])  # type: ignore
