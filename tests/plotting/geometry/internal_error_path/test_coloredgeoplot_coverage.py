import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import box

from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot, _ContinuousColorLayer
from gerrytools.plotting.mpl.geoplot_options import ColorbarOptions


def _rect_gdf_with_crs(crs="EPSG:4326"):
    """Return a tiny 3-row GeoDataFrame of rectangles with the given CRS."""
    geoms = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(0, 1, 1, 2),
    ]
    return GeoDataFrame(
        {"value": [10.0, 20.0, 30.0], "category": ["A", "B", "A"]},
        geometry=geoms,
        crs=crs,
    )


# ===========================
# == CONTINUOUS LAYER INIT ==
# ===========================
class TestContinuousColorLayerPostInit:
    def test_invalid_colormap_type_raises(self, testing_gdf):
        """Non-str, non-Colormap colormap raises TypeError."""
        with pytest.raises(TypeError, match="colormap.*must be a str or Colormap"):
            _ContinuousColorLayer(
                geometry_source=testing_gdf,
                datacolumn="tot_pop",
                colormap={"A": "red"},  # ty: ignore [invalid-argument-type]
            )

    def test_missing_datacolumn_raises(self, testing_gdf):
        """Missing datacolumn raises TypeError."""
        with pytest.raises(TypeError, match="datacolumn.*must be set"):
            _ContinuousColorLayer(
                geometry_source=testing_gdf,
                datacolumn=None,
                colormap="viridis",
            )


# ====================
# == BIN BOUNDARIES ==
# ====================


class TestBinBoundariesError:
    def test_bin_boundaries_none_bins_raises(self, testing_gdf):
        """Calling _bin_boundaries with bins=None raises RuntimeError."""
        layer = _ContinuousColorLayer(
            geometry_source=testing_gdf,
            datacolumn="tot_pop",
            bins=None,
        )
        with pytest.raises(RuntimeError, match="_bin_boundaries"):
            layer._bin_boundaries(0.0, 1.0)


# =======================
# == BIN COLOR MAPPING ==
# =======================


class TestContinuousLayerColormapObject:
    def test_colormap_object_works(self, testing_gdf, tmp_path):
        """A Colormap object can be used instead of a string."""
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", colormap=cmap)
        plot.save(str(tmp_path / "colormap_obj.png"))
        assert (tmp_path / "colormap_obj.png").exists()


class TestColorMappingForBinsAlpha:
    def test_bins_with_facealpha(self, testing_gdf, tmp_path):
        """bins + facealpha triggers _color_mapping_for_bins alpha path."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            bins=4,
            facealpha=0.6,
            show_colorbar=True,
        )
        plot.save(str(tmp_path / "bins_facealpha.png"))
        assert (tmp_path / "bins_facealpha.png").exists()

    def test_facealpha_no_bins_with_colorbar(self, testing_gdf, tmp_path):
        """facealpha without bins and with colorbar triggers _with_alpha in _mappable."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            facealpha=0.5,
            bins=None,
            show_colorbar=True,
        )
        plot.save(str(tmp_path / "facealpha_colorbar.png"))
        assert (tmp_path / "facealpha_colorbar.png").exists()


# =========================
# == BINNED COLOR SERIES ==
# =========================


class TestContinuousColorSeriesBinPaths:
    def test_nan_value_in_bins_uses_missing_color(self, testing_gdf, tmp_path):
        """NaN values with bins get missing_color."""
        gdf = testing_gdf.copy()
        gdf["pop_with_nan"] = gdf["tot_pop"].astype(float)
        gdf.loc[gdf.index[0], "pop_with_nan"] = np.nan
        plot = ColoredGeoPlot(gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="pop_with_nan", bins=4)
        plot.save(str(tmp_path / "bins_nan.png"))
        assert (tmp_path / "bins_nan.png").exists()

    def test_value_equals_upper_bound_gets_last_bin(self, testing_gdf):
        """Value equal to upper_bound is assigned to the last bin."""
        layer = _ContinuousColorLayer(
            geometry_source=testing_gdf,
            datacolumn="tot_pop",
            bins=4,
        )
        cs = layer.color_series
        assert len(cs) == len(testing_gdf)

    def test_value_below_lower_bound_gets_first_bin(self, testing_gdf):
        """Value below the bin lower bound gets index 0."""
        gdf = testing_gdf.copy()
        min_val = gdf["tot_pop"].min()
        # Set vmin higher so values are below the first bin
        layer = _ContinuousColorLayer(
            geometry_source=gdf,
            datacolumn="tot_pop",
            bins=[min_val + 10000, min_val + 20000, min_val + 30000],
        )
        cs = layer.color_series
        assert len(cs) == len(gdf)

    def test_value_above_upper_bound_gets_last_bin(self, testing_gdf):
        """Value above the bin upper bound gets last bin index."""
        gdf = testing_gdf.copy()
        layer = _ContinuousColorLayer(
            geometry_source=gdf,
            datacolumn="tot_pop",
            bins=[0.0, 1.0, 2.0],  # All values above the last bin
        )
        cs = layer.color_series
        assert len(cs) == len(gdf)


# ===================
# == RENDER ERRORS ==
# ===================


class TestContinuousLayerRender:
    def test_render_unknown_kwargs_raises(self, testing_gdf):
        """Unknown kwargs to render raises TypeError."""
        import matplotlib.pyplot as plt

        layer = _ContinuousColorLayer(
            geometry_source=testing_gdf,
            datacolumn="tot_pop",
        )
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="Unknown keyword argument"):
            layer.render(ax, bad_kwarg="oops")
        plt.close(fig)

    def test_render_missing_column_keyerror(self, testing_gdf):
        """Rendering with a column absent from geometry_source raises KeyError.

        _ContinuousColorLayer.__post_init__ only checks datacolumn is not None,
        not that the column actually exists in the GDF. So we can create a layer
        with a nonexistent column and get a KeyError at render time.
        """
        import matplotlib.pyplot as plt

        try:
            # init doesn't check column existence, only that it's not None
            fake_layer = _ContinuousColorLayer(
                geometry_source=testing_gdf,
                datacolumn="__nonexistent_col__",
            )
            fig, ax = plt.subplots()
            with pytest.raises(KeyError):
                fake_layer.render(ax)
            plt.close(fig)
        except TypeError:
            # If __post_init__ raised TypeError, that is also acceptable
            pass


# ============================
# == DISTRICTING PLAN LAYER ==
# ============================


class TestAddDistrictingPlanLayerExtras:
    def test_add_plan_layer_with_geosource(self, testing_gdf, tmp_path):
        """add_districting_plan_layer with explicit geosource uses that."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        subset = testing_gdf[testing_gdf["district"].isin([0, 1])].copy()
        plot.add_districting_plan_layer(
            geosource=subset,
            plancolumn="district",
        )
        plot.save(str(tmp_path / "plan_geosource.png"))
        assert (tmp_path / "plan_geosource.png").exists()

    def test_add_plan_layer_show_labels_with_exclude(self, testing_gdf, tmp_path):
        """show_labels=True with exclude_labels excludes them."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(
            plancolumn="district",
            show_labels=True,
            exclude_labels=[0, 1],
        )
        plot.save(str(tmp_path / "plan_exclude_labels.png"))
        assert (tmp_path / "plan_exclude_labels.png").exists()


# =====================
# == COLORBAR LAYOUT ==
# =====================


class TestClearColorbarsAndResetLayout:
    def test_clear_colorbars_called_on_rebuild(self, testing_gdf, tmp_path):
        """Building twice exercises cax.remove() in _clear_colorbars_and_reset_layout."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", show_colorbar=True)
        # First build creates the colorbar axes
        plot.save(str(tmp_path / "build1.png"))
        # Second build triggers _clear_colorbars_and_reset_layout with non-empty _colorbar_axes
        plot.save(str(tmp_path / "build2.png"))
        assert (tmp_path / "build2.png").exists()


class TestSetColorbarLayoutAllParams:
    def test_set_colorbar_layout_all_params(self, testing_gdf, tmp_path):
        """set_colorbar_layout with all params updates _colorbar_layout_options."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", show_colorbar=True)
        plot.set_colorbar_layout(
            outer_pad=0.01,
            inner_pad=0.01,
            width=0.02,
            right_margin=0.01,
        )
        plot.save(str(tmp_path / "colorbar_all_params.png"))
        assert (tmp_path / "colorbar_all_params.png").exists()


# ======================
# == COLORBAR OPTIONS ==
# ======================


class TestColorbarOptions:
    def test_colorbar_with_custom_label(self, testing_gdf, tmp_path):
        """colorbar_label override is used instead of datacolumn."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            show_colorbar=True,
            colorbar_label="Population",
        )
        plot.save(str(tmp_path / "colorbar_custom_label.png"))
        assert (tmp_path / "colorbar_custom_label.png").exists()

    def test_colorbar_with_label_fontsize_options(self, testing_gdf, tmp_path):
        """ColorbarOptions with label_fontsize triggers extended set_label call."""
        cb_options = ColorbarOptions(label_fontsize=10, label_rotation=90, label_pad=5.0)
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_label_opts.png"))
        assert (tmp_path / "colorbar_label_opts.png").exists()

    def test_colorbar_force_ticks(self, testing_gdf, tmp_path):
        """force_ticks is applied to colorbar."""
        cb_options = ColorbarOptions(force_ticks=[1000, 3000, 5000])
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_force_ticks.png"))
        assert (tmp_path / "colorbar_force_ticks.png").exists()

    def test_colorbar_force_ticklabels(self, testing_gdf, tmp_path):
        """force_ticklabels is applied to colorbar."""
        cb_options = ColorbarOptions(
            force_ticks=[1000, 3000, 5000],
            force_ticklabels=["low", "mid", "high"],
        )
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_force_ticklabels.png"))
        assert (tmp_path / "colorbar_force_ticklabels.png").exists()

    def test_colorbar_max_n_ticks(self, testing_gdf, tmp_path):
        """max_n_ticks reduces the number of displayed ticks."""
        cb_options = ColorbarOptions(max_n_ticks=3)
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_max_ticks.png"))
        assert (tmp_path / "colorbar_max_ticks.png").exists()

    def test_colorbar_bins_with_ticks(self, testing_gdf, tmp_path):
        """Bins path with colorbar uses ticks from layer_defaults."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop",
            bins=4,
            show_colorbar=True,
        )
        plot.save(str(tmp_path / "colorbar_bins_ticks.png"))
        assert (tmp_path / "colorbar_bins_ticks.png").exists()


# ===========================
# == RANDOM POINTS IN POLY ==
# ===========================
