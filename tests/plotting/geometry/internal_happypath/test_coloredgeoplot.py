from collections.abc import Hashable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot


# ========================
# == CRS / REPROJECTION ==
# ========================
class TestCategoricalColorLayerColorSeries:
    """Cover __map_unique_values_to_colors and color_series branches."""

    def test_districtr_colormap_default(self, testing_gdf, tmp_path):
        """Default 'districtr' colormap should build without error."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(plancolumn="district")
        plot.save(str(tmp_path / "districtr.png"))
        assert (tmp_path / "districtr.png").exists()

    def test_none_colormap(self, testing_gdf, tmp_path):
        """colormap=None should produce 'none' fill for all geometries."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(plancolumn="district", colormap=None)
        plot.save(str(tmp_path / "none_cmap.png"))
        assert (tmp_path / "none_cmap.png").exists()

    def test_dict_colormap(self, testing_gdf, tmp_path):
        """A dict colormap should map district values to specific colors."""
        districts: list[int] = sorted(map(int, testing_gdf["district"].unique()))
        colors: dict[Hashable, str] = {d: "#FF0000" for d in districts}
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(plancolumn="district", colormap=colors)
        plot.save(str(tmp_path / "dict_cmap.png"))
        assert (tmp_path / "dict_cmap.png").exists()

    def test_too_many_unique_values_raises(self, testing_gdf):
        """Requesting districtr colormap with >max districts should raise ValueError."""
        from gerrytools.plotting.geometry.geoplot import _CategoricalColorLayer

        # Force only 1 color available by monkey-patching via a direct layer construction
        # Instead: make a GDF with more unique values than districtr provides
        big_gdf = testing_gdf.copy()
        # create a column with 200 unique values (districtr only provides up to a limit)
        big_gdf = big_gdf.iloc[:10].copy()
        big_gdf["fake_col"] = list(range(10))

        # districtr will provide up to some number of colors; request more than that
        # The easiest way is to create a layer directly with mismatched color count
        unique_vals = pd.Index(list(range(10)))
        # Only 2 colors provided for 10 unique values
        with pytest.raises(ValueError, match="Not enough colors"):
            _CategoricalColorLayer.__dict__["_CategoricalColorLayer__map_unique_values_to_colors"](
                unique_vals, ["#FF0000", "#00FF00"]
            )


# =======================
# == GeoPlot XY LIMITS ==
# =======================


class TestColoredGeoPlotContinuousLayer:
    def test_choropleth_with_facealpha(self, testing_gdf, tmp_path):
        """facealpha triggers _with_alpha path in _mappable."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", facealpha=0.5)
        plot.save(str(tmp_path / "choropleth_alpha.png"))
        assert (tmp_path / "choropleth_alpha.png").exists()

    def test_choropleth_with_integer_bins(self, testing_gdf, tmp_path):
        """Integer bins triggers _bin_boundaries integer path."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", bins=5)
        plot.save(str(tmp_path / "choropleth_bins_int.png"))
        assert (tmp_path / "choropleth_bins_int.png").exists()

    def test_choropleth_with_list_bins(self, testing_gdf, tmp_path):
        """List bins triggers IntervalIndex.from_breaks path."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", bins=[0, 1000, 3000, 5000, 7000, 10000])
        plot.save(str(tmp_path / "choropleth_bins_list.png"))
        assert (tmp_path / "choropleth_bins_list.png").exists()

    def test_choropleth_with_all_nan_column(self, testing_gdf, tmp_path):
        """All-NaN column hits _effective_bounds empty path."""
        gdf = testing_gdf.copy()
        gdf["nan_col"] = np.nan
        plot = ColoredGeoPlot(gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="nan_col")
        plot.save(str(tmp_path / "choropleth_nan.png"))
        assert (tmp_path / "choropleth_nan.png").exists()

    def test_choropleth_with_vmin_vmax(self, testing_gdf, tmp_path):
        """Explicit vmin/vmax overrides _effective_bounds."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", vmin=1000, vmax=9000)
        plot.save(str(tmp_path / "choropleth_vminvmax.png"))
        assert (tmp_path / "choropleth_vminvmax.png").exists()

    def test_choropleth_geoseries_raises_typeerror(self, testing_gdf):
        """Passing a GeoSeries as geosource to choropleth should raise TypeError."""
        from gerrytools.plotting.geometry.coloredgeoplot import _ContinuousColorLayer

        with pytest.raises(TypeError, match="geosource must be a GeoDataFrame"):
            _ContinuousColorLayer(
                geometry_source=testing_gdf.geometry,
                datacolumn="tot_pop",
            )

    def test_choropleth_invalid_colormap_raises(self, testing_gdf):
        """Passing an unknown colormap name should raise ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="not found in matplotlib colormaps"):
            plot.add_choropleth_layer(datacolumn="tot_pop", colormap="this_colormap_does_not_exist")


# =====================================
# == ColoredGeoPlot COLORBAR OPTIONS ==
# =====================================


class TestColoredGeoPlotColorbar:
    def test_choropleth_with_colorbar(self, testing_gdf, tmp_path):
        """show_colorbar=True should create colorbar without error."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", show_colorbar=True)
        plot.save(str(tmp_path / "choropleth_colorbar.png"))
        assert (tmp_path / "choropleth_colorbar.png").exists()

    def test_set_colorbar_layout(self, testing_gdf, tmp_path):
        """set_colorbar_layout modifies options and builds correctly."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", show_colorbar=True)
        plot.set_colorbar_layout(outer_pad=0.02, inner_pad=0.1)
        plot.save(str(tmp_path / "choropleth_colorbar_layout.png"))
        assert (tmp_path / "choropleth_colorbar_layout.png").exists()


# ==================================
# == ColoredGeoPlot WITH MESSAGES ==
# ==================================


class TestColoredGeoPlotSilentFalse:
    def test_build_silent_false_prints(self, testing_gdf, tmp_path, capsys):
        """Building with silent=False triggers print statements."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=False)
        plot.add_choropleth_layer(datacolumn="tot_pop")
        plot.save(str(tmp_path / "silent_false.png"))
        captured = capsys.readouterr()
        assert "Rendering" in captured.out


# ================================
# == ColoredGeoPlot DISTRICTING ==
# ================================


class TestColoredGeoPlotDistrictingPlan:
    def test_add_districting_plan_layer_dissolve_labels(self, testing_gdf, tmp_path):
        """add_districting_plan_layer with dissolve=True and show_labels=True."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(
            plancolumn="district",
            dissolve=True,
            show_labels=True,
        )
        plot.save(str(tmp_path / "plan_dissolve_labels.png"))
        assert (tmp_path / "plan_dissolve_labels.png").exists()


# ====================
# == DotDensityPlot ==
# ====================


class TestStringPlanLabels:
    """coerce_labels except branch with non-numeric string district labels."""

    def test_string_district_labels_hit_coerce_except(self, testing_gdf, tmp_path):
        """Plan column with non-numeric strings triggers the except branch in coerce_labels."""

        from gerrytools.plotting.geometry import ColoredGeoPlot

        str_gdf = testing_gdf.copy()
        str_gdf["str_district"] = str_gdf["district"].map(
            lambda d: chr(65 + int(d))  # 0->"A", 1->"B", etc.
        )

        plot = ColoredGeoPlot(str_gdf, dpi=50)
        plot.add_districting_plan_layer(plancolumn="str_district", show_labels=True)
        out = str(tmp_path / "str_labels.png")
        plot.save(out)
        assert Path(out).exists()


# ===============================
# == DOTDENSITY ZERO-DOT PATHS ==
# ===============================
