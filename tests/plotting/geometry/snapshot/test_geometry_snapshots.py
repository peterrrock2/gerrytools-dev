from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot
from gerrytools.plotting.geometry.dotdensity import DotDensityPlot
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


# =======================
# == GEOPLOT SNAPSHOTS ==
# =======================
class TestGeoPlotSnapshots:
    @pytest.mark.snapshot
    def test_geoplot_basic_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="geoplot_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_geoplot_with_outline_layer_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_outline_layer(edgecolor="black", edgewidth=1.0)
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="geoplot_with_outline",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_geoplot_with_dissolved_outline_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_outline_layer(dissolve_column="district", edgecolor="red", edgewidth=1.5)
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="geoplot_with_dissolved_outline",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_geoplot_with_highlight_layer_snapshot(self, testing_gdf, tmp_path):
        highlighted = testing_gdf[testing_gdf["district"] == 0]

        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_highlight_layer(geosource=highlighted, facecolor="yellow", facealpha=0.6)
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="geoplot_with_highlight",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )


# ===============================
# == COLORED GEOPLOT SNAPSHOTS ==
# ===============================
class TestColoredGeoPlotSnapshots:
    @pytest.mark.snapshot
    def test_choropleth_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", colormap="Purples")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="coloredgeoplot_choropleth",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_choropleth_with_colorbar_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_choropleth_layer(datacolumn="tot_pop", colormap="Blues", show_colorbar=True)
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="coloredgeoplot_choropleth_colorbar",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_choropleth_binned_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_choropleth_layer(
            datacolumn="tot_pop", colormap="Greens", bins=5, show_colorbar=True
        )
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="coloredgeoplot_choropleth_binned",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_districting_plan_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_districting_plan_layer(plancolumn="district", colormap="districtr")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="coloredgeoplot_districting_plan",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_districting_plan_dissolved_snapshot(self, testing_gdf, tmp_path):
        plot = ColoredGeoPlot(testing_gdf, dpi=100, silent=True)
        plot.add_districting_plan_layer(
            plancolumn="district", colormap="districtr", dissolve=True, edgecolor="black"
        )
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="coloredgeoplot_districting_plan_dissolved",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )


# ================================
# == DOT DENSITY PLOT SNAPSHOTS ==
# ================================
class TestDotDensityPlotSnapshots:
    @pytest.mark.snapshot
    def test_dot_density_single_group_snapshot(self, testing_gdf, tmp_path):
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            people_per_dot=200,
            show_labels=False,
            rng_seed=RNG_SEED,
            dpi=100,
            silent=True,
        )
        plot.add_dot_density(column_name="tot_pop", color="denim")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="dotdensity_single_group",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_dot_density_two_groups_snapshot(self, testing_gdf, tmp_path):
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            people_per_dot=200,
            show_labels=False,
            rng_seed=RNG_SEED,
            dpi=100,
            silent=True,
        )
        plot.add_dot_density(column_name="maj_pop", color="denim")
        plot.add_dot_density(column_name="min_pop", color="alizarin")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="dotdensity_two_groups",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
