import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import box

from gerrytools.plotting.geometry.dotdensity import DotDensityPlot, _random_xy_in_poly


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
# == RANDOM POINTS IN POLY ==
# ===========================
class TestRandomXYInPoly:
    def test_zero_area_polygon_raises(self):
        """A degenerate polygon (zero area) raises ValueError."""
        from shapely.geometry import LineString

        rng = np.random.default_rng(42)
        # A line has zero area
        line = LineString([(0, 0), (1, 1)])
        with pytest.raises(ValueError, match="zero area"):
            _random_xy_in_poly(line, 5, rng=rng)

    def test_high_inclusion_probability_uses_small_batch(self):
        """probability_of_inclusion > 0.9 uses batch_size = min(10000, n//5+1)."""
        from shapely.geometry import box as sbox

        rng = np.random.default_rng(42)
        # A very close-to-rectangular polygon has high inclusion probability
        poly = sbox(0, 0, 10, 10)  # perfect rectangle → probability = 1.0
        xs, ys = _random_xy_in_poly(poly, 100, rng=rng)
        assert len(xs) == 100

    def test_low_inclusion_probability_uses_estimated_batch(self):
        """probability_of_inclusion < 0.9 estimates batch from inclusion rate."""

        # Create a thin triangle with low inclusion probability
        from shapely.geometry import Polygon

        rng = np.random.default_rng(42)
        # A narrow triangle has low area/bbox ratio
        poly = Polygon([(0, 0), (100, 0), (50, 1)])  # very flat, low probability
        xs, ys = _random_xy_in_poly(poly, 10, rng=rng)
        assert len(xs) == 10


# ===============
# == ZERO DOTS ==
# ===============


class TestDotDensityZeroDots:
    def test_zero_population_rows_skipped(self, testing_gdf, tmp_path):
        """Rows with population = 0 produce no dots (n_dots <= 0 path)."""
        gdf = testing_gdf.copy()
        gdf["zero_pop"] = 0
        gdf.loc[gdf.index[0], "zero_pop"] = 100  # one row with dots
        plot = DotDensityPlot(
            gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=50,
            show_labels=False,
        )
        plot.add_dot_density(column_name="zero_pop", color="red")
        plot.save(str(tmp_path / "zero_pop.png"))
        assert (tmp_path / "zero_pop.png").exists()

    def test_all_zero_population_empty_result(self, testing_gdf, tmp_path):
        """Very small populations can produce no points for every geometry."""
        gdf = testing_gdf.copy()
        gdf["all_zero"] = 0
        _ = DotDensityPlot(
            gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=1000,
            show_labels=False,
        )
        gdf2 = testing_gdf.copy()
        gdf2["small_pop"] = 1  # very small, all will be 0 with large people_per_dot
        plot2 = DotDensityPlot(
            gdf2,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=10000,  # so large that n_dots = 0 for all rows
            show_labels=False,
        )
        plot2.add_dot_density(column_name="small_pop", color="blue")
        plot2.save(str(tmp_path / "all_zero_dots.png"))
        assert (tmp_path / "all_zero_dots.png").exists()


# ======================
# == EMPTY DOT LAYERS ==
# ======================


class TestDotDensityEmptyDict:
    def test_draw_all_dots_with_no_layers(self, testing_gdf, tmp_path):
        """Building a DotDensityPlot with no dot layers still works."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        # Don't add any dot density layers; _draw_all_dots should return early
        plot.save(str(tmp_path / "no_dots.png"))
        assert (tmp_path / "no_dots.png").exists()


# ====================
# == LEGEND OPTIONS ==
# ====================


class TestDotDensitySetLegendOptions:
    def test_set_legend_options(self, testing_gdf, tmp_path):
        """set_legend_options updates legend and renders correctly."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.set_legend_options(
            loc="upper right",
            ncols=2,
            fontsize=8,
            title="Legend",
            frameon=True,
        )
        plot.add_dot_density(column_name="tot_pop", color="red")
        plot.save(str(tmp_path / "legend_opts.png"))
        assert (tmp_path / "legend_opts.png").exists()
