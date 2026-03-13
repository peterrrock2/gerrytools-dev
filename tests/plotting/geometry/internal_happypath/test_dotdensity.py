from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.plotting.geometry.dotdensity import DotDensityPlot


# ========================
# == CRS / REPROJECTION ==
# ========================
class TestDotDensityRngSeedSetter:
    def test_set_rng_seed(self, testing_gdf):
        """Setting rng_seed property should update internal RNG."""
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        plot.rng_seed = 99
        assert plot.rng_seed == 99


class TestDotDensityMarkerOptions:
    def test_set_marker_options_and_build(self, testing_gdf, tmp_path):
        """set_marker_options followed by a build should not raise."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.set_marker_options(marker="^", markersize=2.0)
        plot.add_dot_density(column_name="tot_pop", color="blue")
        plot.save(str(tmp_path / "dot_marker.png"))
        assert (tmp_path / "dot_marker.png").exists()


class TestDotDensityValidationErrors:
    def test_missing_column_raises(self, testing_gdf):
        """Referencing a non-existent column should raise ValueError."""
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="not found in GeoDataFrame"):
            plot.add_dot_density(column_name="nonexistent_column", color="red")

    def test_negative_values_raises(self, testing_gdf):
        """A column with negative values should raise ValueError."""
        gdf = testing_gdf.copy()
        gdf["neg_col"] = -1
        plot = DotDensityPlot(
            gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="negative values"):
            plot.add_dot_density(column_name="neg_col", color="red")

    def test_nan_values_raises(self, testing_gdf):
        """A column with NaN values should raise ValueError."""
        gdf = testing_gdf.copy()
        gdf["nan_col"] = np.nan
        plot = DotDensityPlot(
            gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="NaN values"):
            plot.add_dot_density(column_name="nan_col", color="red")

    def test_same_column_same_color_warns(self, testing_gdf):
        """Adding the same column+color twice should emit a UserWarning."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
        )
        plot.add_dot_density(column_name="tot_pop", color="red")
        with pytest.warns(UserWarning, match="already exist"):
            plot.add_dot_density(column_name="tot_pop", color="red")

    def test_same_column_different_color_warns(self, testing_gdf):
        """Adding the same column with a different color should emit a UserWarning."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
        )
        plot.add_dot_density(column_name="tot_pop", color="red")
        with pytest.warns(UserWarning, match="Overwriting"):
            plot.add_dot_density(column_name="tot_pop", color="blue")


class TestDotDensityLegend:
    def test_build_with_show_legend(self, testing_gdf, tmp_path):
        """show_legend=True should render legend without error."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.add_dot_density(column_name="maj_pop", color="#1b7837")
        plot.save(str(tmp_path / "dot_legend.png"))
        assert (tmp_path / "dot_legend.png").exists()


class TestDotDensitySilentFalse:
    def test_build_silent_false(self, testing_gdf, tmp_path, capsys):
        """Building DotDensityPlot with silent=False should print progress."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=False,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_dot_density(column_name="tot_pop", color="red")
        plot.save(str(tmp_path / "dot_silent_false.png"))
        captured = capsys.readouterr()
        # Either "Generating dots" or "Rendering ... dots" should appear
        assert "dots" in captured.out.lower() or "rendering" in captured.out.lower()


class TestDotDensitySaveLegend:
    def test_save_legend_creates_file(self, testing_gdf, tmp_path):
        """save_legend should write a file when data has been added."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_dot_density(column_name="maj_pop", color="green")
        legend_path = str(tmp_path / "legend.png")
        plot.save_legend(legend_path)
        assert Path(legend_path).exists()

    def test_save_legend_with_display_names(self, testing_gdf, tmp_path):
        """save_legend with column_to_display_name should rename labels."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_dot_density(column_name="maj_pop", color="purple")
        legend_path = str(tmp_path / "legend_named.png")
        plot.save_legend(legend_path, column_to_display_name={"maj_pop": "Majority Population"})
        assert Path(legend_path).exists()


class TestDotDensitySaveLegendEmpty:
    def test_save_legend_empty_prints_warning(self, testing_gdf, tmp_path, capsys):
        """save_legend on a plot with no data should print a warning."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        legend_path = str(tmp_path / "empty_legend.png")
        plot.save_legend(legend_path)
        captured = capsys.readouterr()
        assert "No legend" in captured.out
        assert not Path(legend_path).exists()


# ========================
# == STRING PLAN LABELS ==
# ========================


class TestDotDensityZeroDots:
    """n_dots <= 0 skip and empty x_parts early return."""

    def test_some_polygons_below_people_per_dot_threshold(self, testing_gdf, tmp_path):
        """Polygons with val < people_per_dot get n_dots=0 and are skipped."""
        from gerrytools.plotting.geometry import DotDensityPlot

        # people_per_dot much larger than most polygon populations
        max_pop = int(testing_gdf["min_pop"].max())
        plot = DotDensityPlot(
            testing_gdf,
            people_per_dot=max_pop * 10,
            outline_column="district",
            dpi=50,
            silent=True,
            show_labels=False,
        )
        plot.add_dot_density(column_name="min_pop", color="blue", n_cores_for_processing=1)
        out = str(tmp_path / "zero_dot_some.png")
        plot.save(out)
        assert Path(out).exists()

    def test_all_polygons_below_threshold_returns_empty(self, testing_gdf, tmp_path):
        """All polygons have n_dots=0 -> x_parts empty -> early return."""

        from gerrytools.plotting.geometry import DotDensityPlot

        max_pop = int(testing_gdf["min_pop"].max())
        plot = DotDensityPlot(
            testing_gdf,
            people_per_dot=max_pop * 1000,  # all polys get 0 dots
            outline_column="district",
            dpi=50,
            silent=True,
            show_labels=False,
        )
        plot.add_dot_density(column_name="min_pop", color="red", n_cores_for_processing=1)
        out = str(tmp_path / "zero_dot_all.png")
        plot.save(out)
        assert Path(out).exists()
