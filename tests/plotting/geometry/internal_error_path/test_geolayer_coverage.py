import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, box

from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot
from gerrytools.plotting.geometry.geoplot import (
    _CategoricalColorLayer,
    _MarkerLayer,
)


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


# ======================
# == CRS REPROJECTION ==
# ======================
class TestGeometriesInCRSReproject:
    """CRS reprojection when source CRS != target CRS."""

    def test_reproject_different_crs(self, testing_gdf):
        """When source and target CRS differ, geometries are reprojected."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        # target_crs must differ from source to trigger the to_crs() call
        plot = ColoredGeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:3857")
        with tempfile.TemporaryDirectory() as tmpdir:
            plot.save(str(Path(tmpdir) / "reproject.png"))


# ================
# == GEO SOURCE ==
# ================


class TestGeoLayerGeoSourceProperty:
    """Cover the geosource property when geometry_mask is applied."""

    def test_geosource_with_mask_geoseries(self, testing_gdf):
        """geometry_mask on a GeoSeries goes through the else branch."""
        gs = testing_gdf.geometry
        mask = pd.Series(
            [True, False, True] + [False] * (len(testing_gdf) - 3), index=testing_gdf.index
        )
        layer = _CategoricalColorLayer(
            geometry_source=gs,
            geometry_mask=mask,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor="black",
        )
        result = layer.geosource
        assert len(result) < len(gs)

    def test_geosource_with_mask_geodataframe(self, testing_gdf):
        """geometry_mask on a GeoDataFrame returns masked GeoDataFrame."""
        mask = testing_gdf["district"] == 0
        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            geometry_mask=mask,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor="black",
        )
        result = layer.geosource
        assert len(result) < len(testing_gdf)


# ============================
# == CATEGORICAL LAYER INIT ==
# ============================


class TestCategoricalColorLayerGeoSeries:
    """GeoSeries geosource with districtr colormap is silently set to 'none'."""

    def test_geoseries_districtr_maps_to_none(self, testing_gdf):
        gs = testing_gdf.geometry
        layer = _CategoricalColorLayer(
            geometry_source=gs,
            colormap="districtr",
            missing_color="none",
            facealpha=0.0,
            edgecolor="black",
        )
        # After __post_init__, colormap should have been changed to "none"
        assert layer.colormap == "none"


# =============================
# == DATACOLUMN REQUIREMENTS ==
# =============================


class TestCategoricalColorLayerNeedsDatacolumn:
    def test_mpl_colormap_without_datacolumn_raises(self, testing_gdf):
        with pytest.raises(TypeError, match="datacolumn.*must be set"):
            _CategoricalColorLayer(
                geometry_source=testing_gdf,
                colormap="viridis",  # valid mpl colormap
                datacolumn=None,
                missing_color="lightgrey",
                facealpha=0.0,
                edgecolor="none",
            )

    def test_dict_colormap_without_datacolumn_raises(self, testing_gdf):
        with pytest.raises(TypeError, match="datacolumn.*must be set"):
            _CategoricalColorLayer(
                geometry_source=testing_gdf,
                colormap={"A": "red"},
                datacolumn=None,
                missing_color="lightgrey",
                facealpha=0.0,
                edgecolor="none",
            )


# ============================
# == UNIQUE VALUE COLOR MAP ==
# ============================


class TestMapUniqueValuesStringSort:
    def test_string_keys_sorted_by_string(self, testing_gdf):
        """Non-integer category keys fall through to string-sort path."""
        gdf = testing_gdf.copy()
        gdf["str_col"] = ["alpha", "beta", "gamma", "delta"] * (len(testing_gdf) // 4) + [
            "alpha"
        ] * (len(testing_gdf) % 4)
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            datacolumn="str_col",
            colormap="districtr",
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        # Should have a dict colormap from string sort
        assert isinstance(layer.colormap, dict)


# ==================
# == COLOR SERIES ==
# ==================


class TestCategoricalColorSeriesBranches:
    def test_colormap_object(self, testing_gdf):
        """Colormap instance colormap branch is covered."""
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            datacolumn="district",
            colormap=cmap,
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        cs = layer.color_series
        assert len(cs) == len(testing_gdf)

    def test_named_mpl_colormap_string(self, testing_gdf):
        """String colormap that is in plt.colormaps() uses Colormap path."""
        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            datacolumn="district",
            colormap="viridis",
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        cs = layer.color_series
        assert len(cs) == len(testing_gdf)

    def test_color_series_invalid_colormap_raises(self, testing_gdf):
        """color_series else branch: invalid colormap type raises TypeError."""
        # _CategoricalColorLayer.__post_init__ doesn't validate colormap type,
        # so we can pass an int and it falls through to the else: raise TypeError
        gdf = testing_gdf.copy()
        # Use a non-string, non-Colormap, non-dict, non-None, non-pd.Series colormap
        # to hit the else branch in color_series
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            datacolumn=None,
            colormap=12345,  # ty: ignore [invalid-argument-type]
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        with pytest.raises(TypeError, match="colormap.*must be one of"):
            _ = layer.color_series

    def test_color_series_with_nan_in_column(self, testing_gdf):
        """NaN in datacolumn uses missing_color."""
        gdf = testing_gdf.copy()
        gdf = gdf.copy()
        # Set some values to NaN for a Colormap path
        gdf["district_with_nan"] = gdf["district"].astype(float)
        gdf.loc[gdf.index[0], "district_with_nan"] = np.nan
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            datacolumn="district_with_nan",
            colormap=cmap,
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        cs = layer.color_series
        assert len(cs) == len(gdf)


# ===================
# == RENDER KWARGS ==
# ===================


class TestCategoricalRenderUnknownKwargs:
    def test_render_unknown_kwargs_raises(self, testing_gdf):
        import matplotlib.pyplot as plt

        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor="black",
        )
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="Unknown keyword argument"):
            layer.render(ax, bad_kwarg="oops")
        plt.close(fig)


# =========================
# == RENDER MISSING DATA ==
# =========================


class TestCategoricalRenderMissingColumn:
    def test_render_missing_datacolumn_raises(self, testing_gdf):
        import matplotlib.pyplot as plt

        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            datacolumn="nonexistent_col",
            colormap={"A": "red"},
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        fig, ax = plt.subplots()
        with pytest.raises(KeyError):
            layer.render(ax)
        plt.close(fig)


# =======================
# == MARKER LAYER INIT ==
# =======================


class TestMarkerLayerPostInit:
    def test_none_point_geometries_raises(self):
        with pytest.raises(TypeError, match="point_geometries"):
            _MarkerLayer(point_geometries=None)  # ty: ignore [invalid-argument-type]

    def test_labels_wrong_length_raises(self, testing_gdf):
        pts = GeoSeries([Point(0, 0), Point(1, 1)])
        with pytest.raises(ValueError, match="same length"):
            _MarkerLayer(
                point_geometries=pts,
                labels=["only_one_label"],
            )

    def test_none_marker_options_uses_default(self, testing_gdf):
        pts = GeoSeries([Point(0, 0)])
        layer = _MarkerLayer(
            point_geometries=pts,
            marker_options=None,  # ty: ignore [invalid-argument-type]
        )
        assert layer.marker_options is not None


# =========================
# == MARKER LAYER RENDER ==
# =========================


class TestMarkerLayerRender:
    def test_render_unknown_kwargs_raises(self):
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(0, 0)])
        layer = _MarkerLayer(point_geometries=pts)
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="Unknown keyword argument"):
            layer.render(ax, bad_kwarg="oops")
        plt.close(fig)

    def test_render_with_crs_reprojection(self):
        """Points with CRS != target_crs triggers to_crs."""
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(-90, 40), Point(-91, 41)], crs="EPSG:4326")
        layer = _MarkerLayer(point_geometries=pts, show_labels=False)
        fig, ax = plt.subplots()
        layer.render(ax, target_crs="EPSG:3857")
        plt.close(fig)

    def test_render_show_labels_false(self):
        """show_labels=False skips the label-drawing branch."""
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(0, 0), Point(1, 1)])
        layer = _MarkerLayer(
            point_geometries=pts,
            show_labels=False,
            labels=None,
        )
        fig, ax = plt.subplots()
        layer.render(ax)
        plt.close(fig)

    def test_render_show_labels_true_no_labels(self):
        """show_labels=True but labels=None still goes to the no-label branch."""
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(0, 0)])
        layer = _MarkerLayer(
            point_geometries=pts,
            show_labels=True,
            labels=None,
        )
        fig, ax = plt.subplots()
        layer.render(ax)
        plt.close(fig)


# ==========================
# == OUTLINE LAYER ERRORS ==
# ==========================
