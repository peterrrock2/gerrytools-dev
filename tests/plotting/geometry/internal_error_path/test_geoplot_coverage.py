import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, box

from gerrytools.plotting.geometry.coloredgeoplot import ColoredGeoPlot
from gerrytools.plotting.geometry.geoplot import (
    _LabelRequest,
)
from gerrytools.plotting.mpl.label_text_options import LabelFontOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions


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


# ==========================
# == OUTLINE LAYER ERRORS ==
# ==========================
class TestAddOutlineLayerErrors:
    def test_dissolve_non_gdf_raises_typeerror(self, testing_gdf):
        """dissolve_column with GeoSeries raises TypeError."""
        gs = testing_gdf.geometry
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(TypeError, match="geosource must be a GeoDataFrame"):
            plot.add_outline_layer(geosource=gs, dissolve_column="district")

    def test_show_labels_without_dissolve_column_raises(self, testing_gdf):
        """show_labels=True without dissolve_column raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="dissolve_column.*must be set"):
            plot.add_outline_layer(show_labels=True)

    def test_show_labels_geoseries_without_dissolve_raises_typeerror(self, testing_gdf):
        """show_labels=True with GeoSeries geosource and no dissolve_column raises TypeError."""
        gs = testing_gdf.geometry
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(TypeError, match="geosource must be a GeoDataFrame"):
            plot.add_outline_layer(geosource=gs, show_labels=True)

    def test_show_labels_with_labelfont_options(self, testing_gdf, tmp_path):
        """show_labels=True with custom labelfont_options uses them."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(
            dissolve_column="district",
            show_labels=True,
            labelfont_options=LabelFontOptions(fontsize=6, fontcolor="red"),
        )
        plot.save(str(tmp_path / "outline_custom_font.png"))
        assert (tmp_path / "outline_custom_font.png").exists()

    def test_show_labels_with_exclude_labels(self, testing_gdf, tmp_path):
        """show_labels=True with exclude_labels excludes them."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(
            dissolve_column="district",
            show_labels=True,
            exclude_labels=[0],
        )
        plot.save(str(tmp_path / "outline_exclude.png"))
        assert (tmp_path / "outline_exclude.png").exists()


# ============================
# == HIGHLIGHT LAYER ERRORS ==
# ============================


class TestAddHighlightLayerErrors:
    def test_show_labels_no_label_column_raises(self, testing_gdf):
        """show_labels=True without label_column raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="label_column"):
            plot.add_highlight_layer(
                geosource=testing_gdf,
                show_labels=True,
                label_column=None,
            )

    def test_show_labels_no_geosource_raises(self, testing_gdf):
        """show_labels=True without geosource raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="geosource"):
            plot.add_highlight_layer(
                show_labels=True,
                label_column="district",
                geosource=None,
            )

    def test_show_labels_geoseries_geosource_raises(self, testing_gdf):
        """show_labels=True with GeoSeries geosource raises TypeError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(TypeError, match="GeoDataFrame"):
            plot.add_highlight_layer(
                geosource=testing_gdf.geometry,
                show_labels=True,
                label_column="district",
            )

    def test_highlight_geosource_none_uses_gdf(self, testing_gdf, tmp_path):
        """When geosource=None, uses base gdf geometry."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer()  # geosource=None
        plot.save(str(tmp_path / "highlight_no_src.png"))
        assert (tmp_path / "highlight_no_src.png").exists()

    def test_highlight_with_geometry_mask(self, testing_gdf, tmp_path):
        """geometry_mask filters highlight geometries."""
        mask = testing_gdf["district"] == 0
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(geometry_mask=mask)
        plot.save(str(tmp_path / "highlight_mask.png"))
        assert (tmp_path / "highlight_mask.png").exists()

    def test_highlight_show_labels_with_mask(self, testing_gdf, tmp_path):
        """geometry_mask with show_labels applies mask to label_gdf."""
        mask = testing_gdf["district"] == 0
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(
            geosource=testing_gdf,
            show_labels=True,
            label_column="district",
            geometry_mask=mask,
        )
        plot.save(str(tmp_path / "highlight_mask_labels.png"))
        assert (tmp_path / "highlight_mask_labels.png").exists()

    def test_highlight_show_labels_with_custom_font(self, testing_gdf, tmp_path):
        """Custom labelfont_options used when provided."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(
            geosource=testing_gdf,
            show_labels=True,
            label_column="district",
            labelfont_options=LabelFontOptions(fontsize=8),
        )
        plot.save(str(tmp_path / "highlight_custom_font.png"))
        assert (tmp_path / "highlight_custom_font.png").exists()


# ==================
# == MARKER LAYER ==
# ==================


class TestAddMarkerLayer:
    def test_add_marker_layer_no_args_raises(self, testing_gdf):
        """Both args None raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="Either"):
            plot.add_marker_layer()

    def test_add_marker_layer_both_args_raises(self, testing_gdf):
        """Both args provided raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        pts = GeoSeries([Point(0, 0)])
        with pytest.raises(ValueError, match="Only one"):
            plot.add_marker_layer(
                points_geoseries=pts,
                latlon_list=[(0.0, 0.0)],
            )

    def test_add_marker_layer_with_lat_lon_list(self, testing_gdf, tmp_path):
        """lat/lon list path builds correctly."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        plot = ColoredGeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:4326")
        plot.add_marker_layer(
            latlon_list=[(40.0, -90.0), (41.0, -91.0)],
        )
        plot.save(str(tmp_path / "marker_latlon.png"))
        assert (tmp_path / "marker_latlon.png").exists()

    def test_add_marker_layer_with_geoseries(self, testing_gdf, tmp_path):
        """GeoSeries path builds correctly."""
        pts = GeoSeries([Point(2.0, 3.0), Point(5.0, 7.0)])
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_marker_layer(points_geoseries=pts, show_labels=False)
        plot.save(str(tmp_path / "marker_geoseries.png"))
        assert (tmp_path / "marker_geoseries.png").exists()

    def test_add_marker_layer_geoseries_with_input_crs(self, testing_gdf, tmp_path):
        """GeoSeries without CRS + input_crs sets the CRS."""
        pts = GeoSeries([Point(-90.0, 40.0)])
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_marker_layer(
            points_geoseries=pts,
            input_crs="EPSG:4326",
            show_labels=False,
        )
        plot.save(str(tmp_path / "marker_input_crs.png"))
        assert (tmp_path / "marker_input_crs.png").exists()

    def test_add_marker_layer_with_all_options(self, testing_gdf, tmp_path):
        """Full marker layer with labels, custom marker options."""
        pts = GeoSeries([Point(2.0, 3.0)])
        marker_options = PointMarkerOptions(
            markerfacecolor="red",
            markerfacealpha=0.8,
            marker="^",
            markersize=4.0,
            markeredgecolor="black",
            markeredgealpha=1.0,
            markeredgewidth=0.5,
        )
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_marker_layer(
            points_geoseries=pts,
            labels=["Test"],
            show_labels=True,
            marker_options=marker_options,
        )
        plot.save(str(tmp_path / "marker_full.png"))
        assert (tmp_path / "marker_full.png").exists()


# =================
# == LABEL LAYER ==
# =================


class TestAddLabelLayer:
    def test_add_label_layer_no_args_raises(self, testing_gdf):
        """Both args None raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="Either"):
            plot.add_label_layer()

    def test_add_label_layer_both_args_raises(self, testing_gdf):
        """Both args provided raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        pts = GeoSeries([Point(0, 0)])
        with pytest.raises(ValueError, match="Only one"):
            plot.add_label_layer(
                points_geoseries=pts,
                latlon_list=[(0.0, 0.0)],
            )

    def test_add_label_layer_with_geoseries(self, testing_gdf, tmp_path):
        """GeoSeries path for add_label_layer builds correctly."""
        pts = GeoSeries([Point(2.0, 3.0), Point(5.0, 7.0)])
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_label_layer(points_geoseries=pts)
        plot.save(str(tmp_path / "label_layer_geoseries.png"))
        assert (tmp_path / "label_layer_geoseries.png").exists()

    def test_add_label_layer_with_lat_lon_list(self, testing_gdf, tmp_path):
        """lat/lon list path for add_label_layer."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        plot = ColoredGeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:4326")
        plot.add_label_layer(latlon_list=[(40.0, -90.0)])
        plot.save(str(tmp_path / "label_layer_latlon.png"))
        assert (tmp_path / "label_layer_latlon.png").exists()

    def test_add_label_layer_with_custom_labels(self, testing_gdf, tmp_path):
        """Custom labels list used instead of default numbering."""
        pts = GeoSeries([Point(2.0, 3.0)])
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_label_layer(
            points_geoseries=pts,
            labels=["MyLabel"],
            labelfont_options=LabelFontOptions(fontsize=6),
        )
        plot.save(str(tmp_path / "label_layer_custom.png"))
        assert (tmp_path / "label_layer_custom.png").exists()


# ================
# == FOCUS AXES ==
# ================


class TestFocusAxesPaths:
    def test_focus_axes_with_geometry_mask(self, testing_gdf):
        """geometry_mask arg filters the geoseries."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        mask = testing_gdf["district"] == 0
        plot.focus_axes(geometry_mask=mask)
        assert plot._xlim is not None

    def test_focus_axes_empty_mask_raises(self, testing_gdf):
        """All-False mask results in empty geoseries → ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        mask = pd.Series([False] * len(testing_gdf), index=testing_gdf.index)
        with pytest.raises(ValueError, match="no geometries"):
            plot.focus_axes(geometry_mask=mask)

    def test_focus_axes_with_crs_reprojection(self):
        """When geoseries.crs != target_crs, to_crs is called."""
        gdf = _rect_gdf_with_crs("EPSG:4326")
        plot = ColoredGeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:3857")
        plot.focus_axes()
        assert plot._xlim is not None

    def test_focus_axes_tuple_pad(self, testing_gdf):
        """Tuple pad is split into (pad_x, pad_y)."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(pad=(0.01, 0.05))
        assert plot._xlim is not None

    def test_focus_axes_invalid_pad_mode_raises(self, testing_gdf):
        """Invalid pad_mode raises ValueError."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="pad_mode must be"):
            plot.focus_axes(pad_mode="invalid_mode")  # ty: ignore [invalid-argument-type]

    def test_focus_axes_geosource_geoseries(self, testing_gdf):
        """GeoSeries as geosource works without error."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(geosource=testing_gdf.geometry)
        assert plot._xlim is not None


# =====================
# == DEFERRED LABELS ==
# =====================


class TestDrawDeferredLabels:
    def test_labels_crs_reprojection(self):
        """Labels with CRS != target_crs triggers to_crs."""
        gdf = _rect_gdf_with_crs("EPSG:4326")
        plot = ColoredGeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:3857")
        plot.add_outline_layer(dissolve_column="category", show_labels=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            plot.save(str(Path(tmpdir) / "labels_crs.png"))

    def test_labels_skip_when_clip_empty(self, testing_gdf, tmp_path):
        """Labels outside the current view are clipped and skipped."""
        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        # Set xlim/ylim to a region outside the data
        plot.set_xlim(1000, 2000)
        plot.set_ylim(1000, 2000)
        plot.save(str(tmp_path / "labels_skip.png"))
        assert (tmp_path / "labels_skip.png").exists()

    def test_label_format_fn_exception_fallback(self, testing_gdf, tmp_path):
        """label_format_fn that raises uses the raw string fallback."""

        def bad_fn(x):
            raise ValueError("intentional error")

        plot = ColoredGeoPlot(testing_gdf, dpi=50, silent=True)
        # Manually insert a label request with a bad format fn
        dissolved = GeoDataFrame(testing_gdf.dissolve(by="district").reset_index())
        req = _LabelRequest(
            gdf=dissolved,
            label_column="district",
            labelfont_options=None,
            labelbox_options=None,
            label_format_fn=bad_fn,
        )
        plot._label_requests.append(req)
        plot.save(str(tmp_path / "label_format_fallback.png"))
        assert (tmp_path / "label_format_fallback.png").exists()


# =====================
# == LABEL POSITIONS ==
# =====================


class TestGetLabelPositionsAsLatLong:
    def test_get_label_positions_as_lat_long(self):
        """as_lat_long=True converts label positions to EPSG:4326."""
        gdf = _rect_gdf_with_crs("EPSG:4326")
        plot = ColoredGeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:4326")
        plot.add_outline_layer(dissolve_column="category", show_labels=True)
        crs_str, positions = plot.get_label_positions(as_lat_long=True)
        assert isinstance(crs_str, str)


# ===========================
# == CONTINUOUS LAYER INIT ==
# ===========================
