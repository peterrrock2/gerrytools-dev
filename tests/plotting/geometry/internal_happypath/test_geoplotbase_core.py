import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


from gerrytools.plotting.geometry.geoplot import GeoPlot


# ========================
# == CRS / REPROJECTION ==
# ========================
class TestGeoLayerCRSReprojection:
    """Cover _geometries_in_crs paths when CRS is missing or already matches."""

    def test_no_crs_returns_as_is(self, testing_gdf):
        """GDF with no CRS should not attempt reprojection."""
        assert testing_gdf.crs is None
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        # If reprojection logic explodes we'd get an error here; just build.
        plot.save(str(Path(tempfile.mkdtemp()) / "out.png"))

    def test_matching_crs_returns_as_is(self, testing_gdf):
        """When source CRS == target CRS, layer is returned unchanged."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        plot = GeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:4326")
        plot.save(str(Path(tempfile.mkdtemp()) / "out.png"))


# ================================
# == GEOMETRY_MASK in _GeoLayer ==
# ================================


class TestGeoLayerGeometryMask:
    """Cover the geometry_mask filtering code path in _GeoLayer."""

    def test_geometry_mask_filters_rows(self, testing_gdf, tmp_path):
        """A boolean geometry_mask should restrict which geometries are rendered."""
        mask = testing_gdf["district"] == 0
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(geometry_mask=mask)
        plot.save(str(tmp_path / "masked.png"))
        assert (tmp_path / "masked.png").exists()


# ===============================================
# == _CategoricalColorLayer COLOR SERIES PATHS ==
# ===============================================


class TestGeoPlotXYLimits:
    """Cover set_xlim, set_ylim, set_xlim, set_ylim, and clear_limits."""

    def test_set_xlim_and_ylim(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        assert plot._xlim == (0.0, 12.0)
        assert plot._ylim == (0.0, 12.0)
        plot.save(str(tmp_path / "limits.png"))

    def test_set_xlim_ylim_aliases(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        assert plot._xlim == (0.0, 12.0)
        assert plot._ylim == (0.0, 12.0)

    def test_clear_limits(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        plot.clear_limits()
        assert plot._xlim is None
        assert plot._ylim is None


# =========================
# == CHECK AXIS FOCUSING ==
# =========================


class TestGeoPlotFocusAxes:
    """Cover focus_axes method paths."""

    def test_focus_axes_default(self, testing_gdf, tmp_path):
        """focus_axes() with no args uses base gdf bounds."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes()
        assert plot._xlim is not None
        assert plot._ylim is not None

    def test_focus_axes_with_geosource_and_pad(self, testing_gdf, tmp_path):
        """focus_axes(geosource=subset, pad=0.1) should limit to subset bounds."""
        subset = testing_gdf[testing_gdf["district"] == 0]
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(geosource=subset, pad=0.1)
        plot.save(str(tmp_path / "focus.png"))

    def test_focus_axes_pad_data_mode(self, testing_gdf):
        """focus_axes with pad_mode='data' should work."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(pad=0.5, pad_mode="data")
        assert plot._xlim is not None


# ======================================
# == VALIDATE _apply_limits EXECUTION ==
# ======================================


class TestGeoPlotApplyLimits:
    """Cover _apply_limits when xlim and ylim are set."""

    def test_apply_limits_on_build(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        plot.save(str(tmp_path / "applied.png"))
        assert (tmp_path / "applied.png").exists()


# =========================
# == SAVE AND SHOW TESTS ==
# =========================


class TestGeoPlotSaveMethod:
    def test_save_creates_file(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        out = str(tmp_path / "out.png")
        plot.save(out)
        assert Path(out).exists()


class TestGeoPlotShowMethod:
    def test_show_does_not_raise(self, testing_gdf, tmp_path, monkeypatch):
        """show() should not raise in non-GUI (Agg) backend."""
        import gerrytools.plotting.geometry.geoplotbase as geoplot_module

        saved = []

        def fake_show(fig, *, non_gui_filename, non_gui_prefix):
            out = tmp_path / non_gui_filename
            fig.savefig(str(out))
            saved.append(str(out))

        monkeypatch.setattr(geoplot_module, "show_figure", fake_show)
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.show()
        assert saved
        assert Path(saved[0]).exists()


# ================================
# == RETRIEVING LABEL POSITIONS ==
# ================================


class TestGeoPlotGetLabelPositions:
    def test_get_label_positions_returns_dict(self, testing_gdf, tmp_path):
        """get_label_positions() should return a (crs_str, dict) tuple."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        crs_str, positions = plot.get_label_positions()
        assert isinstance(crs_str, str)
        assert isinstance(positions, dict)


# ============
# == LABELS ==
# ============


class TestGeoPlotOutlineLayerLabels:
    def test_outline_layer_with_labels(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        plot.save(str(tmp_path / "labeled_outline.png"))
        assert (tmp_path / "labeled_outline.png").exists()

    def test_highlight_layer_with_labels(self, testing_gdf, tmp_path):
        subset = testing_gdf[testing_gdf["district"] == 0].copy()
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(
            geosource=subset,
            show_labels=True,
            label_column="district",
        )
        plot.save(str(tmp_path / "labeled_highlight.png"))
        assert (tmp_path / "labeled_highlight.png").exists()


class TestGeoPlotWithLabels:
    def test_build_with_label_request(self, testing_gdf, tmp_path):
        """Exercising the deferred-label draw path."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="county", show_labels=True)
        plot.save(str(tmp_path / "county_labels.png"))
        assert (tmp_path / "county_labels.png").exists()


# ===============================
# == GeoPlot choropleth ==
# ===============================
