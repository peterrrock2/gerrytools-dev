import warnings

import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.histogram import Histogram


# ===============================
# == CONSTRUCTION AND DEFAULTS ==
# ===============================
class TestHistogramActualBuilds:
    """Every test here calls .ax to exercise the draw path."""

    def test_build_overlay_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = h.ax
        assert ax is not None

    def test_build_stack_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], histtype="stack")
        h.add_histogram([4.0, 5.0, 6.0], histtype="stack")
        ax = h.ax
        assert ax is not None

    def test_build_weave_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], histtype="weave")
        h.add_histogram([4.0, 5.0, 6.0], histtype="weave")
        ax = h.ax
        assert ax is not None

    def test_build_outline_histogram(self):
        h = Histogram()
        h.add_histogram(
            [1.0, 2.0, 3.0],
            histtype="outline",
            facecolor="none",
            edgecolor="black",
            edgewidth=1.0,
        )
        ax = h.ax
        assert ax is not None

    def test_build_with_explicit_bins(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.set_bins(5)
        ax = h.ax
        assert ax is not None

    def test_build_with_binwidth(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.set_bins_by_width(0.5)
        ax = h.ax
        assert ax is not None

    def test_build_with_points_above(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0])
        h.add_points_above(2.5, name="Threshold")
        ax = h.ax
        assert ax is not None

    def test_build_with_centered_bin_alignment(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.center_data_on_bin_edges()
        ax = h.ax
        assert ax is not None

    def test_build_as_density(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        h.transform_to_density()
        ax = h.ax
        assert ax is not None

    def test_build_with_grid(self):
        h = Histogram()
        h.enable_grid()
        h.add_histogram([1.0, 2.0, 3.0])
        ax = h.ax
        assert ax is not None

    def test_build_with_weights(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], weights=[1.0, 2.0, 3.0])
        ax = h.ax
        assert ax is not None

    def test_build_with_point_outside_bin_range(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0])
        h.add_points_above(10.0, name="OutOfRange")
        ax = h.ax
        assert ax is not None

    def test_build_stacked_with_point_shows_stacked_height(self):
        h = Histogram()
        h.add_histogram([1.0, 1.5, 2.0], histtype="stack")
        h.add_histogram([1.0, 1.5, 2.0], histtype="stack")
        h.add_points_above(1.2, name="Plan")
        ax = h.ax
        assert ax is not None


class TestHistogramCenterOnBinEdgeErrors:
    """Non-uniform bins cannot be combined with centered bin labels."""

    def test_non_uniform_bins_with_centering_raises(self):
        h = Histogram()
        h.suppress_warnings()
        h.set_bins([0, 1, 3, 6])
        h.center_data_on_bin_edges()
        h.add_histogram([0.5, 1.5, 4.0])
        with pytest.raises(ValueError, match="Cannot center histogram"):
            h.ax


# ===================================
# == WEAVE/STACK EDGEWIDTH WARNING ==
# ===================================


class TestHistogramWeaveEdgeWidthWarning:
    """Weave and stack modes warn when edge widths are enabled."""

    def test_weave_with_positive_edgewidth_warns(self):

        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], histtype="weave", edgewidth=1.0, edgecolor="black")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.ax
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert any("edgewidth" in str(x.message).lower() for x in user_warnings)


# ==========================================
# == OUTLINE HISTOGRAM WITH CENTERED BINS ==
# ==========================================


class TestHistogramOutlineCenteredBins:
    """Outline histograms still build when centered bins are enabled."""

    def test_outline_histogram_with_centered_bins(self):
        h = Histogram()
        h.suppress_warnings()
        h.center_data_on_bin_edges()
        h.add_histogram(
            [1.0, 1.0, 2.0, 2.0, 3.0],
            histtype="outline",
            edgecolor="black",
            edgewidth=1.0,
        )
        ax = h.ax
        assert ax is not None


# ============================================
# == POINTS ABOVE WITH centered_on_bin=True ==
# ============================================


class TestHistogramPointsAboveCentered:
    """Centered bins adjust the point-above placement path."""

    def test_points_above_centered_on_bin(self):
        h = Histogram()
        h.suppress_warnings()
        h.add_histogram([1.0, 1.0, 2.0, 2.0, 3.0])
        h.add_points_above([1.0, 2.0, 3.0], centered_on_bin=True)
        ax = h.ax
        assert ax is not None


# =======================
# == DEGENERATE MARKER ==
# =======================


class TestMarkerClearanceZeroHeightPath:
    """Zero-height marker paths still compute marker clearance safely."""

    def test_horizontal_bar_marker_hits_zero_height_branch(self):
        """The '_' marker has all vertices at y=0, giving height_u=0."""
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        h.add_points_above(3.0, marker="_")
        ax = h.ax
        assert ax is not None
