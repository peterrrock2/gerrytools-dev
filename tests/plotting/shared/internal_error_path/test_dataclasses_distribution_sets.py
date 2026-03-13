import numpy as np
import pytest

from gerrytools.plotting.data.boxplot import BoxPlotSetData
from gerrytools.plotting.data.histogram import HistogramData
from gerrytools.plotting.data.scatterplot import ScatterData
from gerrytools.plotting.data.sealevel import SeaLevelSetData
from gerrytools.plotting.data.violin import ViolinPlotSetData
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions


# ==============
# == LINEDATA ==
# ==============
class TestBoxPlotSetData:
    def test_default_construction(self):
        bps = BoxPlotSetData(name="test", scores_dict={"A": [1, 2, 3]}, facecolor="blue")
        assert bps.edgewidth == 0.8
        assert bps.percentiles == (1, 99)
        assert bps.showfliers is False

    def test_percentiles_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="within"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(-1, 50),
            )

    def test_percentiles_high_greater_than_100_raises_valueerror(self):
        with pytest.raises(ValueError, match="within"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(5, 101),
            )

    def test_percentiles_low_equals_high_raises_valueerror(self):
        with pytest.raises(ValueError, match="low < high"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(50, 50),
            )

    def test_percentiles_low_greater_than_high_raises_valueerror(self):
        with pytest.raises(ValueError, match="low < high"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                percentiles=(99, 1),
            )

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=-1.0,
            )

    def test_infinite_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            BoxPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=float("inf"),
            )

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        bps = BoxPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            edgecolor="none",
            edgewidth=2.0,
        )
        assert bps.edgewidth == 0.0

    def test_zorder_coerced_to_int(self):
        bps = BoxPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            zorder=3.7,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(bps.zorder, int)

    def test_edgewidth_int_coerced_to_float(self):
        bps = BoxPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            edgewidth=2,
        )
        assert isinstance(bps.edgewidth, float)


# =======================
# == VIOLINPLOTSETDATA ==
# =======================


class TestViolinPlotSetData:
    def test_default_construction(self):
        vpsd = ViolinPlotSetData(name="test", scores_dict={"A": [1, 2, 3]}, facecolor="blue")
        assert vpsd.edgewidth == 0.8

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ViolinPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=-1.0,
            )

    def test_infinite_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ViolinPlotSetData(
                name="test",
                scores_dict={"A": [1]},
                facecolor="blue",
                edgewidth=float("inf"),
            )

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        vpsd = ViolinPlotSetData(
            name="test",
            scores_dict={"A": [1]},
            facecolor="blue",
            edgecolor="none",
            edgewidth=2.0,
        )
        assert vpsd.edgewidth == 0.0


# =====================
# == SEALEVELSETDATA ==
# =====================


class TestSeaLevelSetData:
    def test_default_construction(self):
        sl = SeaLevelSetData(name="test", scores_dict={"A": 0.5}, linecolor="black")
        assert sl.linewidth == 2.0

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            SeaLevelSetData(
                name="test",
                scores_dict={"A": 0.5},
                linecolor="black",
                linewidth=-1.0,
            )

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            SeaLevelSetData(
                name="test",
                scores_dict={"A": 0.5},
                linecolor="black",
                linewidth=float("inf"),
            )

    def test_zorder_coerced_to_int(self):
        sl = SeaLevelSetData(
            name="test",
            scores_dict={"A": 0.5},
            linecolor="black",
            zorder=4.9,  # ty: ignore[invalid-argument-type]
        )
        assert isinstance(sl.zorder, int)


# ===================
# == HISTOGRAMDATA ==
# ===================


class TestHistogramData:
    def test_default_construction_with_valid_data(self):
        hd = HistogramData(
            name="test",
            values=np.array([1.0, 2.0, 3.0]),
            weights=np.array([1.0, 1.0, 1.0]),
        )
        assert hd.values.shape == (3,)
        assert hd.weights.shape == (3,)

    def test_non_finite_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            HistogramData(
                name="test",
                values=np.array([1.0, float("nan")]),
                weights=np.array([1.0, 1.0]),
            )

    def test_infinite_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            HistogramData(
                name="test",
                values=np.array([float("inf"), 2.0]),
                weights=np.array([1.0, 1.0]),
            )

    def test_empty_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="no entries"):
            HistogramData(
                name="test",
                values=np.array([]),
                weights=np.array([]),
            )

    def test_weights_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same length"):
            HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.array([1.0]),
            )

    def test_non_finite_weights_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.array([1.0, float("nan")]),
            )

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            HistogramData(
                name="test",
                values=np.array([1.0]),
                weights=np.array([1.0]),
                edgewidth=-1.0,
            )

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        hd = HistogramData(
            name="test",
            values=np.array([1.0]),
            weights=np.array([1.0]),
            edgecolor="none",
            edgewidth=2.0,
        )
        assert hd.edgewidth == 0.0

    def test_values_are_flattened(self):
        hd = HistogramData(
            name="test",
            values=np.array([[1.0, 2.0, 3.0]]),
            weights=np.array([1.0, 1.0, 1.0]),
        )
        assert hd.values.ndim == 1
        assert hd.values.shape == (3,)


# =================
# == SCATTERDATA ==
# =================


class TestScatterData:
    def test_default_construction(self):
        sd = ScatterData(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            label="test",
            marker_options=PointMarkerOptions(),
        )
        assert sd.x.shape == (2,)
        assert sd.label == "test"

    def test_shape_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same shape"):
            ScatterData(
                x=np.array([1.0, 2.0]),
                y=np.array([3.0]),
                label="test",
                marker_options=PointMarkerOptions(),
            )

    def test_multi_dimensional_arrays_raise_valueerror(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            ScatterData(
                x=np.array([[1.0, 2.0]]),
                y=np.array([[3.0, 4.0]]),
                label="test",
                marker_options=PointMarkerOptions(),
            )

    def test_empty_arrays_raise_valueerror(self):
        with pytest.raises(ValueError, match="not be empty"):
            ScatterData(
                x=np.array([]),
                y=np.array([]),
                label="test",
                marker_options=PointMarkerOptions(),
            )

    def test_none_label_is_valid(self):
        sd = ScatterData(
            x=np.array([1.0]),
            y=np.array([2.0]),
            label=None,
            marker_options=PointMarkerOptions(),
        )
        assert sd.label is None


# ====================
# == SEATSVOTESDATA ==
# ====================
