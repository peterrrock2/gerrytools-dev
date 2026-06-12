import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.data.histogram import Histogram


# ===============================
# == CONSTRUCTION AND DEFAULTS ==
# ===============================
class TestHistogramConstruction:
    def test_default_construction(self):
        h = Histogram()
        assert h.grid is False
        assert h.as_density_plot is False
        assert h._bins is None
        assert h._binwidth is None

    def test_custom_construction(self):
        h = Histogram()
        h.enable_grid()
        h.suppress_warnings()
        assert h.grid is True
        assert h.hide_warnings is True


# ===================
# == ADD HISTOGRAM ==
# ===================


class TestAddHistogram:
    def test_add_single_overlay_histogram(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0, 4.0])
        assert len(h._hist_data_dict["overlay"]) == 1

    def test_add_multiple_histograms_of_different_types(self):
        h = Histogram()
        h.add_histogram([1, 2, 3], histtype="overlay")
        h.add_histogram([4, 5, 6], histtype="stack")
        h.add_histogram([7, 8, 9], histtype="weave")
        h.add_histogram([1, 2], histtype="outline", edgecolor="black", edgewidth=1.0)
        assert len(h._hist_data_dict["overlay"]) == 1
        assert len(h._hist_data_dict["stack"]) == 1
        assert len(h._hist_data_dict["weave"]) == 1
        assert len(h._hist_data_dict["outline"]) == 1

    def test_invalid_histtype_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="Invalid histtype"):
            h.add_histogram([1, 2, 3], histtype="bad")  # ty: ignore[invalid-argument-type]

    def test_auto_name_for_histogram(self):
        h = Histogram()
        h.add_histogram([1, 2, 3])
        assert h._hist_data_dict["overlay"][0].name == "Overlay histogram 1"

    def test_explicit_name_for_histogram(self):
        h = Histogram()
        h.add_histogram([1, 2, 3], name="My Hist")
        assert h._hist_data_dict["overlay"][0].name == "My Hist"

    def test_add_histogram_with_weights(self):
        h = Histogram()
        h.add_histogram([1.0, 2.0, 3.0], weights=[1.0, 2.0, 3.0])
        hdata = h._hist_data_dict["overlay"][0]
        np.testing.assert_array_equal(hdata.weights, [1.0, 2.0, 3.0])

    def test_add_histogram_from_series(self):
        h = Histogram()
        h.add_histogram(pd.Series([1.0, 2.0, 3.0]))
        assert len(h._hist_data_dict["overlay"]) == 1

    def test_add_histogram_from_dataframe_single_column(self):
        h = Histogram()
        h.add_histogram(pd.DataFrame({"vals": [1.0, 2.0, 3.0]}))
        assert len(h._hist_data_dict["overlay"]) == 1

    def test_add_histogram_from_dataframe_with_column(self):
        h = Histogram()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        h.add_histogram(df, column="b")
        hdata = h._hist_data_dict["overlay"][0]
        np.testing.assert_array_equal(hdata.values, [3.0, 4.0])

    def test_add_histogram_filters_non_finite_values(self):
        h = Histogram()
        h.add_histogram([1.0, float("nan"), 3.0, float("inf")])
        hdata = h._hist_data_dict["overlay"][0]
        assert hdata.values.shape == (2,)

    def test_all_non_finite_values_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="at least one finite"):
            h.add_histogram([float("nan"), float("inf")])

    def test_empty_values_raises_valueerror(self):
        h = Histogram()
        with pytest.raises(ValueError, match="at least one entry"):
            h.add_histogram([])


# ========================================
# == OUTLINE HISTOGRAM AUTO-CORRECTIONS ==
# ========================================


class TestOutlineHistogramCorrections:
    def test_outline_with_zero_edgewidth_warns_and_sets_default(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram([1, 2, 3], histtype="outline", edgewidth=0.0)
            assert any("edgewidth" in str(warning.message) for warning in w)
        hdata = h._hist_data_dict["outline"][0]
        assert hdata.edgewidth == 0.8

    def test_outline_with_non_none_facecolor_warns_and_overrides(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram(
                [1, 2, 3],
                histtype="outline",
                facecolor="red",
                edgecolor="black",
                edgewidth=1.0,
            )
            assert any("facecolor" in str(warning.message) for warning in w)
        hdata = h._hist_data_dict["outline"][0]
        assert isinstance(hdata.facecolor, str)
        assert hdata.facecolor.lower() == "none"

    def test_outline_with_none_edgecolor_warns_and_sets_black(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram(
                [1, 2, 3],
                histtype="outline",
                edgecolor="none",
                edgewidth=1.0,
            )
            assert any("edgecolor" in str(warning.message) for warning in w)

    def test_outline_corrections_suppressed_when_hide_warnings(self):
        h = Histogram()
        h.suppress_warnings()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram([1, 2, 3], histtype="outline", edgewidth=0.0)
            outline_warnings = [x for x in w if "edgewidth" in str(x.message)]
            assert len(outline_warnings) == 0

    def test_outline_with_valid_settings_no_warnings(self):
        h = Histogram()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.add_histogram(
                [1, 2, 3],
                histtype="outline",
                facecolor="none",
                edgecolor="black",
                edgewidth=1.5,
            )
            outline_warnings = [
                x
                for x in w
                if any(kw in str(x.message) for kw in ("edgewidth", "facecolor", "edgecolor"))
            ]
            assert len(outline_warnings) == 0


# ==================
# == BIN SETTINGS ==
# ==================


class TestHistogramDataEdgeCases:
    def test_infinite_edgewidth_raises_valueerror(self):
        from gerrytools.plotting.data.histogram import HistogramData

        with pytest.raises(ValueError, match="edgewidth must be finite"):
            HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.array([1.0, 1.0]),
                facecolor="blue",
                edgecolor="none",
                edgewidth=float("inf"),
            )


# ==========================================
# == NON-UNIFORM BINS + CENTER DATA ERROR ==
# ==========================================
