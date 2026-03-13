import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.plotting.data.sealevel import SeaLevel


# ==================
# == CONSTRUCTION ==
# ==================
class TestSeaLevelBuildPreconditions:
    def test_no_labels_raises_valueerror(self):
        sl = SeaLevel()
        with pytest.raises(ValueError, match="No labels"):
            sl.ax

    def test_no_datasets_raises_valueerror(self):
        sl = SeaLevel()
        sl._labels = ["A"]
        with pytest.raises(ValueError, match="No sealevel sets"):
            sl.ax


# ======================
# == CATEGORY CENTERS ==
# ======================


class TestSeaLevelCategoryCenters:
    def test_centers_are_1_indexed(self):
        sl = SeaLevel()
        sl.add_sealevel_set({"A": 0.5, "B": 0.7, "C": 0.3})
        centers = sl._sealevel_centers
        np.testing.assert_array_equal(centers, [1.0, 2.0, 3.0])

    def test_no_labels_returns_empty(self):
        sl = SeaLevel()
        assert len(sl._sealevel_centers) == 0
