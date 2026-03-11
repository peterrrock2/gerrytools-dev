import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from gerrytools.plotting.data.sealevel import SeaLevel
from gerrytools.plotting.geometry.dotdensity import DotDensityPlot


def _build_sealevel_xy(seed: int) -> tuple[np.ndarray, np.ndarray]:
    plot = SeaLevel(jitter_rng_seed=seed)
    plot.add_sealevel_set({"A": 1.0, "B": 2.0}, name="Series A")
    plot.set_max_horizontal_jitter_all(0.15)
    plot.set_max_vertical_jitter_all(0.25)

    line = plot.ax.lines[0]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def _build_dotdensity_offsets(seed: int) -> np.ndarray:
    gdf = gpd.GeoDataFrame(
        {"district": ["A"], "population": [40]},
        geometry=[Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])],
        crs="EPSG:4326",
    )
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=seed,
    )
    try:
        plot.add_dot_density(
            column_name="population",
            color="black",
            force_new_dots=True,
            n_cores_for_processing=1,
            n_chunks=1,
        )
        offsets = np.asarray(plot.ax.collections[-1].get_offsets())
        return offsets.copy()
    finally:
        plot._close()


def test_sealevel_jitter_seed_is_reproducible():
    x1, y1 = _build_sealevel_xy(12345)
    x2, y2 = _build_sealevel_xy(12345)

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_dotdensity_seed_is_reproducible():
    offsets1 = _build_dotdensity_offsets(2024)
    offsets2 = _build_dotdensity_offsets(2024)

    assert np.array_equal(offsets1, offsets2)


def test_dotdensity_different_seeds_change_output():
    offsets1 = _build_dotdensity_offsets(2024)
    offsets2 = _build_dotdensity_offsets(2025)

    assert not np.array_equal(offsets1, offsets2)
