import atexit
import os
import pickle
import random
import tempfile
from pathlib import Path
from warnings import warn

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from joblib import Parallel, delayed
from matplotlib.colors import to_hex
from shapely.geometry import Point

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._gerryplot_option_classes import (
    LabelBoxOptions,
    LabelFontOptions,
    PointMarkerOptions,
)
from gerrytools.plotting.geoplot import GeoPlot
from gerrytools.typing import Color

logger = get_logger(__name__)

MAX_CORES = max(int(os.cpu_count() or 1) - 2, 1)


@staticmethod
def _make_random_points_row(
    row: GeoDataFrame, people_per_dot: int, datacolumn_name: str, color: Color
) -> GeoDataFrame:
    """Generates random points within a polygon."""
    n_points: int = round(row[datacolumn_name] / people_per_dot)
    colors: list[Color] = [color] * n_points
    if n_points == 0:
        gdf = gpd.GeoDataFrame(columns=["geometry", "color"], index=[])
    else:
        poly: GeoSeries = row.geometry
        minx, miny, maxx, maxy = poly.bounds
        points: list[Point] = []
        while len(points) < n_points:
            x: int | float = random.uniform(minx, maxx)
            y: int | float = random.uniform(miny, maxy)
            point = Point(x, y)
            if poly.contains(point):
                points.append(point)

        if len(points) != len(colors):
            raise ValueError(f"Mismatch in number of points {len(points)} and colors {len(colors)}")
        gdf = gpd.GeoDataFrame(data={"geometry": points, "color": colors})

    return gdf


@staticmethod
def _make_random_points(
    gdf: GeoDataFrame, people_per_dot: int, datacolumn_name: str, color: Color, n_jobs=-1
) -> GeoDataFrame:
    """Generates random points within polygons in a GeoDataFrame."""
    use_cores: int = min(MAX_CORES, n_jobs) if n_jobs > 0 else MAX_CORES

    chunk_size = max(1, len(gdf) // (use_cores - 1))

    chunked_gdfs = [
        gdf.iloc[i : min(len(gdf), i + chunk_size)] for i in range(0, len(gdf), chunk_size)
    ]

    def process_chunk(chunk):
        return pd.concat(
            chunk.apply(
                _make_random_points_row,
                axis=1,
                args=(people_per_dot, datacolumn_name, color),
            ).tolist(),
            ignore_index=True,
        )

    results = Parallel(n_jobs=use_cores)(delayed(process_chunk)(chunk) for chunk in chunked_gdfs)

    return gpd.GeoDataFrame(pd.concat(results, ignore_index=True))


class DotDensityPlot(GeoPlot):
    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        plancolumn: str,
        people_per_dot: int = 100,
        show_labels: bool = True,
        exclude_labels: list[str] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        edgecolor: Color = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.6,
        dpi: int = 300,
        show_axis: bool = False,
        target_crs=None,
        include_default_outline: bool = True,
        silent: bool = False,
    ):
        super().__init__(
            gdf=gdf,
            dpi=dpi,
            show_axis=show_axis,
            target_crs=target_crs,
            include_default_outline=include_default_outline,
        )
        self.people_per_dot = people_per_dot

        # outlines for disricts
        self.add_districting_plan_layer(
            plancolumn=plancolumn,
            dissolve=True,
            colormap="none",
            show_labels=show_labels,
            exclude_labels=exclude_labels,
            labelfont_options=labelfont_options
            or LabelFontOptions(
                fontfamily="sans-serif",
                fontsize=4,
                fontweight="bold",
                fontcolor="black",
                outlinecolor="none",
            ),
            labelbox_options=labelbox_options
            or LabelBoxOptions(
                enabled=True,
                boxstyle="circle",
                pad=0.5,
                facecolor="#f1deb8",
                facealpha=1.0,
                edgecolor="black",
                edgealpha=1.0,
                edgewidth=0.5,
            ),
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=100,
        )

        marker_options = PointMarkerOptions(
            marker="o",
            markersize=1.0,
            markerfacecolor="none",
            markerfacealpha=None,
            markeredgecolor="none",
            markeredgealpha=None,
            markeredgewidth=0.0,
        ).to_mpl_scatter_settings_dict()
        self.__global_marker_settings_dict = marker_options

        # Used for caching the dots so that you can iterate quickly when adjusting styles
        self.__temp_dir: tempfile.TemporaryDirectory | None = tempfile.TemporaryDirectory()
        if not getattr(self.__temp_dir, "name", None):
            raise ValueError("tempfile.TemporaryDirectory did not return a valid name attribute.")

        if not os.path.exists(self.__temp_dir.name):
            os.makedirs(self.__temp_dir.name)

        logger.debug(f"Created temporary directory for dot density plot: {self.__temp_dir.name}")
        self.__temp_dir_name = str(self.__temp_dir.name)
        self.__column_to_color_dict: dict[str, Color] = {}
        atexit.register(self._close)
        self._silent = silent

    def _close(self):
        # Safe to call multiple times
        if getattr(self, "_DotDensityPlot__temp_dir", None) is not None:
            logger.debug(f"Cleaning up temporary directory: {self.__temp_dir_name}")
            assert self.__temp_dir is not None
            self.__temp_dir.cleanup()
            self.__temp_dir = None

    def set_marker_options(
        self,
        *,
        marker: str = "o",
        markersize: float = 1.0,
        markeredgecolor: Color = "none",
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.0,
    ):
        options_dict = PointMarkerOptions(
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
        ).to_mpl_scatter_settings_dict()
        self.__global_marker_settings_dict.update(options_dict)

    def add_dot_density(
        self,
        *,
        column_name: str,
        color: Color,
        force_new_dots: bool = False,
        n_cores_for_processing: int = -1,
    ):
        if column_name not in self.gdf.columns:
            raise ValueError(f"Column '{column_name}' not found in GeoDataFrame.")

        if any(self.gdf[column_name] < 0):
            raise ValueError(f"Column '{column_name}' contains negative values.")

        if any(self.gdf[column_name].isna()):
            raise ValueError(f"Column '{column_name}' contains NaN values.")

        color = to_hex(resolve_color_and_alpha(color)[0])
        if self.__column_to_color_dict.get(column_name) == color and not force_new_dots:
            warn(
                f"Dots for column '{column_name}' with the same color already exist. "
                "Use 'force_new_dots=True' to recreate them.",
                UserWarning,
            )
            return

        if (
            column_name in self.__column_to_color_dict
            and self.__column_to_color_dict[column_name] != color
            and not force_new_dots
        ):
            warn(
                f"Overwriting existing dots for column '{column_name}' with new color.",
                UserWarning,
            )
            return

        self.__column_to_color_dict[column_name] = color

        # now to create the dots and cache them
        cache_filepath = (
            Path(self.__temp_dir_name) / f"dots_{column_name}_ppd{self.people_per_dot}.pkl"
        )
        if not cache_filepath.exists() or force_new_dots:
            if not self._silent:
                print(f"Generating dots for column '{column_name}'.")

            dots_gdf = _make_random_points(
                gdf=self.gdf.loc[:, [column_name, "geometry"]],
                people_per_dot=self.people_per_dot,
                datacolumn_name=column_name,
                color=color,
                n_jobs=n_cores_for_processing,
            )
            logger.debug(f"Caching dots for column '{column_name}' at {cache_filepath}")
            with open(cache_filepath, "wb") as f:
                pickle.dump(dots_gdf, f)
            logger.debug(f"Finished caching dots for column '{column_name}'")
            logger.debug(
                "Current cached files: %s", "\n".join(list(os.listdir(self.__temp_dir_name)))
            )

    def _draw_all_dots(self) -> None:
        if len(self.__column_to_color_dict) == 0:
            return
        ax = self._ax
        gdf_list = []
        for column_name, color in self.__column_to_color_dict.items():
            if not self._silent:
                print(f"Drawing dots for column '{column_name}'.")
            cache_filepath = (
                Path(self.__temp_dir_name) / f"dots_{column_name}_ppd{self.people_per_dot}.pkl"
            )
            with open(cache_filepath, "rb") as f:
                dots_gdf = pickle.load(f)
                gdf_list.append(dots_gdf)

        full_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        # Randomly shuffle the points so that no one color is on top
        full_gdf = full_gdf.sample(frac=1, random_state=0).reset_index(drop=True)

        ax.scatter(
            full_gdf.geometry.x,
            full_gdf.geometry.y,
            c=full_gdf["color"],
            **self.__global_marker_settings_dict,
        )

    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and applying settings."""
        super()._build_plot()
        self._draw_all_dots()

    def save_legend(self):
        raise NotImplementedError()
