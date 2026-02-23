import atexit
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from warnings import warn

import numpy as np
import shapely
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from shapely.geometry import Point

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._legend_utils import build_legend_options, save_legend_handles
from gerrytools.plotting.geometry.geoplot import GeoPlot
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import Color, MplCompatibleColor

logger = get_logger(__name__)

MAX_CORES = max(int(os.cpu_count() or 1) - 2, 1)


def _random_points_in_poly(poly: shapely.Geometry, n_points: int, batch_size: int = 4096):
    """Generate random points within a polygon.

    Args:
        poly (shapely.Geometry): The polygon within which to generate points.
        n_points (int): The number of random points to generate.
        batch_size (int, optional): The number of candidate points to generate in each batch.
            Defaults to 4096.
    """
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    while len(pts) < n_points:
        k = max(batch_size, (n_points - len(pts)) * 2)
        xs = np.random.uniform(minx, maxx, size=k)
        ys = np.random.uniform(miny, maxy, size=k)
        cand = shapely.points(xs, ys)
        mask = shapely.contains(poly, cand)
        pts.extend(cand[mask].tolist())
    return pts[:n_points]


def _random_xy_in_poly(poly: shapely.Geometry, n_points: int):
    """Generate random x, y coordinates within a polygon.

    Args:
        poly (shapely.Geometry): The polygon within which to generate points.
        n_points (int): The number of random points to generate.
        batch_size (int, optional): The number of candidate points to generate in each batch.
            Defaults to 4096.
    """
    minx, miny, maxx, maxy = poly.bounds
    xs_out = []
    ys_out = []
    n_points_so_far = 0

    # Each point needs to be checked for inclusion and since we generate the points
    # randomly withing the bounding box, the probabilty of inclusion is area(poly) / area(bbox)
    # we are expected to need roughly n_points / probability_of_inclusion points to get n_points
    # inside of the provided polygon

    probability_of_inclusion = poly.area / ((maxx - minx) * (maxy - miny))
    if probability_of_inclusion <= 0:
        raise ValueError("Polygon has zero area, cannot generate points within it.")
    elif probability_of_inclusion > 0.9:
        # If the polygon is very close to a rectangle, just use a fixed fraction
        # to avoid unnecessarily small batches
        batch_size = min(10_000, (n_points // 5) + 1)
    else:
        estimated_needed_points = int(n_points / probability_of_inclusion) + 1
        batch_size = min(10_000, estimated_needed_points - n_points)

    while n_points_so_far < n_points:
        k = max(batch_size, (n_points - n_points_so_far) * 2)
        xs = np.random.uniform(minx, maxx, size=k)
        ys = np.random.uniform(miny, maxy, size=k)

        cand = shapely.points(xs, ys)
        mask = shapely.contains(poly, cand)

        xs_out.append(xs[mask])
        ys_out.append(ys[mask])
        n_points_so_far += int(mask.sum())

    x = np.concatenate(xs_out)[:n_points]
    y = np.concatenate(ys_out)[:n_points]
    return x, y


def _make_random_points(
    gdf: GeoDataFrame,
    people_per_dot: int,
    datacolumn_name: str,
    color: Color,
    n_jobs=-1,
    n_chunks=10,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generates random points within polygons in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame containing polygons.
        people_per_dot (int): Number of people represented by each dot.
        datacolumn_name (str): The name of the data column to use for dot density.
        color (Color): The color of the dots.
        n_jobs (int): Number of CPU cores to use for parallel processing. Defaults to -1 (all
            available cores minus two).
        n_chunks (int): Number of chunks to split the GeoDataFrame into for parallel processing.
    """
    use_cores: int = min(MAX_CORES, n_jobs) if n_jobs > 0 else MAX_CORES

    chunk_size = max(1, (len(gdf) + n_chunks - 1) // n_chunks)  # ceil
    chunked_gdfs = [
        gdf.iloc[i : min(len(gdf), i + chunk_size)] for i in range(0, len(gdf), chunk_size)
    ]

    def process_chunk(chunk: GeoDataFrame):
        """Generate random dot coordinates for one GeoDataFrame chunk.

        Args:
            chunk (GeoDataFrame): Subset of polygons with density values.

        Returns:
            tuple[NDArray, NDArray, NDArray]: X coordinates, Y coordinates, and polygon ids
                for generated dots.
        """
        x_parts = []
        y_parts = []
        pid_parts = []

        for polyid, (geom, val) in zip(
            chunk.index.to_numpy(), zip(chunk.geometry.values, chunk[datacolumn_name].values)
        ):
            n_dots = int(round(val / people_per_dot))
            if n_dots <= 0:
                continue

            x, y = _random_xy_in_poly(geom, n_dots)
            x_parts.append(x)
            y_parts.append(y)
            pid_parts.append(np.full(n_dots, polyid, dtype=np.int64))

        if not x_parts:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )

        return (
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            np.concatenate(pid_parts),
        )

    results = Parallel(n_jobs=use_cores)(delayed(process_chunk)(chunk) for chunk in chunked_gdfs)

    xs = np.concatenate([r[0] for r in results])
    ys = np.concatenate([r[1] for r in results])
    pids = np.concatenate([r[2] for r in results])
    return xs, ys, pids


class DotDensityPlot(GeoPlot):
    """Class for creating dot density plots from GeoDataFrames.

    Each dot represents a specified number of people, and dots are randomly placed
    within the polygons of the GeoDataFrame. Different data columns can be visualized
    with different colors.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        target_crs: The target CRS for reprojecting geometries.
        silent (bool): Whether to suppress informational output throughout
            the rendering process.
        show_legend (bool): Whether to show the legend.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        outline_column: str,
        dpi: int = 300,
        show_axis: bool = False,
        target_crs=None,
        include_default_outline: bool = False,
        silent: bool = False,
        people_per_dot: int = 100,
        show_labels: bool = True,
        exclude_labels: list[str] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        show_legend: bool = False,
        edgecolor: Color = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.6,
    ):
        """Initialize a DotDensityPlot instance.

        By default, dot density plots will include an outline layer based on the
        `outline_column` provided. This is used to show the boundaries of districts or
        other relevant areas to help provide context for the dot density visualization.

        Args:
            gdf (GeoDataFrame): The base GeoDataFrame for the plot.
            outline_column (str): The column in the GeoDataFrame to use for outlining
                districts or areas.
            people_per_dot (int, optional): Number of people represented by each dot.
                Defaults to 100.
            show_labels (bool, optional): Whether to show labels for the outlined areas.
                Defaults to True.
            exclude_labels (list[str] | None, optional): List of labels to exclude from
                being shown. Defaults to None.
            labelfont_options (LabelFontOptions | None, optional): Font options for labels.
                Defaults to None.
            labelbox_options (LabelBoxOptions | None, optional): Box options for labels.
                Defaults to None.
            edgecolor (Color, optional): Color of the outline edges. Defaults to 'black'.
            edgealpha (float | None, optional): Alpha transparency for the outline edges.
                Defaults to None.
            edgewidth (float, optional): Width of the outline edges. Defaults to 0.6.
            dpi (int, optional): Dots per inch for the plot. Defaults to 300.
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            target_crs (optional): Target CRS for reprojecting geometries. Defaults to None.
            include_default_outline (bool, optional): Whether to include a default outline
                layer. Defaults to False because the outline layer is already being added.
            silent (bool, optional): Whether to suppress informational output throughout
                the rendering process. Defaults to False.
            show_legend (bool, optional): Whether to show the legend. Defaults to False.
        """
        super().__init__(
            gdf=gdf,
            dpi=dpi,
            show_axis=show_axis,
            target_crs=target_crs,
            include_default_outline=include_default_outline,
            silent=silent,
        )
        self.people_per_dot = people_per_dot

        # outlines for districts
        self.add_outline_layer(
            dissolve_column=outline_column,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
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

        self._legend_options = build_legend_options()
        self.show_legend = show_legend

    def _close(self):
        """Clean up temporary directory used for caching dot density points."""
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
        """Set global marker options for all dot density layers.

        This method will set the marker style for all dot density layers in the plot. So all
        dots will share the same marker style with the exception of color, which is set per-layer
        when adding a dot density layer.

        Args:
            marker (str): The marker style (e.g., 'o' for circle, '^' for triangle).
            markersize (float): The size of the markers.
            markeredgecolor (Color): The color of the marker edges.
            markeredgealpha (float | None): The alpha transparency of the marker edges.
            markeredgewidth (float): The width of the marker edges.
        """
        # NOTE: The size of the markers is adjusted to work with ax.scatter in the
        # `to_mpl_scatter_settings_dict` method of PointMarkerOptions.
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
        n_chunks: int = 10,
    ):
        """Add a dot density layer for a specific data column.

        This method generates random dots within the polygons of the GeoDataFrame
        based on the values in the specified data column. Each dot represents a
        certain number of people, defined by `people_per_dot`. The dots are colored
        according to the specified color.

        The Point objects generated are cached in a temporary directory to speed up
        subsequent renderings. If the same column and color are requested again, the
        cached dots will be used unless `force_new_dots` is set to True.

        Args:
            column_name (str): The name of the data column to visualize.
            color (Color): The color of the dots.
            force_new_dots (bool, optional): If True, forces regeneration of dots even if cached.
                Defaults to False.
            n_cores_for_processing (int, optional): Number of CPU cores to use for processing when
                generating dots. Defaults to -1 which will use all available cores minus two.
            n_chunks (int, optional): Number of chunks used to split polygon processing work.
                Defaults to ``10``.
        """
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
                stacklevel=1,
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
                stacklevel=1,
            )
            return

        self.__column_to_color_dict[column_name] = color

        # now to create the dots and cache them
        cache_filepath = (
            Path(self.__temp_dir_name) / f"dots_{column_name}_ppd{self.people_per_dot}.npz"
        )

        if not cache_filepath.exists() or force_new_dots:
            if not self.silent:
                print(f"Generating dots for column '{column_name}'.")

            xs, ys, polyids = _make_random_points(
                gdf=self.gdf.loc[:, [column_name, "geometry"]],
                people_per_dot=self.people_per_dot,
                datacolumn_name=column_name,
                color=color,
                n_jobs=n_cores_for_processing,
                n_chunks=n_chunks,
            )

            np.savez(cache_filepath, x=xs, y=ys, polyids=polyids)

    def __draw_interleaved_scatter_blocks(
        self,
        *,
        layers_xy_polyid: list[tuple[NDArray, NDArray, NDArray]],
        layer_colors: list[MplCompatibleColor],
        block=200_000,
    ):
        """Draw dots from all layers in interleaved blocks for visual mixing.

        Args:
            layers_xy_polyid (list[tuple[NDArray, NDArray, NDArray]]): Per-layer dot data as
                ``(x, y, polygon_id)`` arrays.
            layer_colors (list[MplCompatibleColor]): Per-layer marker colors.
            block (int, optional): Number of points per ``scatter`` call. Defaults to ``200_000``.

        Returns:
            None
        """
        # Build one big table of points with polygon id and a layer id
        xs_all = np.concatenate([x for (x, _, _) in layers_xy_polyid])
        ys_all = np.concatenate([y for (_, y, _) in layers_xy_polyid])
        polyid_all = np.concatenate([polyid for (_, _, polyid) in layers_xy_polyid])

        # layer id per point -> used to map to colors fast
        layer_ids = np.concatenate(
            [np.full(len(x), i, dtype=np.int32) for i, (x, _, _) in enumerate(layers_xy_polyid)]
        )

        palette = np.asarray(layer_colors, dtype=object)
        random_priority = np.random.random(size=len(xs_all))

        # Randomize within each polygon: sort by (polyid, rnd)
        # Lexsort uses last key as primary sort key and avoids copying data
        order = np.lexsort((random_priority, polyid_all))

        xs_all = xs_all[order]
        ys_all = ys_all[order]
        layer_ids = layer_ids[order]

        # Draw in blocks (keeps memory + scatter call size reasonable)
        n = len(xs_all)
        for start in range(0, n, block):
            end = min(n, start + block)
            self._ax.scatter(
                xs_all[start:end],
                ys_all[start:end],
                c=palette[layer_ids[start:end]],
                **self.__global_marker_settings_dict,
            )

    def _draw_all_dots(self) -> None:
        """Draw all dot density layers on the plot."""
        if len(self.__column_to_color_dict) == 0:
            return
        layers_xy_polyid = []
        colors = []

        for column_name, color in self.__column_to_color_dict.items():
            cache_filepath = (
                Path(self.__temp_dir_name) / f"dots_{column_name}_ppd{self.people_per_dot}.npz"
            )
            np_data = np.load(cache_filepath)
            x = np_data["x"]
            y = np_data["y"]
            polyids = np_data["polyids"]
            layers_xy_polyid.append((x, y, polyids))
            colors.append(color)

        if not self.silent:
            n_cols = len(self.__column_to_color_dict)
            cols = list(self.__column_to_color_dict.keys())
            print(
                f"Rendering {sum(len(x) for x, _, _ in layers_xy_polyid):,} dots for column{'s' if n_cols > 1 else ''} '{cols}'..."
            )
        self.__draw_interleaved_scatter_blocks(
            layers_xy_polyid=layers_xy_polyid, layer_colors=colors, block=200_000
        )

    def _draw_legend(self, **legend_kwargs) -> None:
        """Draw the in-axes legend for currently configured dot-density layers.

        Args:
            **legend_kwargs (Any): Extra keyword arguments forwarded to
                ``matplotlib.axes.Axes.legend``.

        Returns:
            None
        """
        if not self.show_legend or not self.__column_to_color_dict:
            return

        settings = {}
        settings["marker"] = self.__global_marker_settings_dict.get("marker", "o")
        settings["markeredgecolor"] = self.__global_marker_settings_dict.get("edgecolor", "none")
        settings["markeredgewidth"] = self.__global_marker_settings_dict.get("linewidths", 0.0)
        if getattr(settings["markeredgewidth"], "__len__", None) is not None:
            settings["markeredgewidth"] = settings["markeredgewidth"][0]

        handles = [
            Line2D(
                [0],
                [0],
                label=label,
                linestyle="",
                markerfacecolor=color,
                **settings,
            )
            for label, color in self.__column_to_color_dict.items()
        ]

        legend_options = self._legend_options.to_dict() | legend_kwargs
        self._ax.legend(handles=handles, **legend_options)

    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and applying settings."""
        super()._build_plot()
        self._draw_all_dots()
        self._draw_legend()

    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Build the plot and apply stored settings like limits."""
        self._build_plot()
        self._apply_limits()
        label_points = self._draw_deferred_labels()
        return label_points

    def set_legend_options(
        self,
        *,
        loc: str | int = "center left",
        bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = (
            1.01,
            0.5,
        ),
        ncols: int = 1,
        fontsize: float | str | None = None,
        frameon: bool = True,
        fancybox: bool = False,
        shadow: bool = False,
        framealpha: float | None = None,
        facecolor: Color | None = None,
        edgecolor: Color | None = None,
        title: str | None = None,
        alignment: Literal["center", "left", "right"] = "center",
        labelspacing: float = 0.5,
        columnspacing: float = 2.0,
    ) -> None:
        """Set legend options used by ``Axes.legend`` during plot build.

        Args:
            loc (str | int, optional): Matplotlib legend location. Defaults to
                ``"center left"``.
            bbox_to_anchor (tuple[float, float] | tuple[float, float, float, float] | None,
                optional): Legend anchor box. Defaults to ``(1.01, 0.5)``.
            ncols (int, optional): Number of legend columns. Defaults to ``1``.
            fontsize (float | str | None, optional): Legend text size. Defaults to None.
            frameon (bool, optional): Whether to draw the legend frame. Defaults to True.
            fancybox (bool, optional): Whether to use a rounded frame. Defaults to False.
            shadow (bool, optional): Whether to draw a shadow. Defaults to False.
            framealpha (float | None, optional): Frame alpha override. Defaults to None.
            facecolor (Color | None, optional): Frame face color. Defaults to None.
            edgecolor (Color | None, optional): Frame edge color. Defaults to None.
            title (str | None, optional): Legend title. Defaults to None.
            alignment (Literal["center", "left", "right"], optional): Legend content
                alignment. Defaults to ``"center"``.
            labelspacing (float, optional): Vertical spacing between entries.
                Defaults to ``0.5``.
            columnspacing (float, optional): Horizontal spacing between columns.
                Defaults to ``2.0``.

        Returns:
            None
        """
        self._legend_options = build_legend_options(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncols=ncols,
            fontsize=fontsize,
            frameon=frameon,
            fancybox=fancybox,
            shadow=shadow,
            framealpha=framealpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            title=title,
            alignment=alignment,
            labelspacing=labelspacing,
            columnspacing=columnspacing,
        )

    def save_legend(
        self,
        filepath: str,
        *,
        column_to_display_name: dict[str, str] | None = None,
        outer_padding: float = 0.07,
        dpi: int | None = None,
        **legend_kwargs: Any,
    ) -> None:
        """Save the legend to a separate file.

        Args:
            filepath (str): The file path to save the legend to.
            column_to_display_name (dict[str, str] | None, optional): A mapping from original
                column names to new display names for the legend. If None, original column names
                are used. Defaults to None.
            dpi (int | None, optional): The DPI to use when saving the legend. If None, uses the
                same DPI as the main figure. Defaults to None.
            outer_padding (float, optional): The outer padding around the legend.
                Defaults to 0.07.
            **legend_kwargs (Any): Additional keyword arguments passed to
                ``matplotlib.axes.Axes.legend``.

        Returns:
            None
        """

        if not self.__column_to_color_dict:
            print("No legend to save.")
            return

        settings = {}
        settings["marker"] = self.__global_marker_settings_dict.get("marker", "o")
        settings["markeredgecolor"] = self.__global_marker_settings_dict.get("edgecolor", "none")
        settings["markeredgewidth"] = self.__global_marker_settings_dict.get("linewidths", 0.0)
        if getattr(settings["markeredgewidth"], "__len__", None) is not None:
            settings["markeredgewidth"] = settings["markeredgewidth"][0]

        column_to_color_dict = self.__column_to_color_dict
        if column_to_display_name is not None:
            column_to_color_dict = {
                column_to_display_name.get(label, label): color
                for label, color in self.__column_to_color_dict.items()
            }

        handles = [
            Line2D(
                [0],
                [0],
                label=label,
                linestyle="",
                markerfacecolor=color,
                **settings,
            )
            for label, color in column_to_color_dict.items()
        ]

        save_legend_handles(
            handles=handles,
            legend_options=self._legend_options,
            filepath=filepath,
            outer_padding=outer_padding,
            dpi=dpi or self.fig.dpi,
            **legend_kwargs,
        )
