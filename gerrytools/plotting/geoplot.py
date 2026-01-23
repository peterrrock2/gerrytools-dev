from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Literal, Union

import geopandas as gpd
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from numpy import linspace
from shapely.geometry import Point, box

from gerrytools.colors import districtr, resolve_color_and_alpha
from gerrytools.plotting.gerryplot import PointMarkerOptions
from gerrytools.typing import Color

GeoSource = GeoDataFrame | GeoSeries


def _as_geoseries(source: GeoSource) -> gpd.GeoSeries:
    return source.geometry if isinstance(source, gpd.GeoDataFrame) else source


@dataclass(frozen=True, slots=True)
class _GeoLayer(ABC):
    """Abstract base class for a geographic layer to be rendered on a GeoPlot.

    Attributes:
        geosource (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        datacolumn (str | None): Optional data column for color mapping. Default is None.
        colormap (str | Color | Colormap | dict[Any, Color] | pd.Series): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (Any): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color): Color for geometry edges. Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
    """

    # Try to keep the GeoSource as a reference so that users don't copy the polygons all the time.
    geometry_source: GeoSource
    geometry_mask: pd.Series | None = None
    datacolumn: str | None = None
    colormap: str | Color | Colormap | dict[Any, Color] | pd.Series = "Purples"
    missing_color: Any = "lightgrey"
    facealpha: float | None = None
    edgecolor: Color = "none"
    edgealpha: float | None = None
    edgewidth: float = 0.5
    zorder: int = 1

    def _geometries_in_crs(self, target_crs) -> gpd.GeoSeries:
        """Return this layer's geometries (respecting mask) reprojected to target_crs.

        Args:
            target_crs: The target CRS to reproject to.

        Returns:
            gpd.GeoSeries: The geometries in the target CRS.
        """
        geoseries = self.geometries

        # If either side has no CRS, don't try to reproject; let GeoPandas/Matplotlib handle.
        if getattr(geoseries, "crs", None) is None or target_crs is None:
            return geoseries

        if geoseries.crs != target_crs:
            return geoseries.to_crs(target_crs)

        return geoseries

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_color",
            resolve_color_and_alpha(
                self.missing_color, alpha=self.facealpha, field="missing_color"
            ),
        )

    @property
    @abstractmethod
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries."""
        raise NotImplementedError

    @property
    def geometries(self) -> gpd.GeoSeries:
        """Get this layer's geometries, applying any geometry mask."""
        gs = _as_geoseries(self.geometry_source)
        if self.geometry_mask is not None:
            gs = gs[self.geometry_mask]
        return gs

    @property
    def geosource(self) -> GeoSource:
        """Get this layer's geosource, applying any geometry mask."""
        if self.geometry_mask is not None:
            if isinstance(self.geometry_source, GeoDataFrame):
                return self.geometry_source[self.geometry_mask]
            else:
                return self.geometry_source[self.geometry_mask]
        return self.geometry_source

    @abstractmethod
    def render(self, ax: Axes, **kwargs) -> Axes:
        """Render this layer onto the given Axes."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _ContinuousColorLayer(_GeoLayer):
    """A geographic layer with continuous color mapping based on a data column.

    Attributes:
        geosource (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        datacolumn (str | None): Optional data column for color mapping. Default is None.
        colormap (str | Color | Colormap | dict[Any, Color] | pd.Series): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (Any): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color): Color for geometry edges. Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
        colormap (str | Colormap): The colormap to use for continuous color mapping.
        vmin (float | None): Lower bound value for color mapping.
        vmax (float | None): Upper bound value for color mapping.
        bins (int | list[float] | None): Optional binning specification for discrete intervals.
    """

    colormap: str | Colormap = "Purples"
    vmin: float | None = None
    vmax: float | None = None
    bins: int | list[float] | None = None

    def __post_init__(self) -> None:
        super(_ContinuousColorLayer, self).__post_init__()
        if not isinstance(self.geometry_source, GeoDataFrame):
            raise TypeError(
                "Tried to create a continuous color layer using geosource of type "
                f"{type(self.geometry_source).__name__!r}; geosource must be a GeoDataFrame",
            )

        if not isinstance(self.colormap, (str, Colormap)):
            raise TypeError(
                "'colormap' must be a str or Colormap for continuous color layers; got "
                f"{type(self.colormap).__name__!r}",
            )
        if isinstance(self.colormap, str) and self.colormap not in plt.colormaps():
            raise ValueError(
                f"Colormap name {self.colormap!r} not found in matplotlib colormaps. "
                f"Available colormaps are: {plt.colormaps()}",
            )
        if self.datacolumn is None:
            raise TypeError("'datacolumn' must be set for color-mapped layers")

    def _data_series(self) -> pd.Series:
        """Get the data series (used in color mapping)."""
        return self.geometry_source[self.datacolumn]

    def _effective_bounds(self, dataseries: pd.Series) -> tuple[float, float]:
        """Determine the effective data bounds for color mapping.

        Args:
            dataseries (pd.Series): The data series to analyze.
        """
        non_na = dataseries.dropna()
        if non_na.empty:
            lo = float(self.vmin if self.vmin is not None else 0.0)
            hi = float(self.vmax if self.vmax is not None else 1.0)
            return lo, hi

        lo = float(non_na.min() if self.vmin is None else self.vmin)
        hi = float(non_na.max() if self.vmax is None else self.vmax)
        return lo, hi

    def _bin_boundaries(self, lower: float, upper: float) -> pd.IntervalIndex:
        """Get the bin boundaries as an IntervalIndex.

        Args:
            lower (float): The lower bound of the data.
            upper (float): The upper bound of the data.
        """
        if self.bins is None:
            raise RuntimeError("Called _bin_boundaries but bins is None")

        if isinstance(self.bins, int):
            return pd.interval_range(
                start=lower,
                end=upper,
                periods=self.bins,
                closed="left",
            )

        return pd.IntervalIndex.from_breaks(self.bins, closed="left")

    def _color_mapping_for_bins(
        self, boundaries: pd.IntervalIndex
    ) -> tuple[list[float], list[str]]:
        """Get the color mapping for the given bin boundaries.

        Args:
            boundaries (pd.IntervalIndex): The bin boundaries.

        Returns:
            tuple[list[float], list[str]]: The edges and corresponding hex colors for the bins.
        """
        cmap = get_cmap(self.colormap, lut=len(boundaries))
        colors = []
        for i in range(len(boundaries)):
            rgba = cmap(i)
            if self.facealpha is not None:
                rgba = (rgba[0], rgba[1], rgba[2], float(self.facealpha))
            colors.append(to_hex(rgba, keep_alpha=True))
        edges = boundaries.left.tolist() + [boundaries.right[-1]]
        return edges, colors

    @staticmethod
    def _with_alpha(cmap: Colormap, alpha: float) -> Colormap:
        """Return a copy of the given colormap with the specified alpha applied.

        Args:
            cmap (Colormap): The original colormap.
            alpha (float): The alpha value to apply (0.0 to 1.0).

        Returns:
            Colormap: A new colormap with the specified alpha applied.
        """
        n = getattr(cmap, "N", 256)
        rgba = cmap(linspace(0, 1, n))
        rgba[:, 3] = float(alpha)
        return ListedColormap(rgba, name=f"{cmap.name}_a{alpha:g}")

    def _mappable(self) -> tuple[ScalarMappable, dict[str, Any]]:
        """Get a ScalarMappable and the colorbar kwargs for this layer.

        Returns:
            tuple[ScalarMappable, dict[str, Any]]: The ScalarMappable and colorbar kwargs.
        """
        s = self._data_series()
        lower, upper = self._effective_bounds(s)

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower, upper)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            listed = ListedColormap(interval_hex_colors)

            norm = BoundaryNorm(edges, ncolors=listed.N, clip=False)

            m = ScalarMappable(norm=norm, cmap=listed)
            m.set_array([])

            cbar_kwargs = {
                "ticks": edges,
                "spacing": "uniform",
                "boundaries": edges,
            }
        else:
            norm = Normalize(vmin=lower, vmax=upper)

            cmap = get_cmap(self.colormap)
            if self.facealpha is not None:
                cmap = self._with_alpha(cmap, self.facealpha)
            m = ScalarMappable(norm=norm, cmap=cmap)
            m.set_array([])
            cbar_kwargs = {}

        return m, cbar_kwargs

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        data_series = self._data_series()
        lower_bound, upper_bound = self._effective_bounds(data_series)

        colors: dict[Any, Any] = {}

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower_bound, upper_bound)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            interval_to_hex = {
                interval: interval_hex_colors[i] for i, interval in enumerate(boundaries)
            }

            for idx, value in data_series.items():
                if pd.isna(value):
                    colors[idx] = self.missing_color
                    continue

                # ensure upper bound gets last bin
                if value == upper_bound:
                    interval_i = len(boundaries) - 1
                else:
                    try:
                        loc = boundaries.get_loc(value)
                        interval_i = int(loc) if not isinstance(loc, slice) else loc.start
                    except KeyError:
                        if value < boundaries.left[0]:
                            interval_i = 0
                        elif value > boundaries.right[-1]:
                            interval_i = len(boundaries) - 1
                        else:
                            interval_i = -1

                if interval_i == -1:
                    colors[idx] = self.missing_color
                else:
                    colors[idx] = resolve_color_and_alpha(
                        interval_to_hex[boundaries[interval_i]],
                        alpha=self.facealpha,
                    )

        else:
            norm = Normalize(vmin=lower_bound, vmax=upper_bound)

            cmap = get_cmap(self.colormap)

            for idx, value in data_series.items():
                if pd.isna(value):
                    color = self.missing_color
                else:
                    normalized_value = norm(value)
                    color: tuple[str, int | float] = resolve_color_and_alpha(
                        to_hex(cmap(normalized_value), keep_alpha=True),
                        alpha=self.facealpha,
                    )
                colors[idx] = color

        return pd.Series(colors).reindex(self.geometries.index)

    def render(self, ax: Axes, *, target_crs=None, **kwargs) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs: The target CRS to reproject geometries to.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Axes: The Axes with the layer rendered.
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        if self.datacolumn not in self.geometry_source.columns:
            raise KeyError(
                f"Column {self.datacolumn!r} not found in GeoDataFrame."
                f" Available columns: {list(self.geometry_source.columns)}"
            )

        edge_color_tup = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
        )

        geoseries = self._geometries_in_crs(target_crs)
        _ = geoseries.plot(
            ax=ax,
            color=self.color_series,
            edgecolor=edge_color_tup,
            linewidth=self.edgewidth,
            zorder=self.zorder,
        )

        return ax


@dataclass(frozen=True, slots=True)
class _CategoricalColorLayer(_GeoLayer):
    """A geographic layer with categorical color mapping based on a data column.

    Attributes:
        geosource (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        datacolumn (str | None): Optional data column for color mapping. Default is None.
        colormap (str | Color | Colormap | dict[Any, Color] | pd.Series): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (Any): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color): Color for geometry edges. Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
        colormap (str | Color | Colormap | dict[Any, Color] | pd.Series): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "districtr".
    """

    colormap: str | Color | Colormap | dict[Any, Color] | pd.Series = "districtr"

    def __post_init__(self) -> None:
        super(_CategoricalColorLayer, self).__post_init__()

        if isinstance(self.geometry_source, GeoSeries) and self.colormap == "districtr":
            object.__setattr__(self, "colormap", "none")

        needs_datacolumn = (
            self.colormap == "districtr"
            or isinstance(self.colormap, (dict, pd.Series, Colormap))
            or (isinstance(self.colormap, str) and self.colormap in plt.colormaps())
        )

        if (
            isinstance(self.geometry_source, GeoDataFrame)
            and needs_datacolumn
            and self.datacolumn is None
        ):
            raise TypeError("'datacolumn' must be set for color-mapped layers")

        if self.colormap == "districtr" and isinstance(self.geometry_source, GeoDataFrame):
            unique_values = self.geometry_source[self.datacolumn].unique()
            districtr_colors = districtr(len(unique_values))
            object.__setattr__(
                self,
                "colormap",
                self.__map_unique_values_to_colors(unique_values, districtr_colors),
            )

    @staticmethod
    def __map_unique_values_to_colors(
        unique_values: pd.Index,
        color_list: list[Color],
    ) -> dict[Any, Color]:
        """Map unique values in the data to colors from the provided list. Filters out NaN values.

        Args:
            unique_values (pd.Index): The unique values to map.
            color_list (list[Color]): The list of colors to use for mapping.
        """
        n_colors = len(color_list)
        non_na_values = list(filter(pd.notna, unique_values))
        if len(non_na_values) > n_colors:
            raise ValueError(
                "Not enough colors provided to map all unique values; "
                f"received {n_colors} colors for {len(unique_values)} unique values",
            )

        # Try to convert to integers and sort by those if possible
        # Just in case the values are something like ["1", "2", "10"]
        # which would incorrectly sort to ["1", "10", "2"] as strings
        try:
            key_int_pairs = [(key, int(key)) for key in non_na_values]
            sorted_keys = sorted(key_int_pairs, key=lambda x: x[1])
            keys_in_order = [k for (k, _) in sorted_keys]
        except (ValueError, TypeError):
            keys_in_order = sorted(non_na_values)

        return {k: color_list[i] for i, k in enumerate(keys_in_order)}

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        ret_colors_series: pd.Series

        if self.colormap is None:
            ret_colors_series = pd.Series(
                ["none"] * len(self.geometry_source), index=self.geometry_source.index
            )

        elif isinstance(self.colormap, str) and (
            self.colormap not in plt.colormaps() or self.datacolumn is None
        ):
            color = resolve_color_and_alpha(self.colormap, alpha=self.facealpha)
            ret_colors_series = pd.Series(
                [color] * len(self.geometry_source), index=self.geometry_source.index
            )

        elif isinstance(self.colormap, pd.Series):
            new_entries = [resolve_color_and_alpha(c, alpha=self.facealpha) for c in self.colormap]
            ret_colors_series = pd.Series(new_entries, index=self.colormap.index)
        elif isinstance(self.colormap, Colormap) or (
            isinstance(self.colormap, str) and self.colormap in plt.colormaps()
        ):
            cmap = self.colormap
            if isinstance(self.colormap, str):
                cmap = get_cmap(self.colormap)

            # Almost all color maps have at most 256 discrete colors (even the "continuous" ones).
            # This is just a safeguard to avoid indexing errors
            n_colors = int(getattr(cmap, "N", 256))

            value_to_color_dict = self.__map_unique_values_to_colors(
                self.geometry_source[self.datacolumn].unique(),
                [to_hex(cmap(i), keep_alpha=True) for i in range(n_colors)],
            )

            new_entries = []
            for val in self.geometry_source[self.datacolumn]:
                new_color = self.missing_color
                if pd.notna(val):
                    # Try to convert to integer index
                    new_color = resolve_color_and_alpha(
                        value_to_color_dict[val], alpha=self.facealpha
                    )
                new_entries.append(new_color)
            ret_colors_series = pd.Series(new_entries, index=self.geometry_source.index)

        elif isinstance(self.colormap, dict):
            new_entries: list[tuple[str, int | float]] = []
            for val in self.geometry_source[self.datacolumn]:
                color = self.colormap.get(val, self.missing_color)
                color_tup = resolve_color_and_alpha(color, alpha=self.facealpha)
                new_entries.append(color_tup)
            ret_colors_series = pd.Series(new_entries, index=self.geometry_source.index)
        else:
            raise TypeError(
                "'colormap' must be one of: None, str (named colormap or color), "
                "Colormap, dict, or pd.Series; got "
                f"{type(self.colormap).__name__!r}",
            )

        return ret_colors_series.reindex(self.geometries.index)

    def render(self, ax: Axes, *, target_crs=None, **kwargs) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs: The target CRS to reproject geometries to.
            **kwargs: Additional keyword arguments (not used but included to satisfy
                render function signature contract).

        Returns:
            Axes: The Axes with the layer rendered.
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        if (
            not isinstance(self.geometry_source, GeoSeries)
            and self.datacolumn is not None
            and self.datacolumn not in self.geometry_source.columns
        ):
            raise KeyError(
                f"Column {self.datacolumn!r} not found in GeoDataFrame."
                f" Available columns: {list(self.geometry_source.columns)}"
            )

        edge_color_tup = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
        )

        geoseries = self._geometries_in_crs(target_crs)
        _ = geoseries.plot(
            ax=ax,
            color=self.color_series,
            edgecolor=edge_color_tup,
            linewidth=self.edgewidth,
            zorder=self.zorder,
        )

        return ax


FontStyle = Literal["normal", "italic", "oblique"]
"""How the glyphs are slanted.

- "normal": Upright (no slant). This is the default for most fonts.
- "italic": Uses the font's *italic face* if it exists (often a distinct, designed italic).
  This typically changes letterforms (e.g., a, f) and the slant.
- "oblique": Applies a *slant* to the regular face (or uses an oblique face if the font has one).
  Oblique is usually a geometric slant rather than a redesigned italic.
"""

FontVariant = Literal["normal", "small-caps"]
"""Glyph variant selection.

- "normal": Standard lowercase/uppercase forms.
- "small-caps": Lowercase letters are drawn as *small capital* forms (if the font supports it).
  If the font does not provide true small-caps, Matplotlib/font rendering may fall back to
  a synthetic approximation or ignore the request depending on backend/font.
"""

FontStretchName = Literal[
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "normal",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
]
FontStretch = Union[FontStretchName, int, float]
"""Width of the font face (condensed/expanded).

Named values (most common):
- "ultra-condensed": Extremely narrow.
- "extra-condensed": Very narrow.
- "condensed": Narrow.
- "semi-condensed": Slightly narrow.
- "normal": Standard width.
- "semi-expanded": Slightly wide.
- "expanded": Wide.
- "extra-expanded": Very wide.
- "ultra-expanded": Extremely wide.

Numeric values:
- Matplotlib also accepts numeric stretch values in the range 0–1000.
  (In practice, named values are more portable; numeric values depend on the font/backend.)
"""

FontWeightName = Literal[
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
]
FontWeight = Union[FontWeightName, int, float]
"""Stroke thickness / darkness of glyphs.

Named weights (portable, when a font provides them):
- "ultralight": Very thin strokes.
- "light": Thin strokes.
- "normal": Default weight.
- "regular": Synonym-ish for normal, depends on font naming.
- "book": Slightly heavier than normal for some typefaces.
- "medium": Between normal and bold.
- "roman": Often synonymous with normal/regular in some families.
- "semibold": Between medium and bold.
- "demibold" / "demi": Another naming convention for semibold-ish weights.
- "bold": Clearly heavier strokes, common emphasis.
- "heavy": Heavier than bold.
- "extra bold": Very heavy (note the space).
- "black": Heaviest strokes in many families.

Numeric weights:
- Matplotlib also accepts numeric weights in the range 0–1000.
  (Common convention: ~400 normal, ~700 bold, but exact mapping is font-dependent.)
"""

GenericFontFamily = Literal[
    "serif",
    "sans-serif",
    "sans serif",
    "sans",
    "monospace",
    "cursive",
    "fantasy",
]
FontFamily = Union[GenericFontFamily, str, Sequence[str]]
"""Font family selection.

- Generic families:
  - "serif": Fonts with serifs (e.g., DejaVu Serif, Times).
  - "sans-serif" / "sans serif" / "sans": Sans fonts (e.g., DejaVu Sans, Arial).
  - "monospace": Fixed-width fonts (e.g., DejaVu Sans Mono, Courier).
  - "cursive": Script-like fonts.
  - "fantasy": Decorative/display fonts.
- Specific font name:
  - Any installed font family name, e.g. "Nunito", "DejaVu Sans", "Arial".
- Fallback list:
  - A list/tuple of names like ["Nunito", "DejaVu Sans", "sans-serif"].
  - Matplotlib will pick the first available font from the list.
"""


@dataclass(frozen=True, slots=True)
class LabelFontOptions:
    """Font options for labels on marker layers.

    This is a thin, typed wrapper around Matplotlib’s text/font controls (used via
    `Axes.text(..., **to_mpl_text_kwargs())`).

    Face selection (what font Matplotlib will actually draw)
    --------------------------------------------------------
    Matplotlib chooses a *font face* by combining several independent knobs:

    1) `fontfamily` (which family to use)
       - You may pass a specific family name like `"Nunito"` or `"DejaVu Sans"`.
       - You may pass a generic family like `"sans-serif"`, `"serif"`, `"monospace"`, etc.
       - You may pass a *fallback list* like `["Nunito", "DejaVu Sans", "sans-serif"]`.
         Matplotlib will pick the first available entry on the current machine.
       - Important: specific font names only work if the font is installed or registered
         with Matplotlib (e.g., via `matplotlib.font_manager.fontManager.addfont()`).

    2) `fontweight` (how thick/dark the strokes are)
       - Named weights like `"normal"`, `"medium"`, `"semibold"`, `"bold"`, `"black"`, etc.
       - Or numeric weights `0–1000` (common convention: ~400 normal, ~700 bold),
         but the exact mapping is font-dependent.

    3) `fontstyle` (whether the glyphs are slanted)
       - `"normal"`: upright.
       - `"italic"`: uses the font’s designed italic face if present.
       - `"oblique"`: slants the regular face (or uses an oblique face if the font provides one).

    4) `fontvariant` (alternate glyph set)
       - `"small-caps"` requests small-cap lowercase forms if the font supports them.
         If not supported, Matplotlib/backends may approximate or ignore it.

    5) `fontstretch` (condensed/expanded width)
       - Requests narrower/wider variants like `"condensed"` or `"expanded"`, if present.
       - Or numeric stretch `0–1000`. Support varies by font.

    Notes & portability
    -------------------
    - The *same* settings can produce different results on different systems because the
      available fonts differ. If you need consistency, bundle a font (e.g. Nunito .ttf)
      and register it at runtime.
    - If a requested face (e.g., italic + semibold + condensed) does not exist in the chosen
      family, Matplotlib may fall back to the closest available face.

    Outline / halo
    --------------
    `outlinecolor` and `outlinewidth` are applied via path effects around the glyphs to
    improve legibility over busy map backgrounds.

    Attributes:
        fontcolor (Color): Fill color of the text.
        fontalpha (float | None): Alpha transparency of the text fill.
        fontsize (float): Font size (points).
        fontfamily (FontFamily | None): Specific family name, generic family, or fallback list.
        fontweight (FontWeight): Named or numeric weight (0–1000).
        fontstyle (FontStyle): Upright/italic/oblique slant selection.
        fontvariant (FontVariant): Normal vs small-caps glyph variant.
        fontstretch (FontStretch | None): Condensed/expanded variant (named or numeric 0–1000).
        outlinecolor (Color): Color of the glyph outline (halo).
        outlinewidth (float): Width of the glyph outline (halo), in points.
    """

    fontcolor: Color = "white"
    fontalpha: float | None = 1.0
    fontsize: float = 6.0

    # --- Style Options ---
    fontweight: FontWeight = "bold"
    fontstyle: FontStyle = "normal"
    fontvariant: FontVariant = "normal"
    fontstretch: FontStretch | None = None
    fontfamily: FontFamily | None = None

    outlinecolor: Color = "black"
    outlinewidth: float = 0.75

    def to_mpl_text_kwargs(self) -> dict:
        """Return kwargs to pass into `ax.text(...)` for font styling.

        This intentionally does NOT include color/alpha/zorder/ha/va/text/etc.
        """
        kw: dict = {
            "color": to_hex(
                resolve_color_and_alpha(self.fontcolor, self.fontalpha), keep_alpha=True
            ),
            "fontsize": float(self.fontsize),
            "fontweight": self.fontweight,
            "fontstyle": self.fontstyle,
            "fontvariant": self.fontvariant,
        }
        if self.fontstretch is not None:
            kw["fontstretch"] = self.fontstretch
        if self.fontfamily is not None:
            kw["fontfamily"] = self.fontfamily
        return kw


@dataclass(frozen=True, slots=True)
class LabelBoxOptions:
    """Background box options for text labels drawn via `Axes.text(..., bbox=...)`.

    This controls the *box behind the text*. The box automatically sizes to the text.

    Notes:
      - `pad` lives inside the `boxstyle` string (e.g., "round,pad=0.25") and is in
        *fraction of the font size* units (Matplotlib convention).
      - Matplotlib's `bbox` patch effectively has a single alpha; if you set separate
        face/edge alphas, the simplest thing is to apply one alpha to the whole patch.

    Attributes:
        enabled (bool): Whether to draw a background box behind the label text.
        boxstyle (str): The style of the background box. Default is "round". Options are:
              - "square"     : Plain rectangle
              - "round"      : Rectangle with rounded corners
              - "round4"     : Alternate rounded-rectangle style
              - "circle"     : Circular box around the text's bounding rectangle
              - "ellipse"    : Elliptical box around the text's bounding rectangle
        pad (float): Padding between text and box, in fraction-of-fontsize units.
        facecolor (Color): Fill color of the box (background).
        facealpha (float | None): Alpha for the box fill. If None, uses the color's inherent
            alpha (if any).
        edgecolor (Color): Edge (stroke) color of the box.
        edgealpha (float | None): Alpha for the box edge. If None, uses the color's inherent
            alpha (if any).
        linewidth (float): Line width of the box edge, in points.
    """

    enabled: bool = True
    boxstyle: Literal["square", "round", "round4", "circle", "ellipse"] = "round4"
    pad: float = 0.25
    facecolor: Color = "black"
    facealpha: float | None = 0.6
    edgecolor: Color = "none"
    edgealpha: float | None = 0.0
    linewidth: float = 0.8

    def to_mpl_bbox(self) -> dict | None:
        """Return a dict suitable for passing as `bbox=` to `Axes.text`.

        Returns:
            dict | None: A Matplotlib bbox properties dict if enabled; otherwise None.
        """
        if not self.enabled:
            return None

        face_color = resolve_color_and_alpha(self.facecolor, alpha=self.facealpha)
        edge_color = resolve_color_and_alpha(self.edgecolor, alpha=self.edgealpha)

        bbox = {
            "boxstyle": f"{self.boxstyle},pad={float(self.pad)}",
            "fc": to_hex(face_color, keep_alpha=True),
            "ec": to_hex(edge_color, keep_alpha=True),
            "lw": float(self.linewidth),
        }

        return bbox


@dataclass(frozen=True, slots=True)
class _MarkerLayer:
    """A layer of point markers with optional labels.

    Attributes:
        point_geometries (GeoSeries): A GeoSeries of Point geometries for the markers.
        labels (Sequence[str] | None): Optional labels for each marker.
        marker_options (PointMarkerOptions): Marker style settings. Uses default constructor if
            not provided.
        show_labels (bool): Whether to show labels on the markers. Default is True.
        font_options (LabelFontOptions): Font options for the labels. Uses default constructor if
            not provided.
        zorder (int): Z-order for rendering. Default is 2.
    """

    point_geometries: GeoSeries

    # Optional labels (same length as point_geometries)
    labels: Sequence[str] | None = None

    # Marker style (shared across the layer)
    marker_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)

    # Label style (centered in marker)
    show_labels: bool = True
    labelfont_options: LabelFontOptions = field(default_factory=LabelFontOptions)
    labelbox_options: LabelBoxOptions = field(default_factory=LabelBoxOptions)
    zorder: int = 2

    def __post_init__(self) -> None:
        if self.point_geometries is None:
            raise TypeError("MarkerLayer requires `point_geometries` (a GeoSeries of Points).")

        if self.labels is not None and len(self.labels) != len(self.point_geometries):
            raise ValueError("`labels` must have the same length as `point_geometries`.")

        if self.marker_options is None:
            object.__setattr__(self, "marker_options", PointMarkerOptions())

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        # required by _GeoLayer, unused for markers
        return pd.Series(dtype=object)

    def render(self, ax: Axes, *, target_crs=None, **kwargs) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs: The target CRS to reproject geometries to.
            **kwargs: Additional keyword arguments (not used).
        """
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s) passed to render: {unknown}")

        point_geometries = self.point_geometries

        # Reproject points if needed
        if getattr(point_geometries, "crs", None) is not None and target_crs is not None:
            if point_geometries.crs != target_crs:
                point_geometries = point_geometries.to_crs(target_crs)

        x_coordinates = point_geometries.x.to_numpy()
        y_coordinates = point_geometries.y.to_numpy()

        # PointMarkerOptions already returns RGBA colors with alpha baked in.
        marker_kwargs = dict(self.marker_options.to_mpl_settings_dict())
        marker_kwargs.pop("zorder", None)

        if not self.show_labels or self.labels is None:
            ax.plot(
                x_coordinates,
                y_coordinates,
                linestyle="None",
                zorder=int(self.zorder),
                **marker_kwargs,
            )
        else:
            outline_color, _ = resolve_color_and_alpha(
                self.labelfont_options.outlinecolor,
                alpha=1.0,
            )
            text_effects = [
                patheffects.Stroke(
                    linewidth=float(self.labelfont_options.outlinewidth),
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]

            text_color, text_alpha = resolve_color_and_alpha(
                self.labelfont_options.fontcolor,
                alpha=self.labelfont_options.fontalpha,
            )

            for x_value, y_value, label_text in zip(x_coordinates, y_coordinates, self.labels):

                ax.plot(
                    x_value,
                    y_value,
                    linestyle="None",
                    zorder=int(self.zorder),
                    **marker_kwargs,
                )

                text_artist = ax.text(
                    float(x_value),
                    float(y_value),
                    str(label_text),
                    ha="center",
                    va="center",
                    zorder=int(self.zorder),
                    bbox=self.labelbox_options.to_mpl_bbox(),
                    clip_on=True,
                    **self.labelfont_options.to_mpl_text_kwargs(),
                )
                text_artist.set_clip_path(ax.patch)
                text_artist.set_path_effects(text_effects)

        return ax


@dataclass(slots=True)
class ColorbarLayoutOptions:
    """Layout options for positioning colorbars in GeoPlot.
    Attributes:
        outer_pad (float): Padding between the colorbars and the plot edges (figure-relative).
        inner_pad (float): Padding between the colorbars and the main plot area (figure-relative).
        width (float): Width of the colorbars (figure-relative).
        right_margin (float): Margin to the right of the colorbars (figure-relative).
    """

    outer_pad: float = 0.03
    inner_pad: float = 0.06
    width: float = 0.02
    right_margin: float = 0.02


@dataclass(slots=True)
class ColorbarOptions:
    """Options for configuring colorbars in GeoPlot.

    Attributes:
        tick_fontsize (float): Font size for colorbar ticks.
        tick_pad (float): Padding for colorbar ticks.
        label_fontsize (float | None): Font size for colorbar label.
        label_rotation (float | None): Rotation angle for colorbar label.
        label_pad (float | None): Padding for colorbar label.
        orientation (Literal["vertical", "horizontal"]): Orientation of the colorbar.
        extend (Literal["neither", "both", "min", "max"]): Extension style for the colorbar.
        format (str | None): Format string for colorbar tick labels.
        fraction (float | None): Fraction of original size for colorbar.
        shrink (float | None): Shrink factor for colorbar.
        aspect (float | None): Aspect ratio for colorbar.
        force_ticks (list[float] | None): Explicit tick locations for the colorbar.
        force_ticklabels (list[str] | None): Explicit tick labels for the colorbar.
        max_n_ticks (int | None): Maximum number of ticks on the colorbar.
    """

    # --- tick appearance (axes.tick_params) ---
    tick_fontsize: float = 8.0
    tick_pad: float = 2.0

    # --- label appearance (cb.set_label) ---
    label_fontsize: float | None = None
    label_rotation: float | None = None
    label_pad: float | None = None

    # --- fig.colorbar behavior ---
    orientation: Literal["vertical", "horizontal"] = "vertical"
    extend: Literal["neither", "both", "min", "max"] = "neither"
    format: str | None = None  # e.g. ".2f"
    fraction: float | None = None  # rarely needed when using cax
    shrink: float | None = None  # rarely needed when using cax
    aspect: float | None = None  # rarely needed when using cax

    # --- explicit overrides (optional) ---
    force_ticks: list[float] | None = None
    force_ticklabels: list[str] | None = None
    max_n_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class _ColorbarRequest:
    layer: _ContinuousColorLayer
    label: str | None = None  # override label shown on the bar
    zorder: int = 0  # used only for ordering colorbars
    options: ColorbarOptions | None = None  # optional per-bar overrides (optional feature)


@dataclass(frozen=True, slots=True)
class _LabelRequest:
    gdf: GeoDataFrame
    label_column: str
    labelfont_options: LabelFontOptions | None
    labelbox_options: LabelBoxOptions | None
    label_format_fn: Callable[[Any], str] | None = None
    zorder: int = 100


class GeoPlot:
    """A class for creating geographic plots with multiple layers.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        show_axis (bool): Whether to show axis lines and labels.
        target_crs: The target CRS for reprojecting geometries.
        show_colorbars (bool): Whether to display colorbars for layers.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        dpi: int = 300,
        show_axis: bool = False,
        target_crs=None,
        include_default_outline: bool = True,
    ) -> None:
        self.gdf = gdf

        self.fig = Figure(dpi=dpi)
        self._canvas = FigureCanvas(self.fig)  # gives the Figure a renderer
        self._ax = self.fig.add_subplot(111)

        self.show_axis = show_axis
        self.target_crs = target_crs if target_crs is not None else getattr(gdf, "crs", None)

        self._xlim: tuple[float, float] | None = None
        self._ylim: tuple[float, float] | None = None

        self._colorbar_layout_options: ColorbarLayoutOptions = ColorbarLayoutOptions()
        self._colorbar_requests: list[_ColorbarRequest] = []

        self._choropleth_layers: list[_ContinuousColorLayer] = []
        self._districting_plan_layers: list[_CategoricalColorLayer] = []
        self._outline_layers: list[_CategoricalColorLayer] = []
        self._highlight_layers: list[_CategoricalColorLayer] = []
        self._marker_layers: list[_MarkerLayer] = []

        self._label_requests: list[_LabelRequest] = []

        self._colorbar_axes: list[Axes] = []

        if include_default_outline:
            fully_dissolved_geos = GeoSeries(gdf.geometry.union_all())
            self.add_outline_layer(
                geosource=fully_dissolved_geos,
                edgecolor="black",
                edgewidth=0.5,
            )

    def add_choropleth_layer(
        self,
        *,
        geosource: GeoDataFrame | None = None,
        datacolumn: str,
        colormap: str | Colormap = "Purples",
        missing_color: Any = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        vmin: float | None = None,
        vmax: float | None = None,
        bins: int | list[float] | None = None,
        show_colorbar: bool = False,
        colorbar_label: str | None = None,
        colorbar_options: ColorbarOptions | None = None,
        zorder: int = 0,
    ) -> None:
        """Add a choropleth layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | None): The GeoDataFrame source for the layer.
                If None, uses the base gdf of the GeoPlot. Default is None.
            datacolumn (str): The data column to use for color mapping.
            colormap (str | Colormap): The colormap to use for color mapping. Default is "Purples".
            missing_color (Any): Color to use for missing data. Default is "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            vmin (float | None): Lower bound for color mapping range. Default is None which then
                uses the minimum value in the data.
            vmax (float | None): Upper bound for color mapping range. Default is None which then
                uses the maximum value in the data.
            show_colorbar (bool): Whether to show a colorbar for this layer. Default is False.
            colorbar_label (str | None): Label for the colorbar. Default is None which then
                uses the datacolumn name.
            colorbar_options (ColorbarOptions | None): Options for customizing the colorbar.
                Default is None.
            bins (int | list[float] | None): Optional binning specification for discrete intervals.
                Default is None.
            zorder (int): Z-order for rendering. Default is 0.
        """
        if geosource is None:
            geosource = self.gdf
        layer = _ContinuousColorLayer(
            geometry_source=geosource,
            datacolumn=datacolumn,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            vmin=vmin,
            vmax=vmax,
            bins=bins,
            zorder=zorder,
        )
        self._choropleth_layers.append(layer)

        if show_colorbar:
            self._colorbar_requests.append(
                _ColorbarRequest(
                    layer=layer,
                    label=colorbar_label,
                    zorder=zorder,
                    options=colorbar_options,
                )
            )

    def add_districting_plan_layer(
        self,
        *,
        geosource: GeoDataFrame | None = None,
        plancolumn: str,
        dissolve: bool = False,
        show_labels: bool = False,
        exclude_labels: list[Any] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        colormap: str | Colormap | dict[Any, Color] | pd.Series = "districtr",
        missing_color: Any = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        zorder: int = 2,
    ) -> None:
        """Add a districting plan layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | None): The GeoDataFrame source for the layer.
                If None, uses the base gdf of the GeoPlot. Default is None.
            plancolumn (str): The column containing district identifiers.
            dissolve (bool): Whether to dissolve geometries by district. Default is False.
            show_labels (bool): Whether to show district labels. Default is False.
            exclude_labels (list[Any] | None): List of district labels to exclude from labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfontoptions (LabelFontOptions | None): Font options for district labels.
                If None, uses default settings. Default is None.
            colormap (str | Colormap | dict[Any, Color] | pd.Series): Color mapping specification.
                Can be a single color, a named colormap, a Colormap object, or a mapping from
                district identifiers to colors. Default is "districtr".
            missing_color (Any): Color to use for missing data. Default is "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            zorder (int): Z-order for rendering. Default is 2.
        """
        if geosource is None:
            plan_gdf = self.gdf
        else:
            plan_gdf = geosource

        if dissolve:
            plan_gdf = GeoDataFrame(plan_gdf.dissolve(by=plancolumn).reset_index())

        layer = _CategoricalColorLayer(
            geometry_source=plan_gdf,
            datacolumn=plancolumn,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )
        self._districting_plan_layers.append(layer)

        if show_labels:
            dissolved_plan_gdf = plan_gdf.dissolve(by=plancolumn).reset_index()

            def coerce_labels(x: Any) -> str:
                try:
                    return str(int(x))
                except Exception:
                    return str(x)

            dissolved_plan_gdf[plancolumn] = dissolved_plan_gdf[plancolumn].apply(coerce_labels)
            new_exclude_labels = (
                list(map(coerce_labels, exclude_labels)) if exclude_labels is not None else []
            )
            dissolved_plan_gdf = GeoDataFrame(
                dissolved_plan_gdf.query(f"`{plancolumn}` not in {new_exclude_labels}")
            )

            self._label_requests.append(
                _LabelRequest(
                    gdf=dissolved_plan_gdf,
                    label_column=plancolumn,
                    labelfont_options=labelfont_options,
                    labelbox_options=labelbox_options,
                    label_format_fn=lambda x: str(int(x)),
                    zorder=zorder + 1,
                )
            )

    def add_outline_layer(
        self,
        *,
        geosource: GeoDataFrame | GeoSeries | None = None,
        geometry_mask: pd.Series | None = None,
        dissolve_column: str | None = None,
        edgecolor: Color = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        show_labels: bool = False,
        exclude_labels: list[Any] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 3,
    ) -> None:
        """Add an outline layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source
                for the layer. If None, uses the base gdf of the GeoPlot. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
                Default is None.
            dissolve_column (str | None): Optional column to dissolve geometries by
                before outlining. Default is None.
            edgecolor (Color): Color for geometry edges. Default is "black".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            show_labels (bool): Whether to show labels on the outlined geometries. Default is False.
            exclude_labels (list[Any] | None): List of labels to exclude from labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfont_options (LabelFontOptions | None): Font options for labels.
                If None, uses the following defaults:
                    - fontcolor="black",
                    - fontsize=4,
                    - fontweight="roman",
                    - outlinecolor="grey",
                    - outlinewidth=0.2.
                Default is None.
            labelbox_options (LabelBoxOptions | None): Box options for labels. If None the box
                is disabled. Default is None.
            zorder (int): Z-order for rendering. Default is 3.
        """
        if geosource is None:
            geosource = self.gdf

        if dissolve_column is not None:
            if not isinstance(geosource, GeoDataFrame):
                raise TypeError(
                    "Tried to dissolve geosource of type "
                    f"{type(geosource).__name__!r}; geosource must be a GeoDataFrame",
                )
            geosource = GeoDataFrame(geosource.dissolve(by=dissolve_column).reset_index())

        layer = _CategoricalColorLayer(
            geometry_source=geosource,
            geometry_mask=geometry_mask,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )
        self._outline_layers.append(layer)

        if show_labels:
            processed_geosource = layer.geosource
            if not isinstance(processed_geosource, GeoDataFrame):
                raise TypeError(
                    "Tried to add labels to geosource of type "
                    f"{type(geosource).__name__!r}; geosource must be a GeoDataFrame",
                )

            if dissolve_column is None:
                raise ValueError(
                    "'dissolve_column' must be set to add labels to an outline layer",
                )

            new_exclude_labels = exclude_labels if exclude_labels is not None else []
            labeled_gdf = GeoDataFrame(
                processed_geosource.query(f"`{dissolve_column}` not in {new_exclude_labels}")
            )

            if labelfont_options is None:
                labelfont_options = LabelFontOptions(
                    fontcolor="black",
                    fontsize=4,
                    fontweight="roman",
                    outlinecolor="grey",
                    outlinewidth=0.2,
                )

            self._label_requests.append(
                _LabelRequest(
                    gdf=labeled_gdf,
                    label_column=dissolve_column,
                    labelfont_options=labelfont_options,
                    labelbox_options=labelbox_options,
                    zorder=zorder + 1,
                )
            )

    def add_highlight_layer(
        self,
        *,
        geosource: GeoDataFrame | GeoSeries | None = None,
        geometry_mask: pd.Series | None = None,
        label_column: str | None = None,
        facecolor: Color = "gray",
        facealpha: float | None = 0.5,
        show_labels: bool = False,
        exclude_labels: list[Any] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 10,
    ) -> None:
        """Add a highlight layer to the GeoPlot.

        Args:
            geosource (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source
                for the layer. If None, uses the base gdf of the GeoPlot. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
                Default is None.
            label_column (str | None): Optional column to label geometries by
                before highlighting. Default is None.
            facecolor (Color): Color for geometry faces. Default is "gray".
            facealpha (float | None): Alpha transparency for face colors. Default is 0.5.
            show_labels (bool): Whether to show labels on the highlighted geometries. Default is
                False.
            exclude_labels (list[Any] | None): List of labels to exclude from labeling.
                If None, no labels are excluded. Does not do anything if show_labels is False.
                Default is None.
            labelfont_options (LabelFontOptions | None): Font options for labels.
                If None uses the following defaults:
                    - fontcolor="black",
                    - fontsize=4,
                    - fontweight="roman",
                    - outlinecolor="grey",
                    - outlinewidth=0.2.
                Default is None.
            labelbox_options (LabelBoxOptions | None): Box options for labels. If None the box
                is disabled. Default is None.
            zorder (int): Z-order for rendering. Default is 10.
        """
        if show_labels:
            if label_column is None:
                raise ValueError(
                    "add_highlight_layer(show_labels=True) requires label_column=... to know "
                    "what to label. Example: dissolve_column='COUNTYFP10'."
                )
            if geosource is None:
                raise ValueError(
                    "add_highlight_layer(show_labels=True) requires geosource=... (a GeoDataFrame) "
                    "so the dissolve_column exists."
                )
            if not isinstance(geosource, GeoDataFrame):
                raise TypeError(
                    "add_highlight_layer(show_labels=True) requires geosource to be a GeoDataFrame "
                    f"so it has the label_column {label_column!r}. "
                    f"You passed {type(geosource).__name__!r}. "
                    "Either pass a GeoDataFrame, or set show_labels=False."
                )

        if geosource is None:
            geometries = self.gdf.geometry
        else:
            geometries = _as_geoseries(geosource)

        if geometry_mask is not None:
            geometries = geometries[geometry_mask]

        geometries = GeoSeries(geometries.union_all())

        layer = _CategoricalColorLayer(
            geometry_source=geometries,
            colormap=facecolor,
            missing_color="none",
            facealpha=facealpha,
            edgecolor="none",
            edgealpha=None,
            edgewidth=0.0,
            zorder=zorder,
        )
        self._highlight_layers.append(layer)

        if show_labels:
            label_gdf = geosource
            if label_gdf is None:
                raise RuntimeError(
                    "An unexpected error occured in add_highlight_layer. "
                    "The geosource was None when trying to add labels."
                )

            if isinstance(label_gdf, GeoSeries):
                raise TypeError(
                    "add_highlight_layer(show_labels=True) requires geosource to be a GeoDataFrame "
                    f"so it has the label_column {label_column!r}. "
                    f"You passed a GeoSeries. Either pass a GeoDataFrame, or set show_labels=False."
                )

            if geometry_mask is not None:
                label_gdf = GeoDataFrame(label_gdf.loc[geometry_mask])

            new_exclude_labels = exclude_labels if exclude_labels is not None else []
            labeled_gdf = GeoDataFrame(
                label_gdf.query(f"`{label_column}` not in {new_exclude_labels}")
            )

            if labelfont_options is None:
                labelfont_options = LabelFontOptions(
                    fontcolor="black",
                    fontsize=4,
                    fontweight="roman",
                    outlinecolor="grey",
                    outlinewidth=0.2,
                )

            if label_column is None:
                raise RuntimeError(
                    "An unexpected error occured in add_highlight_layer. "
                    "The dissolve_column was None when trying to add labels."
                )

            self._label_requests.append(
                _LabelRequest(
                    gdf=labeled_gdf,
                    label_column=label_column,
                    labelfont_options=labelfont_options,
                    labelbox_options=labelbox_options,
                    zorder=zorder + 1,
                )
            )

        return None

    def add_marker_layer(
        self,
        *,
        points_geoseries: gpd.GeoSeries | None = None,
        latitude_longitude_list: Sequence[tuple[float, float]] | None = None,
        input_crs=None,
        marker_options: PointMarkerOptions | None = None,
        show_labels: bool = True,
        labels: Sequence[str] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of markers (points) to the GeoPlot.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latitude_longitude_list` must be provided. Default is None.
            latitude_longitude_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            input_crs: The CRS of the input points if using `latitude_longitude_list`.
                If None, assumes EPSG:4326 (lat/lon). Default is None.
            marker_options (PointMarkerOptions | None): Marker style settings.
                If None, uses the following defaults:
                    - markerfacecolor="white",
                    - markerfacealpha=1.0,
                    - marker="o",
                    - markersize=3.0,
                    - markeredgecolor="black",
                    - markeredgealpha=1.0,
                    - markeredgewidth=0.5.
                Default is None.
            show_labels (bool): Whether to show labels on the markers. Default is True.
            labels (Sequence[str] | None): Optional labels for each marker. Default is None.
            labelfont_options (LabelFontOptions | None): Font options for the labels If None, uses
                default LabelFontOptions().
            labelbox_options (LabelBoxOptions | None): Box options for the labels. If None the
                box is disabled. Default is None.
            zorder (int) Z-order for rendering. Default is 2.
        """
        if marker_options is None:
            marker_options = PointMarkerOptions(
                markerfacecolor="white",
                markerfacealpha=1.0,
                marker="o",
                markersize=3.0,
                markeredgecolor="black",
                markeredgealpha=1.0,
                markeredgewidth=0.5,
            )
        if labelfont_options is None:
            labelfont_options = LabelFontOptions()
        if labelbox_options is None:
            labelbox_options = LabelBoxOptions(enabled=False)

        if points_geoseries is None and latitude_longitude_list is None:
            raise ValueError("Either `points_geoseries` or `latitude_longitude_list` must be set.")
        if points_geoseries is not None and latitude_longitude_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latitude_longitude_list` "
                "may be set at a time.",
            )

        if latitude_longitude_list is not None:
            # crs EPSG:4326 corresponds to lat/lon
            point_geometries = gpd.GeoSeries(
                [
                    Point(float(longitude), float(latitude))
                    for latitude, longitude in latitude_longitude_list
                ],
                crs="EPSG:4326",
            )
            point_geometries = point_geometries.to_crs(
                input_crs if input_crs is not None else self.gdf.crs
            )
        elif points_geoseries is not None:
            point_geometries = points_geoseries
            if getattr(point_geometries, "crs", None) is None and input_crs is not None:
                point_geometries = point_geometries.set_crs(input_crs)
        else:
            raise RuntimeError(
                "An unexpected error occured in add_marker_layer. One of the argurments "
                "'points_geoseries' or 'latitude_longitude_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latitude_longitude_list': {type(latitude_longitude_list).__name__!r}",
            )

        marker_layer = _MarkerLayer(
            point_geometries=point_geometries,
            labels=labels,
            marker_options=marker_options,
            show_labels=show_labels,
            labelfont_options=labelfont_options,
            labelbox_options=labelbox_options,
            zorder=zorder,
        )
        self._marker_layers.append(marker_layer)

    def add_label_layer(
        self,
        *,
        points_geoseries: gpd.GeoSeries | None = None,
        latitude_longitude_list: Sequence[tuple[float, float]] | None = None,
        input_crs=None,
        labels: Sequence[str] | None = None,
        labelfont_options: LabelFontOptions | None = None,
        labelbox_options: LabelBoxOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of markers (points) to the GeoPlot.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latitude_longitude_list` must be provided. Default is None.
            latitude_longitude_list (Sequence[tuple[float, float]] | None): A sequence of
                (latitude, longitude) tuples for the marker locations. If None, `points_geoseries`
                must be provided. Default is None.
            input_crs: The CRS of the input points if using `latitude_longitude_list`.
                If None, assumes EPSG:4326 (lat/lon). Default is None.
            labels (Sequence[str] | None): Optional labels for each marker. Default is None which
                results numerical labels.
            labelfont_options (LabelFontOptions | None): Font options for the labels If None, uses
                the following defaults:
                    - fontcolor="black",
                    - fontsize=4,
                    - fontweight="roman",
                    - outlinecolor="grey",
                    - outlinewidth=0.2.
            labelbox_options (LabelBoxOptions | None): Box options for the labels. If None the
                box is disabled. Default is None.
            zorder (int) Z-order for rendering. Default is 2.
        """
        if points_geoseries is None and latitude_longitude_list is None:
            raise ValueError("Either `points_geoseries` or `latitude_longitude_list` must be set.")
        if points_geoseries is not None and latitude_longitude_list is not None:
            raise ValueError(
                "Only one of `points_geoseries` or `latitude_longitude_list` "
                "may be set at a time.",
            )
        if points_geoseries is None and latitude_longitude_list is not None:
            n_labels = len(list(latitude_longitude_list))
        elif points_geoseries is not None:
            n_labels = len(points_geoseries)
        else:
            raise RuntimeError(
                "An unexpected error occured in add_label_layer. One of the argurments "
                "'points_geoseries' or 'latitude_longitude_list' was likely set incorrectly."
                f"Type of 'points_geoseries': {type(points_geoseries).__name__!r}, "
                f"type of 'latitude_longitude_list': {type(latitude_longitude_list).__name__!r}",
            )

        if labels is None:
            labels = [str(i) for i in range(n_labels)]

        if labelfont_options is None:
            labelfont_options = LabelFontOptions(
                fontcolor="black",
                fontsize=4,
                fontweight="roman",
                outlinecolor="grey",
                outlinewidth=0.2,
            )

        self.add_marker_layer(
            points_geoseries=points_geoseries,
            latitude_longitude_list=latitude_longitude_list,
            input_crs=input_crs,
            marker_options=PointMarkerOptions(
                markerfacecolor="none",
                markerfacealpha=0.0,
                marker="o",
                markersize=0.0,
                markeredgecolor="none",
                markeredgealpha=0.0,
                markeredgewidth=0.0,
            ),
            show_labels=True,
            labels=labels,
            labelfont_options=labelfont_options,
            labelbox_options=labelbox_options,
            zorder=zorder,
        )

    def set_colorbar_layout(
        self,
        *,
        outer_pad: float | None = None,
        inner_pad: float | None = None,
        width: float | None = None,
        right_margin: float | None = None,
    ) -> None:
        """Set the spacing between colorbars in GeoPlot.

        All arguments are optional; only those provided will be updated.

        Args:
            outer_pad (float | None): Padding between the colorbar and the plot edges
                (figure-relative). Default is None.
            inner_pad (float | None): Padding between the colorbar and the main plot area
                (figure-relative). Default is None.
            width (float | None): Width of the colorbar (figure-relative). Default is None.
            right_margin (float | None): Margin to the right of the colorbar (figure-relative).
                Default is None.
        """
        cb_options = self._colorbar_layout_options

        if outer_pad is not None:
            cb_options.outer_pad = float(outer_pad)
        if inner_pad is not None:
            cb_options.inner_pad = float(inner_pad)
        if width is not None:
            cb_options.width = float(width)
        if right_margin is not None:
            cb_options.right_margin = float(right_margin)

    def set_xlim(self, left: float, right: float) -> None:
        """Set the x-axis limits to apply when the plot is built.

        Args:
            left (float): The left x-axis limit.
            right (float): The right x-axis limit.
        """
        self._xlim = (float(left), float(right))

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set the y-axis limits to apply when the plot is built.

        Args:
            bottom (float): The bottom y-axis limit.
            top (float): The top y-axis limit.
        """
        self._ylim = (float(bottom), float(top))

    def clear_limits(self) -> None:
        """Clear any stored x/y limits so autoscaling can occur."""
        self._xlim = None
        self._ylim = None

    def focus_axes(
        self,
        *,
        geosource: GeoSource | None = None,
        geometry_mask: pd.Series | None = None,
        pad: float | tuple[float, float] = 0.02,
        pad_mode: Literal["fraction", "data"] = "fraction",
    ) -> None:
        """Set x/y limits to the (padded) bounding box of a geosource.

        Args:
            geosource: GeoDataFrame or GeoSeries to focus on. Defaults to this plot's gdf.
                If None, will use the base gdf used to initialize GeoPlot. Defaults to None.
            geometry_mask (pd.Series | None): Optional boolean mask aligned to geosource index.
                If None, will use all geometries in geosouce. Defaults to None.
            pad (float | tuple[float, float]): Padding around bounds.
                 - If pad_mode="fraction": fraction of width/height (e.g., 0.02 = 2%)
                 - If pad_mode="data": absolute units in data coords.
                 You can pass a single float or (pad_x, pad_y).
                 Defaults to 0.02 (2%).
            pad_mode (Literal): "fraction" or "data". Defaults to "fraction".
        """
        if geosource is None:
            geosource = self.gdf

        geoseries = _as_geoseries(geosource)

        if geometry_mask is not None:
            geoseries = geoseries[geometry_mask]

        geoseries = geoseries[geoseries.notna()]
        try:
            geoseries = geoseries[~geoseries.is_empty]
        except Exception:
            # older shapely/geopandas combos may not have is_empty reliably; ignore
            pass

        if geoseries.empty:
            raise ValueError("focus_on(): no geometries after applying mask / dropping empties.")

        gs_crs = getattr(geoseries, "crs", None)
        if gs_crs is not None and self.target_crs is not None and gs_crs != self.target_crs:
            geoseries = geoseries.to_crs(self.target_crs)

        minx, miny, maxx, maxy = map(float, geoseries.total_bounds)

        width = maxx - minx
        height = maxy - miny

        if isinstance(pad, tuple):
            pad_x, pad_y = float(pad[0]), float(pad[1])
        else:
            pad_x = pad_y = float(pad)

        if pad_mode == "fraction":
            # If width/height are 0 (single point/line), give a small default pad
            dx = (width * pad_x) if width > 0 else pad_x
            dy = (height * pad_y) if height > 0 else pad_y
        elif pad_mode == "data":
            dx, dy = pad_x, pad_y
        else:
            raise ValueError("pad_mode must be 'fraction' or 'data'.")

        self.set_xlim(minx - dx, maxx + dx)
        self.set_ylim(miny - dy, maxy + dy)

    def _iter_layers_in_draw_order(self) -> list[_GeoLayer | _MarkerLayer]:
        """Iterate over all layers in the order they should be drawn."""

        def _sorted(layers: Sequence[_GeoLayer | _MarkerLayer]) -> list[_GeoLayer | _MarkerLayer]:
            return sorted(layers, key=lambda L: int(L.zorder))

        return (
            _sorted(self._choropleth_layers)
            + _sorted(self._districting_plan_layers)
            + _sorted(self._marker_layers)
            + _sorted(self._outline_layers)
            + _sorted(self._highlight_layers)
        )

    def _clear_colorbars_and_reset_layout(self) -> None:
        """Clear any existing colorbars and reset layout to default."""
        for cax in list(self._colorbar_axes):
            try:
                cax.remove()
            except Exception:
                pass
        self._colorbar_axes = []

        # reset layout so we don't keep a shrunken main axes
        self.fig.subplots_adjust(right=0.98)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _draw_colorbars(self) -> None:
        """Draw colorbars for all requested layers."""
        self._clear_colorbars_and_reset_layout()

        if not self._colorbar_requests:
            return

        # Sort requests by their requested zorder (usually layer.zorder)
        sorted_cb_requests = sorted(self._colorbar_requests, key=lambda r: int(r.zorder))
        n_layers = len(sorted_cb_requests)

        cb_global_options = self._colorbar_layout_options
        outer_pad = float(cb_global_options.outer_pad)
        inner_pad = float(cb_global_options.inner_pad)
        width = float(cb_global_options.width)
        right_margin = float(cb_global_options.right_margin)

        total_width = n_layers * width + (n_layers - 1) * inner_pad + outer_pad + right_margin
        right = max(0.05, 1.0 - total_width)

        self.fig.subplots_adjust(right=right)
        self.fig.canvas.draw_idle()

        main_pos = self._ax.get_position()
        x0 = float(main_pos.x1 + outer_pad)
        y0 = float(main_pos.y0)
        h = float(main_pos.height)

        for i, cb_request in enumerate(sorted_cb_requests):
            layer = cb_request.layer
            cb_options = cb_request.options if cb_request.options is not None else ColorbarOptions()

            xi = x0 + i * (width + inner_pad)
            rect = (float(xi), float(y0), float(width), float(h))
            cb_ax = self.fig.add_axes(rect)
            self._colorbar_axes.append(cb_ax)

            mappable, layer_defaults = layer._mappable()

            cb_kwargs: dict[str, Any] = dict(layer_defaults)
            cb_kwargs["orientation"] = cb_options.orientation
            cb_kwargs["extend"] = cb_options.extend

            if cb_options.format is not None:
                cb_kwargs["format"] = cb_options.format
            if cb_options.fraction is not None:
                cb_kwargs["fraction"] = cb_options.fraction
            if cb_options.shrink is not None:
                cb_kwargs["shrink"] = cb_options.shrink
            if cb_options.aspect is not None:
                cb_kwargs["aspect"] = cb_options.aspect

            colorbar = self.fig.colorbar(mappable, cax=cb_ax, **cb_kwargs)

            # label: request override > datacolumn > none
            label_text = cb_request.label if cb_request.label is not None else layer.datacolumn
            if label_text is not None:
                label_kwargs: dict[str, Any] = {}
                if cb_options.label_fontsize is not None:
                    label_kwargs["fontsize"] = cb_options.label_fontsize
                if cb_options.label_rotation is not None:
                    label_kwargs["rotation"] = cb_options.label_rotation
                if cb_options.label_pad is not None:
                    label_kwargs["labelpad"] = cb_options.label_pad
                colorbar.set_label(str(label_text), **label_kwargs)

            cb_ax.tick_params(labelsize=cb_options.tick_fontsize, pad=cb_options.tick_pad)

            if cb_options.force_ticks is not None:
                colorbar.set_ticks(cb_options.force_ticks)
            if cb_options.force_ticklabels is not None:
                colorbar.set_ticklabels(cb_options.force_ticklabels)

            if cb_options.max_n_ticks is not None and cb_options.force_ticks is None:
                try:
                    ticks = list(colorbar.get_ticks())
                    if len(ticks) > cb_options.max_n_ticks:
                        step = max(1, len(ticks) // cb_options.max_n_ticks)
                        colorbar.set_ticks(ticks[::step])
                except Exception:
                    pass

    def _apply_limits(self) -> None:
        """Apply stored x/y limits to the axes."""
        if self._xlim is not None:
            self._ax.set_xlim(*self._xlim)
        if self._ylim is not None:
            self._ax.set_ylim(*self._ylim)

    def _draw_deferred_labels(self) -> dict[str, Point]:
        """Draw all deferred labels and return their positions.

        Returns:
            dict[str, Point]: A dictionary mapping label text to Point objects.
        """
        label_positions: dict[str, Point] = {}
        if not self._label_requests:
            return label_positions

        ax = self._ax

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        clip_geom = box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))

        for req in self._label_requests:
            # One label per dissolved part
            dissolved = GeoDataFrame(req.gdf.dissolve(by=req.label_column).reset_index())

            # Match plot CRS
            if getattr(dissolved, "crs", None) is not None and self.target_crs is not None:
                if dissolved.crs != self.target_crs:
                    dissolved = dissolved.to_crs(self.target_crs)

            # Clip to current view
            clipped = dissolved.geometry.intersection(clip_geom)
            keep = (~clipped.isna()) & (~clipped.is_empty)
            if not keep.any():
                continue

            dissolved = dissolved.loc[keep].copy()
            dissolved["geometry"] = clipped.loc[keep]

            # Representative points inside the clipped geometry
            pts = dissolved.representative_point()

            labels: list[str] = []
            for raw in dissolved[req.label_column].tolist():
                txt = str(raw)
                if req.label_format_fn is not None:
                    try:
                        txt = str(req.label_format_fn(raw))
                    except Exception:
                        pass
                labels.append(txt)

            # Defaults
            font = (
                req.labelfont_options if req.labelfont_options is not None else LabelFontOptions()
            )
            boxopt = (
                req.labelbox_options
                if req.labelbox_options is not None
                else LabelBoxOptions(enabled=False)
            )

            # Ephemeral label-only marker options (no visible marker)
            label_marker_opts = PointMarkerOptions(
                markerfacecolor="none",
                markerfacealpha=0.0,
                marker="o",
                markersize=0.0,
                markeredgecolor="none",
                markeredgealpha=0.0,
                markeredgewidth=0.0,
            )

            # Create an ephemeral marker layer and render immediately
            tmp = _MarkerLayer(
                point_geometries=pts,
                labels=labels,
                marker_options=label_marker_opts,
                show_labels=True,
                labelfont_options=font,
                labelbox_options=boxopt,
                zorder=req.zorder,
            )
            tmp.render(ax, target_crs=self.target_crs)
            label_positions.update(
                {label: Point(pt.x, pt.y) for label, pt in zip(labels, pts.geometry.tolist())}
            )
        return label_positions

    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and applying settings."""
        self._ax.clear()

        if not self.show_axis:
            self._ax.set_axis_off()

        for layer in self._iter_layers_in_draw_order():
            layer.render(self._ax, target_crs=self.target_crs)

        self._draw_colorbars()

    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Build the plot and apply stored settings like limits."""
        self._build_plot()
        self._apply_limits()
        label_points = self._draw_deferred_labels()
        return label_points

    @property
    def ax(self) -> Axes:
        """The Matplotlib Axes object for the plot."""
        self._build_and_apply_settings()
        return self._ax

    def get_label_positions(self, *, as_lat_long: bool = False) -> tuple[str, dict[str, Point]]:
        """A dictionary mapping label text to Point objects for all labels in the plot."""
        label_points = GeoSeries(self._build_and_apply_settings(), crs=self.target_crs)
        if as_lat_long:
            label_points = label_points.to_crs("EPSG:4326")
        return (
            str(label_points.crs.to_string() if label_points.crs is not None else "undefined"),
            {str(label): Point(pt.x, pt.y) for label, pt in label_points.items()},
        )

    def show(
        self,
    ) -> None:
        """Display the plot inline (e.g., in a Jupyter notebook) or in a window."""
        self._build_and_apply_settings()

        try:
            from IPython.display import Image, display

            # Render to PNG in memory and display inline. We have to do this because we are
            # building the Figure directly.
            buf = BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.fig.dpi)
            buf.seek(0)
            display(Image(data=buf.getvalue()))
        except Exception:
            self.fig.show()

    def save(self, filepath: str, **kwargs: Any) -> None:
        """Save the plot to a file."""
        self._build_and_apply_settings()
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("dpi", self.fig.dpi)
        self.fig.savefig(filepath, **kwargs)
