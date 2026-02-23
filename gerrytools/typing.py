from collections.abc import Hashable, Iterable
from typing import Callable, Literal, Sequence, TypeAlias, TypedDict

from geopandas import GeoDataFrame, GeoSeries
from matplotlib.artist import Artist
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from pandas import DataFrame, Series
from pyproj import CRS

Numeric: TypeAlias = int | float
"""Numeric scalar type used in plotting/color APIs."""

NumericIterable: TypeAlias = Iterable[Numeric]
"""Iterable collection of numeric scalar values."""

RGBColor: TypeAlias = tuple[Numeric, Numeric, Numeric]
"""RGB color tuple in either 0-1 or 0-255 component scale."""

RGBAColor: TypeAlias = tuple[Numeric, Numeric, Numeric, Numeric]
"""RGBA color tuple in either 0-1 or 0-255 component scale."""

MplRGBAColor: TypeAlias = tuple[float, float, float, float]
"""Matplotlib-normalized RGBA tuple with components in ``[0.0, 1.0]``."""

HexColor: TypeAlias = str
"""Hex-encoded color string (for example ``#RRGGBB`` or ``#RRGGBBAA``)."""

Color: TypeAlias = str | RGBColor
"""Public color token type used by GerryTools plotting and LaTeX APIs."""

MplBaseColor: TypeAlias = str | RGBColor | RGBAColor
"""Matplotlib-compatible base color token without explicit external alpha override."""

MplCompatibleColor: TypeAlias = MplBaseColor | tuple[MplBaseColor, Numeric]
"""Color token accepted by Matplotlib conversion helpers, including ``(color, alpha)``."""

ResolvedColor: TypeAlias = tuple[str, float]
"""Resolved color tuple ``(hex6_or_none, alpha)`` returned by color normalizers."""

CategoryKey: TypeAlias = Hashable
"""Hashable category/group key used by categorical and geometry plot APIs."""

CategoryColorMap: TypeAlias = dict[CategoryKey, Color]
"""Mapping from category keys to explicit colors for categorical layers."""

GeoColorMap: TypeAlias = str | Color | Colormap | CategoryColorMap | Series
"""Color mapping specification accepted by geometry plotting layers."""

CRSLike: TypeAlias = CRS | str | int
"""Coordinate reference system token accepted by GeoPandas reprojection methods."""

NumericArrayLike: TypeAlias = Numeric | NumericIterable | NDArray | Series | DataFrame
"""Flexible numeric data input accepted by plotting/data coercion utilities."""

# Format takes in original value and currently rendered string
# and returns original value and new rendered string
TableCellValue: TypeAlias = object
"""Arbitrary DataFrame cell value used by table/formatter pipelines."""

CellWrapper: TypeAlias = Callable[[TableCellValue, str], tuple[TableCellValue, str]]
"""Formatter callback that receives ``(raw_value, rendered_text)`` and returns updated pair."""

# Type alias for tick types in Matplotlib
TickType = Literal["major", "minor", "both"]

# Type alias for histogram
BinsType = int | Sequence[float] | str | NDArray
HistType = Literal["overlay", "stack", "weave", "outline"]

GeoSource = GeoDataFrame | GeoSeries
"""Accepted geometry container type for geometry plotting APIs."""

LegendHandle: TypeAlias = Artist
"""Legend handle artist type returned by plotting classes."""

MplKwargs: TypeAlias = dict[str, object]
"""Generic Matplotlib kwargs dictionary with JSON-like value constraints."""

SavefigKwargValue: TypeAlias = (
    bool | int | float | str | tuple[float, float] | tuple[float, float, float, float] | None
)
"""Commonly used scalar/tuple values accepted by ``Figure.savefig`` kwargs."""

SavefigKwargs: TypeAlias = dict[str, SavefigKwargValue]
"""Keyword arguments accepted by ``Figure.savefig`` wrappers in plotting utilities."""


class PlotMarkerKwargs(TypedDict):
    """Marker kwargs emitted by ``PointMarkerOptions.to_mpl_settings_dict``."""

    markerfacecolor: MplRGBAColor
    marker: str
    markersize: float
    markeredgecolor: MplRGBAColor
    markeredgewidth: float
    zorder: int


class ScatterMarkerKwargs(TypedDict):
    """Marker kwargs emitted by ``PointMarkerOptions.to_mpl_scatter_settings_dict``."""

    marker: str
    s: float
    edgecolor: MplRGBAColor
    linewidths: float
    zorder: int


class AxisLabelKwargs(TypedDict, total=False):
    """Keyword arguments for ``Axes.set_xlabel`` and ``Axes.set_ylabel``."""

    color: MplRGBAColor
    fontsize: float | int
    fontweight: str
    fontstyle: Literal["normal", "italic", "oblique"]
    fontfamily: str
    labelpad: float


class TitleKwargs(TypedDict, total=False):
    """Keyword arguments for ``Axes.set_title``."""

    color: MplRGBAColor
    fontsize: float | int
    fontweight: str
    fontstyle: Literal["normal", "italic", "oblique"]
    fontfamily: str
    loc: Literal["left", "center", "right"]
    pad: float
