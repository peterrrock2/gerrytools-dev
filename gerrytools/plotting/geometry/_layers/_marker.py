"""`_MarkerLayer` — point markers with optional labels."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import matplotlib.patheffects as patheffects
import pandas as pd
from geopandas import GeoSeries
from matplotlib.axes import Axes

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import CRSLike


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
        return pd.Series(
            dtype=object
        )  # pragma: no cover - implemented only to satisfy the abstract interface

    def render(
        self,
        ax: Axes,
        *,
        target_crs: CRSLike | None = None,
        **kwargs: object,
    ) -> Axes:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.
            **kwargs (object): Additional keyword arguments (not used).

        Returns:
            Axes: The Axes with the layer rendered.
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
            text_effects: list[patheffects.AbstractPathEffect] = [
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
