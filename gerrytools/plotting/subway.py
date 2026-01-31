import math
from dataclasses import dataclass
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from gerrytools.colors import convert_color_to_hexa_or_none
from gerrytools.typing import Color


@dataclass(frozen=True)
class SubwaySignOptions:
    """Options for subway sign plotting.

    Attributes:
        radius (float): Radius of each subway sign circle. Default is 0.3.
        edgecolor (Color): Edge color of the subway sign circles. Default is "black".
        linewidth (float): Line width of the subway sign circle edges. Default is 1.5.
        fontsize (float): Font size of the label text. Default is 14.
        fontweight (str): Font weight of the label text. Default is "bold".
        fontcolor (Color): Font color of the label text. Default is "white".
        fontoutlinewidth (float): Width of the outline around the label text. Default is 2.
        horizontalgap (float | None): Horizontal gap between subway signs. If None,
            defaults to 0.3 * radius. Default is None.
        verticalgap (float | None): Vertical gap between subway signs. If None,
            defaults to 0.3 * radius. Default is None.
        padding (float | None): Padding around the entire layout. If None,
            defaults to 0.2 * radius. Default is None.
        ragged (bool): Whether to allow ragged packing of signs when the number of
            signs does not fill the grid completely. Default is True.
        raggededge (Literal["first", "last"]): If ragged packing is enabled, determines whether
            the incomplete band is placed at the start or end of the layout. Default is "last".
    """

    radius: float = 0.3
    edgecolor: Color = "black"
    linewidth: float = 1.5

    fontsize: float = 14
    fontweight: str = "bold"
    fontcolor: Color = "white"
    fontoutlinewidth: float = 2

    horizontalgap: float | None = None
    verticalgap: float | None = None
    padding: float | None = None

    ragged: bool = True
    raggededge: Literal["first", "last"] = "last"


def _validate_subway_settings(
    *,
    colors: list[Color],
    labels: list[str],
    orientation: Literal["vertical", "horizontal"],
    n_bands: int | None,
    max_items_per_band: int | None,
    sign_options: SubwaySignOptions,
):
    """Validate settings for subway sign plotting.

    Raises:
        ValueError: If any settings are invalid.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("`orientation` must be either 'vertical' or 'horizontal'.")
    if n_bands is not None and max_items_per_band is not None:
        raise ValueError("Only one of `n_rows` and `max_items_per_row` may be set.")
    if len(colors) != len(labels):
        raise ValueError("`colors` and `labels` must have the same length.")
    if len(labels) == 0:
        raise ValueError("Nothing to draw: `labels` is empty.")
    if sign_options.radius <= 0:
        raise ValueError("`sign_options.radius` must be > 0.")
    if n_bands is not None and n_bands <= 0:
        raise ValueError("`n_bands` must be > 0.")
    if max_items_per_band is not None and max_items_per_band <= 0:
        raise ValueError("`max_items_per_band` must be > 0.")


@dataclass(frozen=True)
class _SubwayPlotLayout:
    """Layout information for subway sign plotting.

    Attributes:
        item_count (int): Total number of subway signs.
        radius (float): Radius of each subway sign circle.
        diameter (float): Diameter of each subway sign circle.
        horizontalgap (float): Horizontal gap between subway signs.
        verticalgap (float): Vertical gap between subway signs.
        padding (float): Padding around the entire layout.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
    """

    item_count: int
    radius: float
    diameter: float
    horizontalgap: float
    verticalgap: float
    padding: float
    row_count: int
    column_count: int


def _determine_offsets_and_counts(
    *,
    labels: list[str],
    orientation: Literal["vertical", "horizontal"] = "horizontal",
    n_bands: int | None = None,
    max_items_per_band: int | None = None,
    sign_options: SubwaySignOptions,
) -> _SubwayPlotLayout:
    item_count = len(labels)
    radius = float(sign_options.radius)
    diameter = 2.0 * radius

    horizontalgap = (
        sign_options.horizontalgap if sign_options.horizontalgap is not None else 0.3 * radius
    )
    verticalgap = sign_options.verticalgap if sign_options.verticalgap is not None else 0.3 * radius
    padding = sign_options.padding if sign_options.padding is not None else 0.2 * radius

    if max_items_per_band is not None:
        if orientation == "horizontal":
            column_count = int(max_items_per_band)
            row_count = math.ceil(item_count / column_count)
        else:  # vertical
            row_count = int(max_items_per_band)
            column_count = math.ceil(item_count / row_count)

    elif n_bands is not None:
        if orientation == "horizontal":
            row_count = int(n_bands)
            column_count = math.ceil(item_count / row_count)
        else:  # vertical
            column_count = int(n_bands)
            row_count = math.ceil(item_count / column_count)

    else:
        if orientation == "horizontal":
            row_count = 1
            column_count = item_count
        else:  # vertical
            row_count = item_count
            column_count = 1

    return _SubwayPlotLayout(
        item_count=item_count,
        radius=radius,
        diameter=diameter,
        horizontalgap=horizontalgap,
        verticalgap=verticalgap,
        padding=padding,
        row_count=row_count,
        column_count=column_count,
    )


def _determine_grid_position(
    *,
    linear_index: int,
    sign_options: SubwaySignOptions,
    orientation: Literal["vertical", "horizontal"],
    item_count: int,
    row_count: int,
    column_count: int,
) -> tuple[int, int]:
    """Given a linear index, determine the (row, column) position in the grid,
    accounting for ragged edges if necessary.

    Args:
        linear_index (int): The linear index of the sign.
        orientation (Literal["vertical", "horizontal"]): The layout orientation.
        item_count (int): Total number of signs.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
    """
    raggededge = getattr(sign_options, "raggededge", "last")
    if raggededge not in ("first", "last"):
        raise ValueError("`sign_options.raggededge` must be 'first' or 'last'.")

    if orientation == "horizontal":
        full_band_size = column_count
        ragged_band_size = item_count - (row_count - 1) * column_count

        if ragged_band_size == full_band_size or row_count == 1:
            row_index = linear_index // full_band_size
            col_index = linear_index % full_band_size
            return row_index, col_index

        if raggededge == "last":
            row_index = linear_index // full_band_size
            col_index = linear_index % full_band_size
            return row_index, col_index

        if linear_index < ragged_band_size:
            return 0, linear_index

        remaining_index = linear_index - ragged_band_size
        row_index = 1 + (remaining_index // full_band_size)
        col_index = remaining_index % full_band_size
        return row_index, col_index

    else:
        full_band_size = row_count
        ragged_band_size = item_count - (column_count - 1) * row_count

        if ragged_band_size == full_band_size or column_count == 1:
            row_index = linear_index % full_band_size
            col_index = linear_index // full_band_size
            return row_index, col_index

        if raggededge == "last":
            row_index = linear_index % full_band_size
            col_index = linear_index // full_band_size
            return row_index, col_index

        if linear_index < ragged_band_size:
            return linear_index, 0

        remaining_index = linear_index - ragged_band_size
        col_index = 1 + (remaining_index // full_band_size)
        row_index = remaining_index % full_band_size
        return row_index, col_index


def _ragged_edge_offset(
    *,
    row_index: int,
    col_index: int,
    sign_options: SubwaySignOptions,
    orientation: Literal["vertical", "horizontal"],
    item_count: int,
    row_count: int,
    column_count: int,
    x_step: float,
    y_step: float,
) -> tuple[float, float]:
    """Determine the offset for the incomplete outer edge by shifting along primary layout axis.

    Args:
        row_index (int): The row index of the sign.
        col_index (int): The column index of the sign.
        sign_options (SubwaySignOptions): Options for sign appearance.
        orientation (Literal["vertical", "horizontal"]): The layout orientation.
        item_count (int): Total number of signs.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
        x_step (float): The horizontal step size between sign centers.
        y_step (float): The vertical step size between sign centers.

    Returns:
        tuple[float, float]: The (dx, dy) offset to apply to the sign's center.
    """
    raggededge = getattr(sign_options, "raggededge", "last")
    if orientation == "horizontal":
        items_in_partial_row = item_count - (row_count - 1) * column_count
        if items_in_partial_row == column_count:
            return 0.0, 0.0  # no ragged row

        ragged_row_index = 0 if raggededge == "first" else (row_count - 1)
        if row_index != ragged_row_index:
            return 0.0, 0.0

        missing = column_count - items_in_partial_row
        dx = 0.5 * missing * x_step
        return dx, 0.0

    else:
        items_in_partial_col = item_count - (column_count - 1) * row_count
        if items_in_partial_col == row_count:
            return 0.0, 0.0  # no ragged column

        ragged_col_index = 0 if raggededge == "first" else (column_count - 1)
        if col_index != ragged_col_index:
            return 0.0, 0.0

        missing = row_count - items_in_partial_col
        dy = -0.5 * missing * y_step
        return 0.0, dy


def _normalize_colors_and_adjust_item_order(
    *,
    colors: list[Color],
    labels: list[str],
    orientation: Literal["vertical", "horizontal"],
    item_count: int,
    row_count: int,
    column_count: int,
    max_items_per_band: int | None,
    reverse_display_order: bool,
    sign_options: SubwaySignOptions,
):
    """Normalize colors to hex format and adjust item order for ragged edges and reverse display.

    Args:
        colors (list[Color]): List of colors for each sign.
        labels (list[str]): List of labels for each sign.
        orientation (Literal["vertical", "horizontal"]): The layout orientation.
        item_count (int): Total number of signs.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
        max_items_per_band (int | None): Maximum number of items per band (row or column).
        reverse_display_order (bool): Whether to reverse the display order of signs.
        sign_options (SubwaySignOptions): Options for sign appearance.

    """
    new_colors = []
    for c in colors:
        hexa_color = convert_color_to_hexa_or_none(c)
        if hexa_color is None:
            raise ValueError(f"Color {c!r} could not be converted to a valid color.")
        new_colors.append(hexa_color)

    items = list(zip(new_colors, labels))
    raggededge = getattr(sign_options, "raggededge", "last")

    if reverse_display_order:
        new_items = []
        band_count = row_count if orientation == "horizontal" else column_count
        band_width = column_count if orientation == "horizontal" else row_count

        remainder = item_count % band_width

        starting_offset = 0
        if raggededge == "first" and remainder != 0:
            new_items.extend(items[-remainder:])
            starting_offset = remainder

        for band_index in range(band_count):
            start_index = band_index * band_width + starting_offset
            end_index = start_index + band_width
            start_index = len(items) - end_index
            end_index = start_index + band_width
            new_items.extend(items[start_index:end_index])

        if raggededge == "last" and remainder != 0:
            new_items.extend(items[:remainder])

        items = new_items

    return items


def _draw_sign(
    *,
    axes: plt.Axes,
    index: int,
    face_color: Color,
    label_text: str,
    sign_options: SubwaySignOptions,
    orientation: Literal["vertical", "horizontal"],
    item_count: int,
    row_count: int,
    column_count: int,
    x_step: float,
    y_step: float,
    radius: float,
    text_outline_effects: list,
):
    """Draw a single subway sign on the given axes.

    Args:
        axes (plt.Axes): The axes to draw on.
        index (int): The linear index of the sign.
        face_color (Color): The face color of the sign.
        label_text (str): The label text of the sign.
        sign_options (SubwaySignOptions): Options for sign appearance.
        orientation (Literal["vertical", "horizontal"]): The layout orientation.
        item_count (int): Total number of signs.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
        x_step (float): The horizontal step size between sign centers.
        y_step (float): The vertical step size between sign centers.
        radius (float): The radius of the sign circle.
        text_outline_effects (list): Path effects for outlining the text.
    """
    normalized_face_color = mcolors.to_rgba(face_color)

    row_index, col_index = _determine_grid_position(
        linear_index=index,
        sign_options=sign_options,
        orientation=orientation,
        item_count=item_count,
        row_count=row_count,
        column_count=column_count,
    )

    base_x_center = col_index * x_step + radius
    base_y_center = (row_count - 1 - row_index) * y_step + radius

    dx, dy = _ragged_edge_offset(
        row_index=row_index,
        col_index=col_index,
        sign_options=sign_options,
        orientation=orientation,
        item_count=item_count,
        row_count=row_count,
        column_count=column_count,
        x_step=x_step,
        y_step=y_step,
    )

    x_center = base_x_center + dx
    y_center = base_y_center + dy

    circle_patch = Circle(
        (x_center, y_center),
        radius=radius,
        facecolor=normalized_face_color,
        edgecolor=sign_options.edgecolor,
        linewidth=sign_options.linewidth,
    )
    axes.add_patch(circle_patch)

    text_artist = axes.text(
        x_center,
        y_center,
        str(label_text),
        ha="center",
        va="center",
        color=sign_options.fontcolor,
        fontsize=sign_options.fontsize,
        fontweight=sign_options.fontweight,
    )
    text_artist.set_path_effects(text_outline_effects)


def subway_signs(
    colors: list[Color],
    labels: list[str],
    *,
    orientation: Literal["vertical", "horizontal"] = "horizontal",
    n_bands: int | None = None,
    max_items_per_band: int | None = None,
    reverse_display_order: bool = False,
    sign_options: SubwaySignOptions | None = None,
    save_path: str | None = None,
):
    """Draw a grid of colored 'subway signs' with labels.

    Args:
        colors (list[Color]): List of colors for each sign.
        labels (list[str]): List of labels for each sign.
        orientation (Literal["vertical", "horizontal"], optional): Orientation of the layout.
            Defaults to "horizontal".
        n_bands (int | None, optional): Number of bands (rows or columns) in the layout.
            If None, determined by `max_items_per_band` or defaults to 1 band.
            Defaults to None.
        max_items_per_band (int | None, optional): Maximum number of items per band (row or column).
            If None, determined by `n_bands` or defaults to all items in one band.
            Defaults to None.
        reverse_display_order (bool, optional): Whether to reverse the order in which the
            signs are displayed. The general layout (i.e. which side the ragged edge appears on)
            will remain the same. Defaults to False.
        sign_options (SubwaySignOptions | None, optional): Options for sign appearance.
            Defaults to None.
        save_path (str | None, optional): If provided, saves the figure to this path.
            Defaults to None.
    """
    if sign_options is None:
        sign_options = SubwaySignOptions()

    _validate_subway_settings(
        colors=colors,
        labels=labels,
        orientation=orientation,
        n_bands=n_bands,
        max_items_per_band=max_items_per_band,
        sign_options=sign_options,
    )

    layout = _determine_offsets_and_counts(
        labels=labels,
        orientation=orientation,
        n_bands=n_bands,
        max_items_per_band=max_items_per_band,
        sign_options=sign_options,
    )

    x_step = layout.diameter + layout.horizontalgap
    y_step = layout.diameter + layout.verticalgap

    layout_width = layout.column_count * x_step - layout.horizontalgap
    layout_height = layout.row_count * y_step - layout.verticalgap

    fig_width_in = layout_width + 2 * layout.padding
    fig_height_in = layout_height + 2 * layout.padding

    figure, axes = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=200)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)  # allow axes to fill the whole canvas
    axes.set_position((0, 0, 1, 1))  # force axes to fill the whole figure
    axes.set_aspect("equal")
    axes.axis("off")

    text_outline_effects = [
        patheffects.Stroke(linewidth=sign_options.fontoutlinewidth, foreground="black"),
        patheffects.Normal(),
    ]

    items = _normalize_colors_and_adjust_item_order(
        colors=colors,
        labels=labels,
        orientation=orientation,
        item_count=layout.item_count,
        row_count=layout.row_count,
        column_count=layout.column_count,
        max_items_per_band=max_items_per_band,
        reverse_display_order=reverse_display_order,
        sign_options=sign_options,
    )

    for index, (face_color, label_text) in enumerate(items):
        _draw_sign(
            axes=axes,
            index=index,
            face_color=face_color,
            label_text=label_text,
            sign_options=sign_options,
            orientation=orientation,
            item_count=layout.item_count,
            row_count=layout.row_count,
            column_count=layout.column_count,
            x_step=x_step,
            y_step=y_step,
            radius=layout.radius,
            text_outline_effects=text_outline_effects,
        )

    total_width = layout.column_count * x_step - layout.horizontalgap
    total_height = layout.row_count * y_step - layout.verticalgap
    axes.set_xlim(-layout.padding, total_width + layout.padding)
    axes.set_ylim(-layout.padding, total_height + layout.padding)

    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight", pad_inches=0)
