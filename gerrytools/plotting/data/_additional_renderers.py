from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import matplotlib.patheffects as patheffects
from matplotlib.path import Path
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.transforms import Transform

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.data._gerryplot_dataclasses import ArrowData

if TYPE_CHECKING:
    from gerrytools.plotting.data.gerryplot import GerryPlotBase


class _VerticalTextArrowBoxStyle:
    """Centered top/bottom triangular-tip boxstyle for vertical text arrows."""

    def __init__(
        self,
        *,
        direction: Literal["up", "down"],
        pad: float,
        tip_height_scale: float = 0.8,
        tip_width_scale: float = 0.28,
    ) -> None:
        self.direction = direction
        self.pad = float(pad)
        self.tip_height_scale = float(tip_height_scale)
        self.tip_width_scale = float(tip_width_scale)

    def __call__(
        self,
        x0: float,
        y0: float,
        width: float,
        height: float,
        mutation_size: float,
    ) -> Path:
        """Create a box path with a centered triangular tip at top or bottom."""
        pad_pixels = self.pad * mutation_size
        left = x0 - pad_pixels
        bottom = y0 - pad_pixels
        body_width = width + (2.0 * pad_pixels)
        body_height = height + (2.0 * pad_pixels)
        right = left + body_width
        top = bottom + body_height
        center_x = left + (0.5 * body_width)

        tip_height = max(0.0, float(mutation_size) * self.tip_height_scale)
        head_overhang = max(0.0, float(mutation_size) * self.tip_width_scale)

        if self.direction == "up":
            tip_base_y = top
            tip_y = top + tip_height
            vertices = [
                (left, bottom),
                (right, bottom),
                (right, tip_base_y),
                (right + head_overhang, tip_base_y),
                (center_x, tip_y),
                (left - head_overhang, tip_base_y),
                (left, tip_base_y),
                (left, bottom),
            ]
        else:
            tip_base_y = bottom
            tip_y = bottom - tip_height
            vertices = [
                (left, top),
                (right, top),
                (right, tip_base_y),
                (right + head_overhang, tip_base_y),
                (center_x, tip_y),
                (left - head_overhang, tip_base_y),
                (left, tip_base_y),
                (left, top),
            ]

        codes = [Path.MOVETO] + ([Path.LINETO] * (len(vertices) - 2)) + [Path.CLOSEPOLY]
        return Path(vertices, codes)


class _AnnotationArrowRenderer:
    """Internal helper for rendering deferred annotation arrows."""

    def __init__(self, owner: GerryPlotBase) -> None:
        self._owner = owner

    def render_all(self, arrows: Sequence[ArrowData]) -> None:
        """Render all deferred arrows onto the owner axes."""
        for arrow in arrows:
            self._render_arrow(arrow)

    def _render_arrow(self, arrow: ArrowData) -> None:
        owner = self._owner
        placement = arrow.placement
        coordinate_system = placement.coordinate_system
        transform = owner._ax.transData if coordinate_system == "data" else owner._ax.transAxes

        tip_x, tip_y = arrow.arrowtip
        offset_x, offset_y = placement.text_offset

        default_ha, default_va = owner._default_text_alignment_for_direction(arrow.direction)
        ha = (
            arrow.textstyle.horizontalalignment
            if arrow.textstyle.horizontalalignment is not None
            else default_ha
        )
        va = (
            arrow.textstyle.verticalalignment
            if arrow.textstyle.verticalalignment is not None
            else default_va
        )

        if arrow.arrowtype == "text":
            self._render_text_arrow(
                arrow=arrow,
                transform=transform,
                tip_x=tip_x,
                tip_y=tip_y,
                offset_x=offset_x,
                offset_y=offset_y,
                ha=ha,
                va=va,
            )
            return

        self._render_label_arrow(
            arrow=arrow,
            transform=transform,
            coordinate_system=coordinate_system,
            tip_x=tip_x,
            tip_y=tip_y,
            offset_x=offset_x,
            offset_y=offset_y,
            ha=ha,
            va=va,
        )

    def _render_text_arrow(
        self,
        *,
        arrow: ArrowData,
        transform: Transform,
        tip_x: float,
        tip_y: float,
        offset_x: float,
        offset_y: float,
        ha: Literal["left", "center", "right"],
        va: Literal["bottom", "center", "top"],
    ) -> None:
        owner = self._owner
        style = arrow.textarrowstyle
        if style is None:  # pragma: no cover
            raise RuntimeError("Text arrow missing textarrowstyle.")

        boxstyle_base, box_rotation = owner._default_text_arrow_boxstyle_and_rotation(
            arrow.direction
        )
        text_rotation = (
            float(arrow.textstyle.rotation) if arrow.textstyle.rotation is not None else 0.0
        )
        if style.boxstyle is not None:
            boxstyle: str | object = style.boxstyle
        elif boxstyle_base == "__gerryplot_uparrow__":
            boxstyle = _VerticalTextArrowBoxStyle(direction="up", pad=style.boxpad)
        elif boxstyle_base == "__gerryplot_downarrow__":
            boxstyle = _VerticalTextArrowBoxStyle(direction="down", pad=style.boxpad)
        else:
            boxstyle = f"{boxstyle_base},pad={style.boxpad:g}"

        if isinstance(style.arrowoutlinecolor, str) and style.arrowoutlinecolor.lower() == "none":
            edgecolor: str | tuple[float, float, float, float] = "none"
        else:
            edgecolor = owner._resolved_rgba(
                style.arrowoutlinecolor,
                style.arrowoutlinealpha,
                field="annotation_arrow_outlinecolor",
            )

        text_value = arrow.text if arrow.text is not None else "   "
        text_color = owner._resolved_rgba(
            arrow.textstyle.fontcolor,
            arrow.textstyle.fontalpha,
            field="annotation_arrow_text_color",
        )
        face_color = owner._resolved_rgba(
            style.arrowfacecolor,
            style.arrowfacealpha,
            field="annotation_arrow_facecolor",
        )

        if abs(text_rotation - box_rotation) > 1e-8:
            bbox_artist = owner._ax.text(
                tip_x + offset_x,
                tip_y + offset_y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                color=(0.0, 0.0, 0.0, 0.0),
                fontsize=arrow.textstyle.fontsize,
                fontweight=arrow.textstyle.fontweight,
                fontstyle=arrow.textstyle.fontstyle,
                fontfamily=arrow.textstyle.fontfamily,
                rotation=box_rotation,
                clip_on=arrow.placement.clip_on,
                zorder=arrow.placement.zorder,
                bbox=dict(
                    boxstyle=boxstyle,
                    fc=face_color,
                    ec=edgecolor,
                    lw=style.arrowoutlinewidth,
                ),
            )
            owner._align_text_arrow_tip_to_position(
                bbox_artist,
                desired_tip=(tip_x + offset_x, tip_y + offset_y),
                coordinate_transform=transform,
                direction=arrow.direction,
            )
            aligned_x, aligned_y = bbox_artist.get_position()
            text_artist = owner._ax.text(
                aligned_x,
                aligned_y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                color=text_color,
                fontsize=arrow.textstyle.fontsize,
                fontweight=arrow.textstyle.fontweight,
                fontstyle=arrow.textstyle.fontstyle,
                fontfamily=arrow.textstyle.fontfamily,
                rotation=text_rotation,
                clip_on=arrow.placement.clip_on,
                zorder=float(arrow.placement.zorder) + 0.01,
            )
            text_effects = owner._annotation_text_outline_effects(arrow.textstyle)
            if text_effects is not None:
                text_artist.set_path_effects(text_effects)
            return

        text_artist = owner._ax.text(
            tip_x + offset_x,
            tip_y + offset_y,
            text_value,
            transform=transform,
            ha=ha,
            va=va,
            color=text_color,
            fontsize=arrow.textstyle.fontsize,
            fontweight=arrow.textstyle.fontweight,
            fontstyle=arrow.textstyle.fontstyle,
            fontfamily=arrow.textstyle.fontfamily,
            rotation=box_rotation,
            clip_on=arrow.placement.clip_on,
            zorder=arrow.placement.zorder,
            bbox=dict(
                boxstyle=boxstyle,
                fc=face_color,
                ec=edgecolor,
                lw=style.arrowoutlinewidth,
            ),
        )
        owner._align_text_arrow_tip_to_position(
            text_artist,
            desired_tip=(tip_x + offset_x, tip_y + offset_y),
            coordinate_transform=transform,
            direction=arrow.direction,
        )
        text_effects = owner._annotation_text_outline_effects(arrow.textstyle)
        if text_effects is not None:
            text_artist.set_path_effects(text_effects)

    def _render_label_arrow(
        self,
        *,
        arrow: ArrowData,
        transform: Transform,
        coordinate_system: Literal["data", "axes fraction"],
        tip_x: float,
        tip_y: float,
        offset_x: float,
        offset_y: float,
        ha: Literal["left", "center", "right"],
        va: Literal["bottom", "center", "top"],
    ) -> None:
        owner = self._owner
        style = arrow.labelarrowstyle
        if style is None:  # pragma: no cover
            raise RuntimeError("Label arrow missing labelarrowstyle.")

        placement = arrow.placement
        unit_x, unit_y = owner._direction_unit_vector(arrow.direction)
        if placement.arrowtail is None:
            if arrow.arrow_length_percentage is not None:
                owner.fig.canvas.draw()
                axes_bbox = owner._ax.get_window_extent()
                if arrow.direction in {"left", "right"}:
                    direction_span_pixels = float(axes_bbox.width)
                else:
                    direction_span_pixels = float(axes_bbox.height)
                arrow_length_pixels = (
                    float(arrow.arrow_length_percentage) / 100.0
                ) * direction_span_pixels
                tail_x, tail_y = owner._shift_point_along_direction_pixels(
                    (tip_x, tip_y),
                    direction=arrow.direction,
                    signed_pixels=-arrow_length_pixels,
                    coordinate_transform=transform,
                )
            else:
                tail_x = tip_x - (unit_x * placement.tail_length)
                tail_y = tip_y - (unit_y * placement.tail_length)
        else:
            tail_x, tail_y = placement.arrowtail

        if isinstance(style.arrowoutlinecolor, str) and style.arrowoutlinecolor.lower() == "none":
            regular_edgecolor: str | tuple[float, float, float, float] = "none"
        else:
            regular_edgecolor = owner._resolved_rgba(
                style.arrowoutlinecolor,
                style.arrowoutlinealpha,
                field="annotation_arrow_outlinecolor",
            )

        owner._ax.annotate(
            "",
            xy=(tip_x, tip_y),
            xytext=(tail_x, tail_y),
            xycoords=coordinate_system,
            textcoords=coordinate_system,
            clip_on=placement.clip_on,
            zorder=placement.zorder,
            arrowprops=dict(
                arrowstyle=style.arrowstyle,
                connectionstyle=style.connectionstyle,
                mutation_scale=style.arrowhead_scale,
                shrinkA=style.shrink_a,
                shrinkB=style.shrink_b,
                facecolor=owner._resolved_rgba(
                    style.arrowfacecolor,
                    style.arrowfacealpha,
                    field="annotation_arrow_facecolor",
                ),
                edgecolor=regular_edgecolor,
                linewidth=style.arrowoutlinewidth,
                linestyle=style.linestyle,
            ),
        )

        text_value = arrow.text if arrow.text is not None else ""
        if text_value == "":
            return

        if arrow.label_position is None:
            label_anchor_x = tail_x - (unit_x * placement.label_padding)
            label_anchor_y = tail_y - (unit_y * placement.label_padding)
        else:
            label_anchor_x, label_anchor_y = arrow.label_position

        text_x = label_anchor_x + offset_x
        text_y = label_anchor_y + offset_y
        bbox: dict[str, object] | None = (
            None if arrow.labelbox_options is None else arrow.labelbox_options.to_mpl_bbox()
        )

        if arrow.labelfont_options is not None:
            outline_color, _ = resolve_color_and_alpha(
                arrow.labelfont_options.outlinecolor,
                alpha=1.0,
            )
            text_effects: list[AbstractPathEffect] | None = [
                patheffects.Stroke(
                    linewidth=float(arrow.labelfont_options.outlinewidth),
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]
            text_artist = owner._ax.text(
                text_x,
                text_y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                rotation=arrow.textstyle.rotation if arrow.textstyle.rotation is not None else 0.0,
                clip_on=placement.clip_on,
                zorder=placement.zorder,
                bbox=bbox,
                **arrow.labelfont_options.to_mpl_text_kwargs(),
            )
        else:
            text_effects = owner._annotation_text_outline_effects(arrow.textstyle)
            text_artist = owner._ax.text(
                text_x,
                text_y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                rotation=arrow.textstyle.rotation if arrow.textstyle.rotation is not None else 0.0,
                clip_on=placement.clip_on,
                zorder=placement.zorder,
                bbox=bbox,
                color=owner._resolved_rgba(
                    arrow.textstyle.fontcolor,
                    arrow.textstyle.fontalpha,
                    field="annotation_arrow_text_color",
                ),
                fontsize=arrow.textstyle.fontsize,
                fontweight=arrow.textstyle.fontweight,
                fontstyle=arrow.textstyle.fontstyle,
                fontfamily=arrow.textstyle.fontfamily,
            )
        text_artist.set_clip_path(owner._ax.patch)
        if text_effects is not None:
            text_artist.set_path_effects(text_effects)
