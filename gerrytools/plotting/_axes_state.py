"""`_ManagedAxesState` — per-unit ownership tracking for axes-level state.

Gerrytools plots avoid ``ax.clear()`` on rebuild and instead rely on two
cooperating mechanisms:

1. The artist registry removes only the matplotlib artists gerrytools created.
2. This module tracks who most recently set each *axes-level* setting
   (xlim, scale, ticks, label, title, frame, legend, axis visibility) and
   uses that to decide, on every rebuild, whether to reapply gerrytools state
   or yield to direct matplotlib changes the user made.

Resolution is most-recent-wins per unit: whichever party (gerrytools or the
user) touched a given unit last owns it on the next rebuild.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.ticker import AutoLocator, MaxNLocator, NullLocator
from matplotlib.typing import ColorType

# Public unit identifiers used throughout the rebuild flow.
UNIT_X_LIMITS = "x_limits"
UNIT_Y_LIMITS = "y_limits"
UNIT_X_TICKS = "x_ticks"
UNIT_Y_TICKS = "y_ticks"
UNIT_X_TICK_STYLE = "x_tick_style"
UNIT_Y_TICK_STYLE = "y_tick_style"
UNIT_X_LABEL = "x_label"
UNIT_Y_LABEL = "y_label"
UNIT_TITLE = "title"
UNIT_X_SCALE = "x_scale"
UNIT_Y_SCALE = "y_scale"
UNIT_FRAME = "frame"
UNIT_LEGEND = "legend"
UNIT_AXIS_VISIBILITY = "axis_visibility"
UNIT_ASPECT = "aspect"

_AUTOSCALE_PROTECTED: frozenset[str] = frozenset({UNIT_X_LIMITS, UNIT_Y_LIMITS})

_ALL_UNITS: tuple[str, ...] = (
    UNIT_X_LIMITS,
    UNIT_Y_LIMITS,
    UNIT_X_TICKS,
    UNIT_Y_TICKS,
    UNIT_X_TICK_STYLE,
    UNIT_Y_TICK_STYLE,
    UNIT_X_LABEL,
    UNIT_Y_LABEL,
    UNIT_TITLE,
    UNIT_X_SCALE,
    UNIT_Y_SCALE,
    UNIT_FRAME,
    UNIT_LEGEND,
    UNIT_AXIS_VISIBILITY,
    UNIT_ASPECT,
)

# Internal sentinel marking "gerrytools claimed the unit but has not yet
# recorded a concrete applied value." Never escapes to user code; never
# substituted for ``None`` because ``None`` is a real public value for some
# units (e.g. ``ax.get_legend()`` returns ``None`` when there is no legend).
_NO_LAST_APPLIED = object()

OwnershipState = Literal["external", "gerrytools_explicit", "gerrytools_default", "unclaimed"]

# Tolerances for autoscale-protected limit comparisons, compared via
# ``math.isclose(..., rel_tol=1e-9, abs_tol=1e-12)``.
_LIMIT_REL_TOL = 1e-9
_LIMIT_ABS_TOL = 1e-12

# Float tolerance for tick locations and RGBA components.
_VALUE_REL_TOL = 1e-9
_VALUE_ABS_TOL = 1e-12


# ---------------------------------------------------------------------------
# Snapshot dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LabelSnapshot:
    """Atomic text+style snapshot for an axis label.

    All fields read from matplotlib public getters on the underlying ``Text``
    artist, except ``labelpad`` which lives on ``ax.<x|y>axis.labelpad``.
    """

    text: str
    fontsize: float | str | None
    fontweight: str | int | None
    fontstyle: str | None
    fontfamily: tuple[str, ...] | None
    color: tuple[float, float, float, float]
    labelpad: float | None


@dataclass(frozen=True)
class _TitleSnapshot:
    """Atomic text+style snapshot for a title.

    ``pad`` has no stable public getter in matplotlib 3.10.6; the snapshot
    field is always populated from the last applied value, not from
    matplotlib. External direct changes to title pad are therefore not
    detected — a known limitation of matplotlib's title API.
    """

    text: str
    fontsize: float | str | None
    fontweight: str | int | None
    fontstyle: str | None
    fontfamily: tuple[str, ...] | None
    color: tuple[float, float, float, float]
    loc: str | None
    pad: float | None


@dataclass(frozen=True)
class _TickSnapshot:
    """Atomic snapshot of tick locations, labels and label visibility."""

    locations: tuple[float, ...]
    labels: tuple[str, ...]
    labels_visible: bool


@dataclass(frozen=True)
class _TickStyleSnapshot:
    """Snapshot of tick label and tick mark styling.

    Read from major ticks only at this layer; callers that opt into minor or
    "both" must compare with the matching ticktype on the gerrytools side.
    Tick-mark colors come from ``Tick.tick1line.get_color()`` which encodes
    alpha in the RGBA tuple.
    """

    label_size: float | str | None
    label_rotation: float
    label_color: tuple[float, float, float, float]
    label_weight: str | int | None
    label_style: str | None
    label_family: tuple[str, ...] | None
    tick_color: tuple[float, float, float, float]


@dataclass(frozen=True)
class _AxesSnapshot:
    """All observable axes-level state read in one pass."""

    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    x_ticks: _TickSnapshot
    y_ticks: _TickSnapshot
    x_tick_style: _TickStyleSnapshot
    y_tick_style: _TickStyleSnapshot
    x_label: _LabelSnapshot
    y_label: _LabelSnapshot
    title: _TitleSnapshot
    x_scale: str
    y_scale: str
    frame: tuple[bool, bool, bool, bool]  # top, right, bottom, left
    legend: Legend | None
    axis_visibility: bool
    aspect: str | float  # matplotlib's "auto" default or a numeric ratio


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _rgba(color: object) -> tuple[float, float, float, float]:
    # matplotlib artist getters (e.g. ``get_color``) return loosely-typed color
    # values; ``to_rgba`` validates them at runtime.
    r, g, b, a = mcolors.to_rgba(cast(ColorType, color))
    return (float(r), float(g), float(b), float(a))


def _fontfamily_to_tuple(value: object) -> tuple[str, ...] | None:
    """Normalize a matplotlib fontfamily return value to a hashable tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _label_snapshot(ax: Axes, which: Literal["x", "y"]) -> _LabelSnapshot:
    text_artist = ax.xaxis.label if which == "x" else ax.yaxis.label
    return _LabelSnapshot(
        text=text_artist.get_text(),
        fontsize=text_artist.get_fontsize(),
        fontweight=text_artist.get_fontweight(),
        fontstyle=text_artist.get_fontstyle(),
        fontfamily=_fontfamily_to_tuple(text_artist.get_fontfamily()),
        color=_rgba(text_artist.get_color()),
        labelpad=float(ax.xaxis.labelpad if which == "x" else ax.yaxis.labelpad),
    )


def _title_snapshot(ax: Axes, pad: float | None = None) -> _TitleSnapshot:
    text_artist = ax.title
    return _TitleSnapshot(
        text=text_artist.get_text(),
        fontsize=text_artist.get_fontsize(),
        fontweight=text_artist.get_fontweight(),
        fontstyle=text_artist.get_fontstyle(),
        fontfamily=_fontfamily_to_tuple(text_artist.get_fontfamily()),
        color=_rgba(text_artist.get_color()),
        loc=None,
        pad=pad,
    )


def _tick_snapshot(ax: Axes, which: Literal["x", "y"]) -> _TickSnapshot:
    if which == "x":
        locations = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels(minor=False)]
        visible_flags = [t.get_visible() for t in ax.get_xticklabels(minor=False)]
    else:
        locations = ax.get_yticks()
        labels = [t.get_text() for t in ax.get_yticklabels(minor=False)]
        visible_flags = [t.get_visible() for t in ax.get_yticklabels(minor=False)]
    # "Tick labels visible" is a single boolean for the unit: any-visible.
    labels_visible = bool(any(visible_flags)) if visible_flags else True
    return _TickSnapshot(
        locations=tuple(float(loc) for loc in locations),
        labels=tuple(labels),
        labels_visible=labels_visible,
    )


def _tick_style_snapshot(ax: Axes, which: Literal["x", "y"]) -> _TickStyleSnapshot:
    axis = ax.xaxis if which == "x" else ax.yaxis
    major_ticks = axis.get_major_ticks()
    label_texts = (
        ax.get_xticklabels(minor=False) if which == "x" else ax.get_yticklabels(minor=False)
    )
    if label_texts:
        first = label_texts[0]
        label_size = first.get_fontsize()
        label_rotation = float(first.get_rotation())
        label_color = _rgba(first.get_color())
        label_weight = first.get_fontweight()
        label_style = first.get_fontstyle()
        label_family = _fontfamily_to_tuple(first.get_fontfamily())
    else:
        label_size = None
        label_rotation = 0.0
        label_color = (0.0, 0.0, 0.0, 1.0)
        label_weight = None
        label_style = None
        label_family = None
    if major_ticks:
        tick_color = _rgba(major_ticks[0].tick1line.get_color())
    else:
        tick_color = (0.0, 0.0, 0.0, 1.0)
    return _TickStyleSnapshot(
        label_size=label_size,
        label_rotation=label_rotation,
        label_color=label_color,
        label_weight=label_weight,
        label_style=label_style,
        label_family=label_family,
        tick_color=tick_color,
    )


def _frame_snapshot(ax: Axes) -> tuple[bool, bool, bool, bool]:
    return (
        ax.spines["top"].get_visible(),
        ax.spines["right"].get_visible(),
        ax.spines["bottom"].get_visible(),
        ax.spines["left"].get_visible(),
    )


def _is_default_locator(axis_obj: object) -> bool:
    """Heuristic: AutoLocator / MaxNLocator / NullLocator mean default-ish.

    FixedLocator and other concrete locators are taken as evidence that the
    user (or another library) set tick positions explicitly.
    """
    locator = axis_obj.get_major_locator()  # type: ignore[attr-defined]
    return isinstance(locator, (AutoLocator, MaxNLocator, NullLocator))


def _fresh_axes_snapshot() -> _AxesSnapshot:
    """Snapshot the matplotlib default state of an untouched axes.

    Uses ``matplotlib.figure.Figure()`` (not ``plt.subplots()``) so the
    temporary figure does not register with pyplot's global figure manager,
    avoiding Jupyter inline display side effects.
    """
    fig = Figure()
    ax = fig.add_subplot(111)
    try:
        return _read_snapshot(ax, title_pad=None)
    finally:
        plt.close(fig)


def _read_snapshot(ax: Axes, *, title_pad: float | None) -> _AxesSnapshot:
    return _AxesSnapshot(
        x_limits=tuple(float(v) for v in ax.get_xlim()),  # type: ignore[arg-type]
        y_limits=tuple(float(v) for v in ax.get_ylim()),  # type: ignore[arg-type]
        x_ticks=_tick_snapshot(ax, "x"),
        y_ticks=_tick_snapshot(ax, "y"),
        x_tick_style=_tick_style_snapshot(ax, "x"),
        y_tick_style=_tick_style_snapshot(ax, "y"),
        x_label=_label_snapshot(ax, "x"),
        y_label=_label_snapshot(ax, "y"),
        title=_title_snapshot(ax, pad=title_pad),
        x_scale=ax.get_xscale(),
        y_scale=ax.get_yscale(),
        frame=_frame_snapshot(ax),
        legend=ax.get_legend(),
        axis_visibility=bool(ax.axison),
        aspect=ax.get_aspect(),
    )


# ---------------------------------------------------------------------------
# Per-unit comparison helpers
# ---------------------------------------------------------------------------


def _rgba_equal(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return all(
        math.isclose(x, y, rel_tol=_VALUE_REL_TOL, abs_tol=_VALUE_ABS_TOL) for x, y in zip(a, b)
    )


def _limits_equal(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return all(
        math.isclose(x, y, rel_tol=_LIMIT_REL_TOL, abs_tol=_LIMIT_ABS_TOL) for x, y in zip(a, b)
    )


def _tick_locations_equal(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if len(a) != len(b):
        return False
    if not a:
        return True
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=_VALUE_REL_TOL, atol=_VALUE_ABS_TOL))


def _label_equal(a: _LabelSnapshot, b: _LabelSnapshot) -> bool:
    return (
        a.text == b.text
        and a.fontsize == b.fontsize
        and a.fontweight == b.fontweight
        and a.fontstyle == b.fontstyle
        and a.fontfamily == b.fontfamily
        and a.labelpad == b.labelpad
        and _rgba_equal(a.color, b.color)
    )


def _title_equal(a: _TitleSnapshot, b: _TitleSnapshot) -> bool:
    # ``pad`` is intentionally compared (both sides are sourced from the same
    # provenance — the last-applied value — so equality just means "no recorded
    # gerrytools change since last apply"). ``loc`` is similarly best-effort.
    return (
        a.text == b.text
        and a.fontsize == b.fontsize
        and a.fontweight == b.fontweight
        and a.fontstyle == b.fontstyle
        and a.fontfamily == b.fontfamily
        and a.loc == b.loc
        and a.pad == b.pad
        and _rgba_equal(a.color, b.color)
    )


def _tick_equal(a: _TickSnapshot, b: _TickSnapshot) -> bool:
    return (
        _tick_locations_equal(a.locations, b.locations)
        and list(a.labels) == list(b.labels)
        and a.labels_visible == b.labels_visible
    )


def _tick_style_equal(a: _TickStyleSnapshot, b: _TickStyleSnapshot) -> bool:
    return (
        a.label_size == b.label_size
        and math.isclose(a.label_rotation, b.label_rotation, abs_tol=1e-9)
        and a.label_weight == b.label_weight
        and a.label_style == b.label_style
        and a.label_family == b.label_family
        and _rgba_equal(a.label_color, b.label_color)
        and _rgba_equal(a.tick_color, b.tick_color)
    )


def _aspect_equal(a: object, b: object) -> bool:
    """Compare matplotlib aspect values.

    ``ax.get_aspect()`` returns either the string ``"auto"`` or a float
    (``"equal"`` is normalized to ``1.0`` by matplotlib internally). String
    and numeric values are never equal; numeric values use ``math.isclose``.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(
        float(a),  # type: ignore[arg-type]
        float(b),  # type: ignore[arg-type]
        rel_tol=_VALUE_REL_TOL,
        abs_tol=_VALUE_ABS_TOL,
    )


# Dispatch: (unit -> snapshot-field-accessor, equality-fn).
def _snapshot_field(snapshot: _AxesSnapshot, unit: str) -> object:
    return {
        UNIT_X_LIMITS: snapshot.x_limits,
        UNIT_Y_LIMITS: snapshot.y_limits,
        UNIT_X_TICKS: snapshot.x_ticks,
        UNIT_Y_TICKS: snapshot.y_ticks,
        UNIT_X_TICK_STYLE: snapshot.x_tick_style,
        UNIT_Y_TICK_STYLE: snapshot.y_tick_style,
        UNIT_X_LABEL: snapshot.x_label,
        UNIT_Y_LABEL: snapshot.y_label,
        UNIT_TITLE: snapshot.title,
        UNIT_X_SCALE: snapshot.x_scale,
        UNIT_Y_SCALE: snapshot.y_scale,
        UNIT_FRAME: snapshot.frame,
        UNIT_LEGEND: snapshot.legend,
        UNIT_AXIS_VISIBILITY: snapshot.axis_visibility,
        UNIT_ASPECT: snapshot.aspect,
    }[unit]


def _unit_equal(unit: str, a: object, b: object) -> bool:
    if unit in (UNIT_X_LIMITS, UNIT_Y_LIMITS):
        return _limits_equal(a, b)  # type: ignore[arg-type]
    if unit in (UNIT_X_TICKS, UNIT_Y_TICKS):
        return _tick_equal(a, b)  # type: ignore[arg-type]
    if unit in (UNIT_X_TICK_STYLE, UNIT_Y_TICK_STYLE):
        return _tick_style_equal(a, b)  # type: ignore[arg-type]
    if unit in (UNIT_X_LABEL, UNIT_Y_LABEL):
        return _label_equal(a, b)  # type: ignore[arg-type]
    if unit == UNIT_TITLE:
        return _title_equal(a, b)  # type: ignore[arg-type]
    if unit == UNIT_LEGEND:
        # Identity comparison: an external legend placed after last apply is a
        # different object than the one gerrytools tracked.
        return a is b
    if unit == UNIT_ASPECT:
        return _aspect_equal(a, b)
    # Frame, scales, axis_visibility: exact equality.
    return a == b


# ---------------------------------------------------------------------------
# Per-unit ownership record
# ---------------------------------------------------------------------------


@dataclass
class _UnitState:
    ownership: OwnershipState = "unclaimed"
    # last_applied is either a concrete snapshot field value, ``None`` for
    # units whose canonical "no value" is None (e.g. legend), or the
    # ``_NO_LAST_APPLIED`` sentinel for store-and-claim ownership without a
    # recorded value yet.
    last_applied: object = _NO_LAST_APPLIED


# ---------------------------------------------------------------------------
# Public state machine
# ---------------------------------------------------------------------------


class _ManagedAxesState:
    """Per-axes managed-unit ownership and last-applied history.

    One instance lives on each gerrytools plot. Plot configuration (e.g.
    ``self._x_limits``) stays on the plot object; this class only knows about
    "who applied what to which axes most recently" at the matplotlib level.

    See the module docstring for the full contract.
    """

    def __init__(self) -> None:
        self._units: dict[str, _UnitState] = {unit: _UnitState() for unit in _ALL_UNITS}

    # -- initialization & rebind ------------------------------------------------

    def initialize_from_ax(self, ax: Axes) -> None:
        """Classify each unit as externally set or default at bind time.

        Only touches units currently in the ``unclaimed`` state — units already
        reclaimed by explicit plot configuration (e.g. across a
        ``bind_to_ax`` call) are left alone so the explicit gerrytools state
        wins over pre-existing axes state on the new axes.
        """
        default = _fresh_axes_snapshot()

        if self._units[UNIT_X_LIMITS].ownership == "unclaimed":
            if not ax.get_autoscalex_on():
                self._mark_external(UNIT_X_LIMITS, tuple(float(v) for v in ax.get_xlim()))
        if self._units[UNIT_Y_LIMITS].ownership == "unclaimed":
            if not ax.get_autoscaley_on():
                self._mark_external(UNIT_Y_LIMITS, tuple(float(v) for v in ax.get_ylim()))

        label_axis_pairs: tuple[tuple[str, Literal["x", "y"]], ...] = (
            (UNIT_X_LABEL, "x"),
            (UNIT_Y_LABEL, "y"),
        )
        for unit, which in label_axis_pairs:
            if self._units[unit].ownership == "unclaimed":
                current = _label_snapshot(ax, which)
                default_label = default.x_label if which == "x" else default.y_label
                if not _label_equal(current, default_label):
                    self._mark_external(unit, current)

        if self._units[UNIT_TITLE].ownership == "unclaimed":
            current_title = _title_snapshot(ax, pad=None)
            if not _title_equal(current_title, default.title):
                self._mark_external(UNIT_TITLE, current_title)

        for unit, axis_obj in ((UNIT_X_TICKS, ax.xaxis), (UNIT_Y_TICKS, ax.yaxis)):
            if self._units[unit].ownership == "unclaimed":
                if not _is_default_locator(axis_obj):
                    current_ticks = _tick_snapshot(ax, "x" if unit == UNIT_X_TICKS else "y")
                    self._mark_external(unit, current_ticks)

        tick_style_axis_pairs: tuple[tuple[str, Literal["x", "y"]], ...] = (
            (UNIT_X_TICK_STYLE, "x"),
            (UNIT_Y_TICK_STYLE, "y"),
        )
        for unit, which in tick_style_axis_pairs:
            if self._units[unit].ownership == "unclaimed":
                current_style = _tick_style_snapshot(ax, which)
                default_style = default.x_tick_style if which == "x" else default.y_tick_style
                if not _tick_style_equal(current_style, default_style):
                    self._mark_external(unit, current_style)

        if self._units[UNIT_X_SCALE].ownership == "unclaimed":
            if ax.get_xscale() != default.x_scale:
                self._mark_external(UNIT_X_SCALE, ax.get_xscale())
        if self._units[UNIT_Y_SCALE].ownership == "unclaimed":
            if ax.get_yscale() != default.y_scale:
                self._mark_external(UNIT_Y_SCALE, ax.get_yscale())

        if self._units[UNIT_FRAME].ownership == "unclaimed":
            current_frame = _frame_snapshot(ax)
            if current_frame != default.frame:
                self._mark_external(UNIT_FRAME, current_frame)

        if self._units[UNIT_LEGEND].ownership == "unclaimed":
            legend = ax.get_legend()
            if legend is not None:
                self._mark_external(UNIT_LEGEND, legend)

        if self._units[UNIT_AXIS_VISIBILITY].ownership == "unclaimed":
            if not bool(ax.axison):
                self._mark_external(UNIT_AXIS_VISIBILITY, False)

        if self._units[UNIT_ASPECT].ownership == "unclaimed":
            current_aspect = ax.get_aspect()
            if not _aspect_equal(current_aspect, default.aspect):
                self._mark_external(UNIT_ASPECT, current_aspect)

    def reset_history(self) -> None:
        """Clear per-axes last-applied history and external classifications.

        Reclaim flags survive because they describe plot configuration, not
        per-axes history. After ``reset_history``, ``initialize_from_ax``
        classifies only currently-unclaimed units against the new axes.
        """
        for unit, state in self._units.items():
            if state.ownership == "external":
                state.ownership = "unclaimed"
                state.last_applied = _NO_LAST_APPLIED
            elif state.ownership == "gerrytools_default":
                state.ownership = "unclaimed"
                state.last_applied = _NO_LAST_APPLIED
            else:
                # gerrytools_explicit: keep ownership, drop the per-axes value.
                state.last_applied = _NO_LAST_APPLIED

    # -- snapshot & external detection ------------------------------------------

    def snapshot(self, ax: Axes) -> _AxesSnapshot:
        """Read all observable axes-level state in one pass.

        ``title.pad`` is sourced from the last-applied value, not from
        matplotlib, because matplotlib 3.10.6 exposes no public getter.
        """
        last_title = self._units[UNIT_TITLE].last_applied
        title_pad = None
        if isinstance(last_title, _TitleSnapshot):
            title_pad = last_title.pad
        return _read_snapshot(ax, title_pad=title_pad)

    def detect_external_changes(self, snapshot: _AxesSnapshot) -> set[str]:
        """Return units that should be treated as externally owned this rebuild.

        Per-unit dispatch by current ownership state:

        - ``external``: kept in the returned set without value comparison.
          Once a unit yields, it stays external until a gerrytools API
          explicitly reclaims it.
        - ``gerrytools_explicit`` or ``gerrytools_default`` with a recorded
          last-applied value: compare snapshot to last-applied using the
          per-unit equality rules; if they differ, mark external and
          transition ownership so subsequent rebuilds keep it external.
        - ``gerrytools_explicit`` with no recorded value (store-and-claim
          ran but the apply hasn't happened yet): never added.
        - ``unclaimed`` (no last-applied history yet): never added; the
          next apply records the resulting ownership.
        """
        external: set[str] = set()
        for unit, state in self._units.items():
            if state.ownership == "external":
                external.add(unit)
                continue
            if state.ownership == "unclaimed":
                continue
            if state.last_applied is _NO_LAST_APPLIED:
                # gerrytools_explicit with no recorded value yet — store-and-claim.
                continue
            current = _snapshot_field(snapshot, unit)
            if not _unit_equal(unit, current, state.last_applied):
                external.add(unit)
                state.ownership = "external"
                state.last_applied = current
        return external

    # -- write paths ------------------------------------------------------------

    def reclaim_and_mark(self, unit: str, value: object) -> None:
        """Apply-now public setter path. Claim ownership and record value."""
        state = self._units[unit]
        state.ownership = "gerrytools_explicit"
        state.last_applied = value

    def reclaim_without_value(self, unit: str) -> None:
        """Store-and-claim public setter path. Claim without recording yet."""
        state = self._units[unit]
        state.ownership = "gerrytools_explicit"
        state.last_applied = _NO_LAST_APPLIED

    def record_default(self, unit: str, value: object) -> None:
        """Internal default applier path. Record value without claiming."""
        state = self._units[unit]
        # An already-explicit unit stays explicit; defaults never demote.
        if state.ownership != "gerrytools_explicit":
            state.ownership = "gerrytools_default"
        state.last_applied = value

    # -- queries -----------------------------------------------------------------

    def is_reclaimed(self, unit: str) -> bool:
        """True iff the unit is currently ``gerrytools_explicit``.

        Default-owned and externally-owned units return False. Used by apply
        helpers that need to distinguish "gerrytools owns this and may remove
        external content" from "gerrytools has no claim."
        """
        return self._units[unit].ownership == "gerrytools_explicit"

    def last_applied(self, unit: str) -> object:
        """Return the last-applied value or ``_NO_LAST_APPLIED``.

        Mostly useful for debugging and for the title-pad path that needs to
        round-trip the only-stored-locally value.
        """
        return self._units[unit].last_applied

    # -- restore -----------------------------------------------------------------

    def restore_autoscale_protected(
        self,
        ax: Axes,
        pre_redraw: _AxesSnapshot,
        external_units: set[str],
    ) -> None:
        """Restore externally-set xlim/ylim that artist drawing may have clobbered.

        Apply-or-skip units never need this; matplotlib's autoscale only
        affects limits during artist drawing.
        """
        if UNIT_X_LIMITS in external_units and UNIT_X_LIMITS in _AUTOSCALE_PROTECTED:
            ax.set_xlim(*pre_redraw.x_limits)
        if UNIT_Y_LIMITS in external_units and UNIT_Y_LIMITS in _AUTOSCALE_PROTECTED:
            ax.set_ylim(*pre_redraw.y_limits)

    # -- internal ---------------------------------------------------------------

    def _mark_external(self, unit: str, value: object) -> None:
        state = self._units[unit]
        state.ownership = "external"
        state.last_applied = value


# Re-export for the autoscale-protected guard callers may want to consult.
def is_autoscale_protected(unit: str) -> bool:
    return unit in _AUTOSCALE_PROTECTED


def all_units() -> Iterable[str]:
    return iter(_ALL_UNITS)
