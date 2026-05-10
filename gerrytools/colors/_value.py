"""Internal canonical color value for the colors module.

This module is private (`_value`). End users never see `_Color`; the public
surface in `core.py` is unchanged. The point of this file is to collect every
"what counts as a valid color, and how do we normalize it" decision into one
place — replacing the polyglot conversion logic and the five type guards that
previously lived inside ``convert_color_to_hexa_or_none``.

Named-color resolution itself lives in ``_sources.py`` (the registry); this
module just calls into it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias, TypeGuard, cast

import matplotlib.colors as mcolors

from gerrytools.colors._sources import _resolve_named_color
from gerrytools.colors.latex import get_color_from_latex_string
from gerrytools.typing import Color, HexColor, MplBaseColor, MplCompatibleColor, MplRGBAColor

_MplBaseColorWithAlpha: TypeAlias = tuple[MplBaseColor, Real]


# ---------------------------------------------------------------------------
# Private type guards. Used only by _Color.from_any.
# ---------------------------------------------------------------------------


def _is_real(x: object) -> TypeGuard[Real]:
    return (
        isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(float(x))
    )  # filters NaN too


def _is_rgb_tuple(x: object) -> TypeGuard[tuple[int | float, int | float, int | float]]:
    return isinstance(x, tuple) and len(x) == 3 and all(_is_real(v) for v in x)


def _is_rgba_tuple(
    x: object,
) -> TypeGuard[tuple[int | float, int | float, int | float, int | float]]:
    return isinstance(x, tuple) and len(x) == 4 and all(_is_real(v) for v in x)


def _is_mpl_base_color(x: object) -> TypeGuard[MplBaseColor]:
    return isinstance(x, str) or _is_rgb_tuple(x) or _is_rgba_tuple(x)


def _is_mpl_base_color_with_alpha(x: object) -> TypeGuard[_MplBaseColorWithAlpha]:
    return isinstance(x, tuple) and len(x) == 2 and _is_mpl_base_color(x[0]) and _is_real(x[1])


def _validate_alpha(alpha: float, *, field: str = "alpha") -> float:
    """Validate that alpha is convertible to a float and between 0.0 and 1.0."""
    a = float(alpha)
    if not math.isfinite(a):
        raise ValueError(f"{field} must be finite")
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"{field} must be between 0.0 and 1.0")
    return a


# ---------------------------------------------------------------------------
# The canonical internal color value.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Color:
    """Internal canonical color value.

    ``hex6`` is a lowercased ``"#rrggbb"`` string, or the literal ``"none"`` for
    the transparent sentinel returned for ``None`` / ``"none"`` input. ``alpha``
    is in ``[0.0, 1.0]`` and is forced to ``0.0`` for the ``"none"`` sentinel.
    """

    hex6: str
    alpha: float = 1.0

    @property
    def is_none(self) -> bool:
        return self.hex6 == "none"

    def to_hex8(self) -> HexColor:
        if self.is_none:
            return "none"
        alpha_byte = max(0, min(255, int(round(self.alpha * 255))))
        return f"{self.hex6}{alpha_byte:02x}"

    @classmethod
    def none(cls) -> "_Color":
        return cls(hex6="none", alpha=0.0)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_any(
        cls,
        color: MplCompatibleColor | None,
        *,
        logger: logging.Logger | None = None,
    ) -> "_Color":
        """Build a `_Color` from any supported polyglot input.

        Returns the `_Color.none()` sentinel for `None` or the case-insensitive
        string ``"none"``. Raises ``ValueError`` for anything that isn't a
        recognizable color shape; in that case, individual parser-failure
        diagnostics are emitted at DEBUG to ``logger`` if one is supplied.
        """
        if color is None or (isinstance(color, str) and color.lower() == "none"):
            return cls.none()

        if isinstance(color, str):
            return cls._from_string(color, logger=logger)

        if _is_mpl_base_color_with_alpha(color):
            return cls._from_base_with_alpha(color)

        if _is_rgba_tuple(color):
            return cls._from_rgba_tuple(color)

        if _is_rgb_tuple(color):
            return cls._from_rgb_tuple(color)

        raise ValueError(f"Unknown color value: {color!r}")

    # -- private constructors per shape -------------------------------------

    @classmethod
    def _from_string(cls, color_string: str, *, logger: logging.Logger | None) -> "_Color":
        diagnostic_message = ""

        try:
            return cls._from_resolved(_resolve_named_color(color_string))
        except KeyError as named_color_error:
            diagnostic_message += (
                f"Color {color_string!r} is not a known Matplotlib named color string: "
                f"{named_color_error}"
            )

        try:
            return cls._from_resolved(get_color_from_latex_string(color_string))
        except Exception as latex_parse_error:
            diagnostic_message += (
                f" | Color {color_string!r} is not parsable as a LaTeX color string: "
                f"{latex_parse_error}"
            )

        try:
            return cls._from_resolved(mcolors.to_rgba(color_string))
        except Exception as matplotlib_parse_error:
            diagnostic_message += (
                f" | Color {color_string!r} not parseable by Matplotlib: "
                f"{matplotlib_parse_error}"
            )

        if logger is not None:
            logger.debug(diagnostic_message)
        raise ValueError(f"Unknown color value: {color_string!r}")

    @classmethod
    def _from_resolved(cls, resolved_value: Color | MplRGBAColor) -> "_Color":
        """Normalize an already-resolved hex string or RGBA tuple into ``_Color``."""
        hex8_string = mcolors.to_hex(mcolors.to_rgba(resolved_value), keep_alpha=True)
        normalized_hex6 = hex8_string[:7].lower()
        normalized_alpha = int(hex8_string[7:], 16) / 255.0
        return cls(hex6=normalized_hex6, alpha=normalized_alpha)

    @classmethod
    def _from_base_with_alpha(cls, base_alpha_pair: tuple[object, object]) -> "_Color":
        base_color_input, requested_alpha = base_alpha_pair
        validated_alpha = _validate_alpha(
            float(cast(float, requested_alpha)), field="alpha in (base, alpha) color tuple"
        )
        base_color = cls.from_any(cast(MplCompatibleColor, base_color_input))
        if base_color.is_none:
            return cls(hex6="#000000", alpha=0.0)
        rgba_with_override = mcolors.to_rgba(base_color.hex6, alpha=validated_alpha)
        return cls._from_resolved(rgba_with_override)

    @staticmethod
    def _normalize_rgb_components(
        red: float, green: float, blue: float, *, original_input: object
    ) -> tuple[float, float, float]:
        """Normalize RGB to [0,1], detecting 0–255 scale and ambiguity. Pure helper."""
        max_component = max(red, green, blue)
        if max_component > 1.0:
            if any(component < 0.0 for component in (red, green, blue)):
                raise ValueError(f"RGB values must be non-negative: {original_input!r}")
            if max_component > 255.0:
                raise ValueError(
                    f"RGB values must be <=255 when using 0-255 scale: {original_input!r}"
                )
            if max_component < 2.0:
                raise ValueError(
                    f"Ambiguous RGB tuple {original_input!r}: values >1 but <2; "
                    "use 0–1 floats or 0–255 ints."
                )
            red, green, blue = red / 255.0, green / 255.0, blue / 255.0
        return red, green, blue

    @classmethod
    def _from_rgba_tuple(
        cls, rgba_tuple: tuple[int | float, int | float, int | float, int | float]
    ) -> "_Color":
        red, green, blue, raw_alpha = (
            float(rgba_tuple[0]),
            float(rgba_tuple[1]),
            float(rgba_tuple[2]),
            float(rgba_tuple[3]),
        )

        if raw_alpha < 0:
            raise ValueError(f"Alpha must be non-negative: {rgba_tuple!r}")

        red, green, blue = cls._normalize_rgb_components(
            red, green, blue, original_input=rgba_tuple
        )

        if raw_alpha > 1.0:
            if raw_alpha > 255.0:
                raise ValueError(f"Alpha must be <=255 when using 0-255 scale: {rgba_tuple!r}")
            normalized_alpha = raw_alpha / 255.0
        else:
            normalized_alpha = raw_alpha

        return cls._from_resolved((red, green, blue, normalized_alpha))

    @classmethod
    def _from_rgb_tuple(cls, rgb_tuple: tuple[int | float, int | float, int | float]) -> "_Color":
        red, green, blue = float(rgb_tuple[0]), float(rgb_tuple[1]), float(rgb_tuple[2])
        red, green, blue = cls._normalize_rgb_components(red, green, blue, original_input=rgb_tuple)
        return cls._from_resolved((red, green, blue, 1.0))
