import logging
import math

from gerrytools.colors._sources import (
    CITIZEN_BLUE,
    COLOR_CORRECTED_BASESET,
    DEFAULT_GREY,
    ENSEMBLE_COLORS,
    GERRYTOOLS_EXTRA_COLORS_DICT,
    OVERLAYS,
    _resolve_named_color,
    _which_color_source,
    get_all_supported_colors_dict,
)
from gerrytools.colors._value import _Color, _validate_alpha
from gerrytools.logging import get_logger
from gerrytools.typing import Color, HexColor, MplCompatibleColor, ResolvedColor

__all__ = [
    "CITIZEN_BLUE",
    "COLOR_CORRECTED_BASESET",
    "DEFAULT_GREY",
    "ENSEMBLE_COLORS",
    "GERRYTOOLS_EXTRA_COLORS_DICT",
    "OVERLAYS",
    "convert_color_to_hexa_or_none",
    "get_all_supported_colors_dict",
    "get_named_color",
    "resolve_color_and_alpha",
    "which_color_source",
]

gt_logger = get_logger(__name__)


def get_named_color(name: str) -> Color:
    """Get a color value from the supported color names.

    Args:
        name (str): The name of the color.

    Returns:
        Color: The corresponding color value.

    Raises:
        KeyError: If the color name is not recognized.
    """
    return _resolve_named_color(name)


def which_color_source(name: str) -> str:
    """Return the name of the registry source that owns ``name``.

    Useful for diagnosing precedence: when two palettes both define a name,
    this answers which one the resolver actually returns. Source names
    currently include ``"overrides"``, ``"gerrytools"``, ``"color-corrected"``,
    ``"districtr"``, ``"latex"``, and ``"matplotlib"``.

    Args:
        name (str): The name of the color.

    Returns:
        str: The name of the source that resolves the color name.

    Raises:
        KeyError: If the color name is not recognized by any source.
    """
    return _which_color_source(name)


def convert_color_to_hexa_or_none(color: MplCompatibleColor | None) -> HexColor:
    """Convert a color input to a hex8 string or "none".

    Args:
        color (MplCompatibleColor | None): The color input to convert. This can be a named color,
            a LaTeX color string, an RGB(A) tuple, or other formats supported
            by Matplotlib.
    """
    return _Color.from_any(color, logger=gt_logger).to_hex8()


def resolve_color_and_alpha(
    color: MplCompatibleColor | None,
    alpha: float | None = None,
    *,
    allow_none: bool = True,
    field: str = "color",
    owner: str | None = None,
    logger: logging.Logger | None = None,
) -> ResolvedColor:
    """Normalize a (color, alpha) pair into (hex6_or_none, resolved_alpha).

    Rules:
    - color is converted via convert_color_to_hexa_or_none -> "none" or "#RRGGBBAA"
    - if color resolves to "none":
        - allow_none=True  => ("none", 0.0)  (alpha is forced to 0)
        - allow_none=False => ValueError
    - if color resolves to "#RRGGBBAA":
        - returns ("#RRGGBB", alpha_from_color) if alpha is None
        - returns ("#RRGGBB", alpha) if alpha is provided (validated),
          optionally logs when it overrides embedded alpha.


    Args:
        color (MplCompatibleColor | None): The color input to convert.
        alpha (float | None): An optional explicit alpha value between 0.0 and 1.0.
        allow_none (bool): Whether "none" is an acceptable color. Defaults to True.
        field (str): The name of the field being processed, for error messages.
        owner (str | None): An optional owner name for logging context.
        logger (logging.Logger | None): An optional logger for debug messages.

    Returns:
        ResolvedColor: A tuple of (hex6_or_none, resolved_alpha).
    """
    resolved_color = _Color.from_any(color, logger=gt_logger)

    if resolved_color.is_none:
        if not allow_none:
            raise ValueError(f"{field} cannot be 'none'.")
        return "none", 0.0

    if alpha is None:
        return resolved_color.hex6, resolved_color.alpha

    validated_alpha = _validate_alpha(alpha, field=f"{field} alpha")

    explicit_alpha_overrides_embedded = not math.isclose(
        validated_alpha, resolved_color.alpha, abs_tol=1e-4
    )
    if logger is not None and explicit_alpha_overrides_embedded:
        owner_prefix = f"For {owner}: " if owner else ""
        logger.log(
            level=logging.DEBUG,
            msg=(
                f"{owner_prefix}Ignoring alpha embedded in {field} "
                f"{resolved_color.to_hex8()!r} because explicit alpha "
                f"{validated_alpha} was provided."
            ),
        )

    return resolved_color.hex6, validated_alpha
