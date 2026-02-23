"""Shared geometry helpers for matplotlib plotting."""

from __future__ import annotations

import math


def line_segment_through_unit_square(
    slope: float,
) -> tuple[float, float, float, float]:
    """Compute line endpoints inside the unit square for a line through ``(0.5, 0.5)``.

    Args:
        slope (float): Slope of the line.

    Returns:
        tuple[float, float, float, float]: Endpoints in ``(x0, y0, x1, y1)`` order.
    """
    if slope == 0:
        starting_x = 0.0
        ending_x = 1.0
        starting_y = 0.5
        ending_y = 0.5
    elif math.isinf(slope):
        starting_x = 0.5
        ending_x = 0.5
        starting_y = 0.0
        ending_y = 1.0
    elif slope >= 1:
        starting_x = 0.5 - (0.5 / slope)
        starting_y = 0.0
        ending_x = 0.5 + (0.5 / slope)
        ending_y = 1.0
    elif -1 < slope < 1:
        starting_x = 0.0
        starting_y = 0.5 - (0.5 * slope)
        ending_x = 1.0
        ending_y = 0.5 + (0.5 * slope)
    else:
        starting_x = 0.5 - (0.5 / slope)
        starting_y = 0.0
        ending_x = 0.5 + (0.5 / slope)
        ending_y = 1.0

    return (starting_x, starting_y, ending_x, ending_y)
