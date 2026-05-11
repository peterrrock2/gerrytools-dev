"""Shared test helpers for the PaintBall test suite.

Replaces per-file ``_simple_paintball`` helpers that previously lived
duplicated across five test modules.
"""

from __future__ import annotations

from collections.abc import Iterable

from gerrytools.plotting.data.paintball import PaintBall


def simple_paintball(
    voteshare_data: Iterable[float] | None = None,
    seats_data: Iterable[float] | None = None,
    *,
    add_efficiency_gap_line: bool = True,
    add_proportionality_line: bool = True,
    **constructor_kwargs,
) -> PaintBall:
    """Build a ``PaintBall`` with sensible defaults for tests.

    Args:
        voteshare_data: Override the default vote-share values
            (``[0.4, 0.5, 0.6]`` if omitted).
        seats_data: Override the default seat-share values
            (``[0.3, 0.5, 0.7]`` if omitted).
        add_efficiency_gap_line: Whether to add the efficiency-gap guide
            line after construction. Defaults to True.
        add_proportionality_line: Whether to add the proportionality guide
            line after construction. Defaults to True.
        **constructor_kwargs: Forwarded to ``PaintBall(...)``.

    Returns:
        A configured ``PaintBall`` instance ready for further test setup.
    """
    if voteshare_data is None:
        voteshare_data = [0.4, 0.5, 0.6]
    if seats_data is None:
        seats_data = [0.3, 0.5, 0.7]
    plot = PaintBall(**constructor_kwargs)
    plot.add_voteshare_seatshare_data(voteshare_data, seats_data)
    if add_efficiency_gap_line:
        plot.add_efficiency_gap_line()
    if add_proportionality_line:
        plot.add_proportionality_line()
    return plot
