"""Low-level partisan vote arithmetic shared across gerrytools backends.

This module exists so the plotting and LaTeX helpers can share scoring arithmetic without
importing the scoring package, which eagerly loads the plan evaluator and its graph, geometry,
and scoring-engine dependencies. It is pure NumPy. The public scoring API for these formulas is
:mod:`gerrytools.scoring.formulas`, which documents them fully and delegates its core arithmetic
here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _nonnegative(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert an array-like input to finite, nonnegative ``float64`` values.

    The input must have at least one dimension and a nonempty final axis. The returned array keeps
    the input shape. ``name`` is used only to identify the input in validation errors.

    Args:
        values: Values to convert and validate.
        name: Input name to include in error messages.

    Returns:
        A ``float64`` NumPy array with the same shape as ``values``.

    Raises:
        ValueError: If the input is scalar, has an empty final axis, contains a nonfinite value, or
            contains a negative value.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty district axis")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative values")
    return array


def _matching(
    first: ArrayLike,
    first_name: str,
    second: ArrayLike,
    second_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate two nonnegative inputs and require identical shapes.

    Args:
        first: First array-like input.
        first_name: Name used for the first input in validation errors.
        second: Second array-like input.
        second_name: Name used for the second input in validation errors.

    Returns:
        The two inputs as finite, nonnegative ``float64`` arrays.

    Raises:
        ValueError: If either input fails :func:`_nonnegative` validation or their shapes differ.
    """
    first_array = _nonnegative(first, first_name)
    second_array = _nonnegative(second, second_name)
    if first_array.shape != second_array.shape:
        raise ValueError(f"{first_name} and {second_name} must have the same shape")
    return first_array, second_array


def _paired(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate matching party and opposition vote arrays.

    Args:
        party_votes: Nonnegative party vote tallies.
        opposition_votes: Nonnegative opposition tallies with the same shape as ``party_votes``.

    Returns:
        Party and opposition tallies as finite, nonnegative ``float64`` arrays.

    Raises:
        ValueError: If either input fails :func:`_nonnegative` validation or their shapes differ.
    """
    return _matching(party_votes, "party_votes", opposition_votes, "opposition_votes")


def _divide(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Divide broadcast-compatible arrays where the denominator is positive.

    Locations with a zero or negative denominator are returned as ``NaN``. This helper does not
    otherwise validate either array; callers are responsible for rejecting inappropriate inputs.

    Args:
        numerator: Values to divide.
        denominator: Divisors broadcast-compatible with ``numerator``.

    Returns:
        A ``float64`` array with the broadcast shape of both inputs.

    Raises:
        ValueError: If the input shapes cannot be broadcast together.
    """
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def district_vote_shares(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64]:
    """Return the party's two-party share in each district, ``NaN`` where turnout is zero."""
    party, opposition = _paired(party_votes, opposition_votes)
    return _divide(party, party + opposition)


def district_wins(party_votes: ArrayLike, opposition_votes: ArrayLike) -> NDArray[np.bool_]:
    """Identify districts strictly won by the party; ties and zero turnout are not wins."""
    party, opposition = _paired(party_votes, opposition_votes)
    return party > opposition


def swing_breakpoints(shares: ArrayLike, reference: ArrayLike) -> NDArray[np.float64]:
    """Return uniform-swing seats-votes breakpoints ``clip(reference - shares + 0.5, 0, 1)``.

    ``reference`` is the anchoring statewide share: a scalar, or an array broadcastable against
    ``shares`` (e.g. shape ``(..., 1)`` for batched rows). Inputs are not validated; callers own
    share validation.
    """
    shares_array = np.asarray(shares, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return np.clip(reference_array - shares_array + 0.5, 0.0, 1.0)


def overall_vote_share(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    """Return the party's turnout-weighted aggregate two-party share along the district axis."""
    party, opposition = _paired(party_votes, opposition_votes)
    return _divide(np.sum(party, axis=-1), np.sum(party + opposition, axis=-1))
