from collections.abc import Iterable
from numbers import Real
from typing import Any


def _coerce_real_iter(values: Any, *, field: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a number or an iterable of numbers, not a string.")
    # exclude bool (since bool is a Real)
    if isinstance(values, Real) and not isinstance(values, bool):
        return [float(values)]
    if isinstance(values, Iterable):
        out: list[float] = []
        for v in values:
            if isinstance(v, bool) or not isinstance(v, Real):
                raise TypeError(
                    f"All items in {field} must be real numbers; got {type(v).__name__}."
                )
            out.append(float(v))
        return out
    raise TypeError(f"{field} must be a number or an iterable of numbers.")


def sort_elections(elec_list):
    """
    Helper function to sort elections chronologically for plotting. Assumes the last two characters
    in the election name are the year, e.g. "SEN18"
    """
    tuplified_elecs = list(map(lambda x: (x[:-2], x[-2:]), sorted(elec_list)))
    sorted_tuples = sorted(tuplified_elecs, key=lambda x: x[1])
    return [tup[0] + tup[1] for tup in sorted_tuples]
