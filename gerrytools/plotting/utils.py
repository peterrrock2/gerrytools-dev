from collections.abc import Iterable
from numbers import Real

from gerrytools.typing import Numeric, NumericIterable


def _coerce_real_iter(values: Numeric | NumericIterable, *, field: str) -> list[float]:
    """Normalize scalar/iterable numeric input to a list of floats.

    Args:
        values (Numeric | NumericIterable): Scalar real value or iterable of real values.
        field (str): Field name used in validation error messages.

    Returns:
        list[float]: Parsed numeric values as Python floats.

    Raises:
        TypeError: If ``values`` is not numeric or contains non-numeric entries.
    """
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


def sort_elections(elec_list: Iterable[str]) -> list[str]:
    """Sort election identifiers by two-digit year suffix.

    Assumes each identifier ends with a two-character year token such as ``"SEN18"``.

    Args:
        elec_list (Iterable[str]): Election identifier strings to sort.

    Returns:
        list[str]: Election identifiers sorted by year suffix.
    """
    tuplified_elecs = list(map(lambda x: (x[:-2], x[-2:]), sorted(elec_list)))
    sorted_tuples = sorted(tuplified_elecs, key=lambda x: x[1])
    return [tup[0] + tup[1] for tup in sorted_tuples]
