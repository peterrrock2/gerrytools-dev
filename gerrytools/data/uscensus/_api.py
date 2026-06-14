"""Shared HTTP/query plumbing for the Census API (ACS and decennial PL).

This module holds the pieces common to both the ACS fetchers (``acs.py``) and the decennial PL
fetchers (``census.py``, ``block_cvap.py``): the request timeout, base-URL templates, the rate-limit
error, geography-query builders, API-key handling, the JSON-to-DataFrame adapter, and the GEO_ID
stub stripper. Keeping it here means ``census.py`` no longer imports private helpers out of
``acs.py``.
"""

import logging
import os
from typing import cast

import httpx
import pandas as pd
import us

# _api is the census package's shared internal plumbing: every name below is imported by the
# sibling modules (acs.py, census.py, block_cvap.py). Declaring them in __all__ marks them as
# this module's exports so pyright stops complaining about unused imports.
__all__ = [
    "ACS_BASE_URL",
    "CensusRateLimitError",
    "PL_BASE_URL",
    "REQUEST_TIMEOUT",
    "TRACE",
    "_add_census_api_key",
    "_construct_in_query",
    "_response_to_frame",
    "_strip_geoid_prefix",
    "_validate_year",
]

# Custom level for high-volume per-request log lines (below DEBUG).
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")

# Shared request timeout for every Census API call.
REQUEST_TIMEOUT = httpx.Timeout(120)

# Base-URL templates. The decennial PL endpoint takes only a year; the ACS endpoint additionally
# takes the survey ("acs1" or "acs5").
PL_BASE_URL = "https://api.census.gov/data/{year}/dec/pl"
ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/{survey}"


class CensusRateLimitError(RuntimeError):
    """Raised when the Census API returns HTTP 429 Too Many Requests.

    As of 12 May 2026, an API key is required for all requests made to the Census API. The exception
    message includes the offending URL and the ``Retry-After`` header (when present), plus a pointer
    to ``https://api.census.gov/data/key_signup.html`` for obtaining an API key.
    """


def _validate_year(year: int) -> None:
    """Validate that ``year`` is a plausible 4-digit Census data year.

    Args:
        year (int): Candidate Census data year.

    Raises:
        ValueError: If ``year`` is not an integer between 2000 and 2050 inclusive.
    """

    if not isinstance(year, int) or not (2000 <= year <= 2050):
        raise ValueError(f"Year must be a 4-digit integer in [2000, 2050]. Received {year}.")


def _construct_in_query(
    curr_query: dict,
    state: us.states.State,
    geometry: str,
    county_fips: str | None = None,
) -> None:
    """Set the Census API ``for`` / ``in`` query parameters for a geography.

    Args:
        curr_query (dict): Query parameter dictionary to mutate in place.
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``,
            ``"block group"``, or ``"block"``.
        county_fips (str | None): Required when ``geometry`` is ``"block"``; scopes the block query
            to a single county (the Census API does not support ``county:*`` for blocks).

    Raises:
        ValueError: If ``geometry`` is ``"block"`` without ``county_fips``, or ``geometry`` is not a
            recognized geometry level.

    Warning:
        Modifies ``curr_query`` in place.
    """

    if geometry == "state":
        curr_query["for"] = f"state:{state.fips}"
        return

    if geometry == "county":
        curr_query["in"] = f"state:{state.fips}"
        return

    if geometry == "tract":
        curr_query["in"] = [f"state:{state.fips}", "county:*"]
        return

    if geometry == "block group":
        curr_query["in"] = [f"state:{state.fips}", "county:*", "tract:*"]
        return

    if geometry == "block":
        if county_fips is None:
            raise ValueError("Block-level queries must be scoped to a county.")
        curr_query["in"] = [
            f"state:{state.fips}",
            f"county:{county_fips}",
            "tract:*",
        ]
        return

    raise ValueError(
        "Invalid geometry level. Must be one of 'state', 'county', 'tract', "
        "'block group', or 'block'."
    )


def _add_census_api_key(params: dict, api_key: str | None = None) -> None:
    """Add a Census API key to query parameters.

    Resolution order:

    1. ``api_key`` argument, when not ``None``.
    2. ``CENSUS_API_KEY`` environment variable, when set and non-empty.

    As of 12 May 2026 the Census API requires a key for all requests;
    unauthenticated calls 429 fast.

    Args:
        params (dict): Query parameter dictionary to mutate in place. The resolved key is written
            under the ``"key"`` query-string name expected by the Census API.
        api_key (str | None): Explicit Census API key. When ``None``, the environment variable
            ``CENSUS_API_KEY`` is consulted as a fallback.

    Raises:
        ValueError: If neither source yields a key.

    Warning:
        Modifies ``params`` in place.
    """

    if api_key is not None:
        params["key"] = api_key
        return

    if key := os.getenv("CENSUS_API_KEY"):
        params["key"] = key
        return

    raise ValueError(
        "No Census API key provided. Supply one via the api_key argument "
        "or the CENSUS_API_KEY environment variable. Register at "
        "https://api.census.gov/data/key_signup.html."
    )


def _response_to_frame(response: httpx.Response) -> pd.DataFrame:
    """Convert a Census API list-of-lists JSON response into a DataFrame.

    Args:
        response (httpx.Response): Response object from a Census API call.

    Returns:
        pd.DataFrame: DataFrame where the first row of the JSON payload is used as column names and
        the remaining rows become data rows.

    Raises:
        CensusRateLimitError: When the Census API returns 429 Too Many Requests.
        httpx.HTTPStatusError: Propagated from ``response.raise_for_status()`` for other HTTP
            errors.
    """

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise CensusRateLimitError(
            "Census API returned 429 Too Many Requests for "
            f"{response.request.url}. "
            + (f"Retry-After: {retry_after}s. " if retry_after else "")
            + "Unauthenticated requests are capped at 500/day per IP; "
            "register a key at https://api.census.gov/data/key_signup.html "
            "and pass it via api_key=/CENSUS_API_KEY to lift the cap."
        )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError(
            "Unexpected Census API response (expected a non-empty list of rows): "
            f"{repr(data)[:200]}"
        )
    return pd.DataFrame(data[1:], columns=data[0])


def _strip_geoid_prefix(frame: pd.DataFrame, column: str = "GEO_ID") -> pd.Series:
    """Strip the Census summary-level stub from a frame's ``GEO_ID`` column.

    Census ``GEO_ID`` values carry a summary-level prefix ending in ``US`` (e.g. ``1000000US55`` for
    a state or ``0500000US55001`` for a county); the substring after ``US`` is the bare GEOID used
    for downstream joins.

    Args:
        frame (pd.DataFrame): Frame holding the raw ``GEO_ID`` column.
        column (str): Name of the raw GEO_ID column. Defaults to ``"GEO_ID"``.

    Returns:
        pd.Series: The GEOID substring following the ``US`` stub.
    """

    geo_id = cast(pd.Series, frame[column])
    return geo_id.astype("string").str.split("US").str[-1]
