import logging
import os
import warnings
from typing import Literal

import httpx
import pandas as pd
import us
from census_tables import (
    ACS_SOURCE_SUFFIX,
    RACE_PREFIXES,
    ACSTableInfo,
    CVAPTableInfo,
    DecennialPLTableInfo,
    PLBlockVAPTableInfo,
    TotPopTableInfo,
    VAPTableInfo,
    append_source_suffix,
    shorten_acs5_column_names,
)

from gerrytools.logging import get_logger

LOGGER = get_logger(__name__)

ACSSurvey = Literal["acs1", "acs5"]

PL_BASE_URL = "https://api.census.gov/data/{year}/dec/pl"

DEFAULT_BLOCK_CVAP_ACS_YEAR = 2024
DEFAULT_BLOCK_CVAP_PL_YEAR = 2020
DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD = 20


class CensusRateLimitError(RuntimeError):
    """
    Raised when the Census API returns HTTP 429 Too Many Requests.

    As of 12 May 2026, an API key is required for all requests made to the
    Census API. The exception message includes the offending URL and the
    ``Retry-After`` header (when present), plus a pointer to
    ``https://api.census.gov/data/key_signup.html`` for obtaining an API key.
    """


def _validate_year(year: int) -> None:
    """
    Validates that ``year`` is a plausible 4-digit Census data year.

    Parameters
    ----------
    year : int
        Candidate Census data year.

    Raises
    ------
    ValueError
        If ``year`` is not an integer between 2000 and 2050 inclusive.
    """

    if not isinstance(year, int) or not (2000 <= year <= 2050):
        raise ValueError(f"Year must be a 4-digit integer in [2000, 2050]. Received {year}.")


def _normalize_acs_survey(survey: str | int) -> ACSSurvey:
    """
    Normalizes common ACS survey-period inputs to the Census API dataset name.

    Parameters
    ----------
    survey : str | int
        The ACS survey period to query. Accepts "acs5", "acs1", 5, or 1
        (case- and separator-insensitive).

    Returns
    -------
    ACSSurvey
        The normalized ACS survey period, either "acs1" or "acs5".

    Raises
    ------
    ValueError
        If ``survey`` is neither a recognized string nor 1/5.
    """

    if isinstance(survey, int):
        survey = str(survey)
    elif not isinstance(survey, str):
        raise ValueError("Invalid ACS survey. Must be one of 'acs1', 'acs5', 1, or 5.")

    normalized_survey = survey.lower().replace("_", "").replace("-", "").replace(" ", "")

    if normalized_survey in {"1", "1year", "acs1", "acs1year"}:
        return "acs1"

    if normalized_survey in {"5", "5year", "acs5", "acs5year"}:
        return "acs5"

    raise ValueError("Invalid ACS survey. Must be one of 'acs1', 'acs5', 1, or 5.")


def _default_acs_tables() -> list[ACSTableInfo]:
    """
    Builds the default set of ACS table definitions.

    Returns
    -------
    list[ACSTableInfo]
        Table definitions for total population, voting-age population, and
        citizen voting-age population.
    """

    return [TotPopTableInfo(), VAPTableInfo(), CVAPTableInfo()]


def _construct_in_query(
    curr_query: dict,
    state: us.states.State,
    level: str,
    county_fips: str | None = None,
) -> None:
    """
    Sets the Census API ``for`` / ``in`` query parameters for a given geography.

    Parameters
    ----------
    curr_query : dict
        Query parameter dictionary to mutate.
    state : us.states.State
        State the query is scoped to.
    level : str
        Geometry level. Must be one of "state", "county", "tract",
        "block group", or "block".
    county_fips : str, optional
        Required when ``level`` is "block"; scopes the block query to a single
        county (the Census API does not support ``county:*`` for blocks).

    Raises
    ------
    ValueError
        If ``level`` is "block" without ``county_fips``, or ``level`` is not a
        recognized geometry level.

    Warnings
    --------
    This function modifies ``curr_query`` in place.
    """

    if level == "state":
        curr_query["for"] = f"state:{state.fips}"
        return

    if level == "county":
        curr_query["in"] = f"state:{state.fips}"
        return

    if level == "tract":
        curr_query["in"] = [f"state:{state.fips}", "county:*"]
        return

    if level == "block group":
        curr_query["in"] = [f"state:{state.fips}", "county:*", "tract:*"]
        return

    if level == "block":
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
    """
    Adds a Census API key to query parameters.

    Resolution order:

    1. ``api_key`` argument, when not ``None``.
    2. ``CENSUS_API_KEY`` environment variable, when set and non-empty.
    3. No key is added; the request proceeds unauthenticated (subject to the
       500-request-per-day-per-IP cap enforced by the Census API).

    Parameters
    ----------
    params : dict
        Query parameter dictionary to mutate. A resolved key, if any, is
        written under the ``"key"`` query-string name expected by the Census
        API.
    api_key : str, optional
        Explicit Census API key. When ``None``, the environment variable
        ``CENSUS_API_KEY`` is consulted as a fallback.

    Warnings
    --------
    This function modifies ``params`` in place.
    """

    if api_key is not None:
        params["key"] = api_key
    elif key := os.getenv("CENSUS_API_KEY", False):
        params["key"] = key
    else:
        raise ValueError(
            "No Census API key provided. Supply one via the api_key argument "
            "or the CENSUS_API_KEY environment variable."
        )

    return


def _response_to_frame(response: httpx.Response) -> pd.DataFrame:
    """
    Converts a Census API list-of-lists JSON response into a DataFrame.

    Parameters
    ----------
    response : httpx.Response
        Response object from a Census API call.

    Returns
    -------
    pd.DataFrame
        A DataFrame where the first row of the JSON payload is used as column
        names and the remaining rows become data rows.

    Raises
    ------
    CensusRateLimitError
        When the Census API returns 429 Too Many Requests.
    httpx.HTTPStatusError
        Propagated from ``response.raise_for_status()`` for other HTTP errors.
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
    return pd.DataFrame(data[1:], columns=data[0])


def _get_acs5_geo_ids(
    client: httpx.Client,
    state: us.states.State,
    year: int,
    level: str,
) -> set[str]:
    """
    Retrieves GEO_IDs from ACS 5-year data for completeness checks.

    Parameters
    ----------
    client : httpx.Client
        HTTP client used to issue the request.
    state : us.states.State
        State the query is scoped to.
    year : int
        ACS 5-year vintage.
    level : str
        Geometry level. Must be one of "state", "county", "tract", or
        "block group".

    Returns
    -------
    set[str]
        GEO_IDs (stripped of the "US" prefix) for the requested geography.
    """

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    query_params = {
        "get": "GEO_ID",
        "for": f"{level}:*",
    }

    _construct_in_query(query_params, state, level)

    df = _response_to_frame(client.get(base_url, params=query_params))
    return set(df["GEO_ID"].astype("string").str.split("US").str[1])


def _warn_if_partial_acs1_data(
    client: httpx.Client,
    data: pd.DataFrame,
    state: us.states.State,
    year: int,
    level: str,
) -> None:
    """
    Warns when ACS 1-year returns only geographies meeting the population
    threshold.

    ACS 1-year estimates are only published for geographies of at least 65,000
    people, so a county-level query may return fewer counties than the state
    actually has. This helper compares the returned set against the complete
    set from ACS 5-year and emits a ``UserWarning`` when any are missing.

    Parameters
    ----------
    client : httpx.Client
        HTTP client used to issue the completeness-check request.
    data : pd.DataFrame
        The ACS 1-year DataFrame returned to the caller, indexed by GEO_ID.
    state : us.states.State
        State the query is scoped to.
    year : int
        ACS data year (5-year vintage or 1-year year).
    level : str
        Geometry level. No-op unless this is "county".
    """

    if level != "county":
        return

    try:
        expected_geo_ids = _get_acs5_geo_ids(client, state, year, level)
    except httpx.HTTPError:
        return

    returned_geo_ids = set(data.index)
    missing_geo_ids = expected_geo_ids - returned_geo_ids

    if not missing_geo_ids:
        return

    warnings.warn(
        (
            f"ACS 1-year returned {len(returned_geo_ids)} of "
            f"{len(expected_geo_ids)} {state.name} counties for {year}. "
            "ACS 1-year estimates are only available for geographies with "
            "at least 65,000 people; use survey='acs5' for complete county coverage."
        ),
        UserWarning,
        stacklevel=3,
    )


def _get_acs_data(
    client: httpx.Client,
    state: us.states.State,
    year: int,
    level: str,
    table: ACSTableInfo,
    survey: ACSSurvey = "acs5",
    suffix="E",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Retrieves raw ACS data from the Census API for one state/level/table.

    Query parameters follow the Census Bureau API user guide
    (https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-user-guide.pdf).
    Variable and example pages are published per dataset, e.g.
    https://api.census.gov/data/2022/acs/acs5/variables.html.

    Parameters
    ----------
    client : httpx.Client
        HTTP client used to issue the request.
    state : us.states.State
        State the query is scoped to.
    year : int
        ACS data year (5-year vintage end year or 1-year year).
    level : str
        Geometry level. Must be one of "state", "county", "tract", or
        "block group". ACS 1-year does not publish "tract" or "block group".
    table : ACSTableInfo
        Table definition supplying the Census variable names to request.
    survey : ACSSurvey, optional
        Normalized ACS survey period, either "acs5" or "acs1". Defaults to
        "acs5".
    suffix : str, optional
        Census variable suffix: "E" for estimates, "M" for margins of error.
        Defaults to "E".
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
        environment variable. As of 12 May 2026, an API key is required
        for all requests made to the census API.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by GEO_ID (with the "US" prefix stripped) and
        containing the requested variables cast to ``float``.

    Raises
    ------
    ValueError
        If ``survey`` is "acs1" and ``level`` is "tract" or "block group".
    """

    if survey == "acs1" and level in {"tract", "block group"}:
        raise ValueError(
            "ACS 1-year data are not available for 'tract' or 'block group' geometry. "
            "Use ACS 5-year data for those geometry levels."
        )

    base_url = f"https://api.census.gov/data/{year}/acs/{survey}"

    cols = list(table.construct_long_names(suffix=suffix, year=year).keys())

    query_cols = ["GEO_ID"] + cols

    query_params = {
        "get": ",".join(query_cols),
        "for": f"{level}:*",
    }

    _add_census_api_key(query_params, api_key)
    _construct_in_query(query_params, state, level)

    LOGGER.log(
        TRACE,
        "ACS %s %s %s %s for %s (%d vars).",
        survey,
        year,
        table.table_name or "<unnamed>",
        "EST" if suffix == "E" else "MOE",
        state.abbr,
        len(cols),
    )
    new_df = _response_to_frame(client.get(base_url, params=query_params))
    new_df["GEO_ID"] = new_df["GEO_ID"].astype("string").str.split("US").str[1]
    new_df.set_index("GEO_ID", inplace=True)
    return new_df[cols].astype(float)


def acs_full(
    state: us.states.State,
    geometry: str,
    table: ACSTableInfo,
    year: int,
    rename_columns=True,
    survey: str | int = "acs5",
    warn_on_partial_acs1=True,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves full (ungrouped) ACS data for one state, geometry, and table.

    Parameters
    ----------
    state : us.states.State
        State the query is scoped to.
    geometry : str
        Geometry level. Must be one of "state", "county", "tract", or
        "block group".
    table : ACSTableInfo
        Table definition describing the variables to request.
    year : int
        ACS data year (5-year vintage end year or 1-year year).
    rename_columns : bool, optional
        Whether to rename raw Census variable names to long English-language
        descriptions. Defaults to True.
    survey : str | int, optional
        ACS survey period. Accepts "acs5", "acs1", 5, or 1. Defaults to "acs5".
    warn_on_partial_acs1 : bool, optional
        Whether to emit a warning when an ACS 1-year county query returns only
        the geographies meeting the 65,000-population threshold. Defaults to
        True.
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
        environment variable. As of 12 May 2026, an API key is required
        for all requests made to the census API.


    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Estimate (EST) and margin-of-error (MOE) DataFrames, in that order.

    Raises
    ------
    ValueError
        If ``year`` is not a 4-digit integer in [2000, 2050], or ``survey`` is
        not recognized.
    """

    _validate_year(year)
    survey = _normalize_acs_survey(survey)

    with httpx.Client(timeout=httpx.Timeout(120)) as client:
        est_data = _get_acs_data(
            client,
            state,
            year,
            geometry,
            table,
            survey=survey,
            suffix="E",
            api_key=api_key,
        )
        moe_data = _get_acs_data(
            client,
            state,
            year,
            geometry,
            table,
            survey=survey,
            suffix="M",
            api_key=api_key,
        )

        if survey == "acs1" and warn_on_partial_acs1:
            _warn_if_partial_acs1_data(client, est_data, state, year, geometry)

    if rename_columns:
        source_suffix = survey.upper()
        est_data = est_data.rename(
            columns=table.construct_long_names(
                suffix="E",
                year=year,
                source_suffix=source_suffix,
            )
        )
        moe_data = moe_data.rename(
            columns=table.construct_long_names(
                suffix="M",
                year=year,
                source_suffix=source_suffix,
            )
        )

    return est_data, moe_data


def _condense(
    data: pd.DataFrame,
    table: ACSTableInfo,
    suffix: str,
    label: str,
) -> pd.DataFrame:
    """
    Collapses raw ACS columns on ``data`` into the grouped sums declared by
    ``table.condense_group_dict``.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of raw ACS variables (Census variable names with the suffix
        already applied).
    table : ACSTableInfo
        Table definition whose ``condense_group_dict`` drives the grouping.
    suffix : str
        Census variable suffix ("E" or "M") used to reconstruct source column
        names from the table's group specifications.
    label : str
        Suffix appended to each output group column name ("_EST" or "_MOE").

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per group whose full column set was present in
        ``data``. Groups whose source columns are missing are silently skipped.
    """

    columns = set(data.columns)
    result = pd.DataFrame(index=data.index)
    for group, variables in table.condense_group_dict.items():
        source_cols = [variable + suffix for variable in variables]
        if set(source_cols).issubset(columns):
            result[group + label] = data[source_cols].sum(axis=1)
    return result


def _acs(
    state: us.states.State,
    geometry: str,
    table: ACSTableInfo,
    year: int,
    short_names=False,
    survey: str | int = "acs5",
    warn_on_partial_acs1=True,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves ACS data for one table and consolidates it per
    ``table.condense_group_dict``.

    Parameters
    ----------
    state : us.states.State
        State the query is scoped to.
    geometry : str
        Geometry level. Must be one of "state", "county", "tract", or
        "block group".
    table : ACSTableInfo
        Table definition describing the variables to request and the groups to
        sum into.
    year : int
        ACS data year (5-year vintage end year or 1-year year).
    short_names : bool, optional
        Whether to shorten the long English-language group names to their
        canonical abbreviations. Defaults to False.
    survey : str | int, optional
        ACS survey period. Accepts "acs5", "acs1", 5, or 1. Defaults to "acs5".
    warn_on_partial_acs1 : bool, optional
        Whether to emit a warning when an ACS 1-year county query returns only
        the geographies meeting the 65,000-population threshold. Defaults to
        True.
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
        environment variable; if that is also unset, requests go out
        unauthenticated and are subject to the 500-per-day-per-IP cap (see
        ``CensusRateLimitError``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Condensed estimate (EST) and margin-of-error (MOE) DataFrames, in that
        order.
    """

    est_data, moe_data = acs_full(
        state,
        geometry,
        table,
        year,
        rename_columns=False,
        survey=survey,
        warn_on_partial_acs1=warn_on_partial_acs1,
        api_key=api_key,
    )

    short_est_data = _condense(est_data, table, suffix="E", label="_EST")
    short_moe_data = _condense(moe_data, table, suffix="M", label="_MOE")

    source_suffix = _normalize_acs_survey(survey).upper()
    if short_names:
        shorten_acs5_column_names(short_est_data, source_suffix=source_suffix)
        shorten_acs5_column_names(short_moe_data, source_suffix=source_suffix)
    else:
        short_est_data.rename(
            columns={
                column: append_source_suffix(column, source_suffix)
                for column in short_est_data.columns
            },
            inplace=True,
        )
        short_moe_data.rename(
            columns={
                column: append_source_suffix(column, source_suffix)
                for column in short_moe_data.columns
            },
            inplace=True,
        )

    return short_est_data, short_moe_data


def acs(
    state: us.states.State,
    geometry: str,
    year: int,
    tables: list[ACSTableInfo] | None = None,
    short_names=True,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves ACS data for one or more tables and concatenates the condensed
    results column-wise.

    Each table is consolidated per its own ``condense_group_dict``; the
    resulting group columns from all tables are concatenated into the output
    DataFrames.

    Parameters
    ----------
    state : us.states.State
        State the query is scoped to.
    geometry : str
        Geometry level. Must be one of "state", "county", "tract", or
        "block group".
    year : int
        ACS data year (5-year vintage end year or 1-year year).
    tables : list[ACSTableInfo], optional
        Tables to query. Defaults to
        ``[TotPopTableInfo(), VAPTableInfo(), CVAPTableInfo()]``.
    short_names : bool, optional
        Whether to shorten the long English-language group names to their
        canonical abbreviations. Defaults to True.
    survey : str | int, optional
        ACS survey period. Accepts "acs5", "acs1", 5, or 1. Defaults to "acs5".
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
        environment variable. As of 12 May 2026, an API key is required
        for all requests made to the census API.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Condensed estimate (EST) and margin-of-error (MOE) DataFrames, in that
        order, with columns from every requested table joined side by side.

    Raises
    ------
    ValueError
        If ``year`` is not a 4-digit integer in [2000, 2050], if ``tables`` is
        empty, or if ``survey`` is not recognized.
    TypeError
        If any element of ``tables`` is not an ``ACSTableInfo``.
    """

    _validate_year(year)

    if tables is None:
        tables = _default_acs_tables()

    if not tables:
        raise ValueError("Must provide at least one table.")

    for table in tables:
        if not isinstance(table, ACSTableInfo):
            raise TypeError(f"Each table must be an ACSTableInfo; got {type(table).__name__}.")

    survey = _normalize_acs_survey(survey)

    est_frames: list[pd.DataFrame] = []
    moe_frames: list[pd.DataFrame] = []
    for index, table in enumerate(tables):
        est_data, moe_data = _acs(
            state,
            geometry,
            table,
            year,
            short_names=short_names,
            survey=survey,
            warn_on_partial_acs1=index == 0,
            api_key=api_key,
        )
        est_frames.append(est_data)
        moe_frames.append(moe_data)

    return pd.concat(est_frames, axis=1), pd.concat(moe_frames, axis=1)


def _fetch_decennial_pl_county_fips(
    client: httpx.Client,
    state: us.states.State,
    pl_year: int,
    api_key: str | None = None,
) -> list[str]:
    """
    Fetches the county FIPS codes for a state from the decennial PL API.

    The decennial PL94-171 API does not support ``county:*`` at block
    geography, so callers that want block VAP for an entire state must first
    enumerate the state's counties and then issue one block query per
    county. This helper performs that enumeration by asking the PL API for
    ``NAME`` at county geography.

    Parameters
    ----------
    client : httpx.Client
        HTTP client used to issue the request. Sharing a single client
        across all PL calls for one state lets httpx reuse the underlying
        TCP/TLS connection, which matters because per-block-query latency
        dominates when there are many counties.
    state : us.states.State
        State the query is scoped to.
    pl_year : int
        Decennial PL vintage to query (e.g. ``2020``).
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY`` environment variable.
        As of 12 May 2026, an API key is required for all requests made to the census API.

    Returns
    -------
    list[str]
        County FIPS codes (3-digit strings, state-local), sorted
        lexicographically so the subsequent per-county fetch proceeds in a
        deterministic order.

    Raises
    ------
    CensusRateLimitError
        Propagated from ``_response_to_frame`` if the API returns HTTP 429.
    httpx.HTTPStatusError
        Propagated from ``_response_to_frame`` for other HTTP errors.
    """

    params = {
        "get": "NAME",
        "for": "county:*",
        "in": f"state:{state.fips}",
    }
    _add_census_api_key(params, api_key)

    counties = _response_to_frame(client.get(PL_BASE_URL.format(year=pl_year), params=params))
    return sorted(counties["county"].tolist())


def _get_decennial_pl_data(
    client: httpx.Client,
    state: us.states.State,
    year: int,
    level: str,
    table: DecennialPLTableInfo,
    county_fips: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Retrieves decennial PL data for one table and geography level.

    Unlike ACS variables, PL variables do not carry estimate/MOE suffixes;
    this function requests the raw variable names declared by ``table`` and
    renames them to the table's short local names. The Census-returned
    ``GEO_ID`` column (which is prefixed with a summary-level stub like
    ``"1000000US"``) is split on ``"US"`` and the trailing GEOID substring
    is stored on the returned DataFrame as ``GEOID``.

    Parameters
    ----------
    client : httpx.Client
        HTTP client used to issue the request. Reused across per-county
        block queries to amortize TCP/TLS setup.
    state : us.states.State
        State the query is scoped to.
    year : int
        Decennial PL vintage to query.
    level : str
        Geometry level. Must be one of "state", "county", "tract",
        "block group", or "block". When ``level`` is "block", ``county_fips``
        is required because the PL API does not support ``county:*`` at
        block geography.
    table : DecennialPLTableInfo
        Table definition supplying the Census variable names to request
        and the short names to rename them to.
    county_fips : str, optional
        3-digit county FIPS code. Required when ``level`` is "block";
        ignored for coarser geometries.
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY`` environment variable.
        As of 12 May 2026, an API key is required for all requests made to the census API.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per geography at ``level`` within the
        requested scope, with the table's short column names and a
        ``GEOID`` column replacing the Census-native ``GEO_ID``.

    Raises
    ------
    ValueError
        If ``level`` is "block" without ``county_fips``.
    CensusRateLimitError
        Propagated from ``_response_to_frame`` if the API returns HTTP 429.
    httpx.HTTPStatusError
        Propagated from ``_response_to_frame`` for other HTTP errors.
    """

    if level == "block" and county_fips is None:
        raise ValueError("Decennial PL block queries must be scoped to a county.")

    query_columns = ["GEO_ID"] + list(table.construct_variable_names())
    params = {
        "get": ",".join(query_columns),
        "for": f"{level}:*",
    }
    _add_census_api_key(params, api_key)
    _construct_in_query(params, state, level, county_fips=county_fips)

    data = _response_to_frame(client.get(PL_BASE_URL.format(year=year), params=params))
    data["GEOID"] = data["GEO_ID"].astype("string").str.split("US").str[1]
    data.drop(columns=["GEO_ID"], inplace=True)
    table.rename_columns(data)

    return data


def _fetch_block_pl_vap_for_county(
    client: httpx.Client,
    state: us.states.State,
    county_fips: str,
    pl_year: int,
    table: PLBlockVAPTableInfo,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetches block-level voting-age population by race from decennial PL data
    for a single county.

    The raw PL columns are coerced to numeric (the Census API returns all
    values as JSON strings) and the 15-character block GEOID is split into
    ``STATEFP`` (2), ``COUNTYFP`` (3), ``TRACTCE`` (6), ``BLOCKCE`` (4),
    and ``TRACT_GEOID`` (first 11 chars) so that downstream code can join
    the block VAP frame against ACS tract-level CVAP/VAP rates by
    ``TRACT_GEOID``.

    Parameters
    ----------
    client : httpx.Client
        HTTP client used to issue the request. The same client is reused
        for every per-county call when invoked from
        ``block_cvap_estimates``.
    state : us.states.State
        State the query is scoped to.
    county_fips : str
        3-digit county FIPS code identifying the county to fetch blocks
        within. The decennial PL API requires block queries to be scoped to
        a single county.
    pl_year : int
        Decennial PL vintage to query.
    table : PLBlockVAPTableInfo
        Table definition describing the block-level VAP variables to
        request (P3 race-by-VAP and P4 Hispanic-by-VAP variables, named
        per ``table.variable_to_short_name``).
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY`` environment variable.
        As of 12 May 2026, an API key is required for all requests made to the census API.

    Returns
    -------
    pd.DataFrame
        Block-level DataFrame with one row per block in the county. Columns
        include ``GEOID``, the parsed GEOID components (``STATEFP``,
        ``COUNTYFP``, ``TRACTCE``, ``BLOCKCE``, ``TRACT_GEOID``), and the
        race-specific VAP columns named per ``table.construct_short_names()``
        (e.g. ``TOT_VAP_P3``, ``WHITE_VAP_P3``, ``HISP_VAP_P4``).

    Raises
    ------
    CensusRateLimitError
        Propagated from ``_response_to_frame`` if the API returns HTTP 429.
    httpx.HTTPStatusError
        Propagated from ``_response_to_frame`` for other HTTP errors.
    """

    blocks = _get_decennial_pl_data(
        client,
        state,
        pl_year,
        "block",
        table,
        county_fips=county_fips,
        api_key=api_key,
    )

    for column in table.construct_short_names():
        blocks[column] = pd.to_numeric(blocks[column], errors="raise")

    blocks["STATEFP"] = blocks["GEOID"].str[:2]
    blocks["COUNTYFP"] = blocks["GEOID"].str[2:5]
    blocks["TRACTCE"] = blocks["GEOID"].str[5:11]
    blocks["BLOCKCE"] = blocks["GEOID"].str[11:15]
    blocks["TRACT_GEOID"] = blocks["GEOID"].str[:11]

    return blocks


def _fetch_acs_vap_cvap(
    state: us.states.State,
    geometry: str,
    acs_year: int,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetches ACS 5-year VAP and CVAP estimates used for block citizenship rates.

    Parameters
    ----------
    state : us.states.State
        State the query is scoped to.
    geometry : str
        Geometry level. Typically "tract" (for per-tract rates) or "state" (for
        the statewide fallback rate).
    acs_year : int
        ACS 5-year vintage end year.
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY`` environment variable.
        As of 12 May 2026, an API key is required for all requests made to the census API.

    Returns
    -------
    pd.DataFrame
        Condensed, short-named estimate DataFrame with ``{RACE}_VAP_ACS5`` and
        ``{RACE}_CVAP_ACS5`` columns.
    """

    est, _ = acs(
        state,
        geometry,
        acs_year,
        tables=[VAPTableInfo(), CVAPTableInfo()],
        short_names=True,
        survey="acs5",
        api_key=api_key,
    )
    return est


def _state_cvap_rates(
    state_est: pd.DataFrame,
    race_prefixes: tuple[str, ...] = RACE_PREFIXES,
    acs_source_suffix: str = ACS_SOURCE_SUFFIX,
) -> dict[str, float]:
    """
    Computes statewide race-specific CVAP/VAP fallback rates.

    Parameters
    ----------
    state_est : pd.DataFrame
        Single-row state-level ACS estimate DataFrame with source-suffixed
        ``{RACE}_VAP`` and ``{RACE}_CVAP`` columns.
    race_prefixes : tuple[str, ...], optional
        Race prefixes to compute rates for. Defaults to ``RACE_PREFIXES``.
    acs_source_suffix : str, optional
        Source suffix used on ACS columns. Defaults to ``"ACS5"``.

    Returns
    -------
    dict[str, float]
        Mapping from race prefix to statewide ``CVAP / VAP`` rate. Rates for
        races with zero statewide VAP are reported as 0.0.
    """

    state_row = state_est.iloc[0]
    rates = {}
    for race in race_prefixes:
        vap = state_row[f"{race}_VAP_{acs_source_suffix}"]
        cvap = state_row[f"{race}_CVAP_{acs_source_suffix}"]
        rates[race] = float(cvap / vap) if vap > 0 else 0.0
    return rates


def _tract_cvap_rates(
    tract_est: pd.DataFrame,
    denominator_threshold: int,
    race_prefixes: tuple[str, ...] = RACE_PREFIXES,
    acs_source_suffix: str = ACS_SOURCE_SUFFIX,
) -> dict[str, pd.Series]:
    """
    Computes tract-level CVAP/VAP rates, masking low ACS VAP denominators.

    When an ACS tract's VAP for a race is below ``denominator_threshold``, the
    resulting rate is ``NaN``, signalling to the caller that a fallback rate
    should be used for that tract.

    Parameters
    ----------
    tract_est : pd.DataFrame
        Tract-level ACS estimate DataFrame (indexed by tract GEOID) with
        source-suffixed ``{RACE}_VAP`` and ``{RACE}_CVAP`` columns.
    denominator_threshold : int
        Minimum tract VAP required for a tract rate to be computed.
    race_prefixes : tuple[str, ...], optional
        Race prefixes to compute rates for. Defaults to ``RACE_PREFIXES``.
    acs_source_suffix : str, optional
        Source suffix used on ACS columns. Defaults to ``"ACS5"``.

    Returns
    -------
    dict[str, pd.Series]
        Mapping from race prefix to a Series of tract rates (indexed by tract
        GEOID), with ``NaN`` for tracts below the denominator threshold.
    """

    rates = {}
    for race in race_prefixes:
        vap = tract_est[f"{race}_VAP_{acs_source_suffix}"]
        cvap = tract_est[f"{race}_CVAP_{acs_source_suffix}"]
        denominator = vap.where(vap >= denominator_threshold)
        rates[race] = cvap.divide(denominator)
    return rates


def _estimate_block_cvap_from_inputs(
    blocks: pd.DataFrame,
    tract_est: pd.DataFrame,
    state_est: pd.DataFrame,
    denominator_threshold: int,
    block_vap_columns: tuple[str, ...],
    block_vap_column_by_race: dict[str, str],
    race_prefixes: tuple[str, ...] = RACE_PREFIXES,
    acs_source_suffix: str = ACS_SOURCE_SUFFIX,
) -> pd.DataFrame:
    """
    Estimates block CVAP by applying ACS citizenship rates to PL block VAP.

    For each race ``r``:

    - ``tract_rate[r]`` is ``ACS5 tract r_CVAP_ACS5 / ACS5 tract
      r_VAP_ACS5`` when the tract's ACS VAP is at least
      ``denominator_threshold``; otherwise NaN.
    - ``state_rate[r]`` is ``ACS5 state r_CVAP_ACS5 / ACS5 state r_VAP_ACS5``.
    - Each block's estimated ``r_CVAP`` is ``PL block source-suffixed
      r_VAP * selected_rate``,
      where ``selected_rate`` is ``tract_rate[r]`` for the block's tract if
      available, and ``state_rate[r]`` otherwise.

    Parameters
    ----------
    blocks : pd.DataFrame
        Block-level DataFrame (from ``_fetch_block_pl_vap_for_county``) with
        ``TRACT_GEOID``, the parsed GEOID components, and source-suffixed
        ``{RACE}_VAP`` columns.
    tract_est : pd.DataFrame
        Tract-level ACS estimate DataFrame used for ``tract_rate``.
    state_est : pd.DataFrame
        Single-row state-level ACS estimate DataFrame used for ``state_rate``.
    denominator_threshold : int
        Minimum ACS tract VAP required to use a tract-level rate.
    block_vap_columns : tuple[str, ...]
        Block VAP column names to retain in the output.
    block_vap_column_by_race : dict[str, str]
        Mapping from race prefix to the source-suffixed PL block VAP column
        used as the CVAP allocation base.
    race_prefixes : tuple[str, ...], optional
        Race prefixes to estimate CVAP for. Defaults to ``RACE_PREFIXES``.
    acs_source_suffix : str, optional
        Source suffix used on ACS columns. Defaults to ``"ACS5"``.

    Returns
    -------
    pd.DataFrame
        Block-level DataFrame with GEOID components, block VAP, and one
        estimated ``{RACE}_CVAP`` column per race.
    """

    tract_rates = _tract_cvap_rates(
        tract_est,
        denominator_threshold,
        race_prefixes,
        acs_source_suffix=acs_source_suffix,
    )
    state_rates = _state_cvap_rates(
        state_est,
        race_prefixes,
        acs_source_suffix=acs_source_suffix,
    )
    estimated = blocks.copy()

    total_tract_blocks = 0
    total_fallback_blocks = 0
    for race in race_prefixes:
        tract_rate = estimated["TRACT_GEOID"].map(tract_rates[race])
        fallback_mask = tract_rate.isna()
        selected_rate = tract_rate.fillna(state_rates[race])
        block_vap_column = block_vap_column_by_race[race]
        estimated[f"{race}_CVAP"] = estimated[block_vap_column] * selected_rate

        fallback_blocks = int(fallback_mask.sum())
        tract_rate_blocks = int((~fallback_mask).sum())
        total_tract_blocks += tract_rate_blocks
        total_fallback_blocks += fallback_blocks

        LOGGER.debug(
            "%s_CVAP: tract rate for %s blocks, state fallback for %s (rate=%.6f).",
            race,
            tract_rate_blocks,
            fallback_blocks,
            state_rates[race],
        )

    total = total_tract_blocks + total_fallback_blocks
    if total:
        fallback_pct = 100.0 * total_fallback_blocks / total
        LOGGER.info(
            "Estimated %s block × race CVAP cells (%.1f%% used state fallback rate).",
            total,
            fallback_pct,
        )

    output_columns = (
        ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLOCKCE", "TRACT_GEOID"]
        + list(block_vap_columns)
        + [f"{race}_CVAP" for race in race_prefixes]
    )
    return estimated[output_columns]


def block_cvap_estimates(
    state: us.states.State,
    acs_year: int = DEFAULT_BLOCK_CVAP_ACS_YEAR,
    pl_year: int = DEFAULT_BLOCK_CVAP_PL_YEAR,
    denominator_threshold: int = DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Estimates block-level Citizen Voting-Age Population (CVAP) by race for a
    single state.

    Decennial PL data publishes voting-age population (VAP) at the block level
    but does not publish CVAP at any sub-tract geography. This function
    estimates block CVAP by applying ACS 5-year citizenship rates to the
    decennial PL block VAP counts.

    Formula
    -------
    For each race ``r`` in ``PLBlockVAPTableInfo.race_prefixes`` and each
    decennial PL block ``b`` in tract ``t`` of state ``s``:

    1. Compute the **tract rate**::

           tract_rate[t, r] = ACS5 tract[t] r_CVAP_ACS5 / ACS5 tract[t] r_VAP_ACS5

       but only when ``ACS5 tract[t] r_VAP_ACS5 >= denominator_threshold``.
       When the tract's ACS VAP is below the threshold (or the tract is
       missing from the ACS table entirely, which can happen for very small
       tracts), the tract rate is undefined.

    2. Compute the **state fallback rate**::

           state_rate[s, r] = ACS5 state[s] r_CVAP_ACS5 / ACS5 state[s] r_VAP_ACS5

       This is always defined; if ``ACS5 state r_VAP_ACS5 == 0`` the rate is
       taken to be 0.

    3. Select the per-block rate::

        if tract_rate[t, r] is defined:
            selected_rate[b, r] = tract_rate[t, r]
        else:
            selected_rate[b, r] = state_rate[s, r]

    4. Estimate the block CVAP::

           block[b] r_CVAP = PL block[b] source-suffixed r_VAP * selected_rate[b, r]

    Block VAP counts come from the decennial PL94-171 Census API (vintage
    ``pl_year``), queried one county at a time because the API does not
    support ``county:*`` at block geography. Tract and state VAP/CVAP come
    from ACS 5-year (vintage ``acs_year``); picking an ACS vintage centered
    on the PL year (e.g. ACS 2024 5-year centered on 2020 PL) keeps the
    citizenship rates temporally aligned with the block VAP counts.

    Parameters
    ----------
    state : us.states.State
        State to estimate block CVAP for.
    acs_year : int, optional
        ACS 5-year vintage end year supplying tract and state VAP/CVAP.
        Defaults to ``DEFAULT_BLOCK_CVAP_ACS_YEAR``.
    pl_year : int, optional
        Decennial PL vintage supplying block VAP. Defaults to
        ``DEFAULT_BLOCK_CVAP_PL_YEAR``.
    denominator_threshold : int, optional
        Minimum ACS tract VAP required to use the tract rate rather than the
        state fallback rate. Defaults to
        ``DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD``. Raising this trades bias
        (more blocks fall back to the state rate) against variance (tract rates
        from tiny ACS denominators are noisy).
    api_key : str, optional
        Census API key used for both the ACS and decennial PL requests. If
        omitted, falls back to the ``CENSUS_API_KEY`` environment variable.
        As of 12 May 2026, an API key is required for all requests made to the
        census API.

    Returns
    -------
    pd.DataFrame
        One row per decennial PL block in the state, with GEOID components
        (``GEOID``, ``STATEFP``, ``COUNTYFP``, ``TRACTCE``, ``BLOCKCE``,
        ``TRACT_GEOID``), source-suffixed decennial PL ``{RACE}_VAP`` counts
        (for example ``WHITE_VAP_P3`` and ``HISP_VAP_P4``), and estimated
        ``{RACE}_CVAP`` values. Returns an empty DataFrame if the state has no
        counties in the PL table (e.g. a territory not covered by the query).

    Raises
    ------
    ValueError
        If ``acs_year`` or ``pl_year`` is not a 4-digit integer in
        [2000, 2050].
    CensusRateLimitError
        If the Census API returns HTTP 429. This typically means the caller
        is unauthenticated and has exceeded the 500-request daily IP cap;
        supplying an ``api_key`` (or setting ``CENSUS_API_KEY``) lifts it.
    """

    _validate_year(acs_year)
    _validate_year(pl_year)

    table = PLBlockVAPTableInfo()
    race_prefixes = table.race_prefixes

    LOGGER.info(
        "Computing block CVAP for %s (ACS5 %s, decennial PL %s).",
        state.name,
        acs_year,
        pl_year,
    )

    LOGGER.info("Fetching ACS5 %s tract and state VAP/CVAP for %s.", acs_year, state.name)
    tract_est = _fetch_acs_vap_cvap(state, "tract", acs_year, api_key=api_key)
    state_est = _fetch_acs_vap_cvap(state, "state", acs_year, api_key=api_key)
    LOGGER.log(
        TRACE,
        "ACS5 tract VAP/CVAP: %s tracts for %s.",
        len(tract_est),
        state.abbr,
    )

    county_frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=httpx.Timeout(120)) as client:
        county_fips_values = _fetch_decennial_pl_county_fips(
            client, state, pl_year, api_key=api_key
        )
        total_counties = len(county_fips_values)
        LOGGER.info(
            "Fetching %s decennial PL block VAP across %s counties in %s.",
            pl_year,
            total_counties,
            state.name,
        )
        for index, county_fips in enumerate(county_fips_values, start=1):
            blocks = _fetch_block_pl_vap_for_county(
                client,
                state,
                county_fips,
                pl_year,
                table,
                api_key=api_key,
            )
            LOGGER.log(
                TRACE,
                "[%s %s/%s] County %s: %s blocks.",
                state.abbr,
                index,
                total_counties,
                county_fips,
                len(blocks),
            )
            county_frames.append(blocks)

    if not county_frames:
        return pd.DataFrame()

    blocks = pd.concat(county_frames, ignore_index=True)
    LOGGER.info(
        "Fetched %s PL block rows across %s counties for %s.",
        len(blocks),
        len(county_frames),
        state.name,
    )

    return _estimate_block_cvap_from_inputs(
        blocks,
        tract_est,
        state_est,
        denominator_threshold,
        table.construct_short_names(),
        {race: table.source_name_for_short_name(f"{race}_VAP") for race in race_prefixes},
        race_prefixes=race_prefixes,
    )


def cvap(
    state: us.states.State,
    geometry: str,
    year: int,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves Citizen Voting-Age Population (CVAP) data for one state and
    geometry from ACS 5-year or ACS 1-year tables.

    Parameters
    ----------
    state : us.states.State
        State the query is scoped to.
    geometry : str
        Geometry level. Must be one of "state", "county", "tract", or
        "block group".
    year : int
        ACS data year (5-year vintage end year or 1-year year).
    survey : str | int, optional
        ACS survey period. Accepts "acs5", "acs1", 5, or 1. Defaults to "acs5".
    api_key : str, optional
        Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
        environment variable. As of 12 May 2026, an API key is required for all
        requests made to the census API.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Condensed CVAP estimate (EST) and margin-of-error (MOE) DataFrames, in
        that order.
    """

    survey = _normalize_acs_survey(survey)

    est_data, moe_data = acs(
        state,
        geometry,
        year,
        tables=[CVAPTableInfo()],
        short_names=True,
        survey=survey,
        api_key=api_key,
    )

    return est_data, moe_data
