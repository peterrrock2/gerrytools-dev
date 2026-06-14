import warnings
from typing import Literal, cast

import httpx
import pandas as pd
import us

from gerrytools.logging import get_logger

from ._api import (
    ACS_BASE_URL,
    REQUEST_TIMEOUT,
    TRACE,
    CensusRateLimitError,
    _add_census_api_key,
    _construct_in_query,
    _response_to_frame,
    _strip_geoid_prefix,
    _validate_year,
)
from .census_tables import (
    ACSCVAPTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
    append_source_suffix,
    shorten_acs_column_names,
)

logger = get_logger(__name__)

ACSSurvey = Literal["acs1", "acs5"]


def _normalize_acs_survey(survey: str | int) -> ACSSurvey:
    """Normalize common ACS survey-period inputs to the Census API dataset name.

    Args:
        survey (str | int): ACS survey period to query. Accepts ``"acs5"``, ``"acs1"``, ``5``, or
            ``1`` (case- and separator-insensitive).

    Returns:
        ACSSurvey: The normalized ACS survey period, either ``"acs1"`` or ``"acs5"``.

    Raises:
        ValueError: If ``survey`` is neither a recognized string nor ``1``/``5``.
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


def _get_acs5_geo_ids(
    client: httpx.Client,
    state: us.states.State,
    year: int,
    geometry: str,
    api_key: str | None = None,
) -> set[str]:
    """Retrieve GEOIDs from ACS 5-year data for completeness checks.

    Args:
        client (httpx.Client): HTTP client used to issue the request.
        state (us.states.State): State the query is scoped to.
        year (int): ACS 5-year vintage.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable.

    Returns:
        set[str]: GEOIDs (stripped of the ``"US"`` prefix) for the requested geography.
    """

    base_url = ACS_BASE_URL.format(year=year, survey="acs5")
    query_params = {
        "get": "GEO_ID",
        "for": f"{geometry}:*",
    }

    _add_census_api_key(query_params, api_key)
    _construct_in_query(query_params, state, geometry)

    df = _response_to_frame(client.get(base_url, params=query_params))
    return set(_strip_geoid_prefix(df))


def _warn_if_partial_acs1_data(
    client: httpx.Client,
    data: pd.DataFrame,
    state: us.states.State,
    year: int,
    geometry: str,
    api_key: str | None = None,
) -> None:
    """Warn when ACS 1-year returns only the population-threshold geographies.

    ACS 1-year estimates are only published for geographies of at least 65,000 people, so a
    county-level query may return fewer counties than the state actually has. This helper compares
    the returned set against the complete set from ACS 5-year and emits a ``UserWarning`` when any
    are missing.

    Args:
        client (httpx.Client): HTTP client used to issue the completeness-check request.
        data (pd.DataFrame): ACS 1-year DataFrame returned to the caller, indexed by GEOID.
        state (us.states.State): State the query is scoped to.
        year (int): ACS data year (5-year vintage or 1-year year).
        geometry (str): Geometry level. No-op unless this is ``"county"``.
        api_key (str | None): Census API key forwarded to the completeness check. If omitted, falls
            back to the ``CENSUS_API_KEY`` environment variable.
    """

    if geometry != "county":
        return

    try:
        expected_geo_ids = _get_acs5_geo_ids(client, state, year, geometry, api_key=api_key)
    except (httpx.HTTPError, CensusRateLimitError):
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
    geometry: str,
    year: int,
    table: ACSTableInfo,
    survey: ACSSurvey = "acs5",
    suffix: str = "E",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve raw ACS data from the Census API for one state/geometry/table.

    Query parameters follow the Census Bureau API user guide
    (https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-user-guide.pdf).
    Variable and example pages are published per dataset, e.g.
    https://api.census.gov/data/2022/acs/acs5/variables.html.

    Args:
        client (httpx.Client): HTTP client used to issue the request.
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``. ACS 1-year does not publish ``"tract"`` or ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        table (ACSTableInfo): Table definition supplying the Census variable names to request.
        survey (ACSSurvey): Normalized ACS survey period, either ``"acs5"`` or ``"acs1"``. Defaults
            to ``"acs5"``.
        suffix (str): Census variable suffix: ``"E"`` for estimates, ``"M"`` for margins of error.
            Defaults to ``"E"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: DataFrame indexed by GEOID (with the ``"US"`` prefix stripped) and containing
        the requested variables cast to ``float``.

    Raises:
        ValueError: If ``survey`` is ``"acs1"`` and ``geometry`` is ``"tract"`` or ``"block
            group"``.
    """

    if survey == "acs1" and geometry in {"tract", "block group"}:
        raise ValueError(
            "ACS 1-year data are not available for 'tract' or 'block group' geometry. "
            "Use ACS 5-year data for those geometry levels."
        )

    base_url = ACS_BASE_URL.format(year=year, survey=survey)

    cols = list(table.construct_long_names(suffix=suffix, year=year).keys())

    query_cols = ["GEO_ID"] + cols

    query_params = {
        "get": ",".join(query_cols),
        "for": f"{geometry}:*",
    }

    _add_census_api_key(query_params, api_key)
    _construct_in_query(query_params, state, geometry)

    logger.log(
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
    new_df["GEOID"] = _strip_geoid_prefix(new_df)
    new_df.drop(columns=["GEO_ID"], inplace=True)
    new_df.set_index("GEOID", inplace=True)
    return cast(pd.DataFrame, new_df[cols].astype(float))


def acs_full(
    state: us.states.State,
    geometry: str,
    year: int,
    table: ACSTableInfo,
    rename_columns: bool = True,
    survey: str | int = "acs5",
    warn_on_partial_acs1: bool = True,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve full (ungrouped) ACS data for one state, geometry, and table.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        table (ACSTableInfo): Table definition describing the variables to request.
        rename_columns (bool): Whether to rename raw Census variable names to long English-language
            descriptions. Defaults to ``True``.
        survey (str | int): ACS survey period. Accepts ``"acs5"``, ``"acs1"``, ``5``, or ``1``.
            Defaults to ``"acs5"``.
        warn_on_partial_acs1 (bool): Whether to emit a warning when an ACS 1-year county query
            returns only the geographies meeting the 65,000-population threshold. Defaults to
            ``True``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Estimate (EST) and margin-of-error (MOE) DataFrames, in
        that order.

    Raises:
        ValueError: If ``year`` is not a 4-digit integer in [2000, 2050], or ``survey`` is not
            recognized.
    """

    _validate_year(year)
    survey = _normalize_acs_survey(survey)

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        est_data = _get_acs_data(
            client,
            state,
            geometry,
            year,
            table,
            survey=survey,
            suffix="E",
            api_key=api_key,
        )
        moe_data = _get_acs_data(
            client,
            state,
            geometry,
            year,
            table,
            survey=survey,
            suffix="M",
            api_key=api_key,
        )

        if survey == "acs1" and warn_on_partial_acs1:
            _warn_if_partial_acs1_data(client, est_data, state, year, geometry, api_key=api_key)

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
    """Collapse raw ACS columns into the grouped sums in ``condense_group_dict``.

    Args:
        data (pd.DataFrame): DataFrame of raw ACS variables (Census variable names with the suffix
            already applied).
        table (ACSTableInfo): Table definition whose ``condense_group_dict`` drives the grouping.
        suffix (str): Census variable suffix (``"E"`` or ``"M"``) used to reconstruct source column
            names from the table's group specifications.
        label (str): Suffix appended to each output group column name (``"_EST"`` or ``"_MOE"``).

    Returns:
        pd.DataFrame: DataFrame with one column per group whose full column set was present in
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
    year: int,
    table: ACSTableInfo,
    short_names: bool = False,
    survey: ACSSurvey = "acs5",
    warn_on_partial_acs1: bool = True,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve ACS data for one table and consolidate per ``condense_group_dict``.

    ``survey`` must already be normalized to ``"acs1"`` or ``"acs5"`` — this is a private helper
    called only from ``acs()``, which performs the normalization once.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level.
        year (int): ACS data year.
        table (ACSTableInfo): Table to query.
        short_names (bool): Whether to shorten long English-language group names to canonical
            abbreviations. Defaults to ``False``.
        survey (ACSSurvey): Normalized ACS survey period. Defaults to ``"acs5"``.
        warn_on_partial_acs1 (bool): Whether to forward the partial-coverage warning from
            ``acs_full``. Defaults to ``True``.
        api_key (str | None): Census API key, or ``CENSUS_API_KEY`` fallback.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Condensed estimate (EST) and margin-of-error (MOE)
        DataFrames, in that order.
    """

    est_data, moe_data = acs_full(
        state,
        geometry,
        year,
        table,
        rename_columns=False,
        survey=survey,
        warn_on_partial_acs1=warn_on_partial_acs1,
        api_key=api_key,
    )

    short_est_data = _condense(est_data, table, suffix="E", label="_EST")
    short_moe_data = _condense(moe_data, table, suffix="M", label="_MOE")

    source_suffix = survey.upper()
    if short_names:
        shorten_acs_column_names(short_est_data, source_suffix=source_suffix)
        shorten_acs_column_names(short_moe_data, source_suffix=source_suffix)
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
    short_names: bool = True,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve ACS data for one or more tables and concatenate column-wise.

    Each table is consolidated per its own ``condense_group_dict``; the resulting group columns from
    all tables are concatenated into the output DataFrames.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        tables (list[ACSTableInfo] | None): Tables to query. Defaults to ``[ACSTotPopTableInfo(),
            ACSVAPTableInfo(), ACSCVAPTableInfo()]``.
        short_names (bool): Whether to shorten long English-language group names to canonical
            abbreviations. Defaults to ``True``.
        survey (str | int): ACS survey period. Accepts ``"acs5"``, ``"acs1"``, ``5``, or ``1``.
            Defaults to ``"acs5"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Condensed estimate (EST) and margin-of-error (MOE)
            DataFrames, in that order, with columns from every requested table joined side by side.

    Raises:
        ValueError: If ``year`` is not a 4-digit integer in [2000, 2050], if ``tables`` is empty, or
            if ``survey`` is not recognized.
        TypeError: If any element of ``tables`` is not an ``ACSTableInfo``.
    """

    _validate_year(year)

    if tables is None:
        tables = [ACSTotPopTableInfo(), ACSVAPTableInfo(), ACSCVAPTableInfo()]

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
            year,
            table,
            short_names=short_names,
            survey=survey,
            warn_on_partial_acs1=index == 0,
            api_key=api_key,
        )
        est_frames.append(est_data)
        moe_frames.append(moe_data)

    return pd.concat(est_frames, axis=1), pd.concat(moe_frames, axis=1)


def cvap(
    state: us.states.State,
    geometry: str,
    year: int,
    survey: str | int = "acs5",
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve Citizen Voting-Age Population (CVAP) data for one state and geometry.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``, or
            ``"block group"``.
        year (int): ACS data year (5-year vintage end year or 1-year year).
        survey (str | int): ACS survey period. Accepts ``"acs5"``, ``"acs1"``, ``5``, or ``1``.
            Defaults to ``"acs5"``.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Condensed CVAP estimate (EST) and margin-of-error (MOE)
        DataFrames, in that order.
    """

    survey = _normalize_acs_survey(survey)

    est_data, moe_data = acs(
        state,
        geometry,
        year,
        tables=[ACSCVAPTableInfo()],
        short_names=True,
        survey=survey,
        api_key=api_key,
    )

    return est_data, moe_data
