from typing import cast

import httpx
import pandas as pd
import us

from gerrytools.logging import get_logger

from ._api import (
    PL_BASE_URL,
    REQUEST_TIMEOUT,
    TRACE,
    _add_census_api_key,
    _construct_in_query,
    _response_to_frame,
    _strip_geoid_prefix,
    _validate_year,
)
from .acs import acs
from .census_tables import (
    ACS_SOURCE_SUFFIX,
    RACE_PREFIXES,
    ACSCVAPTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
    PLTableInfo,
)

logger = get_logger(__name__)

DEFAULT_BLOCK_CVAP_ACS_YEAR = 2024
DEFAULT_BLOCK_CVAP_PL_YEAR = 2020
DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD = 20


def _fetch_decennial_pl_county_fips(
    client: httpx.Client,
    state: us.states.State,
    pl_year: int,
    api_key: str | None = None,
) -> list[str]:
    """Fetch the county FIPS codes for a state from the decennial PL API.

    The decennial PL94-171 API does not support ``county:*`` at block geography, so callers that
    want block VAP for an entire state must first enumerate the state's counties and then issue one
    block query per county. This helper performs that enumeration by asking the PL API for ``NAME``
    at county geography.

    Args:
        client (httpx.Client): HTTP client used to issue the request. Sharing a single client across
            all PL calls for one state lets httpx reuse the underlying TCP/TLS connection, which
            matters because per-block-query latency dominates when there are many counties.
        state (us.states.State): State the query is scoped to.
        pl_year (int): Decennial PL vintage to query (e.g. ``2020``).
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        list[str]: County FIPS codes (3-digit strings, state-local), sorted lexicographically so the
        subsequent per-county fetch proceeds in a deterministic order.

    Raises:
        CensusRateLimitError: Propagated from ``_response_to_frame`` if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from ``_response_to_frame`` for other HTTP errors.
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
    geometry: str,
    table: PLTableInfo,
    county_fips: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve decennial PL data for one table and geometry level.

    Unlike ACS variables, PL variables do not carry estimate/MOE suffixes; this function requests
    the raw variable names declared by ``table`` and renames them to the table's short local names.
    The Census-returned ``GEO_ID`` column (which is prefixed with a summary-level stub like
    ``"1000000US"``) is split on ``"US"`` and the trailing GEOID substring is stored on the returned
    DataFrame as ``GEOID``.

    Args:
        client (httpx.Client): HTTP client used to issue the request. Reused across per-county block
            queries to amortize TCP/TLS setup.
        state (us.states.State): State the query is scoped to.
        year (int): Decennial PL vintage to query.
        geometry (str): Geometry level. Must be one of ``"state"``, ``"county"``, ``"tract"``,
            ``"block group"``, or ``"block"``. When ``geometry`` is ``"block"``, ``county_fips`` is
            required because the PL API does not support ``county:*`` at block geography.
        table (PLTableInfo): Table definition supplying the Census variable names to request and the
            short names to rename them to.
        county_fips (str | None): 3-digit county FIPS code. Required when ``geometry`` is
            ``"block"``; ignored for coarser geometries.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: DataFrame containing one row per geography at ``geometry`` within the
        requested scope, with the table's short column names and a ``GEOID`` column replacing the
        Census-native ``GEO_ID``.

    Raises:
        ValueError: If ``geometry`` is ``"block"`` without ``county_fips``.
        CensusRateLimitError: Propagated from ``_response_to_frame`` if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from ``_response_to_frame`` for other HTTP errors.
    """

    if geometry == "block" and county_fips is None:
        raise ValueError("Decennial PL block queries must be scoped to a county.")

    query_columns = ["GEO_ID"] + list(table.construct_variable_names())
    params = {
        "get": ",".join(query_columns),
        "for": f"{geometry}:*",
    }
    _add_census_api_key(params, api_key)
    _construct_in_query(params, state, geometry, county_fips=county_fips)

    data = _response_to_frame(client.get(PL_BASE_URL.format(year=year), params=params))
    data["GEOID"] = _strip_geoid_prefix(data)
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
    """Fetch block-level VAP by race from decennial PL data for one county.

    The raw PL columns are coerced to numeric (the Census API returns all values as JSON strings)
    and the 15-character block GEOID is split into ``STATEFP`` (2), ``COUNTYFP`` (3), ``TRACTCE``
    (6), ``BLOCKCE`` (4), and ``TRACT_GEOID`` (first 11 chars) so that downstream code can join the
    block VAP frame against ACS tract-level CVAP/VAP rates by ``TRACT_GEOID``.

    Args:
        client (httpx.Client): HTTP client used to issue the request. The same client is reused for
            every per-county call when invoked from ``block_cvap_estimates``.
        state (us.states.State): State the query is scoped to.
        county_fips (str): 3-digit county FIPS code identifying the county to fetch blocks within.
            The decennial PL API requires block queries to be scoped to a single county.
        pl_year (int): Decennial PL vintage to query.
        table (PLBlockVAPTableInfo): Table definition describing the block-level VAP variables to
            request (P3 race-by-VAP and P4 Hispanic-by-VAP variables, named per
            ``table.variable_to_short_name``).
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: Block-level DataFrame with one row per block in the county. Columns include
        ``GEOID``, the parsed GEOID components (``STATEFP``, ``COUNTYFP``, ``TRACTCE``, ``BLOCKCE``,
        ``TRACT_GEOID``), and the race-specific VAP columns named per
        ``table.construct_short_names()`` (e.g. ``TOT_VAP_P3``, ``WHITE_VAP_P3``, ``HISP_VAP_P4``).

    Raises:
        CensusRateLimitError: Propagated from ``_response_to_frame`` if the API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from ``_response_to_frame`` for other HTTP errors.
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
    """Fetch ACS 5-year VAP and CVAP estimates for block citizenship rates.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Typically ``"tract"`` (for per-tract rates) or ``"state"``
            (for the statewide fallback rate).
        acs_year (int): ACS 5-year vintage end year.
        api_key (str | None): Census API key. If omitted, falls back to the ``CENSUS_API_KEY``
            environment variable. As of 12 May 2026, an API key is required for all requests made to
            the Census API.

    Returns:
        pd.DataFrame: Condensed, short-named estimate DataFrame with ``{RACE}_VAP_ACS5`` and
        ``{RACE}_CVAP_ACS5`` columns.
    """

    est, _ = acs(
        state,
        geometry,
        acs_year,
        tables=[ACSVAPTableInfo(), ACSCVAPTableInfo()],
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
    """Compute statewide race-specific CVAP/VAP fallback rates.

    Args:
        state_est (pd.DataFrame): Single-row state-level ACS estimate DataFrame with source-suffixed
            ``{RACE}_VAP`` and ``{RACE}_CVAP`` columns.
        race_prefixes (tuple[str, ...]): Race prefixes to compute rates for. Defaults to
            ``RACE_PREFIXES``.
        acs_source_suffix (str): Source suffix used on ACS columns. Defaults to ``"ACS5"``.

    Returns:
        dict[str, float]: Mapping from race prefix to statewide ``CVAP / VAP`` rate. Rates for races
        with zero statewide VAP are reported as ``0.0``.
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
    """Compute tract-level CVAP/VAP rates, masking low ACS VAP denominators.

    When an ACS tract's VAP for a race is below ``denominator_threshold``, the resulting rate is
    ``NaN``, signalling to the caller that a fallback rate should be used for that tract.

    Args:
        tract_est (pd.DataFrame): Tract-level ACS estimate DataFrame (indexed by tract GEOID) with
            source-suffixed ``{RACE}_VAP`` and ``{RACE}_CVAP`` columns.
        denominator_threshold (int): Minimum tract VAP required for a tract rate to be computed.
        race_prefixes (tuple[str, ...]): Race prefixes to compute rates for. Defaults to
            ``RACE_PREFIXES``.
        acs_source_suffix (str): Source suffix used on ACS columns. Defaults to ``"ACS5"``.

    Returns:
        dict[str, pd.Series]: Mapping from race prefix to a Series of tract rates (indexed by tract
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
    table: PLBlockVAPTableInfo,
    acs_source_suffix: str = ACS_SOURCE_SUFFIX,
) -> pd.DataFrame:
    """Estimate block CVAP by applying ACS citizenship rates to PL block VAP.

    For each race ``r`` in ``table.race_prefixes``:

    - ``tract_rate[r]`` is ``ACS5 tract r_CVAP_ACS5 / ACS5 tract r_VAP_ACS5``
      when the tract's ACS VAP is at least ``denominator_threshold``;
      otherwise NaN.
    - ``state_rate[r]`` is ``ACS5 state r_CVAP_ACS5 / ACS5 state r_VAP_ACS5``.
    - Each block's estimated ``r_CVAP`` is
      ``PL block source-suffixed r_VAP * selected_rate``, where
      ``selected_rate`` is ``tract_rate[r]`` for the block's tract if
      available, and ``state_rate[r]`` otherwise.

    Args:
        blocks (pd.DataFrame): Block-level DataFrame (from ``_fetch_block_pl_vap_for_county``) with
            ``TRACT_GEOID``, the parsed GEOID components, and source-suffixed ``{RACE}_VAP``
            columns.
        tract_est (pd.DataFrame): Tract-level ACS estimate DataFrame used for ``tract_rate``.
        state_est (pd.DataFrame): Single-row state-level ACS estimate DataFrame used for
            ``state_rate``.
        denominator_threshold (int): Minimum ACS tract VAP required to use a tract-level rate.
        table (PLBlockVAPTableInfo): Table definition providing the race prefixes and
            source-suffixed PL block VAP column names used as the CVAP allocation base.
        acs_source_suffix (str): Source suffix used on ACS columns. Defaults to ``"ACS5"``.

    Returns:
        pd.DataFrame: Block-level DataFrame with GEOID components, block VAP, and one estimated
        ``{RACE}_CVAP`` column per race.
    """

    race_prefixes = table.race_prefixes
    block_vap_columns = table.construct_short_names()
    block_vap_column_by_race = {
        race: table.source_name_for_short_name(f"{race}_VAP") for race in race_prefixes
    }

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
        tract_rate = cast(pd.Series, estimated["TRACT_GEOID"]).map(tract_rates[race])
        fallback_mask = tract_rate.isna()
        selected_rate = tract_rate.fillna(state_rates[race])
        block_vap_column = block_vap_column_by_race[race]
        estimated[f"{race}_CVAP"] = estimated[block_vap_column] * selected_rate

        fallback_blocks = int(fallback_mask.sum())
        tract_rate_blocks = int((~fallback_mask).sum())
        total_tract_blocks += tract_rate_blocks
        total_fallback_blocks += fallback_blocks

        logger.debug(
            "%s_CVAP: tract rate for %s blocks, state fallback for %s (rate=%.6f).",
            race,
            tract_rate_blocks,
            fallback_blocks,
            state_rates[race],
        )

    total = total_tract_blocks + total_fallback_blocks
    if total:
        fallback_pct = 100.0 * total_fallback_blocks / total
        logger.info(
            "Estimated %s block × race CVAP cells (%.1f%% used state fallback rate).",
            total,
            fallback_pct,
        )

    output_columns = (
        ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLOCKCE", "TRACT_GEOID"]
        + list(block_vap_columns)
        + [f"{race}_CVAP" for race in race_prefixes]
    )
    return cast(pd.DataFrame, estimated[output_columns])


def block_cvap_estimates(
    state: us.states.State,
    acs_year: int = DEFAULT_BLOCK_CVAP_ACS_YEAR,
    pl_year: int = DEFAULT_BLOCK_CVAP_PL_YEAR,
    denominator_threshold: int = DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Estimate block-level Citizen Voting-Age Population (CVAP) by race.

    Decennial PL data publishes voting-age population (VAP) at the block level but does not publish
    CVAP at any sub-tract geography. This function estimates block CVAP by applying ACS 5-year
    citizenship rates to the decennial PL block VAP counts.

    For each race ``r`` in ``PLBlockVAPTableInfo.race_prefixes`` and each decennial PL block ``b``
    in tract ``t`` of state ``s``:

    1. Compute the **tract rate**::

           tract_rate[t, r] = ACS5 tract[t] r_CVAP_ACS5 / ACS5 tract[t] r_VAP_ACS5

       but only when ``ACS5 tract[t] r_VAP_ACS5 >= denominator_threshold``.
       When the tract's ACS VAP is below the threshold (or the tract is
       missing from the ACS table entirely, which can happen for very small
       tracts), the tract rate is undefined.

    2. Compute the **state fallback rate**::

           state_rate[s, r] = ACS5 state[s] r_CVAP_ACS5 / ACS5 state[s] r_VAP_ACS5

       This is always defined; if ``ACS5 state r_VAP_ACS5 == 0`` the rate is
       taken to be ``0``.

    3. Select the per-block rate::

           if tract_rate[t, r] is defined:
               selected_rate[b, r] = tract_rate[t, r]
           else:
               selected_rate[b, r] = state_rate[s, r]

    4. Estimate the block CVAP::

           block[b] r_CVAP = PL block[b] source-suffixed r_VAP * selected_rate[b, r]

    Block VAP counts come from the decennial PL94-171 Census API (vintage ``pl_year``), queried one
    county at a time because the API does not support ``county:*`` at block geography. Tract and
    state VAP/CVAP come from ACS 5-year (vintage ``acs_year``); picking an ACS vintage centered on
    the PL year (e.g. ACS 2024 5-year centered on 2020 PL) keeps the citizenship rates temporally
    aligned with the block VAP counts.

    Args:
        state (us.states.State): State to estimate block CVAP for.
        acs_year (int): ACS 5-year vintage end year supplying tract and state VAP/CVAP. Defaults to
            ``DEFAULT_BLOCK_CVAP_ACS_YEAR``.
        pl_year (int): Decennial PL vintage supplying block VAP. Defaults to
            ``DEFAULT_BLOCK_CVAP_PL_YEAR``.
        denominator_threshold (int): Minimum ACS tract VAP required to use the tract rate rather
            than the state fallback rate. Defaults to ``DEFAULT_BLOCK_CVAP_DENOMINATOR_THRESHOLD``.
            Raising this trades bias (more blocks fall back to the state rate) against variance
            (tract rates from tiny ACS denominators are noisy).
        api_key (str | None): Census API key used for both the ACS and decennial PL requests. If
            omitted, falls back to the ``CENSUS_API_KEY`` environment variable. As of 12 May 2026,
            an API key is required for all requests made to the Census API.

    Returns:
        pd.DataFrame: One row per decennial PL block in the state, with GEOID components (``GEOID``,
        ``STATEFP``, ``COUNTYFP``, ``TRACTCE``, ``BLOCKCE``, ``TRACT_GEOID``), source-suffixed
        decennial PL ``{RACE}_VAP`` counts (for example ``WHITE_VAP_P3`` and ``HISP_VAP_P4``), and
        estimated ``{RACE}_CVAP`` values. Returns an empty DataFrame if the state has no counties in
        the PL table (e.g. a territory not covered by the query).

    Raises:
        ValueError: If ``acs_year`` or ``pl_year`` is not a 4-digit integer in [2000, 2050].
        CensusRateLimitError: If the Census API returns HTTP 429. This typically means the caller is
            unauthenticated and has exceeded the 500-request daily IP cap; supplying an ``api_key``
            (or setting ``CENSUS_API_KEY``) lifts it.
    """

    _validate_year(acs_year)
    _validate_year(pl_year)

    table = PLBlockVAPTableInfo()

    logger.info(
        "Computing block CVAP for %s (ACS5 %s, decennial PL %s).",
        state.name,
        acs_year,
        pl_year,
    )

    logger.info("Fetching ACS5 %s tract and state VAP/CVAP for %s.", acs_year, state.name)
    tract_est = _fetch_acs_vap_cvap(state, "tract", acs_year, api_key=api_key)
    state_est = _fetch_acs_vap_cvap(state, "state", acs_year, api_key=api_key)
    logger.log(
        TRACE,
        "ACS5 tract VAP/CVAP: %s tracts for %s.",
        len(tract_est),
        state.abbr,
    )

    county_frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        county_fips_values = _fetch_decennial_pl_county_fips(
            client, state, pl_year, api_key=api_key
        )
        total_counties = len(county_fips_values)
        logger.info(
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
            logger.log(
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
    logger.info(
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
        table,
    )
