import httpx
import pandas as pd
import us

from gerrytools.logging import get_logger

from .acs import (
    PL_BASE_URL,
    TRACE,
    _add_census_api_key,
    _construct_in_query,
    _response_to_frame,
    _validate_year,
)
from .census_tables import (
    PL_POP_TABLES,
    PL_POP_YEARS,
    PLTableInfo,
    pl_pop_table,
)

logger = get_logger(__name__)


def census(
    state: us.states.State,
    geometry: str = "state",
    year: int = 2020,
    table: PLTableInfo | str = "P1",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve decennial PL94-171 Census data for one state, geometry, and table.

    Issues a single ``get=group({table})`` request to the decennial PL Census
    API, drops empty and ``*err`` columns, and renames the raw Census variable
    columns to the short local names declared by ``table`` (e.g.
    ``TOT_VAP_P3``). The returned DataFrame is indexed by ``geo_id`` with the
    Census-stub ``"US"`` prefix stripped.

    Args:
        state (us.states.State): State the query is scoped to.
        geometry (str): Geometry level. Must be one of ``"state"``,
            ``"county"``, ``"tract"``, or ``"block group"``. Defaults to
            ``"state"``.
        year (int): Decennial PL vintage to query. ``2010`` and ``2020`` are
            supported. Defaults to ``2020``.
        table (PLTableInfo | str): Either a ``PLTableInfo`` instance or a
            shortcut string (``"P1"``, ``"P2"``, ``"P3"``, ``"P4"``) that is
            resolved via :func:`pl_pop_table` using ``year``. Defaults to
            ``"P1"``.
        api_key (str | None): Census API key. If omitted, falls back to the
            ``CENSUS_API_KEY`` environment variable. As of 12 May 2026, an
            API key is required for all requests made to the Census API.

    Returns:
        pd.DataFrame: One row per geography at ``geometry``, indexed by
        ``geo_id``, with one column per renamed variable from ``table``.

    Raises:
        ValueError: If ``year`` is unsupported, ``table`` is an invalid
            shortcut string, or ``geometry`` is unrecognized.
        CensusRateLimitError: Propagated from the underlying fetch if the
            API returns HTTP 429.
        httpx.HTTPStatusError: Propagated from the underlying fetch for
            other HTTP errors.
    """

    _validate_year(year)
    if year not in PL_POP_YEARS:
        raise ValueError(
            f"Decennial PL data is only available for years {PL_POP_YEARS}; got {year}."
        )

    if isinstance(table, str):
        if table not in PL_POP_TABLES:
            raise ValueError(
                f"Table {table!r} not recognized; allowed PL pop tables are {PL_POP_TABLES}."
            )
        pl_table_label = table
        table = pl_pop_table(table, year)
    else:
        pl_table_label = table.table_name

    base_url = PL_BASE_URL.format(year=year)
    api_table_code = pl_table_label.split("_", maxsplit=1)[0]
    query_params = {
        "get": f"group({api_table_code})",
        "for": f"{geometry}:*",
    }
    _add_census_api_key(query_params, api_key)
    _construct_in_query(query_params, state, geometry)

    logger.log(
        TRACE,
        "Decennial PL %s %s for %s (%s).",
        year,
        api_table_code,
        state.abbr,
        geometry,
    )

    with httpx.Client(timeout=httpx.Timeout(120)) as client:
        df = _response_to_frame(client.get(base_url, params=query_params))

    na_cols = df.columns[df.isna().all()].tolist()
    err_cols = [col for col in df.columns if col.lower().endswith("err")]
    df.drop(columns=na_cols + err_cols, inplace=True)

    table.rename_columns(df)

    df["geo_id"] = df["GEO_ID"].astype("string").str.split("US").str[-1]
    df.drop(columns=["GEO_ID"], inplace=True)
    df.set_index("geo_id", inplace=True)

    return df
