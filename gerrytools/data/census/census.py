import pandas as pd
import requests
import us

from gerrytools.logging import get_logger
from gerrytools.data.census.ptable_column_aliases import COLUMN_ALIASES_PTABLES

logger = get_logger(__name__)


def census(
    state: us.states.State,
    table: str = "P1",
    geometry: str = "state",
    year: int = 2020,
    retry_count: int = 3,
    timeout_seconds: int = 300,
    census_api_key: str | None = None,
) -> pd.DataFrame:
    if year not in [2010, 2020]:
        raise ValueError("Only years 2010 and 2020 are supported.")
    if table not in COLUMN_ALIASES_PTABLES[year]:
        raise ValueError(
            f"Table {table} not recognized. "
            "Only tables 'P1', 'P2', 'P3', and 'P4' are supported"
        )

    URL_BASE = (
        "https://api.census.gov/data/{year}/dec/pl?get=group({table})&for={geometry}:*"
    )

    if geometry == "state":
        pass
    elif geometry == "county":
        URL_BASE += "&in=state:{fips}"
    elif geometry in ["tract", "place"]:
        URL_BASE += "&in=state:{fips}"
        URL_BASE += "&in=county:*"
    elif geometry == "block group":
        URL_BASE += "&in=state:{fips}"
        URL_BASE += "&in=county:*"
        URL_BASE += "&in=tract:*"
    else:
        raise ValueError(
            f'Geometry "{geometry}" not recognized; '
            f"allowed values are 'state', 'county', 'tract', 'block group', and 'place'."
        )

    if census_api_key is not None:
        URL_BASE += f"&key={census_api_key}"

    attempts = 0

    while attempts < retry_count:
        try:
            full_url = URL_BASE.format(
                year=year,
                table=table,
                geometry=geometry.replace(" ", "%20"),
                fips=str(state.fips).zfill(2),
            )
            logger.debug(f"Requesting data from URL: {full_url}")
            response = requests.get(
                full_url,
                timeout=timeout_seconds,
            )
            response.raise_for_status()

            attempts = retry_count  # Success

        except requests.RequestException as e:
            if attempts < retry_count:
                attempts += 1
            else:
                raise ValueError(
                    f"Failed to retrieve data after multiple attempts. Found error: {e}"
                )

    if response is None:
        raise ValueError("Failed to retrieve data; no response received.")

    if response.status_code != 200:
        raise ValueError(
            f"Failed to retrieve data; status code {response.status_code}."
        )

    df = pd.DataFrame(response.json()[1:], columns=response.json()[0])
    na_cols = df.columns[df.isna().all()].tolist()
    err_cols = [col for col in df.columns if col.lower().endswith("err")]
    df.drop(columns=na_cols + err_cols, inplace=True)

    table_columns = COLUMN_ALIASES_PTABLES[year][table]

    column_remap = {k: table_columns.get(k.lower(), k) for k in df.columns}
    df.rename(columns=column_remap, inplace=True)

    if geometry == "state":
        df = df.query(f"state == '{str(state.fips).zfill(2)}'").copy()

    df["geo_id"] = df["GEO_ID"].str.split("US").str[-1]
    df.drop(columns=["GEO_ID"], inplace=True)
    df.set_index("geo_id", inplace=True)

    return df
