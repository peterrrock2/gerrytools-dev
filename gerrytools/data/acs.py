import httpx
import pandas as pd
from gerrytools.utils.acs5_tables import (
    ACS5TableInfo,
    CVAPTableInfo,
    VAPTableInfo,
    TotPopTableInfo,
    shorten_acs5_column_names,
)
import gerrytools.utils as utils


def _construct_in_query(curr_query: dict, state: utils.states.State, level: str):
    """
    Construct the "in" query parameter for the Census API.

    Args:
        curr_query (dict):
            A dictionary containing the current query parameters.

        state (utils.states.State):
            The state for which the query is being constructed.

        level (str):
            The geometry level for which the query is being constructed. Must be
            one of "state", "county", "tract", or "block group".

    Warning:
        This function modifies the input dictionary in place.

    Returns:
        dict: The updated query dictionary.
    """

    if level == "state":
        curr_query["for"] = f"state:{state.fips}"
        return curr_query

    if level == "county":
        curr_query["in"] = f"state:{state.fips}"
        return curr_query

    if level == "tract":
        curr_query["in"] = [f"state:{state.fips}", f"county:*"]
        return curr_query

    if level == "block group":
        curr_query["in"] = [f"state:{state.fips}", f"county:*", f"tract:*"]
        return curr_query

    raise ValueError(
        "Invalid geometry level. Must be one of 'state', 'county', 'tract', or 'block group'."
    )


def _get_acs5_data(
    client: httpx.Client,
    state: utils.states.State,
    year: int,
    level: str,
    table: ACS5TableInfo,
    suffix="E",
):
    """
    Send a GET request to the Census API to retrieve ACS 5-year data.

    This function retrieves ACS 5-year data for a specified state, geometry level, and table
    by constructing the query parameters based on the Census Bureau's API documentation.

    API Documentation:
        https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-user-guide.pdf

    Examples of queries (for the year 2022):
        https://api.census.gov/data/2022/acs/acs5/examples.html

    ACS 5-year data variable explanations:
        https://api.census.gov/data/2022/acs/acs5/variables.html

    Args:
        client (httpx.Client):
            An `httpx.Client` object used for making HTTP requests.

        state (utils.states.State):
            The state for which ACS 5-year data is being retrieved.

        year (int):
            The year of the ACS 5-year data.

        level (str):
            The geometry level for the query. Must be one of "state", "county", "tract", or
            "block group".

        table (ACS5TableInfo):
            An object containing information about the ACS 5-year table being queried.

        suffix (str, optional):
            The suffix for the ACS 5-year data, defaulting to "E" (Estimate).

    Returns:
        pd.DataFrame:
            A DataFrame containing the ACS 5-year data for the specified state, geometry level, and
            table.
    """

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"

    cols = list(table.construct_long_names(suffix=suffix, year=year).keys())

    query_cols = ["GEO_ID"] + cols

    query_params = {
        "get": ",".join(query_cols),
        "for": f"{level}:*",
    }

    _construct_in_query(query_params, state, level)

    response = client.get(base_url, params=query_params)
    response.raise_for_status()
    data = response.json()

    new_df = pd.DataFrame(data[1:], columns=data[0])
    new_df["GEO_ID"] = new_df["GEO_ID"].map(lambda x: x.split("US")[1])
    new_df.set_index("GEO_ID", inplace=True)
    new_df = new_df[cols]

    return new_df[new_df.columns].astype(float)


def acs5_full(
    state: utils.states.State,
    geometry: str,
    table: ACS5TableInfo,
    year: int = 2020,
    rename_columns=True,
):
    """
    Retrieve ACS 5-year data for the specified state, geometry level, and table.

    Args:
        state (utils.states.State):
            The state for which ACS 5-year data is being retrieved.

        geometry (str):
            The geometry level for which ACS 5-year data is being retrieved. Must be one of
            "state", "county", "tract", or "block group".

        table (ACS5TableInfo):
            An object containing information about the ACS 5-year table being queried.

        year (int, optional):
            The year of the ACS 5-year data. Defaults to 2020.

        rename_columns (bool, optional):
            Whether to rename the DataFrame columns to their long, English-language descriptions.
            Defaults to True.

    Returns:
        pd.DataFrame:
            A DataFrame containing the ACS 5-year Estimate (EST) data for the specified state,
            geometry level, and table.

        pd.DataFrame:
            A DataFrame containing the ACS 5-year Margin of Error (MOE) data for the specified state,
            geometry level, and table.
    """

    assert len(str(year)) == 4, f"Year must be 4 digits. Received {year}."
    assert year >= 2000
    assert year <= 2050

    with httpx.Client(timeout=httpx.Timeout(120)) as client:
        est_data = _get_acs5_data(client, state, year, geometry, table, suffix="E")
        moe_data = _get_acs5_data(client, state, year, geometry, table, suffix="M")

    if rename_columns:
        est_data = est_data.rename(
            columns=table.construct_long_names(suffix="E", year=year)
        )
        moe_data = moe_data.rename(
            columns=table.condense_group_dict(suffix="M", year=year)
        )

    return est_data, moe_data


def _acs5(
    state: utils.states.State,
    geometry: str,
    table: ACS5TableInfo,
    year: int,
    short_names=False,
):
    """
    A helper function for retrieving ACS 5-year data for a specified state, geometry level,
    and table.

    This function also consolidates the data according to the groups specified in the
    `condense_group_dict` attribute of the `ACS5TableInfo` object.

    Args:
        state (utils.states.State):
            The state for which ACS 5-year data is being retrieved.

        geometry (str):
            The geometry level for which ACS 5-year data is being retrieved. Must be one of
            "state", "county", "tract", or "block group".

        table (ACS5TableInfo):
            An object containing information about the ACS 5-year table being queried.

        year (int, optional):
            The year of the ACS 5-year data. Defaults to 2020.

        short_names (bool, optional):
            Whether to shorten the long, English-language descriptions of the ACS 5-year data columns
            to shorter, more concise names. Defaults to False.

    Returns:
        pd.DataFrame:
            A DataFrame containing the ACS 5-year data Estimate (EST) for the specified state,
            geometry level, and table.

        pd.DataFrame:
            A DataFrame containing the ACS 5-year Margin of Error (MOE) data for the specified
            state, geometry level, and table.
    """

    assert len(str(year)) == 4, f"Year must be 4 digits. Received {year}."
    assert year >= 2000
    assert year <= 2050

    est_data, moe_data = acs5_full(state, geometry, table, year, rename_columns=False)

    est_cols = set(est_data.columns)

    short_est_data = pd.DataFrame()
    for group, lst in table.condense_group_dict.items():
        lst = [i + f"E" for i in lst]
        if set(lst) - est_cols == set():
            short_est_data[group + "_EST"] = est_data[lst].sum(axis=1)

    moe_cols = set(moe_data.columns)

    short_moe_data = pd.DataFrame()
    for group, lst in table.condense_group_dict.items():
        lst = [i + f"M" for i in lst]
        if set(lst) - moe_cols == set():
            short_moe_data[group + "_MOE"] = moe_data[lst].sum(axis=1)

    if short_names:
        shorten_acs5_column_names(short_est_data)
        shorten_acs5_column_names(short_moe_data)

    return short_est_data, short_moe_data


def acs5(
    state: utils.states.State,
    geometry: str,
    year: int,
    tables: list[ACS5TableInfo] = [TotPopTableInfo(), VAPTableInfo(), CVAPTableInfo()],
    short_names=True,
):
    """
    Retrieves ACS 5-year data for the provided state, geometry level, and tables,
    and will consolidate the data according to the groups specified in the
    `condense_group_dict` attribute of each of the `ACS5TableInfo` objects.

    Args:
        state (utils.states.State):
            The state for which ACS 5-year data is being retrieved.

        geometry (str):
            The geometry level for which ACS 5-year data is being retrieved. Must be one of
            "state", "county", "tract", or "block group".

        table (ACS5TableInfo):
            An object containing information about the ACS 5-year table being queried.

        year (int, optional):
            The year of the ACS 5-year data. Defaults to 2020.

        short_names (bool, optional):
            Whether to shorten the long, English-language descriptions of the ACS 5-year data columns
            to shorter, more concise names. Defaults to False.

    Returns:
        pd.DataFrame:
            A DataFrame containing the ACS 5-year data Estimate (EST) for the specified state,
            geometry level, and table.

        pd.DataFrame:
            A DataFrame containing the ACS 5-year Margin of Error (MOE) data for the specified
            state, geometry level, and table.
    """

    assert len(str(year)) == 4, f"Year must be 4 digits. Received {year}."
    assert year >= 2000
    assert year <= 2050

    assert len(tables) > 0, "Must provide at least one table."

    for table in tables:
        assert isinstance(table, ACS5TableInfo)

    est_df = None
    moe_df = None
    for table in tables:
        est_data, moe_data = _acs5(
            state, geometry, table, year, short_names=short_names
        )
        if est_df is None:
            est_df = est_data
            moe_df = moe_data
        else:
            est_df = est_df.join(est_data)
            moe_df = moe_df.join(moe_data)

    return est_df, moe_df


def cvap(
    state: utils.states.State,
    geometry: str,
    year: int,
):
    """ "
    Retrieves CVAP data for the provided state, geometry level, and year from the
    American Community Survey 5-year (ACS 5) tables.

    Args:
        state (utils.states.State):
            The state for which we're retrieving CVAP data.

        geometry (str):
            The geometry level for which we're retrieving CVAP data. Must be one of
            "state", "county", "tract", or "block group".

        year (int):
            The year of the CVAP data.

    Returns:
        pd.DataFrame
            A DataFrame containing the CVAP data for the specified state, geometry level, and year.
    """

    return acs5(state, geometry, year, tables=[CVAPTableInfo()], short_names=True)
