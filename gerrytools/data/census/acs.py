import io
from urllib.request import urlopen
from zipfile import ZipFile

import us
import pandas as pd
from typing import cast
import requests
from gerrytools.logging import get_logger

from dataclasses import dataclass

logger = get_logger(__name__)


def cvap(
    state: us.states.State,
    geometry: str = "tract",
    year: int = 2020,
) -> pd.DataFrame:
    """
    Retrieves and CSV-formats 5-year CVAP data for the provided state at
    the specified geometry level.

    Variables and descriptions are
    [listed here](https://www2.census.gov/programs-surveys/decennial/rdo/technical-documentation/special-tabulation/CVAP_2015-2019_ACS_documentation.pdf).

    Args:
        state (us.State): The `State` object for which we're retrieving 2019
            ACS CVAP Special Tab.
        geometry (str, optional): Level of geometry for which we're getting
            data. Accepted values are `"block group"` for 2010 Census Block
            Groups, and `"tract"` for 2010 Census Tracts. Defaults to `"tract"`.
        year (int, optional): Year for which data is retrieved. Defaults to
            2020.

    Returns
        A `DataFrame` with a `GEOID` column and corresponding CVAP columns from
        the ACS CVAP Special Tab for the specified year. Also includes columns containing the
        error reported by the census and the estimated citizen population.
    """
    raw: pd.DataFrame = _raw(geometry, year)
    raw.rename(columns={col: col.lower() for col in raw.columns}, inplace=True)
    raw["geoid"] = raw["geoid"].astype(str).str.split("US").str[-1]
    raw.drop(columns=["geoname"], inplace=True)
    raw = cast(
        pd.DataFrame,
        raw[raw["geoid"].str.startswith(state.fips)].reset_index(drop=True),
    )

    yearsuffix = str(year)[-2:]
    new_pop_col_names = {
        "cit_est": f"cpop_{yearsuffix}",
        "cit_moe": f"cpop_{yearsuffix}_err",
        "cvap_est": f"cvap_{yearsuffix}",
        "cvap_moe": f"cvap_{yearsuffix}_err",
    }

    raw.rename(columns=new_pop_col_names, inplace=True)

    new_titles = {
        "Total": "total",
        "Not Hispanic or Latino": "non_hispanic",
        "American Indian or Alaska Native Alone": "amin",
        "Asian Alone": "asian",
        "Black or African American Alone": "black",
        "Native Hawaiian or Other Pacific Islander Alone": "nhpi",
        "White Alone": "white",
        "American Indian or Alaska Native and White": "white_amin",
        "Asian and White": "white_asian",
        "Black or African American and White": "white_black",
        "American Indian or Alaska Native and Black or African American": "black_amin",
        "Remainder of Two or More Race Responses": "two_or_more_remaining",
        "Hispanic or Latino": "hispanic",
    }
    raw["lntitle"] = raw["lntitle"].map(lambda x: new_titles[x])

    wide = raw.pivot(
        index=["geoid"],  # one row per geoid (and geoname)
        columns="lntitle",  # each lntitle becomes a group of columns
        values=new_pop_col_names.values(),
    )
    # Flatten the MultiIndex columns: (measure, lntitle) -> "lntitle_measure"
    wide.columns = [
        f"{lntitle}_{measure}" for measure, lntitle in wide.columns.to_flat_index()
    ]

    # Bring geoid back as normal column
    return wide.reset_index()


@dataclass
class ACS5Variables:

    def __init__(self, year: int):
        if year < 2009 or year > 2022:
            raise ValueError("ACS5Variables only supports years 2009-2022.")

        self.year_suffix = str(year)[-2:]

    @property
    def pop_columns(self) -> dict[str, list[str]]:
        return {
            f"total_pop_{self.year_suffix}": ["B01001_001E"],
            f"non_hispanic_pop_{self.year_suffix}": ["B03002_002E"],
            f"non_hispanic_white_pop_{self.year_suffix}": ["B03002_003E"],
            f"non_hispanic_black_pop_{self.year_suffix}": ["B03002_004E"],
            f"non_hispanic_amin_pop_{self.year_suffix}": ["B03002_005E"],
            f"non_hispanic_asian_pop_{self.year_suffix}": ["B03002_006E"],
            f"non_hispanic_nhpi_pop_{self.year_suffix}": ["B03002_007E"],
            f"non_hispanic_other_pop_{self.year_suffix}": ["B03002_008E"],
            f"non_hispanic_two_or_more_races_pop_{self.year_suffix}": ["B03002_009E"],
        }

    @property
    def vap_columns(self) -> dict[str, list[str]]:
        """Returns a dictionary mapping voting-age population name to columns in ACS5.

        The columns returned here are mainly "<race> (alone)" voting-age populations,
        but do not generally make a distinction between Hispanic and non-Hispanic populations.
        """
        vapnames = [
            f"white_vap_{self.year_suffix}",
            f"black_vap_{self.year_suffix}",
            f"amin_vap_{self.year_suffix}",
            f"asian_vap_{self.year_suffix}",
            f"nhpi_vap_{self.year_suffix}",
            f"other_vap_{self.year_suffix}",
            f"two_or_more_races_vap_{self.year_suffix}",
            f"non_hispanic_white_vap_{self.year_suffix}",
            f"hispanic_vap_{self.year_suffix}",
        ]
        vaptables = list(
            zip(
                vapnames,
                ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            )
        )
        return {
            column: _variables(f"B01001{table}", 7, 16)
            + _variables(f"B01001{table}", 22, 31)
            for column, table in vaptables
        }

    @property
    def cvap_columns(self) -> dict[str, list[str]]:
        cvapnames = [
            "white_cvap",
            "black_cvap",
            "amin_cvap",
            "asian_cvap",
            "nhpi_cvap",
            "other_cvap",
            "two_or_more_races_cvap",
            "non_hispanic_white_cvap",
            "hispanic_cvap",
        ]
        cvaptables = list(
            zip(
                cvapnames,
                ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            )
        )

        return {
            column: _variables(f"B05003{table}", 9, 9)  # native male
            + _variables(f"B05003{table}", 11, 11)  # foreign-born naturalized male
            + _variables(f"B05003{table}", 20, 20)  # native female
            + _variables(f"B05003{table}", 22, 22)  # foreign-born naturalized female
            for column, table in cvaptables
        }


def _retrieve_acs5_grouped_columns(
    state: us.states.State,
    geometry: str,
    variable_dict: dict[str, list[str]],
    url: str,
    retry_count: int = 3,
    timeout_seconds: int = 300,
) -> pd.DataFrame:
    response = None

    df_list = []
    group_df = None
    logger.debug(f"Retrieving ACS5 data for state {state.name} ({state.fips}).")
    for group_name, group_cols in variable_dict.items():
        attempts = 0

        while attempts < retry_count:
            try:
                full_url = url.format(
                    column_list=",".join(["GEO_ID"] + group_cols),
                    geometry_resolution=geometry.replace(" ", "%20"),
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
            logger.debug(
                f"Bad Response for {group_name}. Found message: {response.content}"
            )
            raise ValueError(
                f"Failed to retrieve data; status code {response.status_code}."
            )

        group_df = pd.DataFrame(response.json()[1:], columns=response.json()[0])
        group_df.set_index("GEO_ID", inplace=True)
        group_df = group_df.astype(int).sum(axis=1).to_frame(name=group_name)
        df_list.append(group_df)

    full_df = pd.concat(df_list, axis=1).reset_index()
    full_df["geo_id"] = full_df["GEO_ID"].astype(str).str.split("US").str[-1]
    full_df.drop(columns=["GEO_ID"], inplace=True)

    if geometry == "state":
        full_df = full_df.query(f"geo_id == '{str(state.fips).zfill(2)}'")

    return full_df


def acs5(
    state: us.states.State,
    geometry: str = "tract",
    year: int = 2020,
    include_pop_columns: bool = False,
    include_vap_columns: bool = False,
    include_cvap_columns: bool = True,
    extra_columns: dict[str, list[str]] = {},
    census_api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieves ACS 5-year estimates for the provided state, geometry level, and year. A

    Args:
        state (us.states.State): `State` object for the desired state.
        geometry (str): Geometry level at which data is retrieved.
            Acceptable values are `"tract"` and `"block group"`. Defaults to
            `"tract"`, so data is retrieved at the 2020 Census tract level.
            Defaults to "tract".
        year (int): Year for which data is retrieved. Defaults to 2020.

    Returns:
        A DataFrame containing the formatted data.
    """
    variables = ACS5Variables(year)

    URL_BASE = "https://api.census.gov/data/2020/acs/acs5?get={column_list}&for={geometry_resolution}:*"

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

    if not any(
        [
            include_pop_columns,
            include_vap_columns,
            include_cvap_columns,
            len(extra_columns) > 0,
        ]
    ):
        raise ValueError("At least one set of columns must be included.")

    final_grouped_dict = {}

    if include_pop_columns:
        final_grouped_dict.update(variables.pop_columns)

    if include_vap_columns:
        final_grouped_dict.update(variables.vap_columns)

    if include_cvap_columns:
        final_grouped_dict.update(variables.cvap_columns)

    if len(extra_columns) > 0:
        final_grouped_dict.update(extra_columns)

    return _retrieve_acs5_grouped_columns(
        state,
        geometry,
        final_grouped_dict,
        URL_BASE,
    ).set_index("geo_id")


def _variables(prefix: str, start: int, stop: int, suffix: str = "E") -> list:
    """Returns a set of ACS5 variable names.

    Variable names are determined by the provided prefix, start, stop, and suffix parameters.
    Used to generate batches of names, especially for things like voting-age population.
    Variable names are formatted like `<prefix>_<number identifier><suffix>`, where
    `<prefix>` is a population grouping, `<number identifier>` is the number of the variable
    in that grouping, and `<suffix>` designates the file used.
    [Variables are listed here](https://api.census.gov/data/2019/acs/acs5/variables.html).

    Args:
        prefix (str): Population grouping; typically "B01001." These prefixes
            change based on subpopulation: for example, the prefix for Black
            age-by-sex tables is "B01001B"; for Hispanic and Latino, it is
            "B01001I."
        start (int): Where to start numbering.
        stop (int): Where to stop numbering. Inclusive.
        suffix (str): Suffix designating the file. For most purposes, this is "E".
            Defaults to "E".

    Returns:
        A list of ACS5 variable names.
    """
    return [f"{prefix}_{str(t).zfill(3)}{suffix}" for t in range(start, stop + 1)]


def _retrieve(year: int, geometry: str = "tract"):
    """Downloads and extracts compressed CVAP data for the specified year.

    Args:
        year (int): Year for which we're grabbing CVAP data.
        geometry (str): Geometry level for which we're grabbing CVAP data. Defaults to `"tract"`.

    Returns:
        In-memory text stream of decompressed CSV data.
    """
    # Create a mapping from geometry names to filenames.
    levels = {
        "block group": "BlockGr.csv",
        "tract": "Tract.csv",
        "county": "County.csv",
        "state": "State.csv",
        "place": "Place.csv",
    }

    if geometry not in levels:
        raise ValueError(
            f'Geometry "{geometry}" not recognized; '
            f"allowed values are {list(levels.keys())}."
        )

    # Construct the URL.
    start, stop = year - 4, year
    root = "https://www2.census.gov/programs-surveys/decennial/rdo/datasets/"
    suffix = f"{stop}/{stop}-cvap/CVAP_{start}-{stop}_ACS_csv_files.zip"

    logger.debug(f"Retrieving CVAP data from {root + suffix}.")
    # Make the request and extract only the required files.
    with urlopen(root + suffix) as resource:
        with ZipFile(io.BytesIO(resource.read())) as archive:
            for file in archive.namelist():
                for v in levels.values():
                    if v in file:
                        return archive.read(file).decode(encoding="ISO-8859-1")

    raise ValueError(f"Could not find data for geometry '{geometry}'.")


def _raw(geometry: str, year: int) -> pd.DataFrame:
    """Reads raw CVAP data from the local repository.

    Args:
        geometry (str): Level of geometry for which we're getting 2019 CVAP
            data.
        year (int): Year for which data is retrieved.

    Returns:
        A DataFrame, where each block of 13 rows corresponds to an individual
        geometric unit (2010 Census Block Group, 2010 Census Tract) and
        each row in a given block corresponds to a CVAP statistic for that
        block's geometric unit.

    """
    # Retrieve the data at the specified geometry level and return
    # it as a dataframe.
    return pd.read_csv(io.StringIO(_retrieve(year, geometry)), encoding="ISO-8859-1")
