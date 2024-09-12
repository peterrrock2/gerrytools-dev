from dataclasses import dataclass
from frozendict import frozendict


def _canonical_replacement(name):
    """
    A helper function for canonically shortening the long, English-language descriptions of ACS
    5-year column names in a DataFrame.
    """

    name = name.replace("TOTAL", "TOT")
    name = name.replace("AMERICAN_INDIAN_AND_ALASKAN_NATIVE", "AIAN")
    name = name.replace("NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER", "NHPI")
    name = name.replace("OTHER_RACE", "OTH")
    name = name.replace("TWO_OR_MORE_RACES", "2MORE")
    name = name.replace("NON_HISPANIC_WHITE", "NHWHITE")
    name = name.replace("HISPANIC", "HISP")
    name = name.replace("_ALONE", "")
    name = name.replace("NATIVE", "NAT")
    name = name.replace("FOREIGN", "FRN")
    name = name.replace("FEMALE", "F")
    name = name.replace("MALE", "M")
    name = name.replace("_EST", "")

    return name


def shorten_acs5_column_names(df, year: int | None = None):
    """
    A function for shortening the long, English-language descriptions of ACS 5-year
    column names in a DataFrame. This function makes the following replacements:

    'TOTAL' -> 'TOT'
    'AMERICAN_INDIAN_AND_ALASKAN_NATIVE' -> 'AIAN'
    'NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER' -> 'NHPI'
    'OTHER_RACE' -> 'OTH'
    'TWO_OR_MORE_RACES' -> '2MORE'
    'NON_HISPANIC_WHITE' -> 'NHWHITE'
    'HISPANIC' -> 'HISP'
    '_ALONE' -> ''
    'NATIVE' -> 'NAT'
    'FOREIGN' -> 'FRN'
    'FEMALE' -> 'F'
    'MALE' -> 'M'
    '_EST' -> ''

    Args:
        df (DataFrame):
            The DataFrame containing the ACS 5-year data.

        year (int, optional):
            The year of the ACS 5-year data. If provided, the function will append the year to the
            end of the column names.


    Warnings:
        This function modifies the input DataFrame in place.
    """

    new_col_names = {}
    suffix = ""
    if year is not None:
        suffix = f"{str(year)[-2:]}"

    for col in df.columns:
        new_col_names[col + suffix] = _canonical_replacement(col)

    df.rename(columns=new_col_names, inplace=True)

    return df


@dataclass(frozen=True)
class ACS5TableInfo:
    """
    An abstract class for the American Community Survey 5-year (ACS 5) tables.

    This class provides mechanisms for handling ACS 5-year data tables, including
    mapping table codes to population groups (such as race and ethnicity), constructing
    long variable names for data queries, and supporting the condensation of table
    groups for specific population subsets.

    Attributes:
        table_name (str):
            The Engish name we will give the ACS table (e.g., "B01001" may be called
            "TotPopByAgeSex").

        base_table_strings (tuple):
            A tuple of base table codes (e.g., "B01001", "B02001") to define which ACS tables are
            used.

        table_indices (tuple):
            A tuple of table indices, representing specific variables within the table (e.g., 1, 2).

        groups_tup (tuple):
            A tuple of population groups or race/ethnicity codes (e.g., "A" for White alone, "B"
            for Black alone).

        condense_group_dict (frozendict):
            A frozendict mapping population subsets (like specific racial/ethnic groups) to their
            respective tables, groups, and indices to support data condensation for focused analysis.

        table_to_group_dict (frozendict):
            A frozendict that maps single-letter group identifiers (e.g., "A", "B") to more
            descriptive population group names (e.g., "WHITE_ALONE", "BLACK_ALONE").

        standard_abbreviations (frozendict):
            A frozendict mapping full population group or table names to standardized abbreviations
            (e.g., "HISPANIC" to "HISP", "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER" to "NHPI").

        suffix_to_abbrev_dict (frozendict):
            A frozendict mapping suffixes like "E" (Estimate) or "M" (Margin of Error) to their
            corresponding abbreviations.

    Methods:
        construct_long_names(suffix: str = "E", year: int | None = None):
            Constructs long variable names for querying ACS 5-year data, including table name,
            population group, and year, along with the appropriate suffix (e.g., "EST" for estimate,
            "MOE" for margin of error).
    """

    table_name: str = ""
    base_table_strings: tuple = tuple()
    table_indices: tuple = tuple()
    groups_tup: tuple = tuple()
    condense_group_dict: frozendict = frozendict()

    table_to_group_dict = frozendict(
        {
            "": "TOTAL",
            "A": "WHITE_ALONE",
            "B": "BLACK_ALONE",
            "C": "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_ALONE",
            "D": "ASIAN_ALONE",
            "E": "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_ALONE",
            "F": "OTHER_RACE_ALONE",
            "G": "TWO_OR_MORE_RACES",
            "H": "NON_HISPANIC_WHITE",
            "I": "HISPANIC",
        }
    )

    # The order of these abbreviations is important; they are used to
    # for string substitution and some strings are substrings of other
    # strings earlier in the list.
    standard_abbreviations = frozendict(
        {
            "TOT": "TOTAL",
            "AMERICAN_INDIAN_AND_ALASKAN_NATIVE": "AIAN",
            "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER": "NHPI",
            "OTHER_RACE": "OTH",
            "TWO_OR_MORE_RACES": "2MORE",
            "NON_HISPANIC_WHITE": "NHWHITE",
            "HISPANIC": "HISP",
            "NATIVE": "NAT",
            "FOREIGN": "FRN",
            "FEMALE": "F",
            "MALE": "M",
            "EXCLUDING": "EXC",
            "INCLUDING": "INC",
            "THREE_OR_MORE_RACES": "3MORE",
        }
    )

    suffix_to_abbrev_dict = frozendict(
        {
            "E": "EST",
            "M": "MOE",
            "EA": "EST_ANNOTATION",
            "MA": "MOE_ANNOTATION",
        }
    )

    condense_group_dict = frozendict()

    def construct_long_names(self, suffix: str = "E", year: int | None = None):
        long_name_dict = {}
        for table in self.base_table_strings:
            for group in self.groups_tup:
                for index in self.table_indices:
                    long_name_dict[f"{table}{group}_{index:03d}{suffix}"] = (
                        f"{self.table_to_group_dict[group]}_"
                        f"{self.table_name}_"
                        f"{self.suffix_to_abbrev_dict[suffix]}"
                    )
                    if self.index_to_name_dict[index] != "":
                        long_name_dict[
                            f"{table}{group}_{index:03d}{suffix}"
                        ] += f"_{self.index_to_name_dict[index]}"
                    if year is not None:
                        long_name_dict[
                            f"{table}{group}_{index:03d}{suffix}"
                        ] += f"_{year}"

        return long_name_dict

    def __post_init__(self):
        full_condense_dict = {}

        if self._condense_group_dict is None:
            object.__setattr__(self, "condense_group_dict", frozendict())

        for key, (tables, groups, indices) in self._condense_group_dict.items():
            condense_list = []
            for table in tables:
                for group in groups:
                    for index in indices:
                        condense_list.append(f"{table}{group}_{index:03d}")
            full_condense_dict[key] = tuple(condense_list)

        object.__setattr__(self, "condense_group_dict", frozendict(full_condense_dict))


@dataclass(frozen=True)
class CVAPTableInfo(ACS5TableInfo):
    """
    A dataclass containing the appropriate information for querying Citizen Voting Age Population
    (CVAP) data from the American Community Survey 5-year (ACS 5) tables.

    See the `ACS5TableInfo` class for more details.
    """

    table_name: str = "CVAP"
    base_table_strings: tuple = ("B05003",)
    groups_tup: tuple = ("", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    table_indices: tuple = (9, 11, 20, 22)

    # new_name: (table_tup, group_tup, index_tup)
    _condense_group_dict: frozendict = frozendict(
        {
            "TOTAL_CVAP": (("B05003",), ("",), (9, 11, 20, 22)),
            "WHITE_ALONE_CVAP": (("B05003",), ("A",), (9, 11, 20, 22)),
            "BLACK_ALONE_CVAP": (("B05003",), ("B",), (9, 11, 20, 22)),
            "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_ALONE_CVAP": (
                ("B05003",),
                ("C",),
                (9, 11, 20, 22),
            ),
            "ASIAN_ALONE_CVAP": (("B05003",), ("D",), (9, 11, 20, 22)),
            "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_ALONE_CVAP": (
                ("B05003",),
                ("E",),
                (9, 11, 20, 22),
            ),
            "OTHER_RACE_ALONE_CVAP": (("B05003",), ("F",), (9, 11, 20, 22)),
            "TWO_OR_MORE_RACES_CVAP": (("B05003",), ("G",), (9, 11, 20, 22)),
            "NON_HISPANIC_WHITE_CVAP": (("B05003",), ("H",), (9, 11, 20, 22)),
            "HISPANIC_CVAP": (("B05003",), ("I",), (9, 11, 20, 22)),
        }
    )

    index_to_name_dict: frozendict = frozendict(
        {
            9: "MALE_NATIVE",
            11: "MALE_FOREIGN",
            20: "FEMALE_NATIVE",
            22: "FEMALE_FOREIGN",
        }
    )


@dataclass(frozen=True)
class VAPTableInfo(ACS5TableInfo):
    """
    A dataclass containing the appropriate information for querying Voting Age Population
    (VAP) data from the American Community Survey 5-year (ACS 5) tables.

    See the `ACS5TableInfo` class for more details.
    """

    table_name: str = "VAP"
    base_table_strings: tuple = ("B05003",)
    table_indices: tuple = (8, 19)
    groups_tup: tuple = ("", "A", "B", "C", "D", "E", "F", "G", "H", "I")

    # new_name: (table_tup, group_tup, index_tup)
    _condense_group_dict: frozendict = frozendict(
        {
            "TOTAL_VAP": (("B05003",), ("",), (8, 19)),
            "WHITE_VAP": (("B05003",), ("A",), (8, 19)),
            "BLACK_VAP": (("B05003",), ("B",), (8, 19)),
            "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_VAP": (
                ("B05003",),
                ("C",),
                (8, 19),
            ),
            "ASIAN_VAP": (("B05003",), ("D",), (8, 19)),
            "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_VAP": (
                ("B05003",),
                ("E",),
                (8, 19),
            ),
            "OTHER_RACE_VAP": (("B05003",), ("F",), (8, 19)),
            "TWO_OR_MORE_RACES_VAP": (("B05003",), ("G",), (8, 19)),
            "NON_HISPANIC_WHITE_VAP": (("B05003",), ("H",), (8, 19)),
            "HISPANIC_VAP": (("B05003",), ("I",), (8, 19)),
        }
    )

    index_to_name_dict: frozendict = frozendict(
        {
            8: "MALE",
            19: "FEMALE",
        }
    )


@dataclass(frozen=True)
class HispByRaceTableInfo(ACS5TableInfo):
    """
    A dataclass containing the appropriate information for querying Hispanic by Race
    population data from the American Community Survey 5-year (ACS 5) tables.

    See the `ACS5TableInfo` class for more details.
    """

    table_name: str = "HispByRace"
    base_table_strings: tuple = ("B03002",)
    table_indices: tuple = tuple(list(range(2, 22)))
    groups_tup: tuple = ("",)

    _condense_group_dict: frozendict = frozendict(
        {
            "TOTAL_NHISP": (("B03002",), ("",), (2,)),
            "WHITE_NHISP": (("B03002",), ("",), (3,)),
            "BLACK_NHISP": (("B03002",), ("",), (4,)),
            "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_NHISP": (("B03002",), ("",), (5,)),
            "ASIAN_NHISP": (("B03002",), ("",), (6,)),
            "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_NHISP": (
                ("B03002",),
                ("",),
                (7,),
            ),
            "OTHER_RACE_NHISP": (("B03002",), ("",), (8,)),
            "TWO_OR_MORE_RACES_NHISP": (("B03002",), ("",), (9,)),
            "TWO_OR_MORE_RACES_INCLUDING_OTHER_RACE_NHISP": (("B03002",), ("",), (10,)),
            "TWO_OR_MORE_RACES_EXCLUDING_OTHER_RACE_INCLUDING_THREE_OR_MORE_RACES_NHISP": (
                ("B03002",),
                ("",),
                (11,),
            ),
            "TOTAL_HISP": (("B03002",), ("",), (12,)),
            "WHITE_HISP": (("B03002",), ("",), (13,)),
            "BLACK_HISP": (("B03002",), ("",), (14,)),
            "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_HISP": (("B03002",), ("",), (15,)),
            "ASIAN_HISP": (("B03002",), ("",), (16,)),
            "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_HISP": (
                ("B03002",),
                ("",),
                (17,),
            ),
            "OTHER_RACE_HISP": (("B03002",), ("",), (18,)),
            "TWO_OR_MORE_RACES_HISP": (("B03002",), ("",), (19,)),
            "TWO_OR_MORE_RACES_INCLUDING_OTHER_RACE_HISP": (("B03002",), ("",), (20,)),
            "TWO_OR_MORE_RACES_EXCLUDING_OTHER_RACE_INCLUDING_THREE_OR_MORE_RACES_HISP": (
                ("B03002",),
                ("",),
                (21,),
            ),
        }
    )

    index_to_name_dict: frozendict = frozendict(
        {
            2: "TOTAL_NHISP",
            3: "WHITE_NHISP",
            4: "BLACK_NHISP",
            5: "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_NHISP",
            6: "ASIAN_NHISP",
            7: "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_NHISP",
            8: "OTHER_RACE_NHISP",
            9: "TWO_OR_MORE_RACES_NHISP",
            10: "TWO_OR_MORE_RACES_INCLUDING_OTHER_RACE_NHISP",
            11: "TWO_OR_MORE_RACES_EXCLUDING_OTHER_RACE_INCLUDING_THREE_OR_MORE_RACES_NHISP",
            12: "TOTAL_HISP",
            13: "WHITE_HISP",
            14: "BLACK_HISP",
            15: "AMERICAN_INDIAN_AND_ALASKAN_NATIVE_HISP",
            16: "ASIAN_HISP",
            17: "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER_HISP",
            18: "OTHER_RACE_HISP",
            19: "TWO_OR_MORE_RACES_HISP",
            20: "TWO_OR_MORE_RACES_INCLUDING_OTHER_RACE_HISP",
            21: "TWO_OR_MORE_RACES_EXCLUDING_OTHER_RACE_INCLUDING_THREE_OR_MORE_RACES_HISP",
        }
    )


@dataclass(frozen=True)
class TotPopTableInfo(ACS5TableInfo):
    """
    A dataclass containing the appropriate information for querying Total Population
    data from the American Community Survey 5-year (ACS 5) tables.

    See the `ACS5TableInfo` class for more details.
    """

    table_name: str = "TotPop"
    base_table_strings: tuple = ("B01001",)
    groups_tup: tuple = ("",)
    table_indices: tuple = (1,)

    _condense_group_dict: frozendict = frozendict(
        {
            "TOTAL_POP": (("B01001",), ("",), (1,)),
        }
    )

    index_to_name_dict: frozendict = frozendict(
        {
            1: "",
        }
    )
