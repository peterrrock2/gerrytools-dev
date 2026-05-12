from dataclasses import dataclass
from itertools import product

import pandas as pd
from frozendict import frozendict

# Canonical race prefixes shared between ACS VAP/CVAP and decennial PL block
# VAP columns. Each prefix combines with a measure (e.g. ``WHITE_VAP``) and
# then a source suffix (e.g. ``WHITE_VAP_ACS5`` or ``WHITE_VAP_P3``). Keep
# this tuple in sync with the base names produced by ``VAPTableInfo``,
# ``CVAPTableInfo``, and ``PLBlockVAPTableInfo``.
RACE_PREFIXES = (
    "TOT",
    "WHITE",
    "BLACK",
    "AIAN",
    "ASIAN",
    "NHPI",
    "OTH",
    "2MORE",
    "NHWHITE",
    "HISP",
)

ACS_SOURCE_SUFFIX = "ACS5"


def append_source_suffix(name: str, source: str | None) -> str:
    """
    Appends a Census table/source suffix to a local column name.

    Examples
    --------
    ``append_source_suffix("TOT_VAP", "ACS5")`` returns ``"TOT_VAP_ACS5"``.
    ``append_source_suffix("TOT_POP", "P1")`` returns ``"TOT_POP_P1"``.
    """

    if not source:
        return name
    return f"{name}_{source}"


def decennial_pl_source_from_variable(variable: str) -> str:
    """
    Returns the decennial PL table prefix for a raw Census variable.

    Examples
    --------
    ``"P1_001N"`` maps to ``"P1"`` and ``"P3_001N"`` maps to ``"P3"``.
    """

    return variable.split("_", maxsplit=1)[0]


def _canonical_replacement(name: str) -> str:
    """
    Shortens a long, English-language ACS 5-year column name to its canonical
    short form.

    Substitutions are driven by ``ACSTableInfo.standard_abbreviations`` so
    there is a single source of truth. The order of that mapping matters:
    longer phrases must be collapsed before any shorter phrase that is a
    substring of them (e.g. ``AMERICAN_INDIAN_AND_ALASKAN_NATIVE`` before
    ``NATIVE``).

    Parameters
    ----------
    name : str
        Long column name to shorten.

    Returns
    -------
    str
        Canonical short form of ``name`` with ``_ALONE`` and ``_EST`` suffixes
        stripped.
    """

    for long, short in ACSTableInfo.standard_abbreviations.items():
        name = name.replace(long, short)
    name = name.replace("_ALONE", "")
    name = name.replace("_EST", "")
    return name


def shorten_acs5_column_names(
    df: pd.DataFrame,
    source_suffix: str | None = ACS_SOURCE_SUFFIX,
) -> None:
    """
    Shortens the long English-language ACS 5-year column names on ``df`` in
    place using ``_canonical_replacement`` and appends the source suffix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose column names will be rewritten.
    source_suffix : str, optional
        Source suffix appended to each shortened column. Defaults to
        ``"ACS5"`` so ACS-derived short columns are distinguishable from
        decennial PL columns such as ``*_P1`` or ``*_P3``.

    Warnings
    --------
    This function modifies ``df`` in place.
    """

    df.rename(
        columns={
            col: append_source_suffix(_canonical_replacement(col), source_suffix)
            for col in df.columns
        },
        inplace=True,
    )


@dataclass(frozen=True)
class DecennialPLTableInfo:
    """
    Base table definition for decennial Public Law 94-171 (PL) Census API
    tables.

    Unlike ACS table definitions, PL variables do not use estimate/MOE suffixes
    or grouped variants. A PL table definition directly maps Census variable
    names to the base local column names used downstream. Public short names
    are source-suffixed from the raw PL variable prefix, so ``P1_001N`` mapped
    to ``"TOTAL_POP"`` would become ``"TOTAL_POP_P1"``, while ``P3_001N``
    becomes ``"TOT_VAP_P3"``.

    Attributes
    ----------
    table_name : str
        Human-readable name for the logical PL table.
    variable_to_short_name : frozendict
        Mapping from raw Census variable name (e.g. ``"P3_001N"``) to the
        base local column name before source suffixing (e.g. ``"TOT_VAP"``).
    """

    table_name: str = ""
    variable_to_short_name: frozendict = frozendict()

    def construct_variable_names(self) -> tuple[str, ...]:
        """
        Returns the raw Census API variable names for this PL table.

        Returns
        -------
        tuple[str, ...]
            Census API variable names (e.g. ``"P3_001N"``) in definition order.
        """

        return tuple(self.variable_to_short_name.keys())

    def construct_short_names(self) -> tuple[str, ...]:
        """
        Returns the source-suffixed local short column names for this PL table.

        Returns
        -------
        tuple[str, ...]
            Short local column names (e.g. ``"TOT_VAP_P3"``) in definition
            order, paired 1:1 with ``construct_variable_names``.
        """

        return tuple(self.construct_rename_map().values())

    def construct_base_short_names(self) -> tuple[str, ...]:
        """
        Returns local short column names before source suffixing.
        """

        return tuple(self.variable_to_short_name.values())

    def construct_rename_map(self) -> dict[str, str]:
        """
        Returns raw Census variable to source-suffixed local column mapping.
        """

        return {
            variable: append_source_suffix(
                short_name,
                decennial_pl_source_from_variable(variable),
            )
            for variable, short_name in self.variable_to_short_name.items()
        }

    def source_name_for_short_name(self, short_name: str) -> str:
        """
        Returns the source-suffixed column name for a base local short name.
        """

        for variable, candidate in self.variable_to_short_name.items():
            if candidate == short_name:
                return append_source_suffix(
                    candidate,
                    decennial_pl_source_from_variable(variable),
                )
        raise KeyError(f"{short_name} is not defined for {self.table_name}.")

    def rename_columns(self, df: pd.DataFrame) -> None:
        """
        Renames raw Census API variables in ``df`` to local short names.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame whose columns will be renamed.

        Warnings
        --------
        This function modifies ``df`` in place.
        """

        df.rename(columns=self.construct_rename_map(), inplace=True)


@dataclass(frozen=True)
class PLBlockVAPTableInfo(DecennialPLTableInfo):
    """
    Decennial PL block-level voting-age population (VAP) table definition.

    Wraps Census PL tables P3 (VAP by race) and P4 (VAP by Hispanic origin) at
    block geography. Output column names preserve the source table suffix, so
    race-by-VAP columns from P3 become ``*_VAP_P3`` and Hispanic-origin VAP
    columns from P4 become ``*_VAP_P4``.

    Attributes
    ----------
    table_name : str
        Inherited from ``DecennialPLTableInfo``; set to ``"PLBlockVAP"``.
    race_prefixes : tuple[str, ...]
        Race prefixes this table covers, defaulting to the module-level
        ``RACE_PREFIXES``. These are the prefixes downstream block-CVAP
        estimation iterates over when pairing PL source-suffixed
        ``{RACE}_VAP`` columns with ACS source-suffixed ``{RACE}_VAP`` /
        ``{RACE}_CVAP`` columns.
    variable_to_short_name : frozendict
        Inherited from ``DecennialPLTableInfo``; maps the specific P3/P4
        variable names above to their base ``{RACE}_VAP`` short names. The
        source suffix is added by ``DecennialPLTableInfo.construct_rename_map``.
    """

    table_name: str = "PLBlockVAP"
    race_prefixes: tuple[str, ...] = RACE_PREFIXES
    variable_to_short_name: frozendict = frozendict(
        {
            "P3_001N": "TOT_VAP",
            "P3_003N": "WHITE_VAP",
            "P3_004N": "BLACK_VAP",
            "P3_005N": "AIAN_VAP",
            "P3_006N": "ASIAN_VAP",
            "P3_007N": "NHPI_VAP",
            "P3_008N": "OTH_VAP",
            "P3_009N": "2MORE_VAP",
            "P4_005N": "NHWHITE_VAP",
            "P4_002N": "HISP_VAP",
        }
    )


@dataclass(frozen=True)
class ACSTableInfo:
    """
    Base class for American Community Survey 5-year (ACS 5) table definitions.

    Subclasses specify the Census base tables, indices, and demographic groups
    that make up one logical "table" (e.g. VAP, CVAP), plus a
    ``_condense_group_dict`` describing how to collapse raw Census variables
    into named group sums. This base class then provides:

    - ``construct_long_names`` to build the Census API variable names and the
      matching descriptive column names;
    - ``condense_group_dict`` (materialized in ``__post_init__``) to drive the
      actual summation step in ``acs._condense``.

    Attributes
    ----------
    table_name : str
        Human-readable name for the logical ACS table (e.g. "CVAP", "VAP").
    base_table_strings : tuple[str, ...]
        Census base table codes used (e.g. ``("B05003",)``).
    table_indices : tuple[int, ...]
        Numeric indices within each base table that correspond to the
        variables this table cares about.
    groups_tup : tuple[str, ...]
        Single-letter Census group suffixes (e.g. ``""`` for all races,
        ``"A"`` for White alone) that this table requests.
    _condense_group_dict : frozendict
        Subclass-defined mapping
        ``{output_group_name: (tables, groups, indices)}`` describing how to
        build each condensed group from raw variables. Materialized into
        ``condense_group_dict`` in ``__post_init__``.
    index_to_name_dict : frozendict
        Subclass-defined mapping from a Census index to the descriptive
        suffix appended to the long column name (e.g. ``{8: "MALE",
        19: "FEMALE"}`` for VAP).
    condense_group_dict : frozendict
        Derived mapping ``{output_group_name: (raw_variable_names, ...)}``.
    table_to_group_dict : frozendict
        Class-level mapping of group suffix letter to long English-language
        group name (e.g. ``"A" -> "WHITE_ALONE"``).
    standard_abbreviations : frozendict
        Class-level long-to-short substring mapping consumed by
        ``shorten_acs5_column_names``.
    suffix_to_abbrev_dict : frozendict
        Class-level mapping of Census variable suffixes ("E", "M", "EA", "MA")
        to their descriptive short forms ("EST", "MOE", ...).
    """

    table_name: str = ""
    base_table_strings: tuple = tuple()
    table_indices: tuple = tuple()
    groups_tup: tuple = tuple()
    _condense_group_dict: frozendict = frozendict()
    index_to_name_dict: frozendict = frozendict()
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

    # Maps the long English-language substring to its canonical short form.
    # Order matters: longer phrases must appear before any shorter phrase that
    # is a substring of them (e.g. ``AMERICAN_INDIAN_AND_ALASKAN_NATIVE`` must
    # be collapsed to ``AIAN`` before the bare ``NATIVE -> NAT`` rule runs,
    # and the two-word races must collapse before ``MALE -> M`` eats the
    # ``MALE`` inside ``FEMALE``).
    standard_abbreviations = frozendict(
        {
            "TOTAL": "TOT",
            "AMERICAN_INDIAN_AND_ALASKAN_NATIVE": "AIAN",
            "NATIVE_HAWAIIAN_AND_OTHER_PACIFIC_ISLANDER": "NHPI",
            "OTHER_RACE": "OTH",
            "THREE_OR_MORE_RACES": "3MORE",
            "TWO_OR_MORE_RACES": "2MORE",
            "NON_HISPANIC_WHITE": "NHWHITE",
            "HISPANIC": "HISP",
            "FEMALE": "F",
            "MALE": "M",
            "NATIVE": "NAT",
            "FOREIGN": "FRN",
            "EXCLUDING": "EXC",
            "INCLUDING": "INC",
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

    def construct_long_names(
        self,
        suffix: str = "E",
        year: int | None = None,
        source_suffix: str | None = None,
    ) -> dict[str, str]:
        """
        Constructs descriptive ACS column names for this table definition.

        Parameters
        ----------
        suffix : str, optional
            Census variable suffix: "E" for estimates, "M" for margins of
            error. Defaults to "E".
        year : int, optional
            ACS year to append to the descriptive column names. If not
            provided, generated names do not include a year suffix.
        source_suffix : str, optional
            Source suffix appended to the descriptive column name, such as
            ``"ACS5"`` or ``"ACS1"``. If not provided, no source suffix is
            appended.

        Returns
        -------
        dict[str, str]
            Mapping from Census API variable names (e.g. ``"B05003_008E"``) to
            descriptive column names (e.g. ``"TOTAL_VAP_EST_MALE_ACS1"`` when
            ``source_suffix="ACS1"``).
        """

        long_name_dict = {}

        for table, group, index in product(
            self.base_table_strings, self.groups_tup, self.table_indices
        ):
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
                long_name_dict[f"{table}{group}_{index:03d}{suffix}"] += f"_{year}"
            if source_suffix is not None:
                long_name_dict[f"{table}{group}_{index:03d}{suffix}"] = append_source_suffix(
                    long_name_dict[f"{table}{group}_{index:03d}{suffix}"],
                    source_suffix,
                )

        return long_name_dict

    def __post_init__(self) -> None:
        """
        Materializes ``condense_group_dict`` from the subclass's
        ``_condense_group_dict`` after dataclass initialization.

        The subclass form specifies each group as
        ``(tables, groups, indices)``; this expands those tuples into the
        cartesian product of raw Census variable names ``{table}{group}_{index:03d}``
        that ``acs._condense`` will sum.
        """

        full_condense_dict = {}

        for key, (tables, groups, indices) in self._condense_group_dict.items():
            condense_list = []
            for table in tables:
                for group in groups:
                    for index in indices:
                        condense_list.append(f"{table}{group}_{index:03d}")
            full_condense_dict[key] = tuple(condense_list)

        object.__setattr__(self, "condense_group_dict", frozendict(full_condense_dict))


@dataclass(frozen=True)
class CVAPTableInfo(ACSTableInfo):
    """
    ACS 5-year Citizen Voting-Age Population (CVAP) table definition.

    Sources variables from Census base table B05003 and sums male/female
    native/foreign-born citizen counts (indices 9, 11, 20, 22) into per-race
    CVAP totals.

    See ``ACSTableInfo`` for the attribute reference.
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
class VAPTableInfo(ACSTableInfo):
    """
    ACS 5-year Voting-Age Population (VAP) table definition.

    Sources variables from Census base table B05003 and sums the male/female
    voting-age subtotals (indices 8 and 19) into per-race VAP totals.

    See ``ACSTableInfo`` for the attribute reference.
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
class HispByRaceTableInfo(ACSTableInfo):
    """
    ACS 5-year Hispanic-by-Race table definition.

    Sources variables from Census base table B03002, which cross-tabulates
    Hispanic origin with race. Condenses into both ``*_NHISP`` (non-Hispanic)
    and ``*_HISP`` (Hispanic) groups for each race category.

    See ``ACSTableInfo`` for the attribute reference.
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
class TotPopTableInfo(ACSTableInfo):
    """
    ACS 5-year total-population table definition.

    Pulls the total-population row (index 1) from Census base table B01001 into
    a single ``TOTAL_POP`` condensed group.

    See ``ACSTableInfo`` for the attribute reference.
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
