from dataclasses import dataclass, field
from itertools import product
from typing import ClassVar

import pandas as pd
from frozendict import frozendict

# Canonical race prefixes shared between ACS VAP/CVAP and decennial PL block
# VAP columns. Each prefix combines with a measure (e.g. ``WHITE_VAP``) and
# then a source suffix (e.g. ``WHITE_VAP_ACS5`` or ``WHITE_VAP_P3``). Keep
# this tuple in sync with the base names produced by ``ACSVAPTableInfo``,
# ``ACSCVAPTableInfo``, and ``PLBlockVAPTableInfo``.
RACE_PREFIXES: tuple[str, ...] = (
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
    """Append a Census table/source suffix to a local column name.

    Args:
        name (str): Base local column name.
        source (str | None): Source suffix (e.g. ``"ACS5"``, ``"P1"``). When
            falsy, ``name`` is returned unchanged.

    Returns:
        str: ``"{name}_{source}"`` when ``source`` is truthy, otherwise ``name``.

    Examples:
        ``append_source_suffix("TOT_VAP", "ACS5")`` returns ``"TOT_VAP_ACS5"``.
        ``append_source_suffix("TOT_POP", "P1")`` returns ``"TOT_POP_P1"``.
    """

    if not source:
        return name
    return f"{name}_{source}"


def decennial_pl_source_from_variable(variable: str) -> str:
    """Return the decennial PL table prefix for a raw Census variable.

    Args:
        variable (str): Raw Census API variable name (e.g. ``"P1_001N"``).

    Returns:
        str: The table prefix preceding the first underscore (e.g. ``"P1"``).

    Examples:
        ``"P1_001N"`` maps to ``"P1"`` and ``"P3_001N"`` maps to ``"P3"``.
    """

    return variable.split("_", maxsplit=1)[0]


def _canonical_replacement(name: str) -> str:
    """Shorten a long English-language ACS column name to its canonical form.

    Substitutions are driven by ``ACSTableInfo.standard_abbreviations`` so
    there is a single source of truth. The order of that mapping matters:
    longer phrases must be collapsed before any shorter phrase that is a
    substring of them (e.g. ``AMERICAN_INDIAN_AND_ALASKAN_NATIVE`` before
    ``NATIVE``).

    Args:
        name (str): Long column name to shorten.

    Returns:
        str: Canonical short form of ``name`` with ``_ALONE`` and ``_EST``
        suffixes stripped.
    """

    for long, short in ACSTableInfo.standard_abbreviations.items():
        name = name.replace(long, short)
    name = name.replace("_ALONE", "")
    name = name.replace("_EST", "")
    return name


def shorten_acs_column_names(
    df: pd.DataFrame,
    source_suffix: str | None = ACS_SOURCE_SUFFIX,
) -> None:
    """Shorten ACS column names on ``df`` in place and append a source suffix.

    Each column is run through ``_canonical_replacement`` and then
    ``append_source_suffix`` with ``source_suffix``.

    Args:
        df (pd.DataFrame): DataFrame whose column names will be rewritten in place.
        source_suffix (str | None): Source suffix appended to each shortened
            column. Defaults to ``"ACS5"`` so ACS-derived short columns are
            distinguishable from decennial PL columns such as ``*_P1`` or
            ``*_P3``.

    Warning:
        Modifies ``df`` in place.
    """

    df.rename(
        columns={
            col: append_source_suffix(_canonical_replacement(col), source_suffix)
            for col in df.columns
        },
        inplace=True,
    )


PL_POP_TABLES: tuple[str, ...] = ("P1", "P2", "P3", "P4")
PL_POP_YEARS: tuple[int, ...] = (2010, 2020)


def pl_pop_table(table: str, year: int) -> "PLTableInfo":
    """Build a PLTableInfo for one of the decennial PL P1/P2/P3/P4 pop tables.

    Wraps the legacy column-alias data in ``ptable_column_aliases.py`` so
    that decennial pop tables can be passed to the same fetch path as any
    other ``PLTableInfo``.

    Args:
        table (str): One of ``"P1"``, ``"P2"``, ``"P3"``, ``"P4"``.
        year (int): ``2010`` or ``2020``.

    Returns:
        PLTableInfo: Table info whose ``variable_to_short_name`` carries
        every Census variable form present in the alias dict for the given
        ``(year, table)``. For 2010 this includes both the zero-padded
        (``P0010001``) and short (``P001001``) forms so renames work
        whichever the API returns.

    Raises:
        ValueError: If ``table`` is not one of ``"P1"``–``"P4"`` or ``year``
            is not ``2010`` or ``2020``.
    """

    if year not in PL_POP_YEARS:
        raise ValueError(f"PL year must be one of {PL_POP_YEARS}; got {year}.")
    if table not in PL_POP_TABLES:
        raise ValueError(f"PL table must be one of {PL_POP_TABLES}; got {table!r}.")

    from .ptable_column_aliases import COLUMN_ALIASES_PTABLES

    raw = COLUMN_ALIASES_PTABLES[year][table]
    variable_to_short_name = frozendict({k.upper(): v for k, v in raw.items()})
    return PLTableInfo(
        table_name=f"{table}_{year}",
        variable_to_short_name=variable_to_short_name,
    )


@dataclass(frozen=True)
class PLTableInfo:
    """Base table definition for decennial Public Law 94-171 (PL) Census API tables.

    Unlike ACS table definitions, PL variables do not use estimate/MOE suffixes
    or grouped variants. A PL table definition directly maps Census variable
    names to the base local column names used downstream. Public short names
    are source-suffixed from the raw PL variable prefix, so ``P1_001N`` mapped
    to ``"TOTAL_POP"`` would become ``"TOTAL_POP_P1"``, while ``P3_001N``
    becomes ``"TOT_VAP_P3"``.

    Attributes:
        table_name (str): Human-readable name for the logical PL table.
        variable_to_short_name (frozendict): Mapping from raw Census variable
            name (e.g. ``"P3_001N"``) to the base local column name before
            source suffixing (e.g. ``"TOT_VAP"``).
    """

    table_name: str = ""
    variable_to_short_name: frozendict = field(default_factory=frozendict)

    def construct_variable_names(self) -> tuple[str, ...]:
        """Return the raw Census API variable names for this PL table.

        Returns:
            tuple[str, ...]: Census API variable names in definition order.
        """

        return tuple(self.variable_to_short_name.keys())

    def construct_short_names(self) -> tuple[str, ...]:
        """Return the source-suffixed local short column names for this PL table.

        Returns:
            tuple[str, ...]: Source-suffixed local column names (e.g.
            ``"TOT_VAP_P3"``) paired 1:1 with ``construct_variable_names``.
        """

        return tuple(self.construct_rename_map().values())

    def construct_base_short_names(self) -> tuple[str, ...]:
        """Return local short column names before source suffixing.

        Returns:
            tuple[str, ...]: Base local column names (e.g. ``"TOT_VAP"``)
            paired 1:1 with ``construct_variable_names``.
        """

        return tuple(self.variable_to_short_name.values())

    def construct_rename_map(self) -> dict[str, str]:
        """Return the raw-variable-to-source-suffixed-column rename mapping.

        Returns:
            dict[str, str]: Mapping suitable for ``DataFrame.rename`` from raw
            Census variable names to source-suffixed local column names.
        """

        return {
            variable: append_source_suffix(
                short_name,
                decennial_pl_source_from_variable(variable),
            )
            for variable, short_name in self.variable_to_short_name.items()
        }

    def _short_name_reverse_map(self) -> dict[str, str]:
        return {
            short_name: append_source_suffix(
                short_name,
                decennial_pl_source_from_variable(variable),
            )
            for variable, short_name in self.variable_to_short_name.items()
        }

    def source_name_for_short_name(self, short_name: str) -> str:
        """Return the source-suffixed column name for a base local short name.

        Args:
            short_name (str): Base local short name (e.g. ``"WHITE_VAP"``).

        Returns:
            str: Source-suffixed column name (e.g. ``"WHITE_VAP_P3"``).

        Raises:
            KeyError: If ``short_name`` is not defined for this table.
        """

        reverse_map = self._short_name_reverse_map()
        if short_name not in reverse_map:
            raise KeyError(f"{short_name} is not defined for {self.table_name}.")
        return reverse_map[short_name]

    def rename_columns(self, df: pd.DataFrame) -> None:
        """Rename raw Census API variables in ``df`` to local short names.

        Args:
            df (pd.DataFrame): DataFrame whose columns will be renamed in place.

        Warning:
            Modifies ``df`` in place.
        """

        df.rename(columns=self.construct_rename_map(), inplace=True)


@dataclass(frozen=True)
class PLBlockVAPTableInfo(PLTableInfo):
    """Decennial PL block-level voting-age population (VAP) table definition.

    Wraps Census PL tables P3 (VAP by race) and P4 (VAP by Hispanic origin) at
    block geography. Output column names preserve the source table suffix, so
    race-by-VAP columns from P3 become ``*_VAP_P3`` and Hispanic-origin VAP
    columns from P4 become ``*_VAP_P4``.

    Attributes:
        table_name (str): ``"PLBlockVAP"`` by default.
        race_prefixes (tuple[str, ...]): Race prefixes this table covers,
            defaulting to ``RACE_PREFIXES``. Block-CVAP estimation iterates
            over these when pairing PL source-suffixed ``{RACE}_VAP`` columns
            with ACS source-suffixed ``{RACE}_VAP`` and ``{RACE}_CVAP`` columns.
        variable_to_short_name (frozendict): Maps the specific P3/P4 variable
            names above to their base ``{RACE}_VAP`` short names. The source
            suffix is added by ``PLTableInfo.construct_rename_map``.
    """

    table_name: str = "PLBlockVAP"
    race_prefixes: tuple[str, ...] = RACE_PREFIXES
    variable_to_short_name: frozendict = field(
        default_factory=lambda: frozendict(
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
    )


@dataclass(frozen=True)
class ACSTableInfo:
    """Base class for American Community Survey (ACS) table definitions.

    Subclasses specify the Census base tables, indices, and demographic groups
    that make up one logical "table" (e.g. VAP, CVAP). The base class derives
    both the long Census-variable-to-descriptive-name mapping and the
    ``condense_group_dict`` (raw variable groupings to sum) from those fields.

    Subclasses override ``condense_group_dict`` only when the standard
    "group letter × indices → per-race sum" derivation does not fit, as is
    the case for ``ACSHispByRaceTableInfo``.

    Attributes:
        table_name (str): Human-readable name for the logical ACS table
            (e.g. ``"CVAP"``, ``"VAP"``).
        base_table_strings (tuple[str, ...]): Census base table codes used
            (e.g. ``("B05003",)``).
        table_indices (tuple[int, ...]): Numeric indices within each base
            table that correspond to the variables this table cares about.
        groups_tup (tuple[str, ...]): Single-letter Census group suffixes
            (e.g. ``""`` for all races, ``"A"`` for White alone) that this
            table requests.
        index_to_name_dict (frozendict): Mapping from a Census index to the
            descriptive suffix appended to the long column name (e.g.
            ``{8: "MALE", 19: "FEMALE"}`` for VAP).

    Class Attributes:
        table_to_group_dict (frozendict): Long English-language name for each
            Census group-suffix letter.
        standard_abbreviations (frozendict): Long-to-short substring
            substitutions consumed by ``shorten_acs_column_names``.
        suffix_to_abbrev_dict (frozendict): Census variable suffixes (``"E"``,
            ``"M"``, ``"EA"``, ``"MA"``) mapped to their short forms.
    """

    table_name: str = ""
    base_table_strings: tuple[str, ...] = ()
    table_indices: tuple[int, ...] = ()
    groups_tup: tuple[str, ...] = ()
    index_to_name_dict: frozendict = field(default_factory=frozendict)

    table_to_group_dict: ClassVar[frozendict] = frozendict(
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
    standard_abbreviations: ClassVar[frozendict] = frozendict(
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

    suffix_to_abbrev_dict: ClassVar[frozendict] = frozendict(
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
        """Construct descriptive ACS column names for this table definition.

        Args:
            suffix (str): Census variable suffix: ``"E"`` for estimates, ``"M"``
                for margins of error. Defaults to ``"E"``.
            year (int | None): ACS year to append to the descriptive column
                names. When omitted, generated names do not include a year suffix.
            source_suffix (str | None): Source suffix appended to the descriptive
                column name, such as ``"ACS5"`` or ``"ACS1"``. When omitted, no
                source suffix is appended.

        Returns:
            dict[str, str]: Mapping from Census API variable names (e.g.
            ``"B05003_008E"``) to descriptive column names (e.g.
            ``"TOTAL_VAP_EST_MALE_ACS1"`` when ``source_suffix="ACS1"``).
        """

        long_name_dict = {}

        for table, group, index in product(
            self.base_table_strings, self.groups_tup, self.table_indices
        ):
            variable = f"{table}{group}_{index:03d}{suffix}"
            parts = [
                self.table_to_group_dict[group],
                self.table_name,
                self.suffix_to_abbrev_dict[suffix],
            ]
            index_name = self.index_to_name_dict.get(index, "")
            if index_name:
                parts.append(index_name)
            if year is not None:
                parts.append(str(year))
            long_name_dict[variable] = append_source_suffix("_".join(parts), source_suffix)

        return long_name_dict

    @property
    def condense_group_dict(self) -> frozendict:
        """Output group name to raw Census variable names to sum into it.

        Default behaviour: for each Census base table × group-suffix letter,
        produce one output group ``{long_group_name}_{table_name}`` that sums
        all ``table_indices`` for that (base table, group) pair. Subclasses
        whose structure does not fit (e.g. ``ACSHispByRaceTableInfo``, which uses
        the per-index name as the output key) override this property.

        Returns:
            frozendict: Mapping from output group name to a tuple of raw
            Census variable names (without estimate/MOE suffix) whose values
            should be summed.
        """

        result = {}
        for base_table in self.base_table_strings:
            for group in self.groups_tup:
                key = f"{self.table_to_group_dict[group]}_{self.table_name}"
                result[key] = tuple(
                    f"{base_table}{group}_{index:03d}" for index in self.table_indices
                )
        return frozendict(result)


@dataclass(frozen=True)
class ACSCVAPTableInfo(ACSTableInfo):
    """ACS Citizen Voting-Age Population (CVAP) table definition.

    Sources variables from Census base table B05003 and sums male/female
    native/foreign-born citizen counts (indices 9, 11, 20, 22) into per-race
    CVAP totals.
    """

    table_name: str = "CVAP"
    base_table_strings: tuple[str, ...] = ("B05003",)
    table_indices: tuple[int, ...] = (9, 11, 20, 22)
    groups_tup: tuple[str, ...] = ("", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
            {
                9: "MALE_NATIVE",
                11: "MALE_FOREIGN",
                20: "FEMALE_NATIVE",
                22: "FEMALE_FOREIGN",
            }
        )
    )


@dataclass(frozen=True)
class ACSVAPTableInfo(ACSTableInfo):
    """ACS Voting-Age Population (VAP) table definition.

    Sources variables from Census base table B05003 and sums the male/female
    voting-age subtotals (indices 8 and 19) into per-race VAP totals.
    """

    table_name: str = "VAP"
    base_table_strings: tuple[str, ...] = ("B05003",)
    table_indices: tuple[int, ...] = (8, 19)
    groups_tup: tuple[str, ...] = ("", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
            {
                8: "MALE",
                19: "FEMALE",
            }
        )
    )


@dataclass(frozen=True)
class ACSTotPopTableInfo(ACSTableInfo):
    """ACS total-population table definition.

    Pulls the total-population row (index 1) from Census base table B01001
    into a single ``TOTAL_POP`` condensed group.
    """

    table_name: str = "POP"
    base_table_strings: tuple[str, ...] = ("B01001",)
    table_indices: tuple[int, ...] = (1,)
    groups_tup: tuple[str, ...] = ("",)
    index_to_name_dict: frozendict = field(default_factory=lambda: frozendict({1: ""}))


@dataclass(frozen=True)
class ACSHispByRaceTableInfo(ACSTableInfo):
    """ACS Hispanic-by-Race table definition.

    Sources variables from Census base table B03002, which cross-tabulates
    Hispanic origin with race. Condenses into both ``*_NHISP`` (non-Hispanic)
    and ``*_HISP`` (Hispanic) groups for each race category. Because the
    output keys are per-index rather than per-group, this subclass overrides
    ``condense_group_dict``.
    """

    table_name: str = "HispByRace"
    base_table_strings: tuple[str, ...] = ("B03002",)
    table_indices: tuple[int, ...] = tuple(range(2, 22))
    groups_tup: tuple[str, ...] = ("",)
    index_to_name_dict: frozendict = field(
        default_factory=lambda: frozendict(
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
    )

    @property
    def condense_group_dict(self) -> frozendict:
        """Map each per-index name to a single-variable tuple.

        Returns:
            frozendict: Mapping from each ``index_to_name_dict`` value (e.g.
            ``"WHITE_NHISP"``) to a one-element tuple containing the raw
            Census variable name without estimate/MOE suffix (e.g.
            ``("B03002_003",)``).
        """

        result = {}
        for base_table in self.base_table_strings:
            for index, name in self.index_to_name_dict.items():
                result[name] = (f"{base_table}_{index:03d}",)
        return frozendict(result)
