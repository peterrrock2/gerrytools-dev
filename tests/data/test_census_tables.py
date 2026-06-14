import pandas as pd
import pytest

from gerrytools.data.uscensus.census_tables import (
    PL_POP_TABLES,
    PL_POP_YEARS,
    ACSCVAPTableInfo,
    ACSHispByRaceTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
    append_source_suffix,
    decennial_pl_source_from_variable,
    pl_pop_table,
    shorten_acs_column_names,
)
from gerrytools.data.uscensus.census_tables import (
    _canonical_replacement as canonical_replacement,
)

# ==================================
# == SUFFIX / SOURCE HELPERS      ==
# ==================================


class TestAppendSourceSuffix:
    def test_appends_truthy_source(self):
        assert append_source_suffix("TOT_VAP", "ACS5") == "TOT_VAP_ACS5"
        assert append_source_suffix("TOT_POP", "P1") == "TOT_POP_P1"

    @pytest.mark.parametrize("falsy", [None, ""])
    def test_falsy_source_returns_name_unchanged(self, falsy):
        assert append_source_suffix("TOT_VAP", falsy) == "TOT_VAP"


class TestDecennialPlSource:
    @pytest.mark.parametrize(
        "variable,expected",
        [("P1_001N", "P1"), ("P3_009N", "P3"), ("P4_002N", "P4")],
    )
    def test_returns_table_prefix(self, variable, expected):
        assert decennial_pl_source_from_variable(variable) == expected


# ==================================
# == ACS NAME SHORTENING          ==
# ==================================


class TestCanonicalReplacement:
    @pytest.mark.parametrize(
        "long_name,expected",
        [
            ("TOTAL_VAP_EST_MALE", "TOT_VAP_M"),
            # FEMALE must collapse before MALE so it is not corrupted to FE+M.
            ("TOTAL_VAP_EST_FEMALE", "TOT_VAP_F"),
            ("WHITE_ALONE_CVAP", "WHITE_CVAP"),
            # The longest race phrase must win over the bare NATIVE rule.
            ("AMERICAN_INDIAN_AND_ALASKAN_NATIVE_ALONE_CVAP", "AIAN_CVAP"),
            ("TWO_OR_MORE_RACES_VAP", "2MORE_VAP"),
            ("NON_HISPANIC_WHITE_VAP", "NHWHITE_VAP"),
        ],
    )
    def test_shortens_to_canonical_form(self, long_name, expected):
        assert canonical_replacement(long_name) == expected


class TestShortenAcsColumnNames:
    def test_renames_in_place_and_appends_source(self):
        frame = pd.DataFrame(columns=pd.Index(["TOTAL_VAP_EST_MALE", "WHITE_ALONE_CVAP"]))

        shorten_acs_column_names(frame, source_suffix="ACS5")

        assert list(frame.columns) == ["TOT_VAP_M_ACS5", "WHITE_CVAP_ACS5"]


# ==================================
# == ACS LONG NAMES + CONDENSE    ==
# ==================================


class TestACSLongNames:
    def test_vap_total_group_male_estimate(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="E")
        assert names["B05003_008E"] == "TOTAL_VAP_EST_MALE"

    def test_vap_white_alone_group(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="E")
        assert names["B05003A_008E"] == "WHITE_ALONE_VAP_EST_MALE"

    def test_year_and_source_suffix_appended(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="E", year=2022, source_suffix="ACS1")
        assert names["B05003_008E"] == "TOTAL_VAP_EST_MALE_2022_ACS1"

    def test_totpop_index_with_empty_name_omits_suffix(self):
        names = ACSTotPopTableInfo().construct_long_names(suffix="E")
        assert names["B01001_001E"] == "TOTAL_POP_EST"

    def test_moe_suffix_changes_abbreviation(self):
        names = ACSVAPTableInfo().construct_long_names(suffix="M")
        assert names["B05003_008M"] == "TOTAL_VAP_MOE_MALE"


class TestCondenseGroupDict:
    def test_vap_groups_sum_both_indices(self):
        groups = ACSVAPTableInfo().condense_group_dict
        assert groups["TOTAL_VAP"] == ("B05003_008", "B05003_019")
        assert groups["WHITE_ALONE_VAP"] == ("B05003A_008", "B05003A_019")

    def test_totpop_single_variable(self):
        groups = ACSTotPopTableInfo().condense_group_dict
        assert groups["TOTAL_POP"] == ("B01001_001",)

    def test_cvap_sums_four_citizen_indices(self):
        groups = ACSCVAPTableInfo().condense_group_dict
        assert groups["TOTAL_CVAP"] == (
            "B05003_009",
            "B05003_011",
            "B05003_020",
            "B05003_022",
        )

    def test_hisp_by_race_override_uses_per_index_keys(self):
        groups = ACSHispByRaceTableInfo().condense_group_dict
        # The override maps each per-index name to a single source variable.
        assert groups["WHITE_NHISP"] == ("B03002_003",)
        assert groups["WHITE_HISP"] == ("B03002_013",)


# ==================================
# == DECENNIAL PL TABLES          ==
# ==================================


class TestPlPopTable:
    @pytest.mark.parametrize("table_name", PL_POP_TABLES)
    @pytest.mark.parametrize("year", PL_POP_YEARS)
    def test_builds_for_every_supported_table_and_year(self, table_name, year):
        table = pl_pop_table(table_name, year)
        assert table.table_name == f"{table_name}_{year}"
        assert len(table.construct_variable_names()) > 0

    def test_rejects_unknown_year(self):
        with pytest.raises(ValueError, match="PL year"):
            pl_pop_table("P1", 1999)

    def test_rejects_unknown_table(self):
        with pytest.raises(ValueError, match="PL table"):
            pl_pop_table("P9", 2020)


class TestPLBlockVAPTableInfo:
    def test_short_names_carry_source_suffix(self):
        short_names = PLBlockVAPTableInfo().construct_short_names()
        assert "TOT_VAP_P3" in short_names
        assert "HISP_VAP_P4" in short_names

    def test_rename_map_maps_raw_to_suffixed_short(self):
        rename_map = PLBlockVAPTableInfo().construct_rename_map()
        assert rename_map["P3_001N"] == "TOT_VAP_P3"
        assert rename_map["P4_002N"] == "HISP_VAP_P4"

    @pytest.mark.parametrize(
        "short,expected",
        [
            ("TOT_VAP", "TOT_VAP_P3"),
            ("WHITE_VAP", "WHITE_VAP_P3"),
            ("NHWHITE_VAP", "NHWHITE_VAP_P4"),
            ("HISP_VAP", "HISP_VAP_P4"),
        ],
    )
    def test_source_name_for_short_name(self, short, expected):
        assert PLBlockVAPTableInfo().source_name_for_short_name(short) == expected

    def test_unknown_short_name_raises_key_error(self):
        with pytest.raises(KeyError):
            PLBlockVAPTableInfo().source_name_for_short_name("NOPE_VAP")


# ==================================
# == PUBLIC EXPORTS               ==
# ==================================

# Table definitions and helpers that acs()/census() document as accepted arguments and that
# should therefore be reachable without importing from the private census_tables module.
PUBLIC_NAMES = (
    "ACSTableInfo",
    "ACSTotPopTableInfo",
    "ACSVAPTableInfo",
    "ACSCVAPTableInfo",
    "ACSHispByRaceTableInfo",
    "PLTableInfo",
    "PLBlockVAPTableInfo",
    "pl_pop_table",
    "CensusRateLimitError",
)


class TestPublicExports:
    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_reachable_from_data_package(self, name):
        import gerrytools.data as data

        assert name in data.__all__
        assert hasattr(data, name)

    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_reachable_from_uscensus_package(self, name):
        # The subpackage is named uscensus (not census), so it no longer collides with the
        # re-exported census() function on gerrytools.data and is reachable directly.
        import gerrytools.data.uscensus as uscensus_pkg

        assert name in uscensus_pkg.__all__
        assert hasattr(uscensus_pkg, name)

    def test_hidden_table_is_now_usable(self):
        # The previously unreachable Hispanic-by-race table can be constructed from the surface.
        from gerrytools.data import ACSHispByRaceTableInfo

        assert ACSHispByRaceTableInfo().table_name == "HispByRace"
