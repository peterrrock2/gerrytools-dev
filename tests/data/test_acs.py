import warnings
from typing import cast

import pandas as pd
import pytest
import us

from gerrytools.data.uscensus.acs import (
    _condense,
    _normalize_acs_survey,
    acs,
    acs_full,
    cvap,
)
from gerrytools.data.uscensus.census_tables import (
    ACSCVAPTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
)
from tests.data._helpers import MockHTTP, acs_api_payload, geo_id_payload

# ==========================
# == SURVEY NORMALIZATION ==
# ==========================


class TestNormalizeAcsSurvey:
    @pytest.mark.parametrize("value", ["acs5", "5", 5, "5year", "ACS-5", "acs 5"])
    def test_normalizes_to_acs5(self, value):
        assert _normalize_acs_survey(value) == "acs5"

    @pytest.mark.parametrize("value", ["acs1", "1", 1, "1year", "ACS_1"])
    def test_normalizes_to_acs1(self, value):
        assert _normalize_acs_survey(value) == "acs1"

    @pytest.mark.parametrize("value", ["acs7", "yearly", 3.5, None])
    def test_rejects_unknown_survey(self, value):
        with pytest.raises(ValueError, match="Invalid ACS survey"):
            _normalize_acs_survey(value)


# ==============
# == CONDENSE ==
# ==============


class TestCondense:
    def test_sums_group_indices(self):
        data = pd.DataFrame(
            {"B05003_008E": [3.0, 1.0], "B05003_019E": [4.0, 1.0]},
            index=pd.Index(["55001", "55003"]),
        )
        result = _condense(data, ACSVAPTableInfo(), suffix="E", label="_EST")
        assert result["TOTAL_VAP_EST"].tolist() == [7.0, 2.0]

    def test_groups_with_missing_columns_are_skipped(self):
        # Only the TOTAL group's columns are present; WHITE_ALONE etc. dropped.
        data = pd.DataFrame({"B05003_008E": [1.0], "B05003_019E": [1.0]}, index=pd.Index(["55001"]))
        result = _condense(data, ACSVAPTableInfo(), suffix="E", label="_EST")
        assert list(result.columns) == ["TOTAL_VAP_EST"]


# ===============
# == ACS FETCH ==
# ===============


class TestAcs:
    def test_totpop_county_estimates(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100, "55003": 200}),
        )

        est, moe = acs(
            us.states.WI,
            "county",
            2020,
            tables=[ACSTotPopTableInfo()],
            api_key="k",
        )

        assert list(est.index) == ["55001", "55003"]
        assert est["TOT_POP_ACS5"].tolist() == [100.0, 200.0]
        assert "TOT_POP_MOE_ACS5" in moe.columns

    def test_short_names_false_keeps_long_names(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 5}),
        )

        est, _ = acs(
            us.states.WI,
            "county",
            2020,
            tables=[ACSTotPopTableInfo()],
            short_names=False,
            api_key="k",
        )

        assert "TOTAL_POP_EST_ACS5" in est.columns

    def test_default_tables_concatenate_pop_vap_cvap(self, mock_http: MockHTTP):
        tables = [ACSTotPopTableInfo(), ACSVAPTableInfo(), ACSCVAPTableInfo()]
        mock_http.route(url_contains="/acs/", json=acs_api_payload(tables, {"55001": 1}))

        est, _ = acs(us.states.WI, "county", 2020, api_key="k")

        # POP has 1 source variable, VAP sums 2 indices, CVAP sums 4.
        assert est.loc["55001", "TOT_POP_ACS5"] == 1.0
        assert est.loc["55001", "TOT_VAP_ACS5"] == 2.0
        assert est.loc["55001", "TOT_CVAP_ACS5"] == 4.0

    def test_acs1_tract_is_rejected(self, mock_http: MockHTTP):
        with pytest.raises(ValueError, match="not available for 'tract'"):
            acs(
                us.states.WI,
                "tract",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

    def test_acs1_partial_county_coverage_warns(self, mock_http: MockHTTP):
        # ACS1 returns one county; ACS5 (completeness check) knows of two.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", json=geo_id_payload(["55001", "55003"]))

        with pytest.warns(UserWarning, match="ACS 1-year returned 1 of 2"):
            acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

    def test_acs1_complete_county_coverage_does_not_warn(self, mock_http: MockHTTP):
        # ACS1 and ACS5 agree on the county set, so no warning should fire.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", json=geo_id_payload(["55001"]))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

        assert not any("ACS 1-year returned" in str(w.message) for w in caught)

    def test_acs1_completeness_check_failure_is_silent(self, mock_http: MockHTTP):
        # The ACS5 completeness probe fails; the ACS1 result is still returned
        # and no warning is raised.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 100}),
        )
        mock_http.route(url_contains="/acs/acs5", status_code=500, text="boom")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est, _ = acs(
                us.states.WI,
                "county",
                2020,
                tables=[ACSTotPopTableInfo()],
                survey="acs1",
                api_key="k",
            )

        assert list(est.index) == ["55001"]
        assert not any("ACS 1-year returned" in str(w.message) for w in caught)

    def test_acs1_non_county_skips_completeness_check(self, mock_http: MockHTTP):
        # State-level ACS1 should not issue the county-only ACS5 probe.
        mock_http.route(
            url_contains="/acs/acs1",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55": 100}),
        )

        acs(
            us.states.WI,
            "state",
            2020,
            tables=[ACSTotPopTableInfo()],
            survey="acs1",
            api_key="k",
        )

        assert all("/acs/acs5" not in url for url in mock_http.urls)


class TestAcsFull:
    def test_renames_to_long_english_names(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSTotPopTableInfo()], {"55001": 7}),
        )

        est, moe = acs_full(us.states.WI, "county", 2020, ACSTotPopTableInfo(), api_key="k")

        assert "TOTAL_POP_EST_2020_ACS5" in est.columns
        assert "TOTAL_POP_MOE_2020_ACS5" in moe.columns
        assert est.loc["55001", "TOTAL_POP_EST_2020_ACS5"] == 7.0


class TestAcsValidation:
    def test_invalid_year_raises(self):
        with pytest.raises(ValueError, match="4-digit integer"):
            acs(us.states.WI, "county", 1999, api_key="k")

    def test_empty_tables_raises(self):
        with pytest.raises(ValueError, match="at least one table"):
            acs(us.states.WI, "county", 2020, tables=[], api_key="k")

    def test_non_table_element_raises_type_error(self):
        with pytest.raises(TypeError, match="must be an ACSTableInfo"):
            acs(
                us.states.WI,
                "county",
                2020,
                tables=cast(list[ACSTableInfo], ["not a table"]),
                api_key="k",
            )


# ==================
# == CVAP WRAPPER ==
# ==================


class TestCvap:
    def test_returns_condensed_cvap(self, mock_http: MockHTTP):
        mock_http.route(
            url_contains="/acs/",
            json=acs_api_payload([ACSCVAPTableInfo()], {"55001": 1}),
        )

        est, moe = cvap(us.states.WI, "county", 2020, api_key="k")

        assert est.loc["55001", "TOT_CVAP_ACS5"] == 4.0
        assert "TOT_CVAP_MOE_ACS5" in moe.columns
