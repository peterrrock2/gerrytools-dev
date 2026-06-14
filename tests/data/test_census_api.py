import httpx
import pandas as pd
import pytest
import us

from gerrytools.data.uscensus._api import (
    CensusRateLimitError,
    _add_census_api_key,
    _construct_in_query,
    _response_to_frame,
    _strip_geoid_prefix,
    _validate_year,
)

# ==================================
# == YEAR VALIDATION              ==
# ==================================


class TestValidateYear:
    @pytest.mark.parametrize("year", [2000, 2010, 2020, 2050])
    def test_accepts_valid_years(self, year):
        _validate_year(year)  # should not raise

    @pytest.mark.parametrize("year", [1999, 2051, 2020.0, "2020", None])
    def test_rejects_invalid_years(self, year):
        with pytest.raises(ValueError, match="4-digit integer"):
            _validate_year(year)


# ==================================
# == GEOGRAPHY QUERY BUILDER      ==
# ==================================


class TestConstructInQuery:
    def test_state_sets_for(self):
        query: dict = {}
        _construct_in_query(query, us.states.WI, "state")
        assert query == {"for": f"state:{us.states.WI.fips}"}

    def test_county_sets_in_state(self):
        query: dict = {}
        _construct_in_query(query, us.states.WI, "county")
        assert query == {"in": f"state:{us.states.WI.fips}"}

    def test_tract_scopes_to_all_counties(self):
        query: dict = {}
        _construct_in_query(query, us.states.WI, "tract")
        assert query["in"] == [f"state:{us.states.WI.fips}", "county:*"]

    def test_block_group_scopes_through_tract(self):
        query: dict = {}
        _construct_in_query(query, us.states.WI, "block group")
        assert query["in"] == [f"state:{us.states.WI.fips}", "county:*", "tract:*"]

    def test_block_requires_county(self):
        with pytest.raises(ValueError, match="scoped to a county"):
            _construct_in_query({}, us.states.WI, "block")

    def test_block_with_county_scopes_to_it(self):
        query: dict = {}
        _construct_in_query(query, us.states.WI, "block", county_fips="001")
        assert query["in"] == [f"state:{us.states.WI.fips}", "county:001", "tract:*"]

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Invalid geometry"):
            _construct_in_query({}, us.states.WI, "precinct")


# ==================================
# == API KEY RESOLUTION           ==
# ==================================


class TestAddCensusApiKey:
    def test_explicit_key_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "from-env")
        params: dict = {}
        _add_census_api_key(params, "explicit")
        assert params["key"] == "explicit"

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CENSUS_API_KEY", "from-env")
        params: dict = {}
        _add_census_api_key(params, None)
        assert params["key"] == "from-env"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CENSUS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No Census API key"):
            _add_census_api_key({}, None)


# ==================================
# == RESPONSE -> DATAFRAME        ==
# ==================================


def _response(status_code: int, json=None, headers=None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=json,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.census.gov/data/2020/dec/pl"),
    )


class TestResponseToFrame:
    def test_first_row_becomes_header(self):
        payload = [["GEO_ID", "P1_001N"], ["1000000US55", "5000"]]
        frame = _response_to_frame(_response(200, json=payload))
        assert list(frame.columns) == ["GEO_ID", "P1_001N"]
        assert frame.iloc[0]["P1_001N"] == "5000"

    def test_429_raises_rate_limit_error_with_retry_after(self):
        with pytest.raises(CensusRateLimitError, match="Retry-After: 7s"):
            _response_to_frame(_response(429, headers={"Retry-After": "7"}))

    def test_other_error_status_raises_http_status_error(self):
        with pytest.raises(httpx.HTTPStatusError):
            _response_to_frame(_response(500, json=[["GEO_ID"]]))

    @pytest.mark.parametrize("payload", [{"error": "bad request"}, [], "oops"])
    def test_non_list_payload_raises_value_error(self, payload):
        # A 200 with an unexpected (non-list-of-rows) body fails loudly, not with an opaque slice.
        with pytest.raises(ValueError, match="Unexpected Census API response"):
            _response_to_frame(_response(200, json=payload))


# ==================================
# == GEOID PREFIX STRIPPING       ==
# ==================================


class TestStripGeoidPrefix:
    def test_strips_summary_level_stub(self):
        frame = pd.DataFrame({"GEO_ID": ["1000000US55", "0500000US55001"]})
        result = _strip_geoid_prefix(frame)
        assert list(result) == ["55", "55001"]

    def test_honors_custom_column_name(self):
        frame = pd.DataFrame({"RAW": ["0500000US06037"]})
        assert list(_strip_geoid_prefix(frame, "RAW")) == ["06037"]
