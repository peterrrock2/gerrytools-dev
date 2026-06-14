import pandas as pd
import pytest
import us

from gerrytools.data.uscensus.block_cvap import (
    _estimate_block_cvap_from_inputs,
    _state_cvap_rates,
    _tract_cvap_rates,
    block_cvap_estimates,
)
from gerrytools.data.uscensus.census_tables import (
    ACSCVAPTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
)
from tests.data._helpers import MockHTTP, acs_api_payload

# ==========================
# == STATE FALLBACK RATES ==
# ==========================


class TestStateCvapRates:
    def test_rate_is_cvap_over_vap(self):
        state_est = pd.DataFrame([{"TOT_VAP_ACS5": 1000, "TOT_CVAP_ACS5": 600}])
        rates = _state_cvap_rates(state_est, race_prefixes=("TOT",))
        assert rates["TOT"] == pytest.approx(0.6)

    def test_zero_vap_yields_zero_rate(self):
        state_est = pd.DataFrame([{"WHITE_VAP_ACS5": 0, "WHITE_CVAP_ACS5": 0}])
        rates = _state_cvap_rates(state_est, race_prefixes=("WHITE",))
        assert rates["WHITE"] == 0.0


# =============================
# == TRACT RATES + THRESHOLD ==
# =============================


class TestTractCvapRates:
    def test_below_threshold_is_masked_to_nan(self):
        tract_est = pd.DataFrame(
            {"TOT_VAP_ACS5": [100, 10], "TOT_CVAP_ACS5": [80, 5]},
            index=pd.Index(["T1", "T2"]),
        )
        rates = _tract_cvap_rates(tract_est, denominator_threshold=20, race_prefixes=("TOT",))
        assert rates["TOT"]["T1"] == pytest.approx(0.8)
        assert bool(pd.isna(rates["TOT"]["T2"]))


# ===========================
# == BLOCK CVAP ESTIMATION ==
# ===========================


class TestEstimateBlockCvap:
    def _inputs(self):
        table = PLBlockVAPTableInfo(race_prefixes=("TOT", "WHITE"))

        blocks = pd.DataFrame(
            {
                "GEOID": ["b1", "b2"],
                "STATEFP": ["55", "55"],
                "COUNTYFP": ["001", "001"],
                "TRACTCE": ["000100", "000200"],
                "BLOCKCE": ["1000", "2000"],
                # b1 is in tract T1 (rate available), b2 in T2 (falls back).
                "TRACT_GEOID": ["T1", "T2"],
            }
        )
        # Every PL VAP column must be present for the output projection.
        for name in table.construct_short_names():
            blocks[name] = [0, 0]
        blocks["TOT_VAP_P3"] = [50, 5]
        blocks["WHITE_VAP_P3"] = [50, 5]

        tract_est = pd.DataFrame(
            {
                "TOT_VAP_ACS5": [100, 10],
                "TOT_CVAP_ACS5": [80, 5],
                "WHITE_VAP_ACS5": [100, 10],
                "WHITE_CVAP_ACS5": [80, 5],
            },
            index=pd.Index(["T1", "T2"]),
        )
        state_est = pd.DataFrame(
            [
                {
                    "TOT_VAP_ACS5": 1000,
                    "TOT_CVAP_ACS5": 600,
                    "WHITE_VAP_ACS5": 1000,
                    "WHITE_CVAP_ACS5": 600,
                }
            ]
        )
        return blocks, tract_est, state_est, table

    def test_uses_tract_rate_then_state_fallback(self):
        blocks, tract_est, state_est, table = self._inputs()

        result = _estimate_block_cvap_from_inputs(
            blocks, tract_est, state_est, denominator_threshold=20, table=table
        )
        by_geoid = result.set_index("GEOID")

        # b1's tract T1 has rate 0.8 (VAP 100 >= 20): 50 * 0.8 = 40.
        assert by_geoid.loc["b1", "TOT_CVAP"] == pytest.approx(40.0)
        # b2's tract T2 is below threshold, so the 0.6 state rate applies:
        # 5 * 0.6 = 3.0.
        assert by_geoid.loc["b2", "TOT_CVAP"] == pytest.approx(3.0)
        assert by_geoid.loc["b1", "WHITE_CVAP"] == pytest.approx(40.0)

    def test_output_carries_geoid_components_and_cvap_columns(self):
        blocks, tract_est, state_est, table = self._inputs()

        result = _estimate_block_cvap_from_inputs(
            blocks, tract_est, state_est, denominator_threshold=20, table=table
        )

        for column in ("GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLOCKCE", "TRACT_GEOID"):
            assert column in result.columns
        assert "TOT_CVAP" in result.columns
        assert "WHITE_CVAP" in result.columns


# ==========================
# == BLOCK CVAP ESTIMATES ==
# ==========================


class TestBlockCvapEstimates:
    def test_invalid_year_raises(self):
        with pytest.raises(ValueError, match="4-digit integer"):
            block_cvap_estimates(us.states.WI, acs_year=1999, api_key="k")

    def test_state_with_no_counties_returns_empty_frame(self, mock_http: MockHTTP):
        # ACS tract/state VAP+CVAP fetches succeed...
        mock_http.route(
            url_contains="/acs/acs5",
            json=acs_api_payload([ACSVAPTableInfo(), ACSCVAPTableInfo()], {"55001": 1}),
        )
        # ...but the decennial PL county enumeration returns no counties.
        mock_http.route(url_contains="/dec/pl", json=[["NAME", "state", "county"]])

        result = block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_county_produces_block_estimates(self, mock_http: MockHTTP):
        pl_variables = list(PLBlockVAPTableInfo().construct_variable_names())
        # 15-char block GEOID: state 55 / county 001 / tract 000100 / block 1000.
        block_geoid = "550010001001000"

        mock_http.route(
            url_contains="/acs/acs5",
            json=acs_api_payload([ACSVAPTableInfo(), ACSCVAPTableInfo()], {"55001": 1}),
        )
        # County enumeration returns a single county (for=county:*)...
        mock_http.route(
            url_contains="for=county",
            json=[["NAME", "state", "county"], ["Test County, WI", "55", "001"]],
        )
        # ...whose block query (for=block:*) returns a single block.
        mock_http.route(
            url_contains="for=block",
            json=[
                ["GEO_ID"] + pl_variables,
                [f"1000000US{block_geoid}"] + ["10"] * len(pl_variables),
            ],
        )

        result = block_cvap_estimates(us.states.WI, acs_year=2024, pl_year=2020, api_key="k")

        assert len(result) == 1
        block = result.iloc[0]
        assert block["GEOID"] == block_geoid
        assert block["STATEFP"] == "55"
        assert block["TRACT_GEOID"] == "55001000100"
        assert "TOT_CVAP" in result.columns
        assert pd.api.types.is_numeric_dtype(result["TOT_CVAP"])
