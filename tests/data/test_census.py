import pandas as pd
import pytest
import us

from gerrytools.data.uscensus.census import census
from gerrytools.data.uscensus.census_tables import pl_pop_table
from tests.data._helpers import MockHTTP

# ==================================
# == VALIDATION                   ==
# ==================================


class TestCensusValidation:
    def test_non_4_digit_year_raises(self):
        with pytest.raises(ValueError, match="4-digit integer"):
            census(us.states.WI, year=1999, api_key="k")

    def test_year_without_pl_data_raises(self):
        with pytest.raises(ValueError, match="only available for years"):
            census(us.states.WI, year=2000, api_key="k")

    def test_unknown_table_string_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            census(us.states.WI, table="P9", api_key="k")

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Invalid geometry"):
            census(us.states.WI, geometry="precinct", api_key="k")


# ==================================
# == FETCH + NUMERIC DTYPE        ==
# ==================================


class TestCensusFetch:
    def test_counts_are_numeric_and_indexed_by_geoid(self, mock_http: MockHTTP):
        # Build a P1 group response from two real variable->short-name pairs.
        table = pl_pop_table("P1", 2020)
        (raw_one, short_one), (raw_two, short_two) = list(table.construct_rename_map().items())[:2]

        header = ["GEO_ID", "NAME", raw_one, raw_two, "state"]
        row = ["1000000US55", "Wisconsin", "5000", "1200", "55"]
        mock_http.route(url_contains="/dec/pl", json=[header, row])

        df = census(us.states.WI, geometry="state", year=2020, table="P1", api_key="k")

        # Index is the stripped GEOID, GEO_ID is dropped.
        assert df.index.name == "GEOID"
        assert list(df.index) == ["55"]
        assert "GEO_ID" not in df.columns

        # Regression: count columns come back numeric, not strings.
        assert pd.api.types.is_numeric_dtype(df[short_one])
        assert df.loc["55", short_one] == 5000
        assert df.loc["55", short_two] == 1200

        # Only the renamed count columns are returned; NAME and geography-breakdown columns dropped.
        assert list(df.columns) == [short_one, short_two]
        assert "NAME" not in df.columns
        assert "state" not in df.columns

    def test_accepts_pltableinfo_instance(self, mock_http: MockHTTP):
        table = pl_pop_table("P1", 2020)
        raw, short = next(iter(table.construct_rename_map().items()))
        mock_http.route(
            url_contains="/dec/pl",
            json=[["GEO_ID", raw, "state"], ["1000000US55", "42", "55"]],
        )

        df = census(us.states.WI, geometry="state", year=2020, table=table, api_key="k")

        assert df.loc["55", short] == 42
