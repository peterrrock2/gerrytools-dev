import logging
from pathlib import Path

import httpx
import pytest
import us

from gerrytools.data.geometries import (
    DATA_MGGG_BASE_URL,
    dualgraphs20,
    geometries20,
    vtds20,
)
from tests.data._helpers import MockHTTP

# ==================================
# == DUAL GRAPHS (dualgraphs20)   ==
# ==================================


class TestDualGraphs:
    @pytest.mark.parametrize(
        "geometry,expected_id",
        [
            ("bg", "bg"),
            ("block group", "bg"),
            ("blockgroup", "bg"),
            ("vtd", "vtd"),
            # Geometry is matched case-insensitively.
            ("BG", "bg"),
            ("Block Group", "bg"),
            ("VTD", "vtd"),
        ],
    )
    def test_builds_expected_url(self, mock_http: MockHTTP, tmp_path: Path, geometry, expected_id):
        mock_http.route(content=b"{}")
        out = tmp_path / "dg.json"

        dualgraphs20(us.states.WI, out, geometry=geometry)

        expected = f"{DATA_MGGG_BASE_URL}/dual-graphs/wi-{expected_id}-connected.json"
        assert mock_http.urls == [expected]

    def test_block_group_does_not_leak_a_space_into_the_url(
        self, mock_http: MockHTTP, tmp_path: Path
    ):
        # Regression: the previous implementation interpolated the raw geometry
        # string, producing "wi-block group-connected.json" (a broken URL).
        mock_http.route(content=b"{}")

        dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="block group")

        assert " " not in mock_http.urls[0]
        assert mock_http.urls[0].endswith("wi-bg-connected.json")

    def test_writes_response_body_to_disk(self, mock_http: MockHTTP, tmp_path: Path):
        payload = b'{"nodes": [1, 2, 3]}'
        mock_http.route(content=payload)
        out = tmp_path / "dg.json"

        dualgraphs20(us.states.WI, out, geometry="bg")

        assert out.read_bytes() == payload

    def test_invalid_geometry_raises_without_any_request(self, mock_http: MockHTTP, tmp_path: Path):
        with pytest.raises(ValueError, match="not available as a dual graph"):
            dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="tract")

        assert mock_http.requests == []

    def test_http_error_raises_and_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=404, text="not found")
        out = tmp_path / "dg.json"

        with pytest.raises(httpx.HTTPStatusError):
            dualgraphs20(us.states.WI, out, geometry="bg")

        # raise_for_status fires before the file is opened, so a failed download never leaves a
        # truncated or error-page file behind.
        assert not out.exists()


# =============================
# == VTD SHAPEFILES (vtds20) ==
# =============================


class TestVTDs:
    def test_builds_expected_url_with_uppercase_abbr(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"zipbytes")

        vtds20(us.states.WI, tmp_path / "vtd.zip")

        assert mock_http.urls == [f"{DATA_MGGG_BASE_URL}/vtd-shapefiles/WI_vtd20.zip"]

    def test_writes_response_body_to_disk(self, mock_http: MockHTTP, tmp_path: Path):
        payload = b"PK\x03\x04 zip contents"
        mock_http.route(content=payload)
        out = tmp_path / "vtd.zip"

        vtds20(us.states.WI, out)

        assert out.read_bytes() == payload

    def test_http_error_raises_and_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=500, text="server error")
        out = tmp_path / "vtd.zip"

        with pytest.raises(httpx.HTTPStatusError):
            vtds20(us.states.WI, out)

        assert not out.exists()


# ======================================
# == CENSUS GEOMETRIES (geometries20) ==
# ======================================


class TestGeometries:
    def test_defaults_to_tract(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"zip")

        geometries20(us.states.WI, tmp_path / "geo.zip")

        assert mock_http.urls == [f"{DATA_MGGG_BASE_URL}/census-2020/wi/wi_tract.zip"]

    @pytest.mark.parametrize(
        "geometry,expected_id",
        [
            ("block group", "bg"),
            ("block", "block"),
            ("congress", "cd116"),
            ("county", "county"),
            ("cousub", "cousub"),
            ("place", "place"),
            ("senate", "sldu"),
            ("house", "sldl"),
            ("tract", "tract"),
            ("vtd", "vtd"),
        ],
    )
    def test_maps_geometry_to_identifier(
        self, mock_http: MockHTTP, tmp_path: Path, geometry, expected_id
    ):
        mock_http.route(content=b"zip")

        geometries20(us.states.WI, tmp_path / "geo.zip", geometry=geometry)

        assert mock_http.urls == [f"{DATA_MGGG_BASE_URL}/census-2020/wi/wi_{expected_id}.zip"]

    def test_invalid_geometry_raises_without_any_request(self, mock_http: MockHTTP, tmp_path: Path):
        with pytest.raises(ValueError, match="not allowed"):
            geometries20(us.states.WI, tmp_path / "geo.zip", geometry="nonsense")

        assert mock_http.requests == []

    def test_http_error_raises_and_leaves_no_file(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(status_code=403, text="forbidden")
        out = tmp_path / "geo.zip"

        with pytest.raises(httpx.HTTPStatusError):
            geometries20(us.states.WI, out)

        assert not out.exists()


# ============================
# == CROSS-CUTTING BEHAVIOR ==
# ============================


class TestCommonBehavior:
    def test_endpoint_is_http_only(self):
        # S3 static-website hosting does not terminate TLS, so the base URL is intentionally
        # http://. Pin it so a well-meaning https:// "fix" fails.
        assert DATA_MGGG_BASE_URL.startswith("http://")

    def test_request_uses_http_scheme(self, mock_http: MockHTTP, tmp_path: Path):
        mock_http.route(content=b"{}")

        dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="bg")

        assert mock_http.requests[0].url.scheme == "http"

    def test_accepts_str_path(self, mock_http: MockHTTP, tmp_path: Path):
        payload = b"zip"
        mock_http.route(content=payload)
        out = tmp_path / "geo.zip"

        # Pass the destination as a plain string rather than a Path.
        geometries20(us.states.WI, str(out), geometry="county")

        assert out.read_bytes() == payload

    def test_multichunk_body_is_written_intact(self, mock_http: MockHTTP, tmp_path: Path):
        # A body larger than httpx's internal chunk size exercises the streamed
        # iter_bytes loop across multiple chunks.
        payload = b"abcd" * 100_000
        mock_http.route(content=payload)
        out = tmp_path / "vtd.zip"

        vtds20(us.states.WI, out)

        assert out.read_bytes() == payload

    def test_logs_download_at_info(
        self, mock_http: MockHTTP, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        mock_http.route(content=b"{}")

        with caplog.at_level(logging.INFO, logger="gerrytools.data.geometries"):
            dualgraphs20(us.states.WI, tmp_path / "dg.json", geometry="bg")

        assert any("dual graph" in record.message for record in caplog.records)
