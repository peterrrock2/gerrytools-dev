import os

import httpx
import us
from frozendict import frozendict

from gerrytools.logging import get_logger

logger = get_logger(__name__)

# data.mggg.org is served from an S3 static-website endpoint, which only supports HTTP (S3 website
# hosting does not terminate TLS), so this base URL is intentionally http:// rather than https://.
DATA_MGGG_BASE_URL = "http://data.mggg.org.s3-website.us-east-2.amazonaws.com"

_REQUEST_TIMEOUT = httpx.Timeout(120)

# Dual-graph geometry levels mapped to their data.mggg.org filename identifiers. "block group"
# and "blockgroup" are accepted spellings of "bg".
_DUALGRAPH_GEOMETRY_IDS = frozendict(
    {
        "bg": "bg",
        "block group": "bg",
        "blockgroup": "bg",
        "vtd": "vtd",
    }
)

# Census-2020 geometry levels mapped to their data.mggg.org identifiers.
_CENSUS_GEOMETRY_IDS = frozendict(
    {
        "block group": "bg",
        "block": "block",
        "congress": "cd116",
        "county": "county",
        "cousub": "cousub",
        "place": "place",
        "senate": "sldu",
        "house": "sldl",
        "tract": "tract",
        "vtd": "vtd",
    }
)


def _download_to_file(url: str, filepath: str | os.PathLike[str]) -> None:
    """Stream a GET response body to ``filepath``, raising on HTTP error.

    Streaming keeps memory use flat for large shapefile and dual-graph archives, and
    ``raise_for_status`` ensures an S3 error page is never written to disk in place of the
    requested file.

    Args:
        url (str): URL to download.
        filepath (str | os.PathLike): Destination path for the response body.

    Raises:
        httpx.HTTPStatusError: If the server returns an error status.
    """

    with (
        httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=True) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with open(filepath, "wb") as output_file:
            for chunk in response.iter_bytes():
                output_file.write(chunk)


def dualgraphs20(
    state: us.states.State,
    filepath: str | os.PathLike[str],
    geometry: str = "block group",
) -> None:
    """Download Lab-processed dual graph data for a state and write it to disk.

    Args:
        state (us.states.State): State for which to retrieve data.
        filepath (str | os.PathLike): Destination path for the downloaded dual graph JSON.
        geometry (str): Geometry level. One of ``"bg"`` / ``"block group"`` / ``"blockgroup"`` or
            ``"vtd"``. Defaults to ``"block group"``.

    Raises:
        ValueError: If ``geometry`` is not an available dual-graph level.
        httpx.HTTPStatusError: If the download request fails.

    Warning:
        Writes the downloaded file to ``filepath``.
    """

    geometry_key = geometry.lower()
    if geometry_key not in _DUALGRAPH_GEOMETRY_IDS:
        raise ValueError(
            f"Requested geometry {geometry!r} is not available as a dual graph; "
            f"choose one of {sorted(_DUALGRAPH_GEOMETRY_IDS)}."
        )

    geometry_id = _DUALGRAPH_GEOMETRY_IDS[geometry_key]
    url = f"{DATA_MGGG_BASE_URL}/dual-graphs/{state.abbr.lower()}-{geometry_id}-connected.json"

    logger.info("Downloading %s %s dual graph to %s.", state.abbr, geometry_id, filepath)
    _download_to_file(url, filepath)


def vtds20(state: us.states.State, filepath: str | os.PathLike[str]) -> None:
    """Download Lab-processed 2020 VTD shapefile data and write it to disk.

    Args:
        state (us.states.State): State for which to retrieve data.
        filepath (str | os.PathLike): Destination path for the downloaded zipped shapefile.

    Raises:
        httpx.HTTPStatusError: If the download request fails.

    Warning:
        Writes the downloaded file to ``filepath``.
    """

    url = f"{DATA_MGGG_BASE_URL}/vtd-shapefiles/{state.abbr.upper()}_vtd20.zip"

    logger.info("Downloading %s VTD shapefile to %s.", state.abbr, filepath)
    _download_to_file(url, filepath)


def geometries20(
    state: us.states.State,
    filepath: str | os.PathLike[str],
    geometry: str = "tract",
) -> None:
    """Download Lab-processed 2020 geometric data and write it to disk.

    Args:
        state (us.states.State): State for which to retrieve data.
        filepath (str | os.PathLike): Destination path for the downloaded zipped shapefile.
        geometry (str): Geometry level at which to retrieve data. One of ``"block group"``,
            ``"block"``, ``"congress"``, ``"county"``, ``"cousub"``, ``"place"``, ``"senate"``,
            ``"house"``, ``"tract"``, or ``"vtd"``. Defaults to ``"tract"``.

    Raises:
        ValueError: If ``geometry`` is not a recognized level.
        httpx.HTTPStatusError: If the download request fails.

    Warning:
        Writes the downloaded file to ``filepath``.
    """

    if geometry not in _CENSUS_GEOMETRY_IDS:
        raise ValueError(
            f"Requested geometry {geometry!r} is not allowed; "
            f"choose one of {sorted(_CENSUS_GEOMETRY_IDS)}."
        )

    geometry_id = _CENSUS_GEOMETRY_IDS[geometry]
    state_abbr = state.abbr.lower()
    url = f"{DATA_MGGG_BASE_URL}/census-2020/{state_abbr}/{state_abbr}_{geometry_id}.zip"

    logger.info("Downloading %s %s geometries to %s.", state.abbr, geometry_id, filepath)
    _download_to_file(url, filepath)
