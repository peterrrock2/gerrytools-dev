"""Shared fixtures for plotting tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def testing_gdf():
    """Load the 12×12 grid GeoPackage used for geometry snapshot tests."""
    import geopandas as gpd

    return gpd.read_file(Path("tests/fixtures/testing_12x12.gpkg"))
