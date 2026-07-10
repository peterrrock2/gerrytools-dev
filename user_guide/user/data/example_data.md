# Tutorial data

<div style="text-align: center;"><a class="sd-sphinx-override sd-btn sd-text-wrap sd-btn-primary reference external" href="https://www.dropbox.com/scl/fo/s22x9phl0hldiakn8nbuz/ABKfxHBaak5ra3eBGkNFWMM?rlkey=igpo7qi07oz5tfgjki317o79t&amp;st=gcxkicnc&amp;dl=1">Download tutorial data</a></div>

The tutorials use the datasets in this bundle. Extract it so the `data` directory is next to the
tutorial notebook:

```text
tutorial.ipynb
data/
```

The individual files are also available below.

## Georgia precinct geography

{download}`Download the Georgia precinct GeoPackage <../../_static/data/ga_2016_precincts.gpkg>`
(9.4 MB).

The file contains 2,664 precinct geometries, 2016 presidential and U.S. Senate returns,
demographic counts, and source congressional, state House, and state Senate assignments. The
geometry is projected to EPSG:5070 and simplified for documentation. It is used throughout the
geographic and statistical plotting guides.

After saving it as `data/ga_2016_precincts.gpkg`, open it with:

<!-- docs-test: skip -- requires the reader to download the linked GeoPackage -->
```python
import geopandas as gpd
from pathlib import Path

data_dir = Path("data")
precincts = gpd.read_file(data_dir / "ga_2016_precincts.gpkg")
print(precincts.shape)
print(precincts.crs)
```

## Georgia demonstration ensemble

{download}`Download the Georgia demonstration ensemble
<../../_static/data/ga_congressional_ensemble.json>` (592 KB).

The file contains 1,000 plans with fourteen district-level Black and White voting-age population
shares per plan. Its original chain configuration was not retained, so use it to learn data
shaping and plotting rather than to draw substantive conclusions about an ensemble.

<!-- docs-test: skip -- requires the reader to download the linked JSON file -->
```python
import json

records = json.loads(
    (data_dir / "ga_congressional_ensemble.json").read_text(encoding="utf-8")
)
print(len(records), records[0].keys())
```

## Colorado scoring bundle

{download}`Download the Colorado scoring BENDL bundle
<../../_static/data/co_vtd_scoring_10000.bendl>` (17 MB).

The bundle contains a 10,000-step ReCom chain, its dual graph, projected VTD geometry, population
and election columns, and fixture provenance. The {doc}`BENDL scoring guide <../scoring/bendl>` explains
how to verify and inspect its embedded resources before evaluation.

## Census examples

The Census guides show the live `gerrytools.data` calls and execute against small committed
response extracts. In a project, make the live call with your API key and save the returned frame
locally. The {doc}`data overview <overview>` explains credentials, GEOIDs, and rate limits.
