# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

### Changed

- Huge updates to project structure. Previous project structure:

```console
gerrytools
├── ben
│   ├── binary_ensemble.py
│   ├── docker_manager.py
│   ├── __init__.py
│   ├── parse.py
│   └── reben.py
├── data
│   ├── acs.py
│   ├── AssignmentCompressor.py
│   ├── census.py
│   ├── estimatecvap.py
│   ├── fetch.py
│   ├── geometries.py
│   ├── __init__.py
│   ├── remap.py
│   └── URLs.py
├── geometry
│   ├── compactness.py
│   ├── dataframe.py
│   ├── dissolve.py
│   ├── dualgraph.py
│   ├── __init__.py
│   ├── optimize.py
│   ├── unitmap.py
│   └── updater.py
├── __init__.py
├── mgrp
│   ├── __init__.py
│   ├── run_container.py
│   └── runners
│       ├── forest.py
│       ├── __init__.py
│       ├── recom.py
│       └── smc.py
├── plotting
│   ├── annotation.py
│   ├── bins.py
│   ├── boxplot.py
│   ├── choropleth.py
│   ├── colors.py
│   ├── districtnumbers.py
│   ├── drawgraph.py
│   ├── drawplan.py
│   ├── gifs.py
│   ├── histogram.py
│   ├── __init__.py
│   ├── latexcolors.json
│   ├── multidimensional.py
│   ├── scatterplot.py
│   ├── sealevel.py
│   ├── utils.py
│   └── violin.py
├── scoring
│   ├── contiguity.py
│   ├── demographics.py
│   ├── __init__.py
│   ├── partisan.py
│   ├── population.py
│   ├── scores.py
│   ├── splits.py
│   └── types.py
└── utilities
    ├── __init__.py
    ├── JSON.py
    └── rename.py
```

  new project structure:

```console
REPLACE WHEN FINISHED
```


### Added

### Removed

- ben
    - none
- data
    - none
- geometry
    - none
- mgrp
    - none
- plotting
    - `multidimensional.py`: This only had one function in it and it was broken because the 
      signatures of the functions that it relied on changed. It also just stacked a scatterplot
      on top of a histogram, and our lab has moved away from stacking plots in python in favor of
      making individual plots and stacking them in LaTeX.
    - `gifs.py`: This only made a gif from the plots made by `multidimensional.py`, so it was 
      removed as well.
    - `latexcolors.json`: This was just a json file with color names and their RGB values. This
      was just loaded by another module, so rather than waste time with the load, it has been
      converted into a python dictionary.
      
- scoring
    - none
- utilities -> REMOVED ALL
    - `rename.py`: This just let you rename a file. Why do this in python? This also does nothing
      for our standard data pipelines.
    - `JSON.py`: This just allowed a user to read in a JSON object as a python object. The `json`
      module already does this perfectly well, and the only thing that this added was that it 
      checked that the attribute names in the JSON object were compatible with shapefile 
      attribute specifications. We are going to try and move to geopackages, and fixing columns
      for use in shapefiles is a infrequent enough task that we need not dedicate a function to it.
