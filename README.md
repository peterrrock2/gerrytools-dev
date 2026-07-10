# gerrytools

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/mggg/gerrytools/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/mggg/gerrytools/tree/main)
[![codecov](https://codecov.io/gh/mggg/gerrytools/branch/main/graph/badge.svg?token=O09GYF7C9X)](https://codecov.io/gh/mggg/gerrytools)
[![PyPI version](https://badge.fury.io/py/gerrytools.svg)](https://badge.fury.io/py/gerrytools)
[![docs](https://img.shields.io/badge/%E2%93%98-Documentation-%230099cd)](https://gerrytools.readthedocs.io/en/latest/)
[![website](https://img.shields.io/badge/%F0%9F%8C%90%20-DDRI%20Lab-%230099cd)](https://data-democracy.org)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-d7ff64.svg)](https://docs.astral.sh/ruff/)

A companion to [GerryChain](https://github.com/mggg/GerryChain), GerryTools is a robust suite of
geometric and algorithmic tools to analyze districting plans and related data. GerryTools is
actively developed and used by the [MGGG Redistricting Lab](https://mggg.org) and our collaborators
to prepare accurate, precise, and clean information for our projects. It is distributed under a
[3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).

## Installation

### Using `pip` (recommended)

To install GerryTools from [PyPi](https://pypi.org/project/gerrytools/), run

```console
pip install gerrytools
```

from the command line. The `gerrytools.ben` API is included in the base installation. To use
`mgrp`, invoke

```console
pip install "gerrytools[mgrp]"
```

That optional API requires [Docker Desktop](https://www.docker.com/get-started/) version 4.28.0 or
later. See [our Docker documentation](https://gerrytools.readthedocs.io/en/latest/topics/docker/).

## Usage

GerryTools is split up into multiple sub-packages, each designed to simplify and standardize
redistricting workflows.

- **`gerrytools.ben`** records GerryChain runs directly to self-describing BENDL files containing
  the graph, node permutation, metadata, and assignment stream. Recorded partitions remain
  available for live analysis, zero-based lookup, and binary-ensemble decoder operations.

- **`gerrytools.data`** deals with the retrieval and processing of data. Here, you can find tools
  for grabbing decennial Census ('10 and '20), ACS 5-year ('12-'20), ACS CVAP Special Tab ('12-'20),
  districtr portal, and 2020 decennial Census geometric data. You can also find tools for moving
  CVAP data to other levels of geometry (e.g. prorating 2019 CVAP on 2019 Census tracts to 2020
  blocks).

- **`gerrytools.plan_comparison`** compares plans by population or area, finds optimal district
  relabelings, and measures population dispersion. Geometry preparation used by scoring remains
  internal.

- **`gerrytools.mgrp`** this module uses a Docker container to allow users to access several
  ensemble methods for generating districting plans on a state. In particular our Rust
  implementation of our `gerrychain` library, `rustrecom`, the Julia implementation of
  [Forest Recom](https://arxiv.org/pdf/2008.08054.pdf), and the R/C++ implementation of
  [Sequential Monte Carlo (SMC)](https://github.com/alarm-redist/redist) are available through this
  module.

- **`gerrytools.plotting`** contains methods for generating extremely high-quality Lab-standard data
  visualizations.

- **`gerrytools.scoring`** prepares graph and geometry resources once, evaluates plans with the
  Rust scoring engine, and provides array-based partisan, population, and demographic formulas.

## Contributing

GerryTools is an active project, and has multiple contributors. If you'd like to contribute, here
are a few house rules:

1. Install `task` and `uv`, then run `task setup` from the repository root. Use
   `task --list-all` to see the available development workflows.

1. Run `task check` before opening a pull request. Use `task format` for Ruff formatting and
   import sorting, `task lint` for Ruff linting, `task test` for the test suite, and `task docs`
   to build the documentation locally.

1. **Write tests.** All changes, major or minor, **must** be accompanied by testing code. Code and
   tests will be immediately reviewed by Lab maintainers.

1. Test coverage must stay **at least** the same; this can be checked by running
   `uv run pytest --cov=gerrytools --cov-report=term-missing` after the tests are added to
   `tests/`.

1. **Write documentation.** Document public APIs and non-obvious invariants. Prefer clear names
   and focused functions over comments that merely restate the code.

We look forward to your contributions!
