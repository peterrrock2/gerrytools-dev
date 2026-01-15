- [ ] Make the `ben` and `mgrp` modes able to interact with the stdin of the docker
  container so we don't always have to overwrite.


## Major Rewrites and updates of the modules

These need to be brought up-to-date with the current version of GerryChain and 
Maup. They should also have redundant functionality trimmed.

- [ ] Data
  - [ ] ACS
  - [ ] AssignmentCompressor (probably delete b/c of BEN)
  - [ ] Census
  - [ ] EstimateCVAP
  - [ ] Fetch
  - [ ] Geometries
  - [ ] ReMap
  - [ ] URLs

- [ ] Geometry
  - [ ] Compactness
  - [ ] DataFrame
  - [ ] Dissolve
  - [ ] DualGraph
  - [ ] Optimize
  - [ ] Unit Map
  - [ ] Updater

- [ ] Plotting
  - [ ] Annotation
  - [ ] Bins
  - [ ] BoxPlot
  - [ ] Choropleth
  - [ ] Colors
  - [ ] DistrictNumbers
  - [ ] DrawGraph
  - [ ] DrawPlan
  - [ ] Gifs
  - [ ] Histogram
  - [ ] MultiDimensional
  - [ ] ScatterPlot
  - [ ] SeaLevel
  - [ ] Utils
  - [ ] Violin

- [ ] Scoring
  - [ ] Contiguity
  - [ ] Demogrpahics
  - [ ] Partisan
  - [ ] Population
  - [ ] Scores
  - [ ] Splits
  - [ ] Types

- [ ] Utilities
  - [ ] JSON
  - [ ] rename


### GerryTools

- [x] Update the developer workflow to use a makefile and uv

- [ ] Ben module
    - [ ] Improve integration with PyBen
    - [ ] Add some python bindings for the MSMS parser and the SMC parser
        - [ ] In the SMC parser and runner, maybe just save in the RDS format and then do the
              conversion on top of that. 
        - [ ] In MSMS, allow for arbitrary numbers of levels to be parsed

- [ ] Data Module
    - [ ] Update the ACS5 module to use the new Census API
    - [ ] Try to remove some of the dependencies of this module
    - [ ] Add some fallback mirrors for the data (e.g. Chicago's mirrors)
    - [ ] Change the way that we estimate CVAP (WolfRam?)

- [ ] Geometry Module
    - [ ] Improve the way that we calculate dispersion. Move over to scipy implementation.
    - [ ] Remove the dissolve function
    - [ ] Remove the dualgraph function
    - [ ] Remove the invert function
    - [ ] Update `minimize_dispersion` and `minimize_dispersion_with_parity`
    - [ ] Remove unit map (done in Maup now)

- [ ] MGRP Module
    - [ ] Fix api reference to actually grab all of the functions for this
    - [ ] Update the docker image for all of the methods
    - []

- [ ] Plotting Module
    - [ ] Make examples showing how all of the outputs look in this
    - [ ] Make some nice interfaces for the plots we needed for UT
    - [ ] Change `arrow` to `add_arrow`
    - [ ] Make a colormaps module (some with districtr colors, some with latex colors, etc.)
        - [ ] Add some of the Matplotlib and seaborn colormaps into this module for easy use
        - [ ] Making the names constants would also be good.
    - [ ] Figure out what "gif_multidimentional" is doing
    - [ ] Change `ideal` to `add_vertical_lines` and then add many of them. Include ability to
        jitter the lines if they might overlap
    - [ ] Figure out what multidimensional means
    - [ ] Update scatterplot so that you can add labels to the points and allow for sorting
        the values
    - [ ] Figure out what the sealevel plot is doing
    - [ ] Improve the interface for violin plots

- [ ] Scoring Module
    - [ ] Add function to convert NX graph to dataframe
        - [ ] Should probably use Polars under the hood for this
    - [ ] Rethink the interface and workflow
        - [ ] Parallelize!!
             - [ ] Check free memory, dataframe size, and number of available cores before doing this
        - [ ] Add a way of saving all of this data
    - [ ] Need to make Reock much, much faster. Shapely will probably help here.
    - [ ] Change names of `summarize` and `summarize_many`. Perhaps turn into a class and add a 
        way to print. 
    - [ ] Remove `unassigned_population` and `unassigned_units`. No clue why these exit.

- [ ] Utilities Module
    - [ ] Change name to `gerrytools.utils`
    - [ ] Not sure what the utility of all the functions currently present in here are.

