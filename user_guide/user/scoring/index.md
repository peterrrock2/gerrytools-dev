# Scoring

Choose the scoring workflow that matches the size and location of the plans:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Score a GerryChain run
:link: basic
:link-type: doc

Evaluate a few partitions together and attach the same metrics as GerryChain updaters.
:::

:::{grid-item-card} Score a BENDL ensemble
:link: bendl
:link-type: doc

Inspect selected assignments or stream a complete recording into Parquet result tables.
:::

::::

The scoring module exposes the same metric families through three interfaces:

| Interface | Use |
| --- | --- |
| Lowercase functions | Evaluate one assignment from a GeoDataFrame or GerryChain partition |
| Capitalized metrics with `PlanEvaluator` | Reuse graph and geometry resources across metrics or plans |
| `gerrytools.scoring.formulas` | Apply formulas to arrays already aggregated by district |

The two tutorials use `PlanEvaluator`. The first works with an ordinary GerryChain run, while the
second uses BENDL input and streamed output.

## One plan

Lowercase functions take the source first and the assignment second. Additional arguments name
the columns required by the metric:

<!-- docs-test: skip -- uses the reader's GeoDataFrame -->
```python
import gerrytools.scoring as scoring

population = scoring.tally(precincts, "DISTRICT", columns="TOTPOP")
bvap_share = scoring.demographic_shares(
    precincts,
    "DISTRICT",
    subgroup="BVAP",
    total="VAP",
)
compactness = scoring.polsby_popper(precincts, "DISTRICT")
```

The assignment may be a GeoDataFrame column name, an assignment vector, or a GerryChain
`Partition`, as documented by each function.

### Assignment forms

| Form | Interpretation |
| --- | --- |
| GeoDataFrame column name | District labels are read from that column |
| Mapping | Each graph node is mapped to a district label |
| Sequence | District labels follow graph-node order |
| GerryChain `Partition` | The partition assignment and graph are used together |

Lowercase functions accept the forms supported by that metric's required resources. Geometry
metrics need a GeoDataFrame, while graph metrics can use a graph or partition. `PlanEvaluator`
uses one graph as the common unit order for every registered metric.

## Metric families

Every row below has both a lowercase single-plan function and the corresponding capitalized metric
unless noted otherwise.

### Aggregation, population, and demographics

| Function / metric | Result |
| --- | --- |
| `tally()` / `Tally` | Sums one or more columns by district |
| `tally_by_region()` / `TallyByRegion` | Sums columns by fixed region and district |
| `population_deviations()` / `PopulationDeviations` | Signed district deviations from ideal population |
| `max_absolute_population_deviation()` / `MaxAbsolutePopulationDeviation` | Largest absolute district deviation |
| `max_population_deviation()` / `MaxPopulationDeviation` | Top-to-bottom district population range |
| `demographic_shares()` / `DemographicShares` | District subgroup shares |
| `districts_above_threshold()` / `DistrictsAboveThreshold` | Count of districts above a subgroup threshold |

### Elections and partisan scores

| Function / metric | Result |
| --- | --- |
| `district_vote_shares()` / `DistrictVoteShares` | Two-party district vote shares |
| `district_wins()` / `DistrictWins` | District win indicators |
| `seats()` / `Seats` | Seats won |
| `overall_vote_share()` / `OverallVoteShare` | Aggregate two-party vote share |
| `efficiency_gap()` / `EfficiencyGap` | Wasted-vote efficiency gap |
| `simplified_efficiency_gap()` / `SimplifiedEfficiencyGap` | Equal-turnout seat-vote formula |
| `mean_median()` / `MeanMedian` | Median minus mean district vote share |
| `partisan_bias()` / `PartisanBias` | Partisan bias at 50 percent under uniform swing |
| `partisan_gini()` / `PartisanGini` | Unsigned partisan Gini under uniform swing |

### Scores across elections

These functions accept one party/opposition column pair per election.

| Function / metric | Result |
| --- | --- |
| `competitive_contests()` / `CompetitiveContests` | Contests within a margin around 50 percent |
| `party_wins_by_district()` / `PartyWinsByDistrict` | Party wins by district across elections |
| `swing_districts()` / `SwingDistricts` | Districts not consistently won by either side |
| `party_districts()` / `PartyDistricts` | Districts won by the party in every election |
| `opposition_party_districts()` / `OppositionPartyDistricts` | Districts won by the opposition in every election |
| `aggregate_seats()` / `AggregateSeats` | Wins across every election and district |
| `mean_signed_seat_vote_gap()` / `MeanSignedSeatVoteGap` | Mean signed seat-share minus vote-share gap |
| `mean_absolute_seat_vote_gap()` / `MeanAbsoluteSeatVoteGap` | Mean absolute seat-share minus vote-share gap |

### Regions and compactness

| Function / metric | Result |
| --- | --- |
| `eguia()` / `Eguia` | Seat share against a population-weighted regional benchmark |
| `cut_edges()` / `CutEdges` | Cut-edge count or summed edge weight |
| `region_splits()` / `RegionSplits` | Fixed regions assigned to multiple districts |
| `region_pieces()` / `RegionPieces` | Occupied region-district pairs |
| `region_parts()` / `RegionParts` | Connected region-district parts |
| `polsby_popper()` / `PolsbyPopper` | Polsby-Popper compactness |
| `schwartzberg()` / `Schwartzberg` | Schwartzberg compactness |
| `reock()` / `Reock` | Minimum-enclosing-circle compactness |
| `convex_hull_ratio()` / `ConvexHullRatio` | Convex-hull compactness |
| `state_clipped_convex_hull_ratio()` / `StateClippedConvexHullRatio` | State-clipped convex-hull compactness |
| `population_polygon()` / `PopulationPolygon` | District-owned population divided by the full weight of population polygons intersecting its convex hull |

## Reuse metrics with `PlanEvaluator`

Capitalized metric objects describe their required columns and output names. Register them once,
then evaluate one assignment, an iterable of assignments, or an encoded ensemble:

<!-- docs-test: skip -- uses the reader's graph, geometry, and assignments -->
```python
import gerrytools.scoring as scoring

evaluator = scoring.PlanEvaluator(graph, geometry=precincts)
evaluator.add_metrics(
    scoring.Tally("TOTPOP", "VAP", "BVAP"),
    scoring.PolsbyPopper(source="geometry"),
)

one_plan = evaluator.evaluate(assignment)
many_plans = evaluator.evaluate_many(assignments)
streamed = evaluator.evaluate_stream("plans.bendl", "scores")
```

The evaluator methods cover the common execution modes:

| Method | Result |
| --- | --- |
| `add_metric()` / `add_metrics()` | Register one or several metric descriptions |
| `evaluate()` | Score one assignment or partition |
| `evaluate_many()` | Score an iterable in memory |
| `evaluate_stream()` | Score BEN, XBEN, or BENDL input into Parquet tables |
| `to_updaters()` | Expose the registered metrics as GerryChain updaters |

`evaluate_many()` accepts optional `sample_ids`, progress display, and label-invariant uniqueness
counts. `evaluate_stream()` accepts a sample limit and batch size so the complete assignment
stream does not need to be materialized in Python.

### Result objects

`PlanEvalResult` and `EnsembleEvalResult` are read-only mappings keyed by registered metric name.
Indexing returns a scalar, Series, or DataFrame according to the metric's logical shape:

| Metric shape | One plan | Many plans |
| --- | --- | --- |
| One plan-level value | scalar | Series indexed by sample |
| Several plan-level values | Series | DataFrame |
| One value per district | Series indexed by district | samples-by-district DataFrame |
| Several values per district | DataFrame | DataFrame with metric/district columns |
| Values by region and district | region-indexed DataFrame | sample/region-indexed DataFrame |

Use `.metrics` for registration order and `.array(name)` for the corresponding immutable NumPy
array. `EnsembleEvalResult.summary` reports sample, accepted, and optional uniqueness counts.

`evaluate_stream()` returns an `EvaluationRun`. Its main interface is:

<!-- docs-test: skip -- requires a completed streamed scoring run -->
```python
from gerrytools.scoring import EvaluationRun

run = EvaluationRun.open("scores")
available = run.metrics
seat_counts = run.read("seats", expand_repetitions=True)
frame_metadata = run.frames
```

`read()` reconstructs the same logical pandas shape used by in-memory evaluation. `raw()` exposes
the physical Parquet table when frame offsets or repetitions are needed directly.

Before loading a complete result, `EvaluationRun` estimates the operation's peak memory from the
manifest and Parquet metadata. It emits a warning at 2 GiB and raises `EvaluationMemoryError` at
8 GiB. These fixed thresholds do not depend on currently available machine memory. Pass
`allow_large=True` when the machine deliberately has enough capacity:

<!-- docs-test: skip -- requires a completed streamed scoring run -->
```python
from gerrytools.scoring import EvaluationMemoryError

try:
    seat_counts = run.read("seats")
except EvaluationMemoryError:
    for batch in run.iter_batches("seats"):
        process(batch)
```

On a machine deliberately provisioned for the estimated allocation, override the cap instead:

<!-- docs-test: skip -- requires a completed streamed scoring run -->
```python
seat_counts = run.read("seats", allow_large=True)
```

Process iterator results one batch at a time. Concatenating every batch recreates the same large
allocation the iterator is intended to avoid. `iter_raw_batches()` and `iter_frame_batches()` are
the corresponding physical-table and frame alternatives. When repetitions are expanded,
`batch_size` limits logical sample rows, including when one accepted frame spans several batches.

Result reads decode Parquet columns serially to make peak memory more predictable. This can reduce
throughput for wide metrics; there is no threading option in the initial memory-safe API.
As with other streaming readers, validation can fail after earlier valid batches have been yielded.
Iterators validate each metric's frame sequence independently; only eager reads compare frame
columns across metrics.

## Array formulas

The functions in `gerrytools.scoring.formulas` operate directly on NumPy-compatible arrays. The
last axis is districts; cross-election functions use the preceding axis for elections.

<!-- docs-test: skip -- uses the reader's district-level arrays -->
```python
from gerrytools.scoring import formulas

seat_count = formulas.seats(democratic_votes, republican_votes)
gap = formulas.efficiency_gap(democratic_votes, republican_votes)
deviations = formulas.population_deviations(district_populations)
```

The formula module includes:

| Family | Functions |
| --- | --- |
| Election outcomes | `district_vote_shares`, `district_wins`, `seats`, `overall_vote_share` |
| Partisan scores | `efficiency_gap`, `simplified_efficiency_gap`, `mean_median`, `partisan_bias`, `partisan_gini` |
| Cross-election scores | `competitive_contests`, `party_wins_by_district`, `swing_districts`, `party_districts`, `opposition_party_districts`, `aggregate_seats`, `mean_signed_seat_vote_gap`, `mean_absolute_seat_vote_gap` |
| Population and demographics | `population_deviations`, `max_absolute_population_deviation`, `max_population_deviation`, `demographic_shares`, `districts_above_threshold` |
| Regional and compactness formulas | `eguia`, `schwartzberg` |

Functions preserve every axis before their documented election or district axes, so the same call
can evaluate one plan or a batch. Partisan bias and partisan Gini accept
`turnout_model="equal"` or `turnout_model="observed"`; `TurnoutModel` is the exported type alias
for those values.

The {doc}`scoring API <../../api/scoring>` documents signatures, result shapes, sign conventions,
turnout models, and geometry-source options.

```{toctree}
:hidden:
:maxdepth: 1

Basic GerryChain scoring <basic>
Scoring BENDL files <bendl>
```
