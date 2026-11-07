# Copied code

Source: an earlier, related project (the region as a hard constraint, fewest kinks by MIP) at its commit **0f5d4bb**
(2026-11-02). Some of these files came to it from tautfit (https://github.com/xangregg/tautfit); its own VENDORED.md
records their history and its changes to them. Copied, not shared: record any change here, with the reason, so the
projects can be compared later.

## src/core (pure; each has its test from the earlier project in test/)

| file | what it gives lafspline |
|---|---|
| `rng.js` | seeded PRNG and normals |
| `prefix.js` | prefix sums, window means |
| `data.js` | CSV/TSV parsing, column choice, sorting, ties |
| `datasets.js` | example data sets: `# key:` metadata lines, validation, index entries |
| `signals.js` | built-in test signals (Blocks, Bumps, sine, line, logistic ramp, …) and `simulate` |
| `sigma.js` | σ̂: first differences (eq. 10) and trend-robust GSJS, both tie-aware |
| `derivatives.js` | tie groups, divided-difference rows (any spacing), TV and sup norms |
| `lp-solve.js` | `solveLP` (LP text), `solveModel` (ModelData; returns a failed run's status instead of throwing) |
| `extremes.js` | runs, local extremes, step signs (to count extremes and flat runs of a fit) |

| `region.js` | Aₙ's threshold c = σ√(τ log n), residual check, constraint boxes (phase 4) |
| `intervals.js` | interval families: the paper's blocks, overlapping windows, all; ties never split (phase 4) |
| `calibrate.js` | Monte Carlo calibration of τ (phase 4) |
| `regions.js` | Aₙ as a region: `anRegion`, claim, `checkFit`, `calibrateTau`, `nullSample` (phase 4; trimmed) |

`region.js`, `intervals.js` and `calibrate.js` are unchanged, with their tests. `an.js` (Aₙ as boxes for phase 2) is
lafspline's own.

## src/ui

`theme.js`, `tooltip.js`: unchanged.
`chart.js`, `layout.js`: changed (see below).
`export.js` (+ test): `toCsv` and `exportColumns` unchanged; `atObservations` and `fitTables` added (see below).
`lp.js`: trimmed (see below).
`urlstate.js` (+ test): copied in phase 2 with the field table rewritten (see below).
`main.js` and `controls.js` are lafspline's own, patterned on the earlier project's (the data source, paste/drop and
example loading code, and the URL hash and Copy link wiring, are the earlier project's). `format.js` and `info.js`
(the ⓘ explanations) are lafspline's own.
Not copied: `residual.js` (Aₙ's residual view). Phase 4 draws the boxes of the tightest level on the main chart
instead.

## Other

`src/lp-worker.js` (trimmed), `vendor/highs/` (highs-js 1.15.4, MIT; SHA-256 of highs.wasm f13134e3…cd, the same build
as the earlier project's), `tools/check-page.mjs`, `tools/data-index.mjs`, `data/` (CSVs, `index.json`, README),
`styles.css` (the earlier project's, including its section at the end; unused rules are harmless; lafspline's own
rules are appended at the end), `LICENSE` (0BSD), `.gitignore`, `.nojekyll`.

## Changes since copying

- `src/lp-worker.js`: trimmed to the plain LP request (`{id, lp, options}`) and a new generic `model` task (`{id,
  task: 'model', model, options}` → `solveModel`, for QPs given as ModelData with a Hessian). The earlier project's
  tasks (regularize, exact bounds, MIP, kinks, ladder, smooth joins, band region, calibration) are dropped because
  their modules were not copied.
- `src/ui/lp.js`: the matching trim; `solveModelInWorker(model, options)` added; `solveLP` and `cancelAll`
  unchanged.
- `test/lp.test.js`: rewritten, since the earlier project's used `smokeLP` from `lp-build.js` (not copied): the same
  smoke LP inline, the infeasibility test unchanged, and a new test of `solveModel` on the smoke LP as ModelData.
- `src/core/lp-solve.js`: `solveModel(highs, model, options, infoNames)` takes an optional list of HiGHS info
  items (e.g. `qp_iteration_count`) and returns them as `info`; an item the run did not produce is NaN. Additive.
  `src/lp-worker.js` and `src/ui/lp.js` (`solveModelInWorker`) pass it through.
- `src/ui/chart.js`: a band and a fit may carry their own sorted abscissae `x` (lafspline's check grid, which
  includes every data x): the band ribbon and the fit line are drawn there, such a band also sets the y scale,
  and the tooltip looks their values up at the hovered data x. Without `x`, unchanged.
- `src/core/signals.js`: `signalFunction(name, n)`, a signal as a function of t with the rescaling
  `signalValues` applies for n (true f on the check grid). Additive; `test/signals.test.js` has a test for it.
  Blocks, Bumps and Doppler move from `SIGNALS` (the app's menu) to `TEST_SIGNALS`, which `simulate`,
  `signalValues` and `signalFunction` still accept, for the tests and tools that use them; a `step` signal (three
  equal flat spans at 0, 1 and ½) is added. The narrow box starts at n = 500, not 2000, so its fit stays responsive.
- `src/ui/chart.js`: axis tick labels have no thousands separator when every value is below 10,000 in magnitude
  (years read 1880, not 1,880); larger values keep d3's default grouping. The earlier project still shows 1,880.
- `src/ui/urlstate.js`: the field table is lafspline's state (band settings, ambition, flat share, κ, ε); the
  constraint codec (`encodeConstraints`, `decodeConstraints`, which import the earlier project's `constraints.js`) and
  the old-link rule for τ are dropped; `MAX_N` is 5,000 and a signal starts at its own n up to that (the earlier
  project caps it at 200 for the MIP); `nseg` may be `auto` (null); `penalty` (phase 3: `quadratic` or `sparse`);
  `region`, `family`, `calibrate` and `tau` as the earlier project's, and `boxes` (phase 4: which Aₙ boxes are drawn);
  `region` also takes `boot` (the bootstrap band), with `bootWeights` for its weights. `penalty` also takes `shaped`
  (rounded bends), with `bendLength`, `extremumCost` and `inflectionCost`. `source` also takes `local:<id>`, a data
  set in `localdata/` (main.js lists them when the app runs from this machine). `lambdaMode` also takes `scale`,
  with `featureScale` (a share of the x range). `view` (`single` or `gallery`) and
  `panes` (the gallery's sources, comma-separated, 4, 6 or 8 of them, never pasted data; written only in the gallery,
  with the seed and true-f toggle for its simulated panes) are added; a setting is compared with its default as
  written, so a list equal to its default is left out. The hash and data packing code is unchanged. `test/urlstate.test.js` is the earlier project's with the lafspline fields.
- `styles.css`: slider and Advanced-panel rules appended.
- `src/core/regions.js`: trimmed to Aₙ. The earlier project's smoother band (`bandRegion`, `bandClaim`, which need its
  `smoother.js`) and LP framing (`standardRows`, for its `lp-build.js`) are removed, since lafspline's band is
  `band.js` and phase 2 takes the constraints in box form (`an.js`). `scaleStatistic` and `nullSample` lost their
  band branches. `test/regions.test.js` is the earlier project's without the band, LP and kink tests.
- `src/ui/chart.js`: a `hidden` set in the model leaves layers undrawn ('data', 'truth', 'band', a fit's key) and
  out of the legend, while they still set the scales and stay in the tooltip. The Aₙ box shapes are listed once
  (`boxShapes`) and drawn on the canvas or exported. `toSvg({title})` returns the chart as drawn as a standalone SVG
  document: title, legend (laid out as the page's, long labels wrapped), points and boxes as vector shapes, and the
  SVG layers with their computed styles written in as SVG 1.1 attributes (rgba split into color and opacity).
- `src/ui/export.js`: `atObservations` (grid values at the observations, ties sharing their x's value) and
  `fitTables` (lafspline's per-observation and per-grid-point tables) added; `test/export.test.js` has tests for
  both.
- `src/ui/chart.js`: `createMainChart(container, {compact})` draws a gallery pane: shorter (215 px), no axes, grid
  lines or legend of its own, small margins; the tooltip is unchanged. The legend is drawn by `drawLegend`, exported
  so the gallery draws one legend for all its panes from each chart's `legendItems()`.
- `src/ui/layout.js`: `xScale(edges, width, margin)` takes the margins (the compact chart's), `MARGIN` by default.
