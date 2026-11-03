# Code copied from the earlier project

Source: the earlier project at commit **0f5d4bb** (2026-11-02). Some of these files came to the earlier project from
tautfit (https://github.com/xangregg/tautfit, local ../nicefit); the earlier project's VENDORED.md records their history and
its own changes to them. Copied, not shared: record any change here, with the reason, so the projects can be
compared later.

## src/core (pure; each has its the earlier project test in test/)

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

Not yet copied (phase 4, Aₙ as a region): `region.js`, `regions.js`, `intervals.js`, `calibrate.js`.

## src/ui

`chart.js`, `layout.js`, `theme.js`, `tooltip.js`, `export.js` (+ test): unchanged.
`lp.js`: trimmed (see below).
Not yet copied: `urlstate.js` (phase 2: its field table is the earlier project's and it imports the earlier project's `constraints.js`;
it will be copied with the table rewritten), `residual.js` (Aₙ's residual view; phase 4 if needed).

## Other

`src/lp-worker.js` (trimmed), `vendor/highs/` (highs-js 1.15.4, MIT; SHA-256 of highs.wasm f13134e3…cd, the
same build as the earlier project's), `tools/check-page.mjs`, `tools/data-index.mjs`, `data/` (CSVs, `index.json`,
README), `styles.css` (unchanged, including the earlier project's section at the end; unused rules are harmless and some,
such as the claim text and `[hidden]`, will be used), `LICENSE` (0BSD), `.gitignore`, `.nojekyll`.

## Changes since copying

- `src/lp-worker.js`: trimmed to the plain LP request (`{id, lp, options}`) and a new generic `model` task
  (`{id, task: 'model', model, options}` → `solveModel`, for QPs given as ModelData with a Hessian). The earlier project's
  tasks (regularize, exact bounds, MIP, kinks, ladder, smooth joins, band region, calibration) are dropped
  because their modules were not copied.
- `src/ui/lp.js`: the matching trim; `solveModelInWorker(model, options)` added; `solveLP` and `cancelAll`
  unchanged.
- `test/lp.test.js`: rewritten, since the earlier project's used `smokeLP` from `lp-build.js` (not copied): the same smoke
  LP inline, the infeasibility test unchanged, and a new test of `solveModel` on the smoke LP as ModelData.
