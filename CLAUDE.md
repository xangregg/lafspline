# CLAUDE.md

## Project

**lafspline** ("Least Ambitious Fit" using P-splines): a browser app (served from GitHub Pages) for
nonparametric regression in two phases. A P-spline fit gives a simultaneous confidence region; a second P-spline is
then penalized severely for leaving the region, slightly for distance from its center, and for ambition (pressure
toward flat and straight lines), with a user knob for long-term versus short-term patterns. It grew out of
**tautfit** (https://github.com/xangregg/tautfit, a faithful Davies, Kovac & Meise implementation) by way of an
earlier project that is not public (see VENDORED.md); lafspline is free to depart from both.

The spec, decisions and milestones are in **PLAN.md**. Read it before starting a phase, and tick its checkboxes
as work is completed. Code copied from the earlier project (some of it originally tautfit's) is listed in
**VENDORED.md**.

## Hard rules

- **No bundler, no build step.** Plain ES modules served as static files. Do not add Vite, webpack,
  esbuild, TypeScript, Babel or Jest. Ask before adding any dependency.
- **Every relative import uses an explicit `.js` extension** (required by both browsers and Node ESM).
- **`src/core/` is pure**: no DOM, no `window`, no bare-specifier imports (`'d3'`), no network. It must
  run unchanged in Node for tests and in the browser.
- Bare imports (`d3`) are allowed only in `src/ui/`, resolved by the import map in `index.html`
  (jsDelivr ESM URLs, pinned versions).
- HiGHS (LP/QP solver, WebAssembly), if used, is vendored in `vendor/highs/` (ES-module build `highs.mjs` +
  `highs.wasm`) and used in the browser only through the module worker `src/lp-worker.js`. Node tests load it
  from the dev dependency `highs`.
- Do not add Claude attribution to commit messages.

## Layout

```
index.html              import map, layout, loads src/ui/main.js as a module
styles.css
src/core/               pure algorithms: pspline.js (basis, penalties, solve, λ choice), band.js (phase-1 region),
                        fit.js (phase-2 fit); modules copied from the earlier project (see VENDORED.md)
src/ui/                 D3 charts, controls, state
src/lp-worker.js        module Web Worker wrapping HiGHS (if the QP solver is used)
vendor/highs/           highs.mjs + highs.wasm copied from node_modules/highs/build/
data/                   example data sets (CSV with # label: / # source: lines) and index.json
localdata/              data sets kept on this machine (git ignores all but its README); listed by the app on localhost
tools/check-page.mjs    headless-Chrome page check (console errors, worker exceptions, screenshot)
test/*.test.js          node:test + node:assert/strict
package.json            "type": "module", scripts below
```

## Commands

- Tests: `npm test` (runs `node --test`). Node 20+.
- Serve locally: `npm run serve` (python3 on port **8002**, leaving 8000 and 8001 to related projects), then
  open http://localhost:8002/
- Refresh vendored HiGHS: `npm run vendor:highs`.
- Example data sets: after adding a CSV to `data/`, `npm run data:index` (the test suite fails if
  `data/index.json` is stale). It also writes `localdata/index.json`, which servers that don't list directories
  (WebStorm's built-in one) need to show local data sets.
- Page check (server must be running): `npm run check:page -- http://localhost:8002/ --until '<js expr>'
  --eval '<js expr>' --screenshot out.png`. Exits 1 on any console error, exception, or failed load.

## Conventions

- Plain JS with JSDoc type annotations on every exported function (parameters, return, units).
- Numeric arrays are `Float64Array`. Code indices are **0-based**; papers and the Eilers & Marx book are usually
  1-based. When implementing a formula from a paper or the book, cite its equation or section in a comment and
  state the index conversion.
- Natural log everywhere (`Math.log`).
- Randomness only through `src/core/rng.js` (seeded). Tests must be deterministic.
- Each core module gets a test file. Numerical tests state their tolerance explicitly; where possible, check
  against fixtures computed in R (JOPS or mgcv), with the script that made them kept next to the fixture.
- Prefer clarity over micro-optimization; state the complexity of anything worse than O(n log n) and keep
  O(n²) or exponential paths behind explicit size limits. Penalized systems are banded: solve them as such.
- **Display honesty:** every reported fit says which region it was checked against, whether it stays inside
  (largest ratio, violations), and what the region covers under which assumptions; approximate methods (iterated
  weights, adaptive ridge) say so, and solver results report their status.
- Changes to copied modules are recorded in VENDORED.md (what and why), so they can be compared with the earlier
  project.

## Working style

- Work one phase (or milestone within a phase) at a time. At the end of each, run the tests, make sure the
  page loads without console errors, summarize what changed, and stop for review.
- Don't refactor unrelated code. Don't reformat files you didn't otherwise change.
- For new code, use a separate line for flow of control statements like return, break and continue as well as
  for the bodies of compound statements like if and for.
