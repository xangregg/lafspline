# PLAN.md — lafspline (least ambitious fit, P-spline version)

## 0. Goal

Given data (xᵢ, yᵢ), find a smooth function that is as unambitious (as flat, as straight) as the data allow, in
two phases:

1. **Region.** An ordinary P-spline fit gives a simultaneous confidence region around a center curve.
2. **Fit.** A second P-spline is penalized severely for leaving the region, lightly for distance from the
   region's center, and for ambition (pressure toward flat and straight lines).

lafspline grew out of tautfit (../nicefit, a faithful Davies, Kovac & Meise implementation) and the earlier project
(the earlier project: the region as a hard constraint, counted features minimized by a MIP). It keeps their lessons
(regions as data, coverage by simulation, display honesty) but replaces the MIP with penalized least squares on a
smooth basis. What changes:

| | the earlier project | lafspline |
|---|---|---|
| basis | linear spline at the data x | B-splines on equally spaced knots (P-splines) |
| region | hard constraint | penalty: severe outside, slight inside |
| simplicity | fewest kinks, extremes, … counted exactly (MIP, proven or with a gap) | difference penalties, optionally L0-like (approximate) |
| speed | seconds to minutes; stops at a time limit for large n | milliseconds (small banded systems) |
| tuning | none beyond the region | an ambition knob: long-term versus short-term patterns |

References: Eilers & Marx (1996), *Flexible smoothing with B-splines and penalties*; Eilers & Marx (2021),
*Practical Smoothing: The Joys of P-splines* (R package JOPS); Bollaerts, Eilers & Van Mechelen (2006), shape
constraints by iterated asymmetric penalties; Frommlet & Nuel (2016), the adaptive ridge for L0 penalties; Rippe,
Meulman & Eilers (2012), segmentation with an L0 penalty; Krivobokova, Kneib & Claeskens (2010), simultaneous
bands for penalized splines.

## 1. Concepts

### 1.1 P-spline core

- **Basis:** B-splines of degree q (default 3) on K equally spaced segments over [min x, max x]. With a penalty the
  number of segments matters little once it is large enough; default to about 40, fewer for small m (distinct x).
- **Penalty:** λ‖D_d α‖², D_d the d-th order difference matrix on the coefficients (default d = 2; d = 3 makes
  the λ → ∞ limit a quadratic).
- **Solve:** (BᵀWB + λD_dᵀD_d) α = BᵀWy. The matrix is banded (bandwidth q + 1), so a banded Cholesky is
  O(K q²) after forming BᵀB in O(n q²).
- **λ for phase 1:** GCV or AIC over a grid of log λ first; REML or Fellner–Schall updates later (open question 7).
- **Effective dimension** ED = trace(H), **σ̂** from RSS/(n − ED); the earlier project's first-difference σ̂ (eq. 10,
  tie-aware) and GSJS remain available as robust alternatives.
- **Covariance:** Bayesian V = σ̂²(BᵀB + λDᵀD)⁻¹ by default (Wahba, Nychka: it allows for the smoothing bias on
  average across x); frequentist sandwich σ̂²(BᵀB + λDᵀD)⁻¹BᵀB(BᵀB + λDᵀD)⁻¹ as an option.
- Ties and unequal spacing need nothing special: the basis is evaluated at each x, and weights handle replicates.

### 1.2 Phase 1: the region

- **Check grid:** the distinct data x plus a fine uniform grid (default 4 points per segment), G points in all.
  Checking only at the knots would miss excursions between them, and the denser grid is cheap (G is a few
  hundred, and every grid row of B has only q + 1 nonzeros).
- **Simultaneous band** on the grid: center c_j = (Bα̂)_j, half-width h_j = q_α·se_j. The critical value q_α comes
  from Monte Carlo of max_j |(B_G z)_j| / se_j with z ~ N(0, V), the same machinery as the earlier project's smoother band.
- **What it covers:** the frequentist band covers E f̂ (the smoother's mean), not f. Options: undersmoothing (λ
  divided by a factor; in the earlier project, half the GCV bandwidth raised coverage of f from 0–89% to 92–98%) and the
  Bayesian covariance. A coverage simulation on the signals decides the defaults.
- **Regions as data,** as in the earlier project: center, lower and upper edges on the grid, label, coverage claim with its
  assumptions, `check(g)` (ratios, violations), and how to draw it. Aₙ comes later as a second implementation
  (phase 4).

### 1.3 Phase 2: the fit

A second P-spline g = B̃β (the same or a denser basis), with β minimizing

    S(β) = κ·Σ_j [ (g_j − u_j)₊² + (l_j − g_j)₊² ] / h_j²     outside the region: severe
         + ε·Σ_j ((g_j − c_j) / h_j)²                          inside: a slight pull to the center
         + λ₁‖D₁β‖² + λ₂‖D₂β‖²                                 ambition: toward flat, toward straight

over the check grid j, with l = c − h and u = c + h. Dividing by h_j makes the region terms unitless, so the pull
is stronger where the band is narrow. The data enter phase 2 only through the region (c, h); a data term
‖y − g‖² is not needed while the center pull plays its part (open question 3).

Two ways to solve it:

- **Iteration** (Bollaerts, Eilers & Van Mechelen): fix the indicator weights v_j = [g_j > u_j] and
  w_j = [g_j < l_j], solve the resulting penalized least squares (banded, size K̃), update the weights, and repeat
  until they stop changing (usually a few iterations). Pure JavaScript, no solver.
- **Quadratic programming** (Eilers & Marx later handled some constraints this way instead of by iteration): the
  same objective with slack variables, l − s ≤ g ≤ u + s and κ·Σs², or a hard region with s = 0. HiGHS (vendored in
  the earlier project) solves convex QPs: highs-js accepts a Hessian in ModelData (active-set QP). Exact, no iteration count to
  worry about, and the hard-region variant comes for free. In that variant, λ → ∞ gives the least rough function
  inside the region, the QP analogue of tautfit's `regularize.js` (min TV in Aₙ by LP).

Phase 0 compares the two on the same problems and picks the default.

**User knobs:**
- **ambition:** one multiplier on (λ₁, λ₂), on a log scale: a slider from detail (short-term patterns) to
  simplicity (long-term patterns). Advanced: the balance between flat and straight, or λ₁ and λ₂ separately.
- **region:** coverage α, undersmoothing, and the phase-1 λ (GCV or manual), the region's own long-term versus
  short-term choice.
- **advanced:** κ (how hard the region is), ε (center pull), the check-grid density.

### 1.4 Sparse ambition (optional)

Quadratic difference penalties make the fit straighter everywhere but give no exactly flat or straight pieces.
The adaptive ridge reweights them each iteration, w_r = 1 / ((Dβ)_r² + δ²), which approximates counting the
nonzero differences (an L0 penalty). On cubic B-splines with equal knots, coefficients linear in their index give
a spline that is linear there, so a sparse D₂β gives straight pieces joined by short smooth bends: few kinks, with
the earlier project's "smooth joins" built in. A sparse D₁β gives flat pieces with smooth transitions (few jumps or steps). It
is an approximation, with no proof of optimality like the earlier project's MIP; report it as such.

### 1.5 Reporting (display honesty)

- Every phase-2 fit is re-checked against the region on the check grid and at the data x: largest ratio
  |g − c|/h and the number of violations. A soft region can be violated, and the readout must say so.
- The region's claim: what it covers (f or E f̂), at what level, under which assumptions (independent noise,
  constant σ, the band's λ).
- The knob values, the phase-2 effective dimension, and the solver's status (iterations and whether the weights
  converged, or the QP status).

### 1.6 The ladder

Ideally not needed: as the ambition grows, a D₂ penalty drives the fit toward a straight line and a D₁ penalty
toward a constant, so if one of those fits inside the region, high ambition should find it. Keep it in view as a
report: "a constant (or a straight line) fits inside the region" is one QP each, and tells the user when simpler
is enough. Revisit after phase 2.

## 2. Open questions

1. **Normalizing λ₁, λ₂** so the ambition knob means the same across data sets: relative to the phase-1 λ, to an
   effective dimension, or to the λ at which the fit first leaves the region.
2. **Default κ:** large enough that violations are negligible, small enough to keep the iterated system well
   conditioned. The hard-region QP sidesteps this.
3. **A data term in phase 2?** As specified, the data enter only through the region. Revisit if fits drift from
   the data in ways the center pull does not prevent.
4. **Band type and coverage:** Bayesian or frequentist covariance; undersmoothing by default or not; covering f
   versus E f̂ (the earlier project's band claim words this carefully; reuse it).
5. **Check-grid density** (default 4 per segment) and whether to include the data x always (yes for now).
6. **Aₙ in penalty form:** each Aₙ constraint |Σ_I (y − g)| ≤ c√|I| is linear in β, so it becomes the same kind
   of penalty on the excess; σ̂, interval families and calibrated τ come from the earlier project.
7. **Phase-1 λ:** GCV versus REML or Fellner–Schall; whether to offer it as the long-term/short-term knob instead
   of (or as well as) the ambition multiplier. GCV can be bimodal (a noisy line: a shallow local minimum at
   ED ≈ 6 and the global one at the line, ED → 2), so the grid stays ahead of any local refinement; and it
   undersmooths serially correlated data (Nile: ED ≈ 15).
8. **Local σ:** a band from a constant-σ smoother is too wide where the noise is small and too narrow where it is
   large (the earlier project's dog-ages example). A variance model (P-spline on log squared residuals) would give a
   heteroscedastic band.

## 3. Phases

### Phase 0 — Scaffold and risk checks
- [x] Repository, LICENSE (0BSD), `.gitignore`, `.nojekyll`; copy the reusable modules from the earlier project (list below)
      and record them in VENDORED.md; their tests pass; scaffold page loads without console errors.
- [x] P-spline core (`src/core/pspline.js`): B-spline basis (equally spaced knots, any degree), difference
      matrices, banded penalized solve, hat-matrix diagonal and ED, GCV/AIC over log λ, σ̂, Bayesian and sandwich
      covariance. Tests: partition of unity, polynomial reproduction (λ → ∞ gives the degree-(d − 1) least-squares
      polynomial), agreement with a dense solve, and fixtures from JOPS or mgcv (R) for a few data sets.
- [ ] QP risk check: the phase-2 objective as a HiGHS QP in Node and in the worker, for K̃ up to ~200 and G up to
      ~2000; compare with the iterated weights (same fit within tolerance, time, iterations). Decide the default.

### Phase 1 — The region
- [ ] Simultaneous band on the check grid (`src/core/band.js`), Monte Carlo critical value, undersmoothing;
      coverage simulation over signals × n × noise (as the earlier project's `tools/sim-band.mjs`).
- [ ] UI: data input, the phase-1 fit and band, readouts with the claim (from the earlier project's main.js and controls.js).

### Phase 2 — The second-phase fit
- [ ] Objective and solver (`src/core/fit.js`); re-check against the region. Tests: a fit stays inside when κ is
      large; ambition → ∞ gives the least-squares line (D₂) or constant (D₁) toward the center, when they fit;
      ambition → 0 gives the center c; iteration and QP agree.
- [ ] Knobs in the UI (ambition slider, flat/straight balance, advanced κ and ε); URL state; diagnostics on the
      examples, side by side with the earlier project's fits.

### Phase 3 — Sparse ambition and the ladder report
- [ ] Adaptive-ridge option for D₁ and D₂; count the resulting flat and straight pieces and bends.
- [ ] "A constant / a straight line fits inside the region" report (open question in §1.6).

### Phase 4 — Aₙ as a region
- [ ] Aₙ in penalty form (open question 6), with the earlier project's interval families and calibrated τ.

### Phase 5 — Polish and deploy
- [ ] About panel, CSV export, GitHub Pages, live-site check.

## 4. What to copy from the earlier project

- `src/core`: `rng.js`, `prefix.js`, `data.js`, `datasets.js`, `signals.js`, `sigma.js`, `derivatives.js`
  (tie groups), `lp-solve.js` (`solveModel`, which returns a failed run's status instead of throwing),
  `extremes.js` (to count extremes and flat runs of a fit); for phase 4, `region.js`, `regions.js`,
  `intervals.js`, `calibrate.js`.
- `src/ui`: `chart.js`, `layout.js`, `theme.js`, `tooltip.js`, `export.js`, `urlstate.js`, `lp.js`; `src/lp-worker.js`
  trimmed to the tasks used.
- `tools/check-page.mjs`, `tools/data-index.mjs`; `data/` (the example CSVs, `index.json`, and the README with its
  license section); `styles.css`; `vendor/highs/`.

## 5. Lessons from tautfit and the earlier project worth keeping

- Coverage is measured, not assumed: simulate it for each region and say which applies. τ = 3 under-covers at
  small n; calibrating by Monte Carlo fixes the claim.
- A smoother band covers the smoother's mean; undersmoothing (half the GCV bandwidth) helps it cover f.
- σ̂ from first differences inflates on small, steep data; GSJS is trend-robust; both are tie-aware. A single σ̂
  misfits data whose noise varies (dog-ages).
- Ties: one value per distinct x; intervals and checks never split tied observations.
- HiGHS: ranged rows need ModelData (the LP reader aborts on them); a failed run can throw, so read the model
  status; time limits must be shared across stages.
- Display honesty: every fit says which region it is checked against and how well it fits; approximate methods
  say so.
- Example data: third-party measurements, cited, not under the repository's license; keep files LF.
