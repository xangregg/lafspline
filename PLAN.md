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

**Measured (phase 1, `npm run sim:band`):** 400 replications per row, α = 0.95, B = 1000, σ̂ from the residuals
(ν = n − ED), λ by GCV times the factor, coverage of f checked on the whole check grid. "∋ E f̂" is the band's
coverage of f smoothed with the same λ (the sandwich band's own target); half-width in units of the true σ.

| signal | n | ∋ f: bayes 1 | bayes 1/16 | sandwich 1 | sandwich 1/16 | ∋ E f̂, sandwich 1 | half-width/σ, bayes 1 → 1/16 |
|---|---|---|---|---|---|---|---|
| line | 100 | 94.0% | 94.5% | 88.0% | 91.0% | 88.0% | 0.43 → 0.52 |
| line | 500 | 95.5% | 96.8% | 88.8% | 90.5% | 88.8% | 0.19 → 0.23 |
| sine | 100 | 96.3% | 95.3% | 90.5% | 92.0% | 93.3% | 1.11 → 1.49 |
| sine | 500 | 97.8% | 97.8% | 88.0% | 94.8% | 93.8% | 0.54 → 0.73 |
| ramp | 100 | 87.8% | 96.0% | 75.3% | 91.5% | 92.5% | 0.89 → 1.23 |
| ramp | 500 | 91.3% | 98.0% | 77.5% | 93.3% | 93.8% | 0.46 → 0.63 |

- **Decided: Bayesian covariance, λ = GCV choice × 1/16** (about half the equivalent bandwidth for d = 2, like
  the earlier project's 0.5). It covers f at 94.5–98% on the smooth signals, for a band 1.2–1.4 times as wide as at the
  GCV λ, where the ramp drops to 88–91%.
- The sandwich band covers its own target E f̂ at 94–96% at a fixed λ (test), but only 88–94% when λ is chosen
  on the same data (the line: E f̂ = f, 88%); the claim says λ is treated as fixed. The Bayesian band absorbs
  that too.
- **Jumps are not covered:** HeaviSine and Blocks are covered at 0–4% on the grid, and 0–80% even at the data
  x only, by every configuration. More segments do not help (Blocks n = 500: 0–13% with 100 or 250
  segments): one global λ with a second-difference penalty smooths every jump. The earlier project's local linear band
  reached 96–98% on Blocks only by becoming nearly pointwise. The claim says jumps are not covered; Aₙ (phase 4)
  is the region for such data.
- Too many segments for the data (100 for n = 100) lets GCV choose near-interpolation (ED ≈ 99.6), leaving no
  degrees of freedom for σ̂; `bandRegion` refuses with a reason. The default (≤ m/2 segments) avoids it.

**A bootstrap band (added after phase 4, for comparison; `src/core/bootstrap.js`).** Around the same P-spline,
from B = 1000 refits at the same λ with positive fractional weights on the observations that sum to n, so it
can be asymmetric and wider where the noise is. What the coverage simulation (`tools/sim-band.mjs`, now with
`--noise gauss|skew|hetero` and `boot` configurations) found along the way, 100 replications per row:

- Refits spread like the residuals, which a smoother shrinks: on a sine (n = 200, ED 23) the spread was 0.85 times
  f̂'s. The replicates use residuals divided by √(1 − 2h_i + (HHᵀ)_ii), the exact factor (test).
- A width that follows each point's bootstrap spread undercovers badly (60% for 95%; 22% with skewed noise): the
  spread comes from the few residuals near x. Studentizing each replicate by its own local standard error
  (bootstrap-t) gave 78–86%; a locally studentized band would need q ≈ 5.0 where the true standard error needs
  3.3. Widths pooled over the whole grid overcovered (98–100%). What works: replicates studentized by their global
  σ̂* and the constant-noise standard error, each point standardized by its own RMS for the coverage check, and
  the widths averaged over ±10% of the x range.
- The asymmetry is in the tails: with Exp(1) − 1 noise the true 10⁻³ tails of f̂ − E f̂ (sine, n = 400) were
  1.35–1.6 times as long above as below, but semi-deviations of the replicates saw 1.05. The band's shape is the
  replicates' own quantiles, pooled over the window, at the pointwise tail level that holds for a fraction α of
  them at every point. Its ratio is then 1.29 with matched weights and 1.47 with Dirichlet weights.
- Reflected (bootstrap-t) or unreflected (percentile): within 0–5 points of coverage, in no consistent
  direction; reflected, as theory says.
- Weights: Dirichlet (the Bayesian bootstrap: mean 1, variance 1, skewness 2) against weights matched to a
  resampled count's mean, variance and skewness (1, 1, 1: a generalized gamma, positive but a fifth below 0.1).

Coverage of E f̂ at α = 0.95, λ = GCV × 1/16, over line, sine, ramp and HeaviSine at n = 100 and 500:

| noise | sandwich band | bootstrap, matched (default) | bootstrap, Dirichlet |
|---|---|---|---|
| Gaussian | 88–97% | 91–100% | 88–95% |
| skewed (Exp(1) − 1) | 74–91% | 84–100% (line: 84–86%) | 77–95% |
| heteroscedastic (sd 0.25σ → 1.75σ) | 69–79% | 93–100% | 81–95% |

The matched band is the wider of the two at n = 100 (1.7σ against 1.4σ on the sine) and overcovers there. On
HeaviSine at n = 100 its local widths reach the jump, so it holds f at the data x in 89–98% of samples (sandwich
43–64%). It takes 0.1 s at n = 200, 0.3 s at 1000, 1.4 s at 5000 for 1000 replicates (Bayesian band: 12–36 ms).

**Refits to first order in the weights (default since the rounded-bends work).** On Tukey coal (n = 49, ED 25 at λ ×
1/16: two observations per segment, hat 0.93 at the ends) the band was 15 times the sandwich band's half-width at
1920 and 3.5 times near 1950, with critical values 11–13. About 21% of the matched weights fall below 0.1; when
the few points that pin a segment or an end get such weights, the exact refit A_b⁻¹ barely holds those
coefficients at a small λ and swings far, measuring the fragility of a reweighted design, not f̂'s variability (f̂
is linear in y). Chess recall's end showed the same in milder form (above ÷ below 2.6 at x = 80, so the band's
midpoint rose while f̂ leveled off). The replicate is now A⁻¹Bᵀ(W_b − W)r with the fit's own A, a multiplier
bootstrap: covariance the heteroscedasticity-robust sandwich, tails from the weights' skewness times the residuals'.
Coal 0.33–1.76 times the sandwich band (mean 1.05), chess's end 1.24; elsewhere within a few percent. Coverage of
E f̂ (200 replications, the same rows as above, matched weights): at n = 500 alike (linear 88–98%, exact 88.5–100%);
at n = 100 linear 86–94.5%, exact 88–100% with bands up to 1.6 times as wide (HeaviSine 2.12σ against 3.45σ).
The price: under skewed noise the band leans less (mean reach below ÷ above 1.08–1.18 on a skewed sine, n = 400,
against 1.24–1.30 exact). `refit: 'exact'` remains for comparison (`tools/sim-band.mjs` boot-matched-0.1-exact).
Re-measured with it, the replicate study (B) gives about the same: at B = 2000, edges within a median 0.08–0.11 of
the half-width and the ratio's extremes ±0.02–0.07 from seed to seed.

**App defaults (after the rounded bends):** the bootstrap band, λ × 1/4, rounded bends at 4 decades below a*. The
core defaults (band.js DEFAULT_UNDERSMOOTH 1/16) are unchanged; the claims name the undersmoothing their
simulations used. Coverage at λ × 1/4 (200 replications; line, sine, ramp, HeaviSine; n = 100, 500): the bootstrap
band contained E f̂ in 84–98% (Gaussian 86–96.5%, skewed 84–97%, heteroscedastic 90.5–98%) and f in 79–95.5% on the
smooth three (0–7% on HeaviSine); the Bayesian P-spline band contained f in 94–98.5% on line, sine and ramp (400
replications; sine half-width 1.30σ at n = 100 against 1.49σ at 1/16).

**Phase 2's target.** The pull ε now aims at a target the region may supply (fit.js `target`, default its center).
The bootstrap band supplies the replicates' median reflected as its edges are (the band at level ½), so a long tail
on one side no longer pulls the fit toward it; its edges, and so the check, are unchanged.

**Replicates (B).** `tools/boot-replicates.mjs`: 20 seeds at each B against a band at B = 8000, on neuroimage (n = 28), dog ages (94), ozone
(1197) and simulated sines (n = 200 Gaussian, 400 skewed). The mean half-width is unbiased from B = 500 and varies
by 1.4–3% between seeds at 1000 (1–2% at 2000). The edges vary more: the largest gap to the reference anywhere on
the grid is a median 0.11–0.17 of the half-width at B = 1000 and 0.08–0.12 at 2000 (90th percentile 0.15–0.34 and
0.11–0.22), shrinking as 1/√B. The above ÷ below range, an extreme over the grid, is widened by noise: neuroimage's
upper end is 1.27 at B = 250, 1.11 at 1000 and 1.06 at 2000 against 1.06; the skewed sine's 0.91 at 1000 against
0.82. **Decided:** B = 2000 for n ≤ 1000 (up to about 0.6 s), 1000 above (up to about 1.5 s at n = 5000); the
ratio's tip says its extremes are Monte Carlo estimates.

**Noise check** (`noisecheck.js`, the "σ̂ check" readout of both bands). The band fit's σ̂ against σ̂ from
neighboring observations (sigma.js GSJS: local linear pseudo-residuals, pure error at tied x), which does not depend
on λ, and the lag-1 autocorrelation r₁ of the residuals' means at successive distinct x (judged from 10 distinct x).
Limits from `tools/noise-check-null.mjs` (smooth curves, independent Gaussian noise, λ by GCV × 1 and × 1/4,
n = 25–1000): |log ratio|·√n ≤ 4.0, −5.0 ≤ r₁·√m ≤ 3.0, about the 0.1% points. On the examples at their default
segments: Tukey coal at 80 segments (GCV interpolates, ratio 0.01) reads "follows the noise"; Blocks, Bumps and
Doppler at 40 segments, and HeaviSine at σ = 0.2, "runs" (r₁ 0.58–0.84); ozone "between" (pure error 7.8 against 14.4,
r₁ −0.75: each year adds its own deviation); dog ages "spread" (ratio 1.59, r₁ 0.04); nostalgia "alternates"
(r₁ −0.60, ratio 0.72); the others pass. Nile passes: GCV absorbs its serially correlated noise, which fools the
differences alike. A simpler λ (a rule of thumb from σ̂ and a pilot polynomial) was considered instead of GCV and
not taken: GCV takes 1–8 ms, the pilot fails on the shapes this app is for, and the claims' coverage was simulated
with GCV.

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

**Decision (phase 0): iterated weights,** made a semismooth Newton method with an exact line search
(`src/core/fit.js`; `npm run bench:qp`, and `tools/bench-qp.html` for the browser worker). On 72 cases (sine,
HeaviSine, Blocks, line × K̃ = 43, 103, 203 with G = 200, 800, 2000 × six ambitions), all converged, in at most 96
Newton solves and 17 ms. The full steps of Bollaerts et al. cycled on Blocks at low ambition; with the line search,
S decreases strictly and the sets settle. Once they settle, the fit is the exact minimizer (S is convex and C¹).
HiGHS's QP (highs-js 1.15.4, HiGHS 1.15.1) agrees with it to about 1e-4·h, which is its own tolerance, but this
build has only the active-set QP solver (`ipm` falls back to it, `hipo` is not compiled in): median 12 ms at K̃ =
43, 0.6 s at K̃ = 103 and 6.6 s (up to 20 s) at K̃ = 203, in Node and the worker alike. The hard-region QP is fast
when it works, but 40 of the 72 runs failed, 22 of them on provably feasible problems (the soft fit lay inside): it
reports "non-convex", fails on degeneracy or hits the time limit; presolve off, dropping tiny matrix values and
rescaling the objective do not help. So the QP stays out of the app; it remains a test oracle. A hard region, if
wanted, is κ → ∞ in the iteration (violations shrink like 1/κ).

**User knobs:**
- **ambition:** one multiplier on (λ₁, λ₂), on a log scale: a slider from detail (short-term patterns) to
  simplicity (long-term patterns). Advanced: the balance between flat and straight, or λ₁ and λ₂ separately.
  *Done (phase 2):* the slider is log₁₀(a/a*), from −8 to +4, default 0, and a share θ of a goes toward flat
  (open question 1 has the normalization).
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

**Done (phase 3)** (`fit.js` `fitSparse`, `sparseLambdas`, `sparseMaxAmbition`; `leavingAmbition(spec, {sparse})`):
- λ_d = a·θ_d·G with θ₁ = θ, θ₂ = 1 − θ, so S/G is the region functional plus a per free difference. A join's count
  does not grow with the segments, so there is no nseg factor. δ = 1e-4·nseg^(1−d) of the median half-width.
- Each reweighting is one `fitIterated`, warm-started from the previous fit's sides (5–9 times fewer Newton
  solves). The support (which differences exceed δ) settles within about 10 reweightings, but the nonzero
  differences drift only linearly after that (1e-5 h per step after 100 steps). So the reweighting stops once the
  support is unchanged for 3 steps, and a final refit pins the zeros and frees the rest: the exact minimizer of
  the region terms plus λ·(free differences) for that support.
- The weights are capped as the quadratic penalty is (1e10 times the center pull per coefficient), so a
  difference counts as zero below δ_eff = max(δ, 1/√cap). With δ alone, under κ's pull no difference fell below
  δ at large a, all counted as free, and a* came back as the center.
- Backward elimination: drop each join in turn and keep the drop when the L0 objective falls (up to 3 passes). On
  a ramp the reweighting settled on four straight pieces where three fit.
- The knob is the same as for the quadratic penalty: decades from a*, found the same way. Its top is
  10·S_poly/(G·θ_d), where S_poly is the region term of the polynomial (every difference pinned): above S_poly/G one
  free difference costs more than the polynomial's whole region term, so the polynomial is the L0 optimum, and "capped"
  means a line (or constant) fits.
- Measured: synthetic kinks at 0.3 and 0.7 come back as bends at 0.300 and 0.700 (ED 4 = 2 + 2), smooth steps as
  three flat pieces. Examples at a*: Nile 2 straight pieces (bend ≈ 1913), Tukey coal 3, ozone 7; chess, fibers and
  neuroimage a line. a* takes 4–40 ms on the examples, 250–360 ms on Blocks and Bumps (n = 1024) and 1.9 s on a
  sine with n = 5000 (main thread).
- The count is not monotone in a (non-convexity): a smaller ambition can give fewer joins. The readout calls it
  an approximation, a local optimum.

### 1.4b Rounded bends (a third ambition penalty)

The plain count charges λ per free second difference, so a turn costs least in one difference: at a*, the four tops
of sin(4πt) (n = 500) were one difference each, a triangle wave. `fitShaped` (fit.js) is the same search (the
shared `adaptiveRidge`; `fitSparse` is bit-for-bit unchanged on 30 cases) with each bend, a run of free second
differences, also paying q·Σt². A turn T over m differences then costs about λ(m + qT²/m), least at a constant
curvature √(1/q): a bend widens with its turn, and a small turn stays in one difference, costing about λ as before.
The knob is a length L (fraction of the x range): q = (L·nseg)⁴/4 makes a bend leave its tangent by one median
half-width over L, whatever the segments. Measured at a*: L = 0.05 rounds the sine's tops (largest |D₂β| 2.48 →
0.97 median half-widths at n = 200), turns Tukey coal's sharp hump into one gentle bend and removes Nile's slight
extremum; ozone, whose region forces its shape, barely changes; L ≤ 0.02 changes nothing on these.

**Costs by a bend's shape**, judged by the straight pieces it joins (so a symmetric peak costs the same read from
either end): extra for an extremum (sides sloping in opposite directions; a straight piece that rises less than 5%
of the median half-width over its length is level, so a slight tilt after leveling off is no extremum) and per
inflection (change of bending direction, the wiggle penalty), in units of a plain bend. In the reweighting alone
these only moved a*, never the fit (sine, ozone, Tukey coal, Nile, dog ages, at a* and two decades below): the
refit sets the signs of the free differences from the region, so nothing trades a sign-dependent cost.

*Done:* a shape search on top. Each candidate removes two sign changes of the slopes (a maximum and a minimum) or
of the free second differences (two inflections), or one at an end, as sign constraints on the coefficients'
differences over a stretch (asymmetric penalties with a growing active set, after Bollaerts, Eilers & Van Mechelen
2006); the whole search reruns under them, and the first rerun that lowers the objective is kept (greedy, up to 6).
A candidate is screened by its constrained fit with no ambition: if that leaves the region, or its region terms
already reach the objective, the rerun cannot win. The screen does not depend on the ambition, so the a* search
shares it. Defaults: extremum 2, inflection 1.

- Synthetic: a rise that sags by 0.18 in a region of half-width 0.1 becomes a rise that levels off (1 bend, no
  extremum); a rising line with wiggles of 0.06 at h = 0.05 keeps its forced wiggles (7 inflections) but no
  longer falls between them (extremes 6 → 0).
- Examples: no change at a* on sine, ozone, Nile, dog ages; Tukey coal loses an inflection only with bend length 0
  (0.05 already removes it). At these regions (λ × 1/16) their extremes and inflections are forced.
- Cost of the a* search with the default costs: 40–200 ms on the examples and the sine (2–5 times without), 2.7 s
  on Blocks (n = 1024). Without the screen's early stop and bound it was 0.5 s on Nile and 5.6 s on Blocks.
- The leveling-off discount (cheaper per difference when one side of a bend is level) acts only in the
  reweighting, where it never changed a fit; it stays in fit.js, neutral, and is not in the UI.

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

**Done (phase 3)** (`src/core/ladder.js`): no QP is needed. The weighted minimax constant and line on the check
grid (smallest r with |g − c| ≤ r·h everywhere) are found by bisection on r and a golden-section search on the
slope (the best ratio for a given slope is convex in it). This is exact to about 1e-12 and agrees with a HiGHS LP to
its tolerance, in 3–12 ms for G up to 5,000. It depends only on the region, not on κ, ε or the ambition, so it says
when a simpler shape than any phase-2 fit is enough. On chess recall a constant fits (0.695 of the half-width), though
the default fit, toward straight, is a line.

## 2. Open questions

1. **Normalizing λ₁, λ₂** so the ambition knob means the same across data sets: relative to the phase-1 λ, to an
   effective dimension, or to the λ at which the fit first leaves the region.
   *Decided (phase 2), in two steps* (`fit.js` `ambitionLambdas`, `leavingAmbition`):
   - λ_d = a·G·nseg^(2d−1) with x mapped to [0, 1] and y in units of the median half-width. This makes S/G a
     functional (κ∫dist²/h² + ε∫((g − c)/h)² + aθ∫g'² + a(1 − θ)∫g''²) that does not depend on the units, the
     number of segments or the check-grid density. Measured on a sine: 20 to 80 segments moves the fit by
     ≤ 0.05h, and a 4× denser grid by ≤ 0.01h.
   - That alone is not enough across data sets. The ambition at which the fit first leaves the region by more
     than 1% of h (a*) ranges from 1e-7 (Blocks) to never (chess, fibers, neuroimage, where a line fits inside).
     So the knob is log₁₀(a/a*). At 0 the fit is the simplest inside the region; below 0 it moves toward the
     center (about 8 decades down, it is c); above 0 it leaves the region, and the readout says where.
   - a* is found by scanning up a decade at a time to the first fit that leaves, then bisecting to 0.02 decades:
     15–25 fits, 1–50 ms on the examples and signals.
   - The ratio is not proven monotone in a; a* is the first crossing found.
   - a is capped where λ_d·nb/(ε·G) = 1e10. Above about 1e12 the Newton system lost enough digits that a line
     inside the region came back outside it, with no Cholesky failure. When no ambition up to the cap leaves the
     region, a* is the cap and the fit is the limiting polynomial: the weighted least-squares line of c, to
     2.5e-5 h.
2. **Default κ:** large enough that violations are negligible, small enough to keep the iterated system well
   conditioned. (The hard-region QP would have sidestepped this, but HiGHS's active-set QP fails on it; see §1.3.)
   *For now (phase 2):* κ = 1e4, ε = 1e-2, both per grid point, and both advanced settings. With the knob at a*,
   violations are below 1% of h by construction. A larger κ moves a* up: ozone's a* is 1.9e-6 at κ = 1e4 and 1.9e-3
   at κ = 1e7, with the same ED of 16.5.
3. **A data term in phase 2?** As specified, the data enter only through the region. Revisit if fits drift from
   the data in ways the center pull does not prevent.
4. **Band type and coverage:** decided in phase 1 (§1.2): Bayesian covariance, λ undersmoothed by 1/16. Open:
   whether to offer the sandwich band at all (it covers E f̂, a weaker claim), and a local λ for jumps.
5. **Check-grid density** (default 4 per segment) and whether to include the data x always (yes for now).
6. **Aₙ in penalty form:** each Aₙ constraint |Σ_I (y − g)| ≤ c√|I| is linear in β, so it becomes the same kind
   of penalty on the excess; σ̂, interval families and calibrated τ come from the earlier project.
   *Done (phase 4)* (`src/core/an.js`; `fit.js` rows): divided by |I|, each constraint is a box for the mean of g
   over I, |mean_I(g) − mean_I(y)| ≤ c/√|I|, and mean_I(g) = r_I·β is linear. So Aₙ is a region of boxes exactly like
   the band, one per interval instead of one per grid point. Its ratio is Aₙ's own |w(I)|/c (tested against
   `regions.checkFit` at the data), and phase 2 (the objective, a*, the sparse ambition, the ladder) takes it
   unchanged. The ε pull is toward mean_I(y) in every box, a multiscale data term. Findings:
   - Long intervals make the Newton system dense (nb × nb). On the examples a* takes 1–11 ms (quadratic) and 4–60 ms
     (sparse); on simulated data up to 0.5 s (HeaviSine n = 1024, every-position windows, quadratic) and 2 s (sine
     n = 5000, sparse). At 400 segments the sparse a* took 44 s, so Aₙ's basis is capped at 100 segments.
   - A cubic spline on equal knots cannot always get inside Aₙ. On Blocks n = 1024 the largest |w(I)|/c with no
     ambition is 5.4 at 40 segments, 2.1 at 100 and 1.2 at 400: Aₙ's singletons ask for jumps sharper than the basis
     can make. Blocks n = 256 fits at 100 segments (seed 7: 10 flat pieces at a*, against 12 true; with seed 1 even
     that is out). Ozone (47 distinct x, 1,197 observations) cannot get inside either. `leavingAmbition` reports
     `unreachable`, and the readout says why instead of giving an a*.
   - Aₙ looks only at the observations, so a coefficient whose support holds none was free: the system was singular
     (chess recall, fibers). A second-difference floor of 1e-12 of the strongest penalty present fixes it.
   - At steps of Blocks the flat sparse fit overshoots (a free run of differences allows a bump): Aₙ does not
     forbid it, and the count of steps does not see it.
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
- [x] QP risk check: the phase-2 objective as a HiGHS QP in Node and in the worker, for K̃ up to ~200 and G up to
      ~2000; compare with the iterated weights (same fit within tolerance, time, iterations). Decide the default.

### Phase 1 — The region
- [x] Simultaneous band on the check grid (`src/core/band.js`), Monte Carlo critical value, undersmoothing;
      coverage simulation over signals × n × noise (as the earlier project's `tools/sim-band.mjs`).
- [x] UI: data input, the phase-1 fit and band, readouts with the claim (from the earlier project's main.js and controls.js).

### Phase 2 — The second-phase fit
- [x] Objective and solver (`src/core/fit.js`); re-check against the region. Tests: a fit stays inside when κ is
      large; ambition → ∞ gives the least-squares line (D₂) or constant (D₁) toward the center, when they fit;
      ambition → 0 gives the center c; iteration and QP agree.
- [x] Knobs in the UI (ambition slider, flat/straight balance, advanced κ and ε); URL state.
- [ ] Diagnostics on the examples, side by side with the earlier project's fits. (lafspline's side is done: every example
      fits in milliseconds and stays within 1% of h at a*. The earlier project's side is still to do.)

### Phase 3 — Sparse ambition and the ladder report
- [x] Adaptive-ridge option for D₁ and D₂; count the resulting flat and straight pieces and bends (§1.4).
- [x] "A constant / a straight line fits inside the region" report (§1.6).

### Phase 4 — Aₙ as a region
- [x] Aₙ in penalty form (open question 6), with the earlier project's interval families and calibrated τ.

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
