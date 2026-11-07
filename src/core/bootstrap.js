// Phase 1, a second band (PLAN §1.2): a simultaneous band from a fractional-weight bootstrap of the P-spline fit. It
// need not be symmetric about the fit, and its width can follow the noise where it changes along x.
//
// The fit f̂ is the band's (band.js bandFit: the same λ choice and undersmoothing). Each replicate refits it at the
// same λ with random fractional weights on the observations: every weight positive (no observation drops out), the
// weights summing to n (so λ keeps its balance with the data). By default the refit is taken to first order in the
// weights (linearChange), which keeps a few small weights from making a sparse stretch swing. Two weight
// distributions:
// - 'matched' (default): w ∝ G^(1/p), G ~ Gamma(k), with k and p chosen so that the weights have the mean,
//   variance and skewness of a resampled count (1, 1, 1). In simulations it held its coverage best.
// - 'dirichlet': w_i = n·E_i/ΣE, E_i ~ Exp(1), the Bayesian bootstrap (Rubin 1981) or fractional-random-weight
//   bootstrap (Xu, Gotwalt, Hong, King & Meeker 2020). Mean and variance as a count's, but skewness 2. With
//   right-skewed noise its band's asymmetry was nearer the true tails' (1.35–1.6 at 10⁻³: Dirichlet 1.47, matched
//   1.29), but its coverage was lower (PLAN §1.2).
//
// What tools/sim-band.mjs showed (PLAN §1.2), and how the band answers it:
// - Residuals are smaller than the noise. A refit with weights W_b changes the coefficients by exactly
//   A_b⁻¹Bᵀ(W_b − W)r (A_b = BᵀW_bB + λDᵀD; Bᵀ(W r) = λDᵀDα̂ at the fit), so its spread is that of the residuals r,
//   which a smoother shrinks: Var r_i = σ²(1 − 2h_i + (HHᵀ)_ii)/w_i. The replicates use r_i divided by the root of
//   that factor (without it, the spread was 0.85 times f̂'s on a sine with ED 23, n = 200).
// - The spread near x comes from the few residuals near x, so it is noisy, and a band whose width follows it point
//   by point undercovers (60% for 95% on that sine; 22% with skewed noise); studentizing each replicate by its own
//   local standard error (bootstrap-t) recovered only part of it (78–86%). So each replicate's deviation
//   d_b(x) = f̂*_b(x) − f̂(x) is studentized by its global σ̂*_b and the constant-noise standard error s(x):
//   T_b(x) = d_b(x)/(σ̂*_b·s(x)). Each point's root mean square τ(x) standardizes T for the coverage check, which then
//   reflects the replicates' shape and correlation along x rather than the noise in τ; the band's widths use τ
//   averaged over a window of ±10% of the x range, where enough residuals pool to follow a trend in the noise.
// - The asymmetry is in the tails. With right-skewed noise the true 0.1% tails of f̂ − E f̂ were 1.35–1.6 times as
//   long above as below, but semi-deviations (second moments) of the replicates saw only 1.05. So the band's shape
//   is the replicates' own quantiles: standardized values Z = T/τ pooled over each window (B times the window's grid
//   points, so that tails at 10⁻³ and beyond are still estimated from many values), at a pointwise tail level chosen
//   so that a fraction α of the replicates lie inside at every point at once.
//
// The band is f̂ − Q_hi·τ̄·σ̂·s below to f̂ − Q_lo·τ̄·σ̂·s above, reflected as a bootstrap-t interval is: replicates
// that run high say f̂ may itself have run high, so the band reaches further below it. Like the sandwich band it
// measures the variability of f̂ at a fixed λ, so it is a band for E f̂ (f smoothed by the penalty).

import { bandFit, checkGrid, gammaDraw, lambdaHowText, DEFAULT_ALPHA } from './band.js';
import {
    bsplineBasis, weightedCrossProducts, addDifferencePenalty, choleskyBanded, solveCholesky, evaluate,
    standardErrors,
} from './pspline.js';
import { createRng } from './rng.js';
import { lambdaSourceName } from './thumb.js';

const DEFAULT_B = 1000; // bootstrap replicates
const DEFAULT_SEED = 1;
const REFITS = ['linear', 'exact'];
// First order in the weights: the exact refit A_b⁻¹ swings far when the few points that pin a segment or an end get
// small weights under a small λ (Tukey coal, ED 25 of n = 49: 15 times the sandwich band at 1920); see linearChange.
const DEFAULT_REFIT = 'linear';
const DEFAULT_WINDOW = 0.1; // half-width of the window the band's widths average over, as a fraction of the x range
const MIN_RESIDUAL_VARIANCE = 1e-9; // factors below this (an interpolated point, h_i ≈ 1): the residual counts as 0
const ZERO_RESIDUAL = 1e-10; // relative to the range of y: residuals no larger leave no spread to resample
const SCALE_FLOOR = 1e-12; // a scale this small (relative) is raised to it, not left at 0
const TAIL_LEVELS = 200; // log-spaced pointwise tail levels the critical level is chosen from
const MIN_TAIL_LEVEL = 1e-7; // the smallest of them, at most
const MAX_STORED = 2e7; // replicate values kept (4 bytes each); B is reduced on very fine grids to stay under it
const DEGREE_NAMES = { 1: 'linear', 2: 'quadratic', 3: 'cubic' };
// Matched weights: X = G^(1/MATCHED_POWER), G ~ Gamma(MATCHED_SHAPE), has coefficient of variation 1 and skewness 1,
// as a resampled count (mean 1, variance 1 − 1/n, third central moment ≈ 1); solved numerically from the moments
// E X^r ∝ Γ(k + r/p)/Γ(k). Its fourth central moment is 3.1 (a count's: 4; Exp(1)'s: 9).
const MATCHED_POWER = 4.7319684555;
const MATCHED_SHAPE = 0.0956207873;
export const WEIGHTS = ['matched', 'dirichlet'];

/**
 * Dirichlet(1, …, 1) weights scaled to sum to total: total·E_i/ΣE with E_i ~ Exp(1). Every weight is positive.
 *
 * @param {number} n
 * @param {import('./rng.js').Rng} rng
 * @param {number} [total]  the weights' sum (n)
 * @returns {Float64Array}
 */
export function dirichletWeights(n, rng, total = n) {
    const w = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
        // 1 − u is in (0, 1], so E is finite; E = 0 (u = 0 exactly) has probability 2⁻³²
        w[i] = -Math.log(1 - rng.uniform());
        sum += w[i];
    }
    for (let i = 0; i < n; i++)
        w[i] *= total / sum;
    return w;
}

/**
 * Fractional weights with a resampled count's mean, variance and skewness, scaled to sum to total. Every weight is
 * positive, but as a count is 0 with probability 1/e, about a fifth of them are below 0.1.
 *
 * @param {number} n
 * @param {import('./rng.js').Rng} rng
 * @param {number} [total]  the weights' sum (n)
 * @returns {Float64Array}
 */
export function matchedWeights(n, rng, total = n) {
    const logX = new Float64Array(n);
    let top = -Infinity;
    for (let i = 0; i < n; i++) {
        // Gamma(k) for k < 1 as Gamma(k + 1)·U^(1/k), in logs: U^(1/k) underflows for k ≈ 0.1
        const logG = Math.log(gammaDraw(rng, MATCHED_SHAPE + 1)) + Math.log(1 - rng.uniform()) / MATCHED_SHAPE;
        logX[i] = logG / MATCHED_POWER;
        top = Math.max(top, logX[i]);
    }
    const w = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
        w[i] = Math.exp(logX[i] - top);
        sum += w[i];
    }
    for (let i = 0; i < n; i++)
        w[i] *= total / sum;
    return w;
}

/**
 * Residuals rescaled to the noise's variance: r_i/√v_i with v_i = 1 − 2h_i + w_i·b_iᵀA⁻¹BᵀWBA⁻¹b_i, where
 * σ²v_i/w_i is the variance of r_i = y_i − f̂_i for a linear smoother with hat diagonal h_i, A = BᵀWB + λDᵀD.
 * O(n·nb·p).
 *
 * @param {import('./pspline.js').PSplineFit} fit
 * @returns {{r: Float64Array, v: Float64Array}}  scaled residuals (0 where v_i is about 0) and the factors v_i
 */
export function scaledResiduals(fit) {
    const { y, w } = fit.system;
    const spread = standardErrors(fit, fit.system.basis, 'sandwich');
    const v = Float64Array.from(y, (_, i) => 1 - 2 * fit.hat[i] + w[i] * (spread[i] / fit.sigma) ** 2);
    const r = Float64Array.from(y, (yi, i) => (v[i] > MIN_RESIDUAL_VARIANCE
        ? (yi - fit.fitted[i]) / Math.sqrt(v[i])
        : 0));
    return { r, v };
}

/**
 * The change in a fit's coefficients when its weights W become Wb, at the same λ: A_b⁻¹Bᵀ(W_b − W)r with
 * A_b = BᵀW_bB + λDᵀD. With r the fit's residuals this is exactly the refit minus the fit. O(n·deg² + nb·p²).
 *
 * @param {import('./pspline.js').PSplineFit} fit
 * @param {ArrayLike<number>} wb  new weights, ≥ 0
 * @param {ArrayLike<number>} r   residuals (the fit's, or rescaled)
 * @returns {Float64Array}  length nb
 */
export function refitChange(fit, wb, r) {
    const { basis, W, w, diff } = fit.system;
    const nb = basis.nb;
    const bw = basis.deg + 1;
    const rhs = new Float64Array(nb);
    for (let i = 0; i < basis.n; i++) {
        const c = (wb[i] - w[i]) * r[i];
        for (let k = 0; k < bw; k++)
            rhs[basis.first[i] + k] += basis.values[i * bw + k] * c;
    }
    const { G: A } = weightedCrossProducts(basis, wb, r, W);
    addDifferencePenalty(A, nb, W, diff, fit.lambda);
    return solveCholesky(choleskyBanded(A, nb, W), nb, W, rhs);
}

/**
 * refitChange to first order in the weights: A⁻¹Bᵀ(W_b − W)r with the fit's own A, a multiplier bootstrap. Its
 * covariance is A⁻¹Bᵀ diag(Var w·r²) BA⁻¹, the heteroscedasticity-robust sandwich, and the weights' skewness carries
 * the residuals' into its tails. The exact refit's A_b changes with the weights; where a few observations pin a
 * segment or an end and λ is small, a small weight on them leaves those coefficients barely held, and the refit
 * measures that reweighted design's fragility rather than f̂'s variability. In simulations (PLAN §1.2) the two
 * covered alike at n = 500; at n = 100 the exact band was up to 1.6 times as wide and overcovered. O(n·deg + nb·p).
 */
function linearChange(fit, wb, r) {
    const { basis, W, w } = fit.system;
    const bw = basis.deg + 1;
    const rhs = new Float64Array(basis.nb);
    for (let i = 0; i < basis.n; i++) {
        const c = (wb[i] - w[i]) * r[i];
        for (let k = 0; k < bw; k++)
            rhs[basis.first[i] + k] += basis.values[i * bw + k] * c;
    }
    return solveCholesky(fit.L, basis.nb, W, rhs);
}

/**
 * The studentized replicates T_bj = d_b(x_j)/(σ̂*_b·s_j) on the grid, replicate b at [b·G, (b + 1)·G).
 * O(B·(n·deg² + nb·p + G·deg)); B·G single-precision values (4 bytes each).
 */
function studentizedReplicates(fit, gb, r, unitSe, { B, seed, weights, refit }) {
    const rng = createRng(seed);
    const { basis, w: prior } = fit.system;
    const n = prior.length;
    const G = gb.n;
    const bw = basis.deg + 1;
    const gbw = gb.deg + 1;
    let total = 0;
    for (let i = 0; i < n; i++)
        total += prior[i];
    const wb = new Float64Array(n);
    const T = new Float32Array(B * G);
    for (let b = 0; b < B; b++) {
        // prior weights times the random weights, rescaled to keep the prior total
        const d = weights === 'dirichlet' ? dirichletWeights(n, rng) : matchedWeights(n, rng);
        let sum = 0;
        for (let i = 0; i < n; i++) {
            wb[i] = prior[i] * d[i];
            sum += wb[i];
        }
        for (let i = 0; i < n; i++)
            wb[i] *= total / sum;
        const delta = refit === 'linear' ? linearChange(fit, wb, r) : refitChange(fit, wb, r);
        // σ̂*: the replicate's weighted root mean square residual, in the bootstrap's world where f̂ is the truth
        // and the rescaled residuals the noise
        let ss = 0;
        for (let i = 0; i < n; i++) {
            let g = 0;
            for (let k = 0; k < bw; k++)
                g += basis.values[i * bw + k] * delta[basis.first[i] + k];
            ss += wb[i] * (r[i] - g) ** 2;
        }
        const sigma = Math.sqrt(ss / total);
        for (let j = 0; j < G; j++) {
            let g = 0;
            for (let k = 0; k < gbw; k++)
                g += gb.values[j * gbw + k] * delta[gb.first[j] + k];
            T[b * G + j] = g / (sigma * unitSe[j]);
        }
    }
    return T;
}

/**
 * Lower and upper quantiles of the standardized replicates Z_bj = T_bj/τ_j pooled over windows, as tables over tail
 * levels: anchors every h along x, each pooling the grid points within ±h, so neighboring windows overlap by half;
 * a grid point's quantile is interpolated linearly between the anchors on either side. Levels are log-spaced from
 * the smallest the pooled sample can resolve to ½. O(B·G·log(B·G)), with one window's values in memory at a time.
 *
 * @returns {{levels: Float64Array, lo: Float64Array[], hi: Float64Array[], anchorOf: Int32Array,
 *     frac: Float64Array}}  lo[k][l], hi[k][l]: anchor k's quantiles at levels[l] and 1 − levels[l]; grid point j
 *     lies between anchors anchorOf[j] and anchorOf[j] + 1, frac[j] of the way
 */
function windowQuantiles(grid, T, tau, B, h) {
    const G = grid.length;
    const x0 = grid[0];
    const span = grid[G - 1] - x0;
    const count = Number.isFinite(h) && h < span ? Math.ceil(span / h) + 1 : 1;
    const step = count > 1 ? span / (count - 1) : 0;
    const smallest = Math.max(1 / (B * Math.min(G, Math.ceil(G * 2 * h / span))), MIN_TAIL_LEVEL);
    const levels = Float64Array.from({ length: TAIL_LEVELS },
        (_, l) => Math.exp(Math.log(smallest) + (Math.log(0.5) - Math.log(smallest)) * l / (TAIL_LEVELS - 1)));
    const lo = [];
    const hi = [];
    for (let k = 0; k < count; k++) {
        const at = x0 + k * step;
        let first = 0;
        while (count > 1 && grid[first] < at - h)
            first++;
        let last = first;
        while (last < G && (count === 1 || grid[last] <= at + h))
            last++;
        const pooled = new Float32Array(B * (last - first));
        let m = 0;
        for (let b = 0; b < B; b++) {
            for (let j = first; j < last; j++)
                pooled[m++] = T[b * G + j] / tau[j];
        }
        pooled.sort();
        const pick = level => pooled[Math.min(m - 1, Math.max(0, Math.round(level * (m - 1))))];
        lo.push(Float64Array.from(levels, pick));
        hi.push(Float64Array.from(levels, level => pick(1 - level)));
    }
    const anchorOf = new Int32Array(G);
    const frac = new Float64Array(G);
    for (let j = 0; j < G; j++) {
        if (count === 1)
            continue;
        const t = (grid[j] - x0) / step;
        anchorOf[j] = Math.min(count - 2, Math.floor(t));
        frac[j] = t - anchorOf[j];
    }
    return { levels, lo, hi, anchorOf, frac };
}

/** A grid point's interpolated lower or upper quantile at table level l. */
function quantileAt(table, side, j, l) {
    const k = table.anchorOf[j];
    const t = table.frac[j];
    const rows = table[side];
    return t === 0 ? rows[k][l] : (1 - t) * rows[k][l] + t * rows[k + 1][l];
}

/**
 * Root mean square of t over the grid points within ±h of each (the grid sorted). O(G).
 */
function windowRms(grid, t, h) {
    const G = grid.length;
    const out = new Float64Array(G);
    let lo = 0;
    let hi = 0;
    let sum = 0;
    for (let j = 0; j < G; j++) {
        while (hi < G && grid[hi] <= grid[j] + h) {
            sum += t[hi] * t[hi];
            hi++;
        }
        while (grid[lo] < grid[j] - h) {
            sum -= t[lo] * t[lo];
            lo++;
        }
        out[j] = Math.sqrt(Math.max(sum, 0) / (hi - lo));
    }
    return out;
}

function round(v) {
    return Number(v.toPrecision(3));
}

function mean(a) {
    return a.reduce((sum, v) => sum + v, 0) / a.length;
}

// Simulated coverage by GCV's undersmoothing factor and by feature scale (tools/sim-band.mjs, 200 replications per
// row; PLAN §1.2 and open question 7)
const SIMULATED_GCV = [
    [1 / 4, 'with λ by GCV × 1/4 it contained E f̂ in 84–98% of samples, least often for a ramp with skewed noise at '
        + 'n = 100 and for lines'],
    [1 / 16, 'with λ by GCV × 1/16 it contained E f̂ in 86–98% of samples, least often for a line with skewed noise'],
];
const SIMULATED_SCALE = [
    [0.1, 'with λ by feature scale at 10% it contained E f̂ in 93–96.5% of samples, and f (but for HeaviSine, whose '
        + 'jumps it never holds) in 92.5–95.5%'],
    [0.2, 'with λ by feature scale at 20% it contained E f̂ in 85–97.5% of samples, least often for a line and a ramp '
        + 'with skewed noise at n = 100, and f as rarely as 7.5% for a sine with a period of 2.5 times the scale'],
];
const SAME = 1e-12; // a simulated setting matches to rounding

function simulated(p) {
    const table = { gcv: SIMULATED_GCV, scale: SIMULATED_SCALE }[p.lambdaSource];
    const setting = p.lambdaSource === 'scale' ? p.rule.scale : p.undersmooth;
    const hit = table?.find(([v]) => Math.abs(v - setting) < SAME);
    if (hit)
        return hit[1];
    const not = { gcv: 'not this undersmoothing', scale: 'not this scale' }[p.lambdaSource] ?? 'not λ chosen this way';
    return `with λ by GCV × 1/4 and 1/16 and by feature scale at 10% and 20% (${not}) it contained E f̂ in 84–98% `
        + 'of samples';
}

/** How the claim treats λ: exact when it does not depend on y. */
function fixedLambda(p) {
    if (p.lambdaSource === 'given')
        return 'λ is given';
    if (p.lambdaSource === 'scale')
        return 'λ depends on x alone, so it is fixed given x';
    return 'λ is treated as fixed';
}

function bootstrapClaim(p) {
    let lambdaHow = `λ chosen by ${lambdaSourceName(p.lambdaSource)}`;
    if (p.lambdaSource === 'given' || p.lambdaSource === 'scale')
        lambdaHow = lambdaHowText(p);
    else if (p.undersmooth !== 1)
        lambdaHow = `${lambdaHow} × ${round(p.undersmooth)} (undersmoothing)`;
    const basis = `${DEGREE_NAMES[p.deg] ?? `degree-${p.deg}`} B-splines on ${p.nseg} segments, ${lambdaHow}; `
        + `effective dimension ${round(p.ed)}`;
    const weights = p.weights === 'dirichlet'
        ? 'Dirichlet weights (the Bayesian bootstrap, twice as skewed as resampled counts)'
        : 'weights distributed like resampled counts in mean, variance and skewness';
    const refits = p.refit === 'exact' ? 'refits' : 'refits (to first order in the weights)';
    return `Simultaneous bootstrap band around a P-spline (${basis}). B = ${p.B} ${refits} at the same λ, each with `
        + `positive fractional weights on the observations that sum to n: ${weights}. The residuals are rescaled to `
        + 'the size of the noise and the replicates studentized by their σ̂; the band\'s reach below and above the '
        + 'fit is the replicates\' upper and lower tail (reflected, as a bootstrap-t interval), pooled over '
        + `±${round(100 * p.window)}% of the x range, at the tail level that holds for a fraction ${p.alpha} of the `
        + `replicates at every point at once. Contains the P-spline's mean, f smoothed by the same penalty, at every `
        + `x of the check grid with probability ≈ ${p.alpha}, for independent observations; the noise need not be `
        + `Gaussian, and its spread may change slowly along x. In simulations at 95% ${simulated(p)} (a `
        + `line, a sine, a ramp and HeaviSine; n = 100 and 500; Gaussian, skewed or heteroscedastic noise). ${
            fixedLambda(p)}, and the band contains f itself only where the smoothing bias is small compared with the `
        + 'half-width.';
}

/**
 * Simultaneous bootstrap band around a P-spline fit of y on x. The result is a band region (band.js) whose center
 * is the band's midpoint, which differs from the fit where the band is asymmetric; its target, for phase 2's pull
 * inside the region, is the replicates' median reflected as the edges are (the band at level ½).
 *
 * @param {ArrayLike<number>} x  any order, ties allowed
 * @param {ArrayLike<number>} y
 * @param {object} [opts]  bandRegion's λ and basis options (lambda, criterion, undersmooth, nseg, deg, diff, w,
 *     gridPerSegment), and:
 * @param {number} [opts.alpha]    coverage level (0.95)
 * @param {number} [opts.B]        bootstrap replicates (1000)
 * @param {number} [opts.seed]
 * @param {'matched'|'dirichlet'} [opts.weights]  ('matched')
 * @param {number} [opts.window]   half-width of the window the widths average over, a fraction of the x range
 *     (0.1); 0 uses each point's own, Infinity one value for the whole grid
 * @param {'exact'|'linear'} [opts.refit]  each replicate the exact reweighted refit, or its first order in the
 *     weights (the fit's own A; see linearChange)
 * @returns {import('./band.js').BandRegion & {target: Float64Array}}  params add fit (f̂ at the grid), se (σ̂·s,
 *     units of y), up and down
 *     (the band's reach above and below f̂), tUp and tDown (t⁺, t⁻ per point)
 */
export function bootstrapBand(x, y, opts = {}) {
    const alpha = opts.alpha ?? DEFAULT_ALPHA;
    const B = opts.B ?? DEFAULT_B;
    const seed = opts.seed ?? DEFAULT_SEED;
    const weights = opts.weights ?? 'matched';
    const window = opts.window ?? DEFAULT_WINDOW;
    if (!WEIGHTS.includes(weights))
        throw new RangeError(`bootstrapBand: unknown weights ${weights}`);
    const { fit, lambdaSource, lambdaChosen, atEdge, undersmooth, deg, diff, rule } = bandFit(x, y, opts);
    const { basis } = fit.system;
    const grid = checkGrid(fit.system.x, basis.xl, basis.xr, basis.nseg, opts.gridPerSegment);
    const gb = bsplineBasis(grid, basis.xl, basis.xr, basis.nseg, basis.deg);
    const G = grid.length;
    const fhat = evaluate(fit, gb);
    const { r } = scaledResiduals(fit);
    // residuals at rounding level count as 0 (data exactly on a polynomial of degree < diff)
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const v of fit.system.y) {
        yMin = Math.min(yMin, v);
        yMax = Math.max(yMax, v);
    }
    if (!r.some(v => Math.abs(v) > ZERO_RESIDUAL * Math.max(yMax - yMin, Number.MIN_VALUE))) {
        throw new RangeError('the residuals are all 0 (the data lie on a polynomial the penalty does not touch), '
            + 'so there is no spread to make a band from');
    }
    const unitSe = standardErrors(fit, gb, 'sandwich').map(v => v / fit.sigma);
    const Bused = Math.max(1, Math.min(B, Math.floor(MAX_STORED / G)));
    const refit = opts.refit ?? DEFAULT_REFIT;
    if (!REFITS.includes(refit))
        throw new RangeError(`bootstrapBand: unknown refit ${refit}`);
    const T = studentizedReplicates(fit, gb, r, unitSe, { B: Bused, seed, weights, refit });
    // τ_j: each point's root mean square, to standardize before pooling and for the coverage check
    const tau = new Float64Array(G);
    for (let b = 0; b < Bused; b++) {
        for (let j = 0; j < G; j++)
            tau[j] += T[b * G + j] ** 2;
    }
    for (let j = 0; j < G; j++)
        tau[j] = Math.max(Math.sqrt(tau[j] / Bused), SCALE_FLOOR);
    const h = window * (basis.xr - basis.xl);
    const table = windowQuantiles(grid, T, tau, Bused, h);
    // each replicate's deepest level (largest l) at which it lies inside the pooled quantiles at every point; the
    // band's level is the one a fraction α of replicates reach. Inside at l implies inside at every smaller l.
    const deepest = new Int32Array(Bused);
    for (let b = 0; b < Bused; b++) {
        const inside = l => {
            for (let j = 0; j < G; j++) {
                const z = T[b * G + j] / tau[j];
                if (z < quantileAt(table, 'lo', j, l) || z > quantileAt(table, 'hi', j, l))
                    return false;
            }
            return true;
        };
        let lo = -1;
        let hi = TAIL_LEVELS - 1;
        while (lo < hi) {
            const mid = Math.ceil((lo + hi + 1) / 2);
            if (inside(mid))
                lo = mid;
            else
                hi = mid - 1;
        }
        deepest[b] = lo;
    }
    const sorted = Int32Array.from(deepest).sort();
    const level = Math.max(0, sorted[Math.max(0, Bused - Math.ceil(alpha * Bused))]);
    const tauBar = windowRms(grid, tau, h);
    const se = unitSe.map(v => v * fit.sigma);
    // reflected, as a bootstrap-t interval: a long upper tail of the replicates makes the band reach further below
    // f̂ (simulations with skewed noise could not tell this from the unreflected band by coverage; theory favors it)
    const qBelow = Float64Array.from(grid, (_, j) => quantileAt(table, 'hi', j, level) * tauBar[j]);
    const qAbove = Float64Array.from(grid, (_, j) => -quantileAt(table, 'lo', j, level) * tauBar[j]);
    const down = se.map((v, j) => qBelow[j] * v);
    const up = se.map((v, j) => qAbove[j] * v);
    const lower = fhat.map((f, j) => f - down[j]);
    const upper = fhat.map((f, j) => f + up[j]);
    // the band at level ½: the replicates' median, reflected as the edges are. Phase 2 pulls toward it rather than
    // toward the midpoint of the edges, which a long tail on one side moves away from the estimate.
    const median = fhat.map((f, j) => f - quantileAt(table, 'hi', j, TAIL_LEVELS - 1) * tauBar[j] * se[j]);
    const params = {
        n: fit.system.n, alpha, B: Bused, seed, covariance: 'bootstrap', weights, window, refit,
        tailLevel: table.levels[level], qBelow, qAbove, q: (mean(qBelow) + mean(qAbove)) / 2,
        criterion: opts.criterion ?? 'gcv', lambdaSource, lambdaChosen, rule, atEdge, undersmooth, lambda: fit.lambda,
        ed: fit.ed, sigma: fit.sigma, df: fit.system.n - fit.ed, xl: basis.xl, xr: basis.xr, nseg: basis.nseg,
        deg, diff, coefficients: fit.alpha, fitted: fit.fitted, fit: fhat, se, up, down,
    };
    return {
        kind: 'band',
        label: `bootstrap band (${round(100 * alpha)}%, ED ${round(fit.ed)})`,
        claim: bootstrapClaim(params),
        draw: 'ribbon',
        grid,
        center: lower.map((l, j) => (l + upper[j]) / 2),
        half: lower.map((l, j) => (upper[j] - l) / 2),
        lower, upper, target: median, params,
    };
}
