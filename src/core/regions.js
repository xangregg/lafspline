// Confidence regions as data (the earlier project PLAN.md §1.1). A region is a set of linear constraints on the fit g at the
// observations, each over a range of observations:
//     |Σ_{i=lo_k..hi_k} g_i − center_k| ≤ scale·width_k
// plus what is needed to report and draw it: a label, a coverage claim, and how to draw it.
//
// Aₙ (paper eqs. 6–7) is one constraint per interval I of the family: center = Σ_I y, width = √|I|,
// scale = c = σ·√(τ log n).
//
// region.js (from tautfit) has Aₙ's threshold, residual check and constraint boxes; this module builds on it.
// Copied from the earlier project with its smoother band (bandRegion, which needs the earlier project's smoother.js) and LP framing
// (standardRows) removed: lafspline's band is band.js, and phase 2 uses the constraints in box form (an.js).

import { familyIntervals, familyText } from './intervals.js';
import { threshold } from './region.js';
import { sigmaMAD } from './sigma.js';
import { simulateMaxStatistics, sigmaPlugIn, tauFromSample } from './calibrate.js';

/** Relative slack used by every check of a solver's fit against a region (solver tolerance ~1e−7). */
export const CHECK_TOL = 1e-6;
// A constraint with zero width (σ = 0) admits only Σ g = center; this relative tolerance on that equality
// keeps its ratio 0 or ∞ instead of 0/0 = NaN.
const ZERO_WIDTH_TOL = 1e-9;
const DEFAULT_TAU = 3;
const DEFAULT_LAMBDA = 2;
const DEFAULT_ALPHA = 0.95; // coverage level for a calibrated τ
const NULL_B = 1000; // Monte Carlo draws of Aₙ's multiscale statistic for p-values
const NULL_SEED = 1;

/**
 * @typedef {object} Region
 * @property {'An'} kind            region type
 * @property {string} label         short name for readouts, e.g. 'Aₙ (τ = 3, λ = 2)'
 * @property {string} claim         coverage sentence shown with every fit inside this region
 * @property {'boxes'} draw        how the UI shows it: constraint boxes
 * @property {Int32Array} lo        first observation of each constraint (0-based, inclusive)
 * @property {Int32Array} hi        last observation (inclusive)
 * @property {Float64Array} center  target of Σ_{i=lo..hi} g_i, units of y
 * @property {Float64Array} width   per-constraint width (Aₙ: √|I|)
 * @property {number} scale         threshold multiplying width (Aₙ: c, units of y)
 * @property {object} params        n, sigma, sigmaSource, tau, lambda, family, calibration
 */

/**
 * @typedef {object} Calibration
 * @property {number} alpha  coverage level τ was calibrated for, 0 < α < 1
 * @property {number} B      Monte Carlo sample size
 * @property {'chosen'|'all'} family  the intervals the calibration covered
 * @property {number} [tauLo]  approximate 95% Monte Carlo interval for τ (calibrateTau, tauFromSample)
 * @property {number} [tauHi]
 */

/**
 * The coverage sentence for Aₙ. τ = 3 numbers are tautfit's measurements (PLAN.md §4).
 *
 * @param {{tau: number, sigmaSource: 'estimated'|'given', calibration: Calibration|null,
 *     blocks: boolean}} p  blocks: the paper's interval family, the one the τ = 3 numbers were measured for
 * @returns {string}
 */
function anClaim(p) {
    const noise = p.sigmaSource === 'given'
        ? 'if the noise is independent Gaussian with the given σ'
        : 'if the noise is independent Gaussian (σ estimated from the data)';
    let coverage;
    if (p.calibration) {
        coverage = `Contains the true f with probability ≈ ${p.calibration.alpha} (τ = ${round(p.tau)} calibrated `
            + `by Monte Carlo, B = ${p.calibration.B}, for this n, design and intervals) ${noise}.`;
    }
    else if (p.tau === DEFAULT_TAU && !p.blocks) {
        coverage = `Coverage of τ = ${DEFAULT_TAU} with these intervals is not known: lower than with the paper's `
            + 'fixed blocks, since more intervals give the noise more chances to exceed c. Calibrate τ for a stated level.';
    }
    else if (p.tau === DEFAULT_TAU) {
        coverage = `Contains the true f with probability about 95% for large n ${noise}; less at small n `
            + '(measured 76% at n = 10, 84–89% at n ≈ 30–50). Calibrate τ for a stated level.';
    }
    else {
        coverage = `Coverage of τ = ${round(p.tau)} is not known; calibrate τ for a stated level. Assumes the noise `
            + `is independent Gaussian${p.sigmaSource === 'given' ? ' with the given σ' : ' (σ estimated from the data)'}.`;
    }
    return `${coverage} No smoothness is assumed about f.`;
}

function round(v) {
    return Number(v.toPrecision(3));
}

/**
 * The multiresolution region Aₙ (paper eqs. 6–7) over x-intervals (ties never split).
 *
 * @param {ArrayLike<number>} x  sorted, user units (ties allowed)
 * @param {ArrayLike<number>} y  observations ordered by x
 * @param {{sigma?: number, tau?: number, lambda?: number, family?: import('./intervals.js').IntervalFamily,
 *     intervals?: {lo: number, hi: number}[], calibration?: Calibration}} [opts]  sigma defaults to σ̂ from first
 *     differences (tie-aware); intervals default to the family, which defaults to the paper's blocks with λ;
 *     calibration describes how τ was chosen, for the claim (see calibrateTau)
 * @returns {Region}
 */
export function anRegion(x, y, opts = {}) {
    const n = y.length;
    const sigmaSource = opts.sigma === undefined ? 'estimated' : 'given';
    const sigma = opts.sigma ?? sigmaMAD(y, x);
    const tau = opts.tau ?? DEFAULT_TAU;
    const lambda = opts.family?.lambda ?? opts.lambda ?? DEFAULT_LAMBDA;
    const family = opts.family ?? { kind: 'blocks', lambda };
    const intervals = opts.intervals ?? familyIntervals(x, family);
    const calibration = opts.calibration ?? null;
    const S = new Float64Array(n + 1);
    for (let i = 0; i < n; i++)
        S[i + 1] = S[i] + y[i];
    const count = intervals.length;
    const lo = new Int32Array(count);
    const hi = new Int32Array(count);
    const center = new Float64Array(count);
    const width = new Float64Array(count);
    intervals.forEach((iv, k) => {
        lo[k] = iv.lo;
        hi[k] = iv.hi;
        center[k] = S[iv.hi + 1] - S[iv.lo];
        width[k] = Math.sqrt(iv.hi - iv.lo + 1);
    });
    const intervalsText = opts.intervals ? 'given intervals' : familyText(family);
    return {
        kind: 'An',
        label: `Aₙ (τ = ${round(tau)}, ${intervalsText})`,
        claim: anClaim({ tau, sigmaSource, calibration, blocks: !opts.intervals && family.kind === 'blocks' }),
        draw: 'boxes',
        lo, hi, center, width,
        scale: threshold(sigma, tau, n),
        params: { n, sigma, sigmaSource, tau, lambda, family: opts.intervals ? null : family, calibration },
    };
}

/**
 * @typedef {object} FitCheck
 * @property {boolean} ok         every ratio ≤ 1 + tol
 * @property {number} maxRatio    largest ratio
 * @property {number} argMax      index of the constraint with the largest ratio
 * @property {number} violations  constraints with ratio > 1 + tol
 * @property {Float64Array} ratios |Σ g − center|/(scale·width) per constraint (Aₙ: |w(I)|/c); for a
 *                                zero-width constraint 0 if Σ g = center to ZERO_WIDTH_TOL, else Infinity
 */

/**
 * Check a fit against a region. O(n + constraints) via prefix sums.
 *
 * @param {Region} region
 * @param {ArrayLike<number>} g  fit at the observations, units of y
 * @param {number} [tol=CHECK_TOL]  relative slack for solver output; 0 for an exact check
 * @returns {FitCheck}
 */
export function checkFit(region, g, tol = CHECK_TOL) {
    const S = new Float64Array(g.length + 1);
    for (let i = 0; i < g.length; i++)
        S[i + 1] = S[i] + g[i];
    const count = region.lo.length;
    const ratios = new Float64Array(count);
    let maxRatio = 0;
    let argMax = -1;
    let violations = 0;
    for (let k = 0; k < count; k++) {
        const sum = S[region.hi[k] + 1] - S[region.lo[k]];
        const diff = Math.abs(sum - region.center[k]);
        const half = region.scale * region.width[k];
        let r;
        if (half > 0)
            r = diff / half;
        else
            r = diff <= ZERO_WIDTH_TOL * Math.max(1, Math.abs(region.center[k])) ? 0 : Infinity;
        ratios[k] = r;
        if (r > maxRatio || argMax < 0) {
            maxRatio = r;
            argMax = k;
        }
        if (r > 1 + tol)
            violations++;
    }
    return { ok: violations === 0, maxRatio, argMax, violations, ratios };
}

/**
 * The region's test statistic at a given scale: the value of the null statistic for which that scale is the
 * threshold, c/σ (= √(τ log n) at the region's own c), compared with M = max_I |Σ_I ε|/(σ̂√|I|). A shape needing
 * scale s is rejected at level p = P(statistic > s's value).
 *
 * @param {Region} region
 * @param {number} scale  units of y
 * @returns {number}
 */
export function scaleStatistic(region, scale) {
    return scale / region.params.sigma;
}

/**
 * Sorted Monte Carlo sample of the region's null statistic (see scaleStatistic): B draws of the multiscale
 * statistic over the region's own intervals, with σ̂ re-estimated as on the data (or σ known when given);
 * O(B · (constraints + n log n)).
 *
 * @param {Region} region
 * @param {ArrayLike<number>} x  observations' x (tie-aware σ̂)
 * @param {{B?: number, seed?: number, onProgress?: function(number): void}} [opts]
 * @returns {Float64Array}
 */
export function nullSample(region, x, opts = {}) {
    const intervals = Array.from(region.lo, (lo, k) => ({ lo, hi: region.hi[k] }));
    const plugSigma = region.params.sigmaSource === 'given' ? false : sigmaPlugIn('diff', Float64Array.from(x));
    return simulateMaxStatistics(region.params.n, intervals,
        { B: opts.B ?? NULL_B, seed: opts.seed ?? NULL_SEED, plugSigma, onProgress: opts.onProgress });
}

/**
 * τ calibrated by Monte Carlo for Aₙ over an interval family, this n and design (paper eq. 8; calibrate.js):
 * τ = q_α(M)²/log n, M the multiscale statistic of pure noise over the same intervals, with σ̂ re-estimated as
 * on the data unless σ is given. Denser families have a larger M, so they need a larger τ for the same coverage;
 * comparing families at a common τ compares regions of different coverage. The sample is the one nullSample
 * would draw for a region over these intervals, so it can be passed to the ladder for its p-values.
 * O(B · (|intervals| + n log n)).
 *
 * @param {ArrayLike<number>} x  sorted (ties allowed)
 * @param {{lo: number, hi: number}[]} intervals  observation ranges, e.g. intervals.familyIntervals
 * @param {{alpha?: number, sigmaGiven?: boolean, B?: number, seed?: number,
 *     onProgress?: function(number): void}} [opts]  alpha: coverage (default 0.95)
 * @returns {{tau: number, tauLo: number, tauHi: number, calibration: Calibration, sample: Float64Array}}
 *     tauLo–tauHi: approximate 95% Monte Carlo interval for τ
 */
export function calibrateTau(x, intervals, opts = {}) {
    const alpha = opts.alpha ?? DEFAULT_ALPHA;
    const B = opts.B ?? NULL_B;
    const plugSigma = opts.sigmaGiven ? false : sigmaPlugIn('diff', Float64Array.from(x));
    const sample = simulateMaxStatistics(x.length, intervals,
        { B, seed: opts.seed ?? NULL_SEED, plugSigma, onProgress: opts.onProgress });
    const { tau, tauLo, tauHi } = tauFromSample(sample, alpha, x.length);
    return { tau, tauLo, tauHi, calibration: { alpha, B, family: 'chosen', tauLo, tauHi }, sample };
}

/**
 * Monte Carlo p-value: the fraction of the null sample above the statistic. 0 means p < 1/B.
 *
 * @param {Float64Array} sorted  ascending
 * @param {number} statistic
 * @returns {number}
 */
export function pValue(sorted, statistic) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] <= statistic)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (sorted.length - lo) / sorted.length;
}
