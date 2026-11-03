// Phase 1 (PLAN §1.2): a simultaneous confidence band around a P-spline fit, on a check grid, as data.
//
// Center c_j = f̂(x_j) and half-width h_j = q·σ̂·s_j on the grid, where s_j is the standard error per unit σ and q
// the α-quantile of the Monte Carlo statistic max_j |(B_G z)_j| / (s_j·σ̂/σ), z ~ N(0, V/σ²):
//
// - 'bayes' (default): V = σ²(BᵀWB + λDᵀD)⁻¹, the posterior covariance when the penalty is read as a prior
//   (Wahba 1983; Nychka 1988). A draw is z = L⁻ᵀe, e ~ N(0, I): one banded back substitution.
// - 'sandwich': V = σ²A⁻¹BᵀWBA⁻¹, the sampling covariance of f̂ about its own mean E f̂ (which is f smoothed by
//   the penalty, not f). A draw is z = A⁻¹BᵀW^½e, e ~ N(0, I_n).
//
// σ̂ is estimated, so each draw also divides by σ̂/σ ~ √(χ²_ν/ν) with ν = n − ED (the usual approximation for a
// linear smoother). λ is treated as fixed: the draws do not redo its choice. Krivobokova, Kneib & Claeskens
// (2010) study such bands for penalized splines.
//
// The check grid is the distinct data x plus GRID_PER_SEGMENT points per segment of the basis, so excursions
// between data points and between knots are checked too.

import { selectLambda, fitPSpline, bsplineBasis, evaluate, standardErrors, DEFAULT_DEGREE, DEFAULT_DIFF }
    from './pspline.js';
import { createRng } from './rng.js';

export const DEFAULT_ALPHA = 0.95;
const DEFAULT_B = 1000; // Monte Carlo draws for the critical value
const DEFAULT_SEED = 1;
const GRID_PER_SEGMENT = 4; // uniform check-grid points per basis segment (PLAN open question 5)
// λ multiplier for the band (1 = the criterion's choice). λ·ρ^(2d) roughly scales the equivalent kernel's
// bandwidth by ρ (Silverman 1984, for smoothing splines), so 1/16 with d = 2 is about half the bandwidth. Chosen by
// tools/sim-band.mjs (PLAN §1.2): with the Bayesian covariance it raises coverage of f on smooth signals from
// 88–98% to 94–98%, for a band 1.2–1.4 times as wide.
const DEFAULT_UNDERSMOOTH = 1 / 16;
const MIN_DF = 2; // residual degrees of freedom n − ED needed to estimate σ
export const CHECK_TOL = 1e-9; // a point is outside when |g − c|/h > 1 + CHECK_TOL
const DEGREE_NAMES = { 1: 'linear', 2: 'quadratic', 3: 'cubic' };
const ORDER_NAMES = { 1: 'first', 2: 'second', 3: 'third', 4: 'fourth' };

/**
 * @typedef {object} BandRegion
 * @property {'band'} kind
 * @property {string} label          short name, e.g. 'P-spline band (95%, ED 7.3)'
 * @property {string} claim          what the band covers, at what level, under which assumptions
 * @property {'ribbon'} draw         drawn as a ribbon from lower to upper over grid
 * @property {Float64Array} grid     check grid, sorted: the distinct data x and a uniform grid
 * @property {Float64Array} center   c_j, the phase-1 fit at the grid
 * @property {Float64Array} half     h_j = q·σ̂·s_j
 * @property {Float64Array} lower    c − h
 * @property {Float64Array} upper    c + h
 * @property {object} params         inputs and intermediate values: n, alpha, B, seed, covariance, criterion,
 *     lambdaSource ('gcv'|'aic'|'given'), lambdaChosen, atEdge, undersmooth, lambda, ed, sigma, df, q, xl, xr,
 *     nseg, deg, diff, coefficients, fitted (at the data x), sample (sorted Monte Carlo statistic)
 */

/**
 * The check grid: the distinct x and perSegment uniform points per segment over [xl, xr], sorted, no duplicates.
 *
 * @param {ArrayLike<number>} x
 * @param {number} xl
 * @param {number} xr
 * @param {number} nseg
 * @param {number} [perSegment]
 * @returns {Float64Array}
 */
export function checkGrid(x, xl, xr, nseg, perSegment = GRID_PER_SEGMENT) {
    const count = nseg * perSegment + 1;
    const all = new Float64Array(x.length + count);
    all.set(x);
    for (let k = 0; k < count; k++)
        all[x.length + k] = k === count - 1 ? xr : xl + (xr - xl) * k / (count - 1);
    all.sort();
    let m = 0;
    for (let i = 0; i < all.length; i++) {
        if (i === 0 || all[i] !== all[m - 1])
            all[m++] = all[i];
    }
    return all.slice(0, m);
}

/**
 * The α-quantile of a sorted sample, as the ⌈αB⌉-th order statistic (1-based), as in the earlier project.
 *
 * @param {Float64Array} sorted
 * @param {number} alpha  0 < α < 1
 * @returns {number}
 */
export function quantile(sorted, alpha) {
    const r = Math.min(sorted.length, Math.max(1, Math.ceil(alpha * sorted.length)));
    return sorted[r - 1];
}

/**
 * Gamma(shape, 1) by Marsaglia & Tsang (2000), shape ≥ 1.
 */
function gammaDraw(rng, shape) {
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
        let z;
        let v;
        do {
            z = rng.normal();
            v = 1 + c * z;
        } while (v <= 0);
        v = v * v * v;
        const u = rng.uniform();
        if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v))
            return d * v;
    }
}

/**
 * Monte Carlo sample of max_j |(B_G z)_j| / (s_j·σ̂/σ), sorted ascending. O(B·(nb·p + G·deg)) for 'bayes';
 * sandwich adds O(B·n·deg).
 *
 * @param {import('./pspline.js').PSplineFit} fit
 * @param {import('./pspline.js').Basis} gridBasis  basis at the check grid (the fit's knots)
 * @param {Float64Array} s   standard errors per unit σ at the grid
 * @param {{covariance?: 'bayes'|'sandwich', B?: number, seed?: number, df?: number}} [opts]  df: degrees of
 *     freedom of σ̂ (Infinity: σ known)
 * @returns {Float64Array}
 */
export function maxTSample(fit, gridBasis, s, opts = {}) {
    const covariance = opts.covariance ?? 'bayes';
    const B = opts.B ?? DEFAULT_B;
    const df = opts.df ?? Infinity;
    if (Number.isFinite(df) && !(df >= 2))
        throw new RangeError(`maxTSample: need ≥ 2 residual degrees of freedom (have ${df})`);
    const rng = createRng(opts.seed ?? DEFAULT_SEED);
    const { L, system } = fit;
    const { W } = system;
    const nb = system.basis.nb;
    const p = W - 1;
    const bw = gridBasis.deg + 1;
    const sample = new Float64Array(B);
    const z = new Float64Array(nb);
    const sqrtW = Float64Array.from(system.w, Math.sqrt);
    const xb = system.basis;
    for (let b = 0; b < B; b++) {
        if (covariance === 'bayes') {
            for (let k = 0; k < nb; k++)
                z[k] = rng.normal();
        }
        else if (covariance === 'sandwich') {
            // z = A⁻¹Bᵀ W^½ e: first forward-substitute L y = Bᵀ W^½ e, then back
            z.fill(0);
            for (let i = 0; i < xb.n; i++) {
                const e = sqrtW[i] * rng.normal();
                for (let r = 0; r < bw; r++)
                    z[xb.first[i] + r] += e * xb.values[i * bw + r];
            }
            for (let i = 0; i < nb; i++) {
                let v = z[i];
                for (let l = Math.max(0, i - p); l < i; l++)
                    v -= L[i * W + (i - l)] * z[l];
                z[i] = v / L[i * W];
            }
        }
        else {
            throw new RangeError(`maxTSample: unknown covariance ${covariance}`);
        }
        // back substitution Lᵀ z = (current z)
        for (let i = nb - 1; i >= 0; i--) {
            let v = z[i];
            for (let k = i + 1; k <= Math.min(nb - 1, i + p); k++)
                v -= L[k * W + (k - i)] * z[k];
            z[i] = v / L[i * W];
        }
        const sigmaRatio = Number.isFinite(df) ? Math.sqrt(2 * gammaDraw(rng, df / 2) / df) : 1;
        let max = 0;
        for (let j = 0; j < gridBasis.n; j++) {
            let g = 0;
            for (let r = 0; r < bw; r++)
                g += gridBasis.values[j * bw + r] * z[gridBasis.first[j] + r];
            max = Math.max(max, Math.abs(g) / s[j]);
        }
        sample[b] = max / sigmaRatio;
    }
    return sample.sort();
}

function round(v) {
    return Number(v.toPrecision(3));
}

function bandClaim(p) {
    const criterion = p.lambdaSource.toUpperCase();
    let lambdaHow = `λ chosen by ${criterion}`;
    if (p.lambdaSource === 'given')
        lambdaHow = 'λ given';
    else if (p.undersmooth !== 1)
        lambdaHow = `λ chosen by ${criterion} × ${round(p.undersmooth)} (undersmoothing)`;
    const basis = `${DEGREE_NAMES[p.deg] ?? `degree-${p.deg}`} B-splines on ${p.nseg} segments, `
        + `${ORDER_NAMES[p.diff] ?? `order-${p.diff}`}-order difference penalty, ${lambdaHow}; effective dimension `
        + `${round(p.ed)}`;
    const fixed = p.lambdaSource === 'given' ? '' : '; λ treated as fixed';
    const level = `with probability ≈ ${p.alpha} (Monte Carlo, B = ${p.B}${fixed})`;
    const where = 'at every x of the check grid (the data x and a fine grid between them)';
    const noise = 'for independent Gaussian noise with constant σ (estimated from the residuals)';
    if (p.covariance === 'sandwich') {
        return `Simultaneous band around a P-spline (${basis}). Contains the P-spline's mean, f smoothed by the same `
            + `penalty, ${where} ${level} ${noise}. It contains f itself only where the smoothing bias is small `
            + 'compared with the half-width, so sharp features of f can fall outside it.';
    }
    return `Simultaneous band around a P-spline (${basis}), from the Bayesian covariance, which widens the band to `
        + `allow for the smoothing bias on average. Contains f ${where} ${level} ${noise}, if f is smooth: in `
        + 'simulations at 95% with the default undersmoothing it contained a line, a sine and a logistic ramp in '
        + '94–98% of samples. It does not contain jumps or corners of f sharper than the penalty allows.';
}

/**
 * Simultaneous band around a P-spline fit of y on x.
 *
 * @param {ArrayLike<number>} x  any order, ties allowed
 * @param {ArrayLike<number>} y
 * @param {object} [opts]
 * @param {number} [opts.alpha]        coverage level (0.95)
 * @param {'bayes'|'sandwich'} [opts.covariance]
 * @param {number} [opts.lambda]       given λ (no selection, no undersmoothing)
 * @param {'gcv'|'aic'} [opts.criterion]  λ criterion when λ is not given (GCV)
 * @param {number} [opts.undersmooth]  multiplier on the chosen λ (1)
 * @param {number} [opts.nseg]         segments (default from the number of distinct x)
 * @param {number} [opts.deg]          B-spline degree (3)
 * @param {number} [opts.diff]         penalty order (2)
 * @param {number} [opts.B]            Monte Carlo draws (1000)
 * @param {number} [opts.seed]
 * @param {number} [opts.gridPerSegment]  uniform check-grid points per segment (4)
 * @param {ArrayLike<number>} [opts.w]  prior weights
 * @returns {BandRegion}
 */
export function bandRegion(x, y, opts = {}) {
    const alpha = opts.alpha ?? DEFAULT_ALPHA;
    const covariance = opts.covariance ?? 'bayes';
    const B = opts.B ?? DEFAULT_B;
    const seed = opts.seed ?? DEFAULT_SEED;
    const deg = opts.deg ?? DEFAULT_DEGREE;
    const diff = opts.diff ?? DEFAULT_DIFF;
    const basisOpts = { w: opts.w, nseg: opts.nseg, deg, diff };
    let lambdaSource;
    let lambdaChosen = NaN;
    let atEdge = false;
    let undersmooth = 1;
    let fit;
    if (opts.lambda !== undefined) {
        lambdaSource = 'given';
        fit = fitPSpline(x, y, { ...basisOpts, lambda: opts.lambda });
    }
    else {
        const criterion = opts.criterion ?? 'gcv';
        const choice = selectLambda(x, y, { ...basisOpts, criterion });
        lambdaSource = criterion;
        lambdaChosen = choice.lambda;
        atEdge = choice.atEdge;
        undersmooth = opts.undersmooth ?? DEFAULT_UNDERSMOOTH;
        fit = undersmooth === 1 ? choice.fit : fitPSpline(x, y, { ...basisOpts, lambda: choice.lambda * undersmooth });
    }
    const { basis } = fit.system;
    const grid = checkGrid(fit.system.x, basis.xl, basis.xr, basis.nseg, opts.gridPerSegment);
    const gridBasis = bsplineBasis(grid, basis.xl, basis.xr, basis.nseg, basis.deg);
    const center = evaluate(fit, gridBasis);
    const se = standardErrors(fit, gridBasis, covariance);
    const s = se.map(v => v / fit.sigma);
    const df = fit.system.n - fit.ed;
    if (!(df >= MIN_DF)) {
        throw new RangeError(`band: the fit uses ${round(fit.ed)} of ${fit.system.n} degrees of freedom, leaving too `
            + 'few to estimate σ; use fewer segments or a larger λ');
    }
    const sample = maxTSample(fit, gridBasis, s, { covariance, B, seed, df });
    const q = quantile(sample, alpha);
    const half = se.map(v => q * v);
    const params = {
        n: fit.system.n, alpha, B, seed, covariance, criterion: opts.criterion ?? 'gcv', lambdaSource, lambdaChosen,
        atEdge, undersmooth, lambda: fit.lambda, ed: fit.ed, sigma: fit.sigma, df, q,
        xl: basis.xl, xr: basis.xr, nseg: basis.nseg, deg, diff,
        coefficients: fit.alpha, fitted: fit.fitted, sample,
    };
    return {
        kind: 'band',
        label: `P-spline band (${round(100 * alpha)}%, ED ${round(fit.ed)})`,
        claim: bandClaim(params),
        draw: 'ribbon',
        grid, center, half,
        lower: center.map((c, j) => c - half[j]),
        upper: center.map((c, j) => c + half[j]),
        params,
    };
}

/**
 * Check a curve given at the band's grid against the band.
 *
 * @param {BandRegion} region
 * @param {ArrayLike<number>} g  values at region.grid
 * @param {number} [tol]          relative slack (CHECK_TOL)
 * @returns {{ok: boolean, maxRatio: number, argMax: number, outside: number, ratios: Float64Array}}
 *     ratios |g − c|/h per grid point
 */
export function checkBand(region, g, tol = CHECK_TOL) {
    const { center, half } = region;
    const ratios = new Float64Array(center.length);
    let maxRatio = 0;
    let argMax = 0;
    let outside = 0;
    for (let j = 0; j < center.length; j++) {
        ratios[j] = Math.abs(g[j] - center[j]) / half[j];
        if (ratios[j] > maxRatio) {
            maxRatio = ratios[j];
            argMax = j;
        }
        if (ratios[j] > 1 + tol)
            outside++;
    }
    return { ok: outside === 0, maxRatio, argMax, outside, ratios };
}
