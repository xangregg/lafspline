// Rules of thumb for λ, for comparison with GCV and AIC. Each rule gives the width h of a Gaussian kernel smoother
// with no search over λ:
//
//   - Silverman's normal reference (Silverman 1986, eq. 3.31): h = 0.9·min(sd, IQR/1.34)·n^(−1/5), from the spread
//     of x alone. It is the width that would be optimal for estimating a normal density of x; in regression it
//     says nothing about the signal or the noise, and is used only as a crude start (as by the np package's
//     normal-reference bandwidths).
//   - Fan & Gijbels' rule of thumb (Fan & Gijbels 1996, §4.2; Ruppert, Sheather & Wand 1995, §2): a pilot quartic
//     by least squares gives σ̃² (its residual variance) and the curvature m̃'', which go into the width that
//     minimizes the asymptotic integrated squared error of local linear regression,
//
//         h = [R(K)·σ̃²·(b − a) / (μ₂(K)²·Σ m̃''(xᵢ)²)]^(1/5),   R(K) = 1/(2√π), μ₂(K) = 1 (Gaussian K).
//
//     It is only as good as the quartic: a signal a quartic cannot follow inflates σ̃² and flattens m̃'', so h is
//     too wide.
//
// A kernel width becomes a P-spline λ by matching effective degrees of freedom, the trace of the hat matrix, as
// smoothers of different kinds are compared (Hastie & Tibshirani 1990, §3.5; Buja, Hastie & Tibshirani 1989): the
// local linear smoother's trace at h, computed exactly on these x, then λ by bisection on log λ, since the
// P-spline's ED falls monotonically as λ grows. That needs no equivalent-kernel constant and holds for any penalty
// order, at the price of comparing the smoothers' overall flexibility rather than their local shape.
//
// The feature scale (scaleLambda) is not a rule of thumb but the reader's choice: the length, as a share of the x
// range, of the shortest features of interest. λ is set so that a sine of that period keeps half its amplitude: a
// longer feature is followed, a shorter one smoothed away. It depends on x only (the design and the basis), not on
// y, so a scale means the same on every data set (PLAN open question 7).

import {
    pSplineSystem, fitSystem, weightedCrossProducts, addDifferencePenalty, choleskyBanded, solveCholesky,
    LOG_LAMBDA_RANGE,
} from './pspline.js';

export const RULES = ['silverman', 'rot'];
const SILVERMAN_FACTOR = 0.9; // Silverman (1986) eq. 3.31
const IQR_PER_SD = 1.34; // interquartile range of a normal distribution, in standard deviations
const GAUSS_ROUGHNESS = 1 / (2 * Math.sqrt(Math.PI)); // R(K) = ∫K² for the standard Gaussian kernel
const PILOT_DEGREE = 4; // Fan & Gijbels' pilot polynomial
const KERNEL_REACH = 5; // Gaussian weights beyond 5h (below 4e-6 of the peak) are dropped from the trace
const LOG_LAMBDA_TOL = 1e-4; // bisection stopping width in log λ (as the GCV search's)
const SCALE_GAIN = 0.5; // feature scale: a sine of that period keeps this share of its amplitude
// the gain is measured over the middle half of the x range: near the ends the fit follows the penalty's own limit (a
// line, a constant) more than the sine, whatever λ
const GAIN_MIDDLE = [0.25, 0.75];
const LOG_PERIOD_TOL = 1e-4; // lambdaScale's bisection width in log period

/** Linear-interpolation quantile (R type 7) of sorted values. */
function quantileSorted(sorted, p) {
    const at = p * (sorted.length - 1);
    const lo = Math.floor(at);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (at - lo) * (sorted[hi] - sorted[lo]);
}

/**
 * Silverman's normal-reference width for x: 0.9·min(sd, IQR/1.34)·n^(−1/5), with sd alone when the IQR is 0.
 *
 * @param {ArrayLike<number>} x  at least 2 values, not all equal
 * @returns {{h: number, sd: number, iqr: number, n: number}}  h, sd and IQR in units of x
 */
export function silvermanWidth(x) {
    const n = x.length;
    const sorted = Float64Array.from(x).sort();
    let mean = 0;
    for (const v of sorted)
        mean += v / n;
    let ss = 0;
    for (const v of sorted)
        ss += (v - mean) ** 2;
    const sd = Math.sqrt(ss / (n - 1));
    const iqr = quantileSorted(sorted, 0.75) - quantileSorted(sorted, 0.25);
    const spread = iqr > 0 ? Math.min(sd, iqr / IQR_PER_SD) : sd;
    if (!(spread > 0))
        throw new RangeError('silvermanWidth: x has no spread');
    return { h: SILVERMAN_FACTOR * spread * n ** (-1 / 5), sd, iqr, n };
}

/** Solve the small dense system A z = b (row-major, k × k) by Gaussian elimination with partial pivoting. */
function solveDense(A, b, k) {
    const M = A.map(row => Float64Array.from(row));
    const z = Float64Array.from(b);
    for (let c = 0; c < k; c++) {
        let p = c;
        for (let r = c + 1; r < k; r++) {
            if (Math.abs(M[r][c]) > Math.abs(M[p][c]))
                p = r;
        }
        if (!(Math.abs(M[p][c]) > 0))
            throw new RangeError('solveDense: singular system');
        [M[c], M[p]] = [M[p], M[c]];
        [z[c], z[p]] = [z[p], z[c]];
        for (let r = c + 1; r < k; r++) {
            const f = M[r][c] / M[c][c];
            for (let j = c; j < k; j++)
                M[r][j] -= f * M[c][j];
            z[r] -= f * z[c];
        }
    }
    for (let c = k - 1; c >= 0; c--) {
        for (let j = c + 1; j < k; j++)
            z[c] -= M[c][j] * z[j];
        z[c] /= M[c][c];
    }
    return z;
}

/**
 * Fan & Gijbels' rule-of-thumb width for local linear regression with a Gaussian kernel, from a pilot quartic
 * (weighted least squares on x mapped to [−1, 1], for conditioning). Infinite when the quartic has no curvature.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @param {ArrayLike<number>} [w]  prior weights (1)
 * @returns {{h: number, sigma: number, curvature: number}}  h in units of x; sigma: σ̃, the quartic's residual
 *     standard deviation (units of y); curvature: √(Σwᵢ m̃''(xᵢ)² / Σwᵢ), the quartic's root-mean-square second
 *     derivative (units of y per x²)
 */
export function ruleOfThumbWidth(x, y, w) {
    const k = PILOT_DEGREE + 1;
    let lo = Infinity;
    let hi = -Infinity;
    let n = 0;
    let wSum = 0;
    for (let i = 0; i < x.length; i++) {
        const wi = w ? w[i] : 1;
        if (wi > 0) {
            lo = Math.min(lo, x[i]);
            hi = Math.max(hi, x[i]);
            n++;
            wSum += wi;
        }
    }
    if (n <= k)
        throw new RangeError(`ruleOfThumbWidth: the pilot quartic needs more than ${k} observations`);
    const mid = (lo + hi) / 2;
    const half = (hi - lo) / 2;
    const A = Array.from({ length: k }, () => new Float64Array(k));
    const b = new Float64Array(k);
    const powers = new Float64Array(k);
    const fill = u => {
        powers[0] = 1;
        for (let j = 1; j < k; j++)
            powers[j] = powers[j - 1] * u;
    };
    for (let i = 0; i < x.length; i++) {
        const wi = w ? w[i] : 1;
        if (!(wi > 0))
            continue;
        fill((x[i] - mid) / half);
        for (let r = 0; r < k; r++) {
            b[r] += wi * powers[r] * y[i];
            for (let c = 0; c < k; c++)
                A[r][c] += wi * powers[r] * powers[c];
        }
    }
    const coef = solveDense(A, b, k);
    let rss = 0;
    let curv = 0;
    for (let i = 0; i < x.length; i++) {
        const wi = w ? w[i] : 1;
        if (!(wi > 0))
            continue;
        const u = (x[i] - mid) / half;
        fill(u);
        let fitted = 0;
        for (let j = 0; j < k; j++)
            fitted += coef[j] * powers[j];
        rss += wi * (y[i] - fitted) ** 2;
        // d²/dx² of Σ cⱼ uʲ with u = (x − mid)/half
        const second = (2 * coef[2] + 6 * coef[3] * u + 12 * coef[4] * u * u) / (half * half);
        curv += wi * second * second;
    }
    const sigma2 = rss / (n - k);
    const h = (GAUSS_ROUGHNESS * sigma2 * (hi - lo) / curv) ** (1 / 5);
    return { h: Number.isFinite(h) ? h : Infinity, sigma: Math.sqrt(sigma2), curvature: Math.sqrt(curv / wSum) };
}

/**
 * Trace of the hat matrix of local linear regression with a Gaussian kernel of width h, on these x and weights:
 * Σᵢ wᵢK(0)·s₂/(s₀s₂ − s₁²) with s_r = Σⱼ wⱼK(xⱼ − xᵢ)(xⱼ − xᵢ)^r, the weight of yᵢ in the fit at xᵢ. Kernel
 * weights beyond KERNEL_REACH·h are dropped. O(n·m) for m points within reach; h = ∞ gives the straight line's 2.
 *
 * @param {ArrayLike<number>} x
 * @param {number} h  > 0, units of x
 * @param {ArrayLike<number>} [w]  prior weights (1)
 * @returns {number}
 */
export function localLinearTrace(x, h, w) {
    const idx = Array.from(x, (_, i) => i).filter(i => (w ? w[i] : 1) > 0).sort((a, b) => x[a] - x[b]);
    const xs = Float64Array.from(idx, i => x[i]);
    const ws = Float64Array.from(idx, i => (w ? w[i] : 1));
    const n = xs.length;
    const reach = KERNEL_REACH * h;
    let trace = 0;
    let from = 0;
    let to = 0;
    for (let i = 0; i < n; i++) {
        while (xs[i] - xs[from] > reach)
            from++;
        while (to < n && xs[to] - xs[i] <= reach)
            to++;
        let s0 = 0;
        let s1 = 0;
        let s2 = 0;
        for (let j = from; j < to; j++) {
            const d = xs[j] - xs[i];
            const kw = ws[j] * Math.exp(-0.5 * (d / h) ** 2);
            s0 += kw;
            s1 += kw * d;
            s2 += kw * d * d;
        }
        const det = s0 * s2 - s1 * s1;
        // one distinct x within reach: the local fit is that x's weighted mean
        trace += det > 1e-12 * s0 * s2 ? ws[i] * s2 / det : ws[i] / s0;
    }
    return trace;
}

/**
 * λ by a rule of thumb: the rule's kernel width, its local linear smoother's trace, and the λ whose P-spline has
 * that effective dimension (bisection on log λ over the GCV search's range; at an end when the trace is out of
 * the P-spline's reach, below the penalty's polynomial or above the basis).
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @param {object} opts  pSplineSystem options plus rule ('silverman' or 'rot')
 * @returns {{criterion: string, lambda: number, fit: import('./pspline.js').PSplineFit, atEdge: boolean,
 *     rule: {h: number, ed: number, sd?: number, iqr?: number, sigma?: number, curvature?: number}}}  rule: the
 *     width h (units of x), the local linear trace ed it asks for, and the rule's inputs
 */
export function ruleLambda(x, y, opts) {
    const rule = opts.rule;
    if (!RULES.includes(rule))
        throw new RangeError(`ruleLambda: unknown rule ${rule}`);
    const system = pSplineSystem(x, y, opts);
    const used = Array.from(system.x).filter((_, i) => system.w[i] > 0);
    const width = rule === 'silverman' ? silvermanWidth(used) : ruleOfThumbWidth(system.x, system.y, system.w);
    const target = localLinearTrace(system.x, width.h, system.w);
    const at = t => {
        try {
            return fitSystem(system, Math.exp(t));
        }
        catch (err) {
            if (err instanceof RangeError)
                return null; // not positive definite: λ too small for this basis
            throw err;
        }
    };
    let [lo, hi] = LOG_LAMBDA_RANGE;
    const hiFit = at(hi);
    let fit = hiFit;
    let atEdge = false;
    if (hiFit.ed >= target) {
        atEdge = true;
    }
    else {
        const loFit = at(lo);
        if (loFit && loFit.ed <= target) {
            fit = loFit;
            atEdge = true;
        }
        else {
            while (hi - lo > LOG_LAMBDA_TOL) {
                const mid = (lo + hi) / 2;
                const midFit = at(mid);
                if (midFit && midFit.ed <= target) {
                    hi = mid;
                    fit = midFit;
                }
                else {
                    lo = mid;
                }
            }
        }
    }
    return { criterion: rule, lambda: fit.lambda, fit, atEdge, rule: { ...width, ed: target } };
}

/**
 * The name of a λ source for readouts and claims.
 *
 * @param {string} source  'gcv', 'aic', 'given' or a rule
 * @returns {string}
 */
export function lambdaSourceName(source) {
    const names = { gcv: 'GCV', aic: 'AIC', given: 'given', silverman: 'Silverman\'s rule',
        rot: 'Fan & Gijbels\' rule of thumb', scale: 'the feature scale' };
    return names[source] ?? source;
}

/**
 * The share of a sine's amplitude that a P-spline at λ keeps, for sines of a given period on the system's x: the
 * least-squares coefficient of the fit on the sine over the middle half of the x range, averaged over the sine and
 * the cosine so the phase does not matter. 1 for a long period (the fit follows it), toward 0 for a short one.
 *
 * @param {ReturnType<typeof pSplineSystem>} system  its y is not used
 * @param {number} lambda
 * @param {number} period  units of x
 * @returns {number}  NaN when the penalized system is not positive definite (λ too small for this basis)
 */
export function sineGain(system, lambda, period) {
    const { basis, W, G, x, w } = system;
    const nb = basis.nb;
    const A = Float64Array.from(G);
    addDifferencePenalty(A, nb, W, system.diff, lambda);
    let L;
    try {
        L = choleskyBanded(A, nb, W);
    }
    catch (err) {
        if (err instanceof RangeError)
            return NaN;
        throw err;
    }
    const span = basis.xr - basis.xl;
    const [from, to] = GAIN_MIDDLE.map(u => basis.xl + u * span);
    const bw = basis.deg + 1;
    let gain = 0;
    for (const phase of [0, Math.PI / 2]) {
        const wave = x.map(v => Math.sin(2 * Math.PI * (v - basis.xl) / period + phase));
        const alpha = solveCholesky(L, nb, W, weightedCrossProducts(basis, w, wave, W).rhs);
        let num = 0;
        let den = 0;
        for (let i = 0; i < x.length; i++) {
            if (x[i] < from || x[i] > to)
                continue;
            let f = 0;
            for (let r = 0; r < bw; r++)
                f += basis.values[i * bw + r] * alpha[basis.first[i] + r];
            num += w[i] * f * wave[i];
            den += w[i] * wave[i] * wave[i];
        }
        gain += den > 0 ? num / den / 2 : NaN;
    }
    return gain;
}

/**
 * λ from a feature scale: the λ at which a sine whose period is `scale` of the x range keeps half its amplitude
 * (sineGain), by bisection on log λ (the gain falls as λ grows). From x alone; the fit returned is the P-spline of y
 * at that λ.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @param {{scale: number, nseg?: number, deg?: number, diff?: number, w?: ArrayLike<number>}} opts  scale in (0, 1]
 * @returns {{criterion: 'scale', lambda: number, fit: object, atEdge: boolean, rule: {scale: number,
 *     period: number}}}  atEdge: the scale is beyond the λ range (shorter than the basis can follow takes the
 *     smallest λ, longer than the largest λ smooths away takes the largest); period in units of x
 */
export function scaleLambda(x, y, opts) {
    const scale = opts.scale;
    if (!(scale > 0 && scale <= 1))
        throw new RangeError('scaleLambda: scale in (0, 1] required');
    const system = pSplineSystem(x, y, opts);
    const period = scale * (system.basis.xr - system.basis.xl);
    const passes = t => sineGain(system, Math.exp(t), period) >= SCALE_GAIN; // NaN (too small a λ) does not
    let [lo, hi] = LOG_LAMBDA_RANGE;
    let atEdge = false;
    let logLambda;
    if (passes(hi)) {
        logLambda = hi;
        atEdge = true;
    }
    else if (!passes(lo)) {
        // even the smallest λ smooths it away, or is not positive definite: the smallest usable λ
        logLambda = lo;
        while (!Number.isFinite(sineGain(system, Math.exp(logLambda), period)) && logLambda < hi)
            logLambda += 1;
        atEdge = true;
    }
    else {
        while (hi - lo > LOG_LAMBDA_TOL) {
            const mid = (lo + hi) / 2;
            if (passes(mid))
                lo = mid;
            else
                hi = mid;
        }
        logLambda = lo;
    }
    const lambda = Math.exp(logLambda);
    return { criterion: 'scale', lambda, fit: fitSystem(system, lambda), atEdge, rule: { scale, period } };
}

/**
 * The feature scale a λ implies: the period, as a share of the x range, of the sine a P-spline at λ keeps half of
 * (sineGain = 1/2), by bisection on log period between one knot spacing and twice the range.
 *
 * @param {ArrayLike<number>} x
 * @param {number} lambda
 * @param {{nseg?: number, deg?: number, diff?: number, w?: ArrayLike<number>}} [opts]
 * @returns {number}  a share of the x range; at the search's end when λ keeps (or removes) every period in it
 */
export function lambdaScale(x, lambda, opts = {}) {
    const system = pSplineSystem(x, new Float64Array(x.length), opts);
    const span = system.basis.xr - system.basis.xl;
    let lo = Math.log(span / system.basis.nseg);
    let hi = Math.log(2 * span);
    while (hi - lo > LOG_PERIOD_TOL) {
        const mid = (lo + hi) / 2;
        if (sineGain(system, lambda, Math.exp(mid)) >= SCALE_GAIN)
            hi = mid;
        else
            lo = mid;
    }
    return Math.exp(hi) / span;
}

