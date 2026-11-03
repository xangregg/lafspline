// Monte Carlo calibration of τ (paper eq. 8, PLAN.md §1.4): τₙ(α) = q_α(M)² / log n, where
//     M = max_{I∈𝓘ₙ} |Σ_{i∈I} Zᵢ| / √|I|,   Z ~ N(0, 1)ⁿ,
// optionally M' = M / σ̂(Z) to account for estimating σ with eq. (10), as the procedure does.

import { createRng } from './rng.js';
import { sigmaMAD, sigmaGSJS } from './sigma.js';

const Z_95 = 1.959963984540054; // standard normal 0.975 quantile, for the Monte Carlo interval

/**
 * The multiscale statistic of one noise vector.
 *
 * @param {ArrayLike<number>} z
 * @param {{lo: number, hi: number}[]} intervals
 * @param {boolean|function(ArrayLike<number>): number} [plugSigma=true]  divide by σ̂(z): true = eq. (10),
 *     a function = that estimator, false = σ known
 * @returns {number}
 */
export function maxStatistic(z, intervals, plugSigma = true) {
    const n = z.length;
    const S = new Float64Array(n + 1);
    for (let i = 0; i < n; i++)
        S[i + 1] = S[i] + z[i];
    let max = 0;
    for (const { lo, hi } of intervals)
        max = Math.max(max, Math.abs(S[hi + 1] - S[lo]) / Math.sqrt(hi - lo + 1));
    if (typeof plugSigma === 'function')
        return max / plugSigma(z);
    return plugSigma ? max / sigmaMAD(z) : max;
}

/**
 * Simulate B values of the statistic; returned sorted ascending so any α can be read off.
 *
 * @param {number} n
 * @param {{lo: number, hi: number}[]} intervals
 * @param {{B?: number, seed?: number, plugSigma?: boolean|function, onProgress?: function(number): void}} [opts]
 *     plugSigma as in maxStatistic
 * @returns {Float64Array}  sorted, length B
 */
export function simulateMaxStatistics(n, intervals, opts = {}) {
    const B = opts.B ?? 2000;
    const rng = createRng(opts.seed ?? 1);
    const plug = opts.plugSigma ?? true;
    const out = new Float64Array(B);
    const z = new Float64Array(n);
    const step = Math.max(1, Math.floor(B / 50));
    for (let b = 0; b < B; b++) {
        for (let i = 0; i < n; i++)
            z[i] = rng.normal();
        out[b] = maxStatistic(z, intervals, plug);
        if (opts.onProgress && (b + 1) % step === 0)
            opts.onProgress((b + 1) / B);
    }
    return out.sort();
}

/**
 * τₙ(α) from a sorted sample, with a Monte Carlo interval from binomial order-statistic bounds.
 *
 * The α-quantile is the ⌈αB⌉-th order statistic (1-based). The interval uses order statistics
 * αB ∓ 1.96·√(Bα(1−α)), an approximate 95% interval for the true quantile.
 *
 * @param {Float64Array} sorted  from simulateMaxStatistics
 * @param {number} alpha         coverage, 0 < α < 1
 * @param {number} n             sample size (for log n)
 * @returns {{tau: number, tauLo: number, tauHi: number, q: number}}
 */
export function tauFromSample(sorted, alpha, n) {
    const B = sorted.length;
    const at = r => sorted[Math.min(B, Math.max(1, r)) - 1];
    const half = Z_95 * Math.sqrt(B * alpha * (1 - alpha));
    const q = at(Math.ceil(alpha * B));
    const toTau = v => (v * v) / Math.log(n);
    return {
        tau: toTau(q),
        tauLo: toTau(at(Math.floor(alpha * B - half))),
        tauHi: toTau(at(Math.ceil(alpha * B + half))),
        q,
    };
}

/**
 * Fraction of the sample at or below the threshold √(τ log n): the simulated coverage of a given τ.
 *
 * @param {Float64Array} sorted
 * @param {number} tau
 * @param {number} n
 * @returns {number}
 */
export function coverageOfTau(sorted, tau, n) {
    const t = Math.sqrt(tau * Math.log(n));
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] <= t)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo / sorted.length;
}

/**
 * The σ̂ plug-in for calibration matching the procedure's own estimator.
 *
 * @param {'diff'|'gsjs'|'none'} kind  eq. (10), GSJS on the design x, or σ known
 * @param {ArrayLike<number>} [x]      design points (needed for 'gsjs'; with ties, also used by 'diff')
 * @returns {boolean|function(ArrayLike<number>): number}
 */
export function sigmaPlugIn(kind, x) {
    if (kind === 'gsjs')
        return z => sigmaGSJS(x, z);
    if (kind === 'diff')
        return x ? (z => sigmaMAD(z, x)) : true; // tie-aware when x is known
    return false;
}

