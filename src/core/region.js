// The multiresolution confidence region Aₙ, paper eqs. (6)–(8).

import { prefixSums } from './prefix.js';

/**
 * Threshold c = σ·√(τ·log n) of eq. (7).
 *
 * @param {number} sigma  noise scale, units of y
 * @param {number} tau    τ (paper default 3)
 * @param {number} n      number of observations
 * @returns {number}  c, units of y
 */
export function threshold(sigma, tau, n) {
    return sigma * Math.sqrt(tau * Math.log(n));
}

/**
 * @typedef {object} RegionCheck
 * @property {boolean} ok          every |w(I)| ≤ c·(1 + tol)
 * @property {number} maxRatio     max over I of |w(I)|/c
 * @property {number} argMax       index into `intervals` of the largest ratio
 * @property {number} violations   number of intervals with ratio > 1 + tol
 * @property {Float64Array} w      w(I) per interval (signed), units of y
 * @property {Float64Array} ratios |w(I)|/c per interval
 */

/**
 * Check g ∈ Aₙ: w(I) = Σ_{i∈I}(yᵢ − gᵢ)/√|I| (eq. 6) and g ∈ Aₙ ⇔ max_I |w(I)| ≤ c (eq. 7).
 * O(n + |𝓘ₙ|) via prefix sums of the residuals.
 *
 * @param {ArrayLike<number>} y         observations, ordered by x
 * @param {ArrayLike<number>} g         candidate values at the design points
 * @param {{lo: number, hi: number}[]} intervals  0-based inclusive index intervals
 * @param {number} c                    threshold from `threshold`, units of y
 * @param {number} [tol=0]              relative slack, e.g. 1e−7 for LP solutions
 * @returns {RegionCheck}
 */
export function checkRegion(y, g, intervals, c, tol = 0) {
    const n = y.length;
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++)
        r[i] = y[i] - g[i];
    const s = prefixSums(r);
    const m = intervals.length;
    const w = new Float64Array(m);
    const ratios = new Float64Array(m);
    let maxRatio = 0;
    let argMax = -1;
    let violations = 0;
    const limit = 1 + tol;
    for (let k = 0; k < m; k++) {
        const { lo, hi } = intervals[k];
        const wk = (s[hi + 1] - s[lo]) / Math.sqrt(hi - lo + 1);
        w[k] = wk;
        // c = 0 happens only for n = 1 (log 1 = 0); then only exact fits are in the region.
        const ratio = c > 0 ? Math.abs(wk) / c : (wk === 0 ? 0 : Infinity);
        ratios[k] = ratio;
        if (ratio > maxRatio || argMax < 0) {
            maxRatio = ratio;
            argMax = k;
        }
        if (ratio > limit)
            violations++;
    }
    return { ok: violations === 0, maxRatio, argMax, violations, w, ratios };
}

/**
 * @typedef {object} ConstraintBoxes
 * @property {Int32Array} lo        first observation of each interval (0-based)
 * @property {Int32Array} hi        last observation (inclusive)
 * @property {Float64Array} yMean   mean of y over the interval
 * @property {Float64Array} half    half-height c/√|I|
 * @property {Float64Array} gMean   mean of g over the interval
 * @property {Float64Array} ratios  |w(I)|/c
 */

/**
 * Aₙ drawn as boxes: each constraint of eq. (7) says the mean of g over I lies in
 *     [mean_I(y) − c/√|I|, mean_I(y) + c/√|I|]
 * (divide |Σ_I(y − g)| ≤ c√|I| by |I|). So g ∈ Aₙ exactly when every box contains g's mean over its
 * interval; gMean inside the box ⇔ ratio ≤ 1.
 *
 * @param {ArrayLike<number>} y
 * @param {ArrayLike<number>} g
 * @param {{lo: number, hi: number}[]} intervals
 * @param {number} c
 * @returns {ConstraintBoxes}
 */
export function constraintBoxes(y, g, intervals, c) {
    const Sy = prefixSums(y);
    const Sg = prefixSums(g);
    const m = intervals.length;
    const out = {
        lo: new Int32Array(m), hi: new Int32Array(m),
        yMean: new Float64Array(m), half: new Float64Array(m), gMean: new Float64Array(m), ratios: new Float64Array(m),
    };
    for (let k = 0; k < m; k++) {
        const { lo, hi } = intervals[k];
        const len = hi - lo + 1;
        const sy = Sy[hi + 1] - Sy[lo];
        const sg = Sg[hi + 1] - Sg[lo];
        out.lo[k] = lo;
        out.hi[k] = hi;
        out.yMean[k] = sy / len;
        out.gMean[k] = sg / len;
        out.half[k] = c / Math.sqrt(len);
        const w = Math.abs(sy - sg) / Math.sqrt(len);
        out.ratios[k] = c > 0 ? w / c : (w === 0 ? 0 : Infinity);
    }
    return out;
}
