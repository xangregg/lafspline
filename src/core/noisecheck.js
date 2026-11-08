// Noise check for a band's P-spline fit: does its residual σ̂ agree with the noise neighboring observations show?
//
// The reference σ̂ is sigma.js's GSJS (local linear pseudo-residuals, pure error for tied x), which does not depend
// on λ. A criterion that follows the noise (GCV with more coefficients than distinct x: Tukey coal at 80 segments,
// ED 49 of 49) leaves residuals far smaller than it; a fit that misses structure (Blocks at 40 segments) leaves
// them larger, in runs along x. The lag-1 autocorrelation of the residuals tells these apart from heavy tails or
// unequal variance (larger, but not in runs) and from variation between x values that the replicates within them do
// not show (larger, alternating: years of daily values recorded at one x). It cannot see serially correlated noise that the criterion has
// absorbed into the fit (Nile): then the fit and the differences are fooled alike.
//
// Limits: quantiles under smooth curves with independent Gaussian noise, fitted as the bands fit them (λ by GCV,
// × 1 or 1/4), from tools/noise-check-null.mjs. log(ratio)·√n and r₁·√m are roughly stable in n there.

import { sigmaGSJS } from './sigma.js';

// Measured (600 fits per n = 25 … 1000, λ × 1 and × 1/4): log ratio·√n from −3.8 at its 0.1% point to 4.4 at its
// 99.9% point; r₁·√m from −5.6 (n = 50, λ/4; −3.3 to −4.2 elsewhere) to 3.0.
const RATIO_Z = 4.0; // |log(σ̂ / σ̂_GSJS)|·√n beyond this is flagged
const LAG_LOW = -5.0; // r₁·√m below this is flagged (alternating residuals)
const LAG_HIGH = 3.0; // r₁·√m above this is flagged (residuals in runs)
const MIN_GROUPS = 10; // distinct x, at least, for r₁ to mean anything

/**
 * @typedef {object} NoiseStatistics
 * @property {number} sigma      the fit's residual σ̂ (as given)
 * @property {number} reference  σ̂ from neighboring observations (sigmaGSJS)
 * @property {number} ratio      sigma / reference
 * @property {number} lag1       lag-1 autocorrelation of the residuals' means at successive distinct x
 * @property {number} groups     distinct x (m)
 */

/**
 * The noise check's statistics, before any judgment.
 *
 * @param {ArrayLike<number>} x       sorted (ties allowed)
 * @param {ArrayLike<number>} y       ordered by x
 * @param {ArrayLike<number>} fitted  the fit at each x
 * @param {number} sigma              the fit's residual σ̂, units of y
 * @returns {NoiseStatistics}
 */
export function noiseStatistics(x, y, fitted, sigma) {
    const reference = sigmaGSJS(x, y);
    const means = [];
    for (let i = 0; i < x.length;) {
        let j = i;
        let s = 0;
        while (j < x.length && x[j] === x[i]) {
            s += y[j] - fitted[j];
            j++;
        }
        means.push(s / (j - i));
        i = j;
    }
    let num = 0;
    let den = 0;
    for (let k = 0; k < means.length; k++) {
        den += means[k] * means[k];
        if (k > 0)
            num += means[k] * means[k - 1];
    }
    return { sigma, reference, ratio: sigma / reference, lag1: den > 0 ? num / den : 0, groups: means.length };
}

/**
 * @typedef {'ok'|'follows'|'runs'|'between'|'spread'|'alternates'} NoiseVerdict
 *   follows: σ̂ far below the neighbors' (the fit follows the noise; the band is too narrow);
 *   runs: residuals in runs along x (the fit misses structure, or the noise is positively correlated);
 *   between: σ̂ above the neighbors' with alternating x-group means and replicates (variation between x values
 *     that the replicates do not show);
 *   spread: σ̂ above the neighbors' alone (heavy tails or unequal variance);
 *   alternates: alternating residuals alone (the fit follows the noise in part, or neighbors are negatively
 *     correlated)
 */

/**
 * The noise check: the statistics, their limits for this n, and a verdict.
 *
 * @param {ArrayLike<number>} x       sorted (ties allowed)
 * @param {ArrayLike<number>} y       ordered by x
 * @param {ArrayLike<number>} fitted  the fit at each x
 * @param {number} sigma              the fit's residual σ̂, units of y
 * @returns {NoiseStatistics & {ratioLimits: [number, number], lagLimits: [number, number]|null,
 *     verdict: NoiseVerdict}}  lagLimits: null (and r₁ not judged) below MIN_GROUPS distinct x
 */
export function checkNoise(x, y, fitted, sigma) {
    const s = noiseStatistics(x, y, fitted, sigma);
    const k = Math.exp(RATIO_Z / Math.sqrt(x.length));
    const ratioLimits = [1 / k, k];
    const lagLimits = s.groups >= MIN_GROUPS
        ? [LAG_LOW / Math.sqrt(s.groups), LAG_HIGH / Math.sqrt(s.groups)] : null;
    const low = !(s.ratio >= ratioLimits[0]);
    const high = s.ratio > ratioLimits[1];
    const runs = lagLimits !== null && s.lag1 > lagLimits[1];
    const alternates = lagLimits !== null && s.lag1 < lagLimits[0];
    const tied = s.groups < x.length;
    let verdict = 'ok';
    if (low)
        verdict = 'follows';
    else if (runs)
        verdict = 'runs';
    else if (high && alternates && tied)
        verdict = 'between';
    else if (high)
        verdict = 'spread';
    else if (alternates)
        verdict = 'alternates';
    return { ...s, ratioLimits, lagLimits, verdict };
}
