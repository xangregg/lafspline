// Aₙ as a phase-2 region (PLAN phase 4, open question 6). Each Aₙ constraint |Σ_{i∈I}(y_i − g_i)| ≤ c·√|I|
// (regions.js anRegion, paper eqs. 6–7) divided by |I| is a box for the mean of g over the interval:
//
//     |mean_I(g) − mean_I(y)| ≤ c/√|I|,
//
// and mean_I(g) = r_I·β is linear in the phase-2 coefficients (fit.js intervalRows). So Aₙ enters phase 2 exactly
// as a band does, with one box per interval instead of one per grid point, and its ratio |mean_I(g) − mean_I(y)|
// / (c/√|I|) is Aₙ's own |w(I)|/c. Everything downstream (the objective, a*, sparse ambition, the ladder) takes
// the boxes unchanged. Aₙ checks the fit at the observations only, so between them the fit is free; the
// ambition penalty, not the region, keeps it smooth there.

import { bsplineBasis } from './pspline.js';
import { intervalRows } from './fit.js';

/**
 * @typedef {object} AnBoxes
 * @property {'An'} kind
 * @property {string} label            the Aₙ region's
 * @property {Float64Array} grid       mean x of each interval (the position of its box, for the ladder and readouts)
 * @property {Float64Array} center     mean_I(y)
 * @property {Float64Array} half       c/√|I|, units of y
 * @property {Float64Array} lower      center − half
 * @property {Float64Array} upper      center + half
 * @property {import('./fit.js').Rows} rows  mean of the phase-2 basis over each interval
 * @property {{xl: number, xr: number, nseg: number, deg: number}} params  the phase-2 basis
 * @property {import('./regions.js').Region} region  the Aₙ region itself (sum form)
 */

/**
 * Aₙ's constraints as boxes for phase 2, with a cubic B-spline basis of nseg segments over [min x, max x].
 * O(Σ|I|) to build: O(n log n) for the paper's blocks.
 *
 * @param {import('./regions.js').Region} region  from anRegion
 * @param {ArrayLike<number>} x  the observations' x, sorted (as given to anRegion)
 * @param {{nseg: number, deg?: number}} basisOpts
 * @returns {AnBoxes}
 */
export function anBoxes(region, x, { nseg, deg = 3 }) {
    const n = x.length;
    const xl = x[0];
    const xr = x[n - 1];
    const basis = bsplineBasis(x, xl, xr, nseg, deg);
    const m = region.lo.length;
    const intervals = Array.from(region.lo, (lo, k) => ({ lo, hi: region.hi[k] }));
    const grid = new Float64Array(m);
    const center = new Float64Array(m);
    const half = new Float64Array(m);
    for (let k = 0; k < m; k++) {
        const { lo, hi } = intervals[k];
        const count = hi - lo + 1;
        let sx = 0;
        for (let i = lo; i <= hi; i++)
            sx += x[i];
        grid[k] = sx / count;
        center[k] = region.center[k] / count;
        half[k] = region.scale * region.width[k] / count;
    }
    return {
        kind: 'An',
        label: region.label,
        grid, center, half,
        lower: center.map((v, k) => v - half[k]),
        upper: center.map((v, k) => v + half[k]),
        rows: intervalRows(basis, intervals),
        params: { xl, xr, nseg, deg },
        region,
    };
}
