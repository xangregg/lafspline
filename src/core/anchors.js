// Value bounds and anchor points for phase 2 (PLAN §1.3b).
//
// Bounds lo ≤ g ≤ hi are applied to the region, not to the fit: each box of a band is intersected with [lo, hi].
// If the true curve is known to lie within the bounds, the intersection covers it whenever the band does, so the
// band's claim carries over unchanged. Only bands are bounded: an Aₙ box constrains an interval mean, and bounding a
// mean does not bound the values.
//
// Anchors are points the fit must pass through, (x_a, y_a), imposed as exact equalities on the coefficients in
// every phase-2 solve (fit.js). Beyond the knot range the fit continues as the tangent line at the nearer end
// (fit.js pointRows), so an anchor there ties the end's value and slope; the basis is not widened. A widened basis
// was tried first: its segments past the data have no region rows, and with little ambition (the a* scan's low
// end, the counting penalties' unpenalized first solve) the system lost positive definiteness for an anchor half
// the data's x range away. The straight bridge is also the least ambitious way to reach the point.

const MIN_BOX_SHARE = 0.01; // a bounded box narrower than this share of its full width counts as a conflict
const MAX_ANCHORS = 20; // more points than this is data, not a constraint

/**
 * @typedef {object} Anchors  points the phase-2 fit must pass through, x distinct and increasing
 * @property {Float64Array} x
 * @property {Float64Array} y
 */

/**
 * Anchors from [x, y] pairs: sorted by x, checked for finite values and distinct x.
 *
 * @param {ArrayLike<[number, number]>} pairs
 * @returns {Anchors|null}  null for no pairs
 * @throws {RangeError} on a non-finite value, a repeated x or more than MAX_ANCHORS pairs
 */
export function makeAnchors(pairs) {
    if (pairs.length === 0)
        return null;
    if (pairs.length > MAX_ANCHORS)
        throw new RangeError(`at most ${MAX_ANCHORS} anchors`);
    const sorted = Array.from(pairs, ([x, y]) => [Number(x), Number(y)]).sort((a, b) => a[0] - b[0]);
    for (let k = 0; k < sorted.length; k++) {
        if (!Number.isFinite(sorted[k][0]) || !Number.isFinite(sorted[k][1]))
            throw new RangeError('anchors need finite x and y');
        if (k > 0 && sorted[k][0] === sorted[k - 1][0])
            throw new RangeError(`two anchors at x = ${sorted[k][0]}`);
    }
    return { x: Float64Array.from(sorted, p => p[0]), y: Float64Array.from(sorted, p => p[1]) };
}

/**
 * A band's boxes intersected with lo ≤ y ≤ hi. The target (the bootstrap band's estimate) is clipped into the new
 * box; lower and upper are recomputed.
 *
 * @param {{grid: Float64Array, center: Float64Array, half: Float64Array, target?: Float64Array}} band
 * @param {{lower?: number|null, upper?: number|null}} bounds  null or undefined: unbounded on that side
 * @returns {{center: Float64Array, half: Float64Array, lower: Float64Array, upper: Float64Array,
 *     target: Float64Array|undefined, clipped: number}}  clipped: boxes the bounds narrowed
 * @throws {RangeError} if lower ≥ upper, or a box lies outside the bounds or keeps less than MIN_BOX_SHARE of its
 *     width (the region and the bounds disagree there)
 */
export function boundBand(band, bounds) {
    const lo = bounds.lower ?? -Infinity;
    const hi = bounds.upper ?? Infinity;
    if (!(lo < hi))
        throw new RangeError(`the lower bound ${lo} must be below the upper bound ${hi}`);
    const G = band.center.length;
    const center = new Float64Array(G);
    const half = new Float64Array(G);
    const lower = new Float64Array(G);
    const upper = new Float64Array(G);
    const target = band.target ? new Float64Array(G) : undefined;
    let clipped = 0;
    for (let j = 0; j < G; j++) {
        const l = Math.max(band.center[j] - band.half[j], lo);
        const u = Math.min(band.center[j] + band.half[j], hi);
        if (!(u - l >= MIN_BOX_SHARE * 2 * band.half[j])) {
            let where = 'barely within the bounds';
            if (band.center[j] - band.half[j] >= hi)
                where = `above the upper bound ${hi}`;
            else if (band.center[j] + band.half[j] <= lo)
                where = `below the lower bound ${lo}`;
            throw new RangeError(`the region lies ${where} at x = ${Number(band.grid[j].toPrecision(4))}, so the `
                + 'data disagree with the bounds');
        }
        if (l > band.center[j] - band.half[j] || u < band.center[j] + band.half[j])
            clipped++;
        lower[j] = l;
        upper[j] = u;
        center[j] = (l + u) / 2;
        half[j] = (u - l) / 2;
        if (target)
            target[j] = Math.min(u, Math.max(l, band.target[j]));
    }
    return { center, half, lower, upper, target, clipped };
}
