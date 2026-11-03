// Interval families 𝓘ₙ over the design indices. Intervals are {lo, hi}: 0-based, inclusive.

/**
 * @typedef {object} Interval
 * @property {number} lo     first index, 0-based inclusive
 * @property {number} hi     last index, 0-based inclusive
 * @property {number} level  resolution level k (window length ≈ λᵏ)
 */

/**
 * The multiresolution scheme of eq. (9), level by level, before de-duplication.
 *
 * Paper (1-based): l(j,k) = ⌊(j−1)λᵏ⌋ + 1, u(j,k) = min(⌊jλᵏ⌋, n), j = 1..⌈n/λᵏ⌉. 0-based here:
 * lo = ⌊(j−1)λᵏ⌋, hi = min(⌊jλᵏ⌋, n) − 1. Levels run k = 0..⌈log n / log λ⌉; the paper starts at
 * k = 1 but k = 0 (singletons) is included per PLAN.md Known issue 2. Each level partitions 0..n−1,
 * which is what the residual view draws as one row.
 *
 * @param {number} n       number of observations, ≥ 1
 * @param {number} lambda  λ > 1 (2 gives dyadic blocks)
 * @returns {Interval[][]}  levels[k] = intervals of level k, ordered by lo
 */
export function multiresolutionLevels(n, lambda = 2) {
    if (!(lambda > 1))
        throw new RangeError('lambda must be > 1');
    const kMax = Math.ceil(Math.log(n) / Math.log(lambda));
    const levels = [];
    for (let k = 0; k <= kMax; k++) {
        const width = Math.pow(lambda, k);
        const jMax = Math.ceil(n / width);
        const level = [];
        for (let j = 1; j <= jMax; j++) {
            const lo = Math.floor((j - 1) * width);
            const hi = Math.min(Math.floor(j * width), n) - 1;
            if (lo <= hi)
                level.push({ lo, hi, level: k });
        }
        levels.push(level);
    }
    return levels;
}

/**
 * The multiresolution family 𝓘ₙ of eq. (9) (with singletons), de-duplicated; when an interval occurs
 * at several levels the lowest level is kept. Size is O(n) (≤ 2n + log₂n for λ = 2).
 *
 * @param {number} n       number of observations, ≥ 1
 * @param {number} lambda  λ > 1
 * @returns {Interval[]}  ordered by level, then lo
 */
export function multiresolution(n, lambda = 2) {
    const seen = new Set();
    const out = [];
    for (const level of multiresolutionLevels(n, lambda)) {
        for (const iv of level) {
            const key = iv.lo * (n + 1) + iv.hi;
            if (seen.has(key))
                continue;
            seen.add(key);
            out.push(iv);
        }
    }
    return out;
}

/**
 * Every interval [i, j], 0 ≤ i ≤ j < n (n(n+1)/2 of them). Reference family for checks on small n.
 * `level` is ⌊log₂ length⌋, so rows group intervals of similar length.
 *
 * @param {number} n
 * @returns {Interval[]}
 */
export function allIntervals(n) {
    const out = [];
    for (let lo = 0; lo < n; lo++) {
        for (let hi = lo; hi < n; hi++)
            out.push({ lo, hi, level: Math.floor(Math.log2(hi - lo + 1)) });
    }
    return out;
}

/**
 * Overlapping multiresolution windows. Each level has one length W = round(λᵏ) (distinct lengths ≤ n, and n
 * itself), with windows of exactly W points starting every max(1, round(W / offsets)) points, plus one ending at
 * the last point. The paper's blocks (multiresolutionLevels) are fixed in position, so a run of residuals that
 * straddles a block boundary is split and can go undetected; shifted copies of each block catch it, at the cost
 * of more constraints (for λ = 2 about 2n blocks, 4n with offsets = 4, n·log₂n with offsets = Infinity) and a
 * larger multiscale statistic, so τ must be larger for the same coverage (calibrate it).
 *
 * @param {number} n        number of points, ≥ 1
 * @param {number} lambda   λ > 1
 * @param {number} offsets  windows per length W, ≥ 1 (Infinity: every start position)
 * @returns {Interval[][]}  levels[k] = windows of the k-th length, ordered by lo; level = k
 */
export function overlappingLevels(n, lambda = 2, offsets = 4) {
    if (!(lambda > 1))
        throw new RangeError('lambda must be > 1');
    if (!(offsets >= 1))
        throw new RangeError('offsets must be ≥ 1');
    const lengths = [];
    for (let w = 1; Math.round(w) < n; w *= lambda) {
        const W = Math.round(w);
        if (W !== lengths[lengths.length - 1])
            lengths.push(W);
    }
    lengths.push(n);
    return lengths.map((W, k) => {
        const step = Math.max(1, Math.round(W / offsets));
        const level = [];
        for (let lo = 0; lo + W <= n; lo += step)
            level.push({ lo, hi: lo + W - 1, level: k });
        if (level[level.length - 1].hi !== n - 1)
            level.push({ lo: n - W, hi: n - 1, level: k });
        return level;
    });
}

/**
 * @typedef {{kind: 'blocks', lambda: number}
 *     | {kind: 'overlapping', lambda: number, offsets: number}
 *     | {kind: 'all'}} IntervalFamily
 *     blocks: the paper's family (eq. 9); overlapping: overlappingLevels; all: every interval (O(m²), limited
 *     to m ≤ ALL_INTERVALS_MAX distinct x)
 */

export const ALL_INTERVALS_MAX = 300; // distinct x: 45,150 intervals, already slow for the MIP

/**
 * A family of x-intervals as observation ranges, level by level (ties never split; levels group intervals of
 * similar length, ⌊log₂ length⌋ for 'all').
 *
 * @param {ArrayLike<number>} x  sorted (ties allowed)
 * @param {IntervalFamily} family
 * @returns {Interval[][]}
 * @throws {RangeError} for 'all' with more than ALL_INTERVALS_MAX distinct x
 */
export function familyLevels(x, family) {
    const { start, m } = tieRanges(x);
    if (family.kind === 'blocks')
        return multiresolutionLevels(m, family.lambda).map(level => toObservations(level, start));
    if (family.kind === 'overlapping')
        return overlappingLevels(m, family.lambda, family.offsets).map(level => toObservations(level, start));
    if (family.kind === 'all') {
        if (m > ALL_INTERVALS_MAX)
            throw new RangeError(`all intervals needs at most ${ALL_INTERVALS_MAX} distinct x (have ${m})`);
        const levels = [];
        for (const iv of allIntervals(m))
            (levels[iv.level] ??= []).push(iv);
        return levels.map(level => toObservations(level, start));
    }
    throw new RangeError(`unknown interval family '${family.kind}'`);
}

/**
 * The family's intervals, de-duplicated (lowest level kept).
 *
 * @param {ArrayLike<number>} x
 * @param {IntervalFamily} family
 * @returns {Interval[]}
 */
export function familyIntervals(x, family) {
    const seen = new Set();
    const out = [];
    for (const level of familyLevels(x, family)) {
        for (const iv of level) {
            const key = `${iv.lo},${iv.hi}`;
            if (seen.has(key))
                continue;
            seen.add(key);
            out.push(iv);
        }
    }
    return out;
}

/**
 * Short description for labels, e.g. 'λ = 2', 'λ = 2, ¼-width steps', 'all intervals'.
 *
 * @param {IntervalFamily} family
 * @returns {string}
 */
export function familyText(family) {
    if (family.kind === 'blocks')
        return `λ = ${family.lambda}`;
    if (family.kind === 'overlapping') {
        const steps = family.offsets === Infinity ? 'every position' : `1/${family.offsets}-width steps`;
        return `λ = ${family.lambda}, ${steps}`;
    }
    return 'all intervals';
}

/**
 * Ranges of tied x: group j covers observations start[j]..start[j+1]−1 (x sorted ascending).
 *
 * @param {ArrayLike<number>} x  sorted
 * @returns {{start: Int32Array, m: number, ties: boolean}}  start has length m + 1, start[m] = n
 */
export function tieRanges(x) {
    const n = x.length;
    const start = [0];
    for (let i = 1; i < n; i++) {
        if (x[i] !== x[i - 1])
            start.push(i);
    }
    start.push(n);
    const m = start.length - 1;
    return { start: Int32Array.from(start), m, ties: m < n };
}

// Map intervals over distinct-x indices to observation ranges.
function toObservations(groupIntervals, start) {
    return groupIntervals.map(({ lo, hi, level }) => ({ lo: start[lo], hi: start[hi + 1] - 1, level }));
}

/**
 * The multiresolution scheme over intervals of x (paper eqs. 6 and 9: I = [t_j, t_k], |I| = number of
 * points in it), level by level. Built on the distinct x values and mapped to observation ranges, so an
 * interval never separates tied observations; without ties it equals multiresolutionLevels(n, λ).
 *
 * @param {ArrayLike<number>} x  sorted (ties allowed)
 * @param {number} lambda
 * @returns {Interval[][]}  observation-index intervals; level k groups λᵏ distinct x values
 */
export function xIntervalLevels(x, lambda = 2) {
    const { start, m } = tieRanges(x);
    return multiresolutionLevels(m, lambda).map(level => toObservations(level, start));
}

/**
 * De-duplicated multiresolution family of x-intervals (see xIntervalLevels). Without ties it equals
 * multiresolution(n, λ).
 *
 * @param {ArrayLike<number>} x  sorted (ties allowed)
 * @param {number} lambda
 * @returns {Interval[]}
 */
export function xIntervalFamily(x, lambda = 2) {
    const { start, m } = tieRanges(x);
    return toObservations(multiresolution(m, lambda), start);
}

/**
 * Every x-interval [t_j, t_k] as an observation range (m(m+1)/2 of them, m distinct x).
 *
 * @param {ArrayLike<number>} x  sorted (ties allowed)
 * @returns {Interval[]}
 */
export function allXIntervals(x) {
    const { start, m } = tieRanges(x);
    return toObservations(allIntervals(m), start);
}
