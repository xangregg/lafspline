// Runs of equal values, local extremes with their plateaus, and monotone sign patterns of a
// piecewise-constant fit (PLAN.md §1.5 "Counting local extremes", §1.6 "combined").

const DEFAULT_REL_TOL = 1e-9; // relative to the range of f

/**
 * @typedef {object} Run
 * @property {number} start  first index (0-based, inclusive)
 * @property {number} end    last index (inclusive)
 * @property {number} value  value of the run (its first element)
 */

/**
 * @typedef {object} Extreme
 * @property {'max'|'min'} type
 * @property {number} start  plateau first index (0-based, inclusive)
 * @property {number} end    plateau last index (inclusive)
 * @property {number} value
 * @property {number} run    index into runs
 */

function tolerance(f, relTol) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of f) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
    }
    return relTol * (hi - lo);
}

/**
 * Compress f into maximal runs of (approximately) equal consecutive values: a new run starts when a
 * value differs from the previous one by more than relTol × range(f).
 *
 * @param {ArrayLike<number>} f
 * @param {number} [relTol=1e-9]
 * @returns {Run[]}
 */
export function runs(f, relTol = DEFAULT_REL_TOL) {
    const n = f.length;
    if (n === 0)
        return [];
    const tol = tolerance(f, relTol);
    const out = [];
    let start = 0;
    for (let i = 1; i <= n; i++) {
        if (i === n || Math.abs(f[i] - f[i - 1]) > tol) {
            out.push({ start, end: i - 1, value: f[start] });
            start = i;
        }
    }
    return out;
}

/**
 * Local extremes of f: a run is a local max if both neighboring runs are lower, a local min if both
 * are higher. Endpoint runs are not counted.
 *
 * @param {ArrayLike<number>} f
 * @param {number} [relTol=1e-9]
 * @returns {{runs: Run[], extremes: Extreme[], nMax: number, nMin: number}}
 */
export function localExtremes(f, relTol = DEFAULT_REL_TOL) {
    const rs = runs(f, relTol);
    const extremes = [];
    let nMax = 0;
    let nMin = 0;
    for (let r = 1; r < rs.length - 1; r++) {
        const v = rs[r].value;
        const left = rs[r - 1].value;
        const right = rs[r + 1].value;
        let type = null;
        if (v > left && v > right)
            type = 'max';
        else if (v < left && v < right)
            type = 'min';
        if (!type)
            continue;
        if (type === 'max')
            nMax++;
        else
            nMin++;
        extremes.push({ type, start: rs[r].start, end: rs[r].end, value: v, run: r });
    }
    return { runs: rs, extremes, nMax, nMin };
}

/**
 * Monotone sign pattern for the steps between consecutive observations: s[i] describes the step from
 * observation i to i + 1 (i = 0..n−2), +1 up, −1 down, 0 free. Steps outside extreme plateaus get the
 * direction of the monotone segment containing them. Steps inside a local extreme's plateau are
 * either left free (plateaus = 'free', PLAN.md §1.6 as first written) or split at the plateau's
 * midpoint (plateaus = 'split': rising then falling for a max, the reverse for a min), which pins the
 * extreme to the midpoint but guarantees a fit obeying the signs has no extra extremes inside the
 * plateau. A constant f gives all zeros.
 *
 * @param {ArrayLike<number>} f
 * @param {{relTol?: number, plateaus?: 'free'|'split'}} [opts]
 * @returns {Int8Array}  length max(0, n − 1)
 */
export function stepSigns(f, opts = {}) {
    const relTol = opts.relTol ?? DEFAULT_REL_TOL;
    const plateaus = opts.plateaus ?? 'free';
    const n = f.length;
    const s = new Int8Array(Math.max(0, n - 1));
    const { runs: rs, extremes } = localExtremes(f, relTol);
    if (rs.length < 2)
        return s;
    const extremeType = new Array(rs.length).fill(null);
    for (const e of extremes)
        extremeType[e.run] = e.type;
    // Direction of each run's monotone segment: the sign of the jump leaving it (or entering it, for
    // the last run). For a non-extreme interior run both agree.
    const dir = new Int8Array(rs.length);
    for (let r = 0; r < rs.length; r++) {
        const next = r + 1 < rs.length ? rs[r + 1].value - rs[r].value : rs[r].value - rs[r - 1].value;
        dir[r] = Math.sign(next);
    }
    for (let r = 0; r < rs.length; r++) {
        const { start, end } = rs[r];
        const type = extremeType[r];
        if (!type) {
            for (let i = start; i < end; i++)
                s[i] = dir[r];
        }
        else if (plateaus === 'split') {
            // steps start..mid−1 lead up to the midpoint observation, mid..end−1 lead away from it
            const mid = Math.floor((start + end) / 2);
            const toward = type === 'max' ? 1 : -1;
            for (let i = start; i < end; i++)
                s[i] = i < mid ? toward : -toward;
        }
        if (r + 1 < rs.length)
            s[end] = Math.sign(rs[r + 1].value - rs[r].value);
    }
    return s;
}
