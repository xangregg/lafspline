// The ladder report (PLAN §1.6): does a constant, or a straight line, fit inside the region? Answered exactly on
// the check grid by a weighted minimax fit: the smallest r such that some constant (or line) g has
// |g_j − c_j| ≤ r·h_j at every grid point. The shape fits inside the region when r ≤ 1, and r says by how much it
// misses or what margin it leaves. This does not depend on the phase-2 knobs (κ, ε, ambition): it is a property
// of the region alone, which tells the user when a simpler answer than any phase-2 fit is enough.
//
// Both are convex problems solved by bisection, without an LP solver:
// - constant: for a given r, a constant fits iff max_j(c_j − r·h_j) ≤ min_j(c_j + r·h_j); that is monotone in r.
// - line: for a slope b, the best intercept is the constant minimax of c_j − b·x_j, and its ratio R(b) is convex
//   in b (a partial minimum of a jointly convex function), so a golden-section search over b finds the minimum.

const CHECK_TOL = 1e-9; // as band.js: a ratio up to 1 + CHECK_TOL counts as inside
const BISECT_STEPS = 200; // cap on bisection halvings; stops earlier at REL_TOL
const GOLDEN_STEPS = 200; // cap on golden-section steps; stops earlier at REL_TOL
const REL_TOL = 1e-13; // relative width at which both searches stop
const GOLDEN = (Math.sqrt(5) - 1) / 2;

/**
 * @typedef {object} LadderRung
 * @property {'constant'|'line'} shape
 * @property {number} ratio       smallest achievable max_j |g_j − c_j|/h_j
 * @property {boolean} fits       ratio ≤ 1 (to CHECK_TOL)
 * @property {number} intercept   g(x) = intercept + slope·(x − x0)
 * @property {number} slope       0 for the constant
 * @property {number} x0          grid midpoint, (min + max)/2
 * @property {Float64Array} g     the minimax shape at the grid
 * @property {number} argMax      grid index where |g − c|/h is largest
 */

/**
 * Weighted minimax constant: min over a of max_j |a − e_j|/h_j, by bisection on r.
 * O(G) per step; about 45 steps to REL_TOL.
 *
 * @param {ArrayLike<number>} e
 * @param {ArrayLike<number>} h  > 0
 * @returns {{ratio: number, value: number}}
 */
export function minimaxConstant(e, h) {
    const G = e.length;
    const feasible = r => {
        let lo = -Infinity;
        let hi = Infinity;
        for (let j = 0; j < G; j++) {
            lo = Math.max(lo, e[j] - r * h[j]);
            hi = Math.min(hi, e[j] + r * h[j]);
        }
        return { ok: lo <= hi, mid: (lo + hi) / 2 };
    };
    // a = e_0 reaches ratio max_j |e_0 − e_j|/h_j, so r* is at most that
    let rHi = 0;
    for (let j = 0; j < G; j++)
        rHi = Math.max(rHi, Math.abs(e[0] - e[j]) / h[j]);
    let rLo = 0;
    if (feasible(0).ok)
        return { ratio: 0, value: feasible(0).mid };
    for (let i = 0; i < BISECT_STEPS && rHi - rLo > REL_TOL * rHi; i++) {
        const m = (rLo + rHi) / 2;
        if (feasible(m).ok)
            rHi = m;
        else
            rLo = m;
    }
    return { ratio: rHi, value: feasible(rHi).mid };
}

/**
 * Weighted minimax line: min over (a, b) of max_j |a + b·(x_j − x0) − c_j|/h_j. The minimizing slope lies within
 * b₀ ± R(b₀)·(h_L + h_R)/(x_R − x_L), with b₀ the slope between the end centers: at the minimum the line's rise
 * between the ends differs from the centers' by at most R·(h_L + h_R), and R there is at most R(b₀).
 * O(G) per bisection step, about 45 steps per slope and 70 slopes.
 *
 * @param {ArrayLike<number>} x  sorted
 * @param {ArrayLike<number>} c
 * @param {ArrayLike<number>} h  > 0
 * @returns {{ratio: number, intercept: number, slope: number, x0: number}}
 */
export function minimaxLine(x, c, h) {
    const G = x.length;
    const x0 = (x[0] + x[G - 1]) / 2;
    if (G < 2 || x[G - 1] === x[0]) {
        const k = minimaxConstant(c, h);
        return { ratio: k.ratio, intercept: k.value, slope: 0, x0 };
    }
    const e = new Float64Array(G);
    const R = b => {
        for (let j = 0; j < G; j++)
            e[j] = c[j] - b * (x[j] - x0);
        return minimaxConstant(e, h);
    };
    const span = x[G - 1] - x[0];
    const b0 = (c[G - 1] - c[0]) / span;
    const reach = R(b0).ratio * (h[0] + h[G - 1]) / span;
    let lo = b0 - reach;
    let hi = b0 + reach;
    let m1 = hi - GOLDEN * (hi - lo);
    let m2 = lo + GOLDEN * (hi - lo);
    let r1 = R(m1).ratio;
    let r2 = R(m2).ratio;
    for (let i = 0; i < GOLDEN_STEPS && hi - lo > REL_TOL * (Math.abs(b0) + reach); i++) {
        if (r1 <= r2) {
            hi = m2;
            m2 = m1;
            r2 = r1;
            m1 = hi - GOLDEN * (hi - lo);
            r1 = R(m1).ratio;
        }
        else {
            lo = m1;
            m1 = m2;
            r1 = r2;
            m2 = lo + GOLDEN * (hi - lo);
            r2 = R(m2).ratio;
        }
    }
    const slope = (lo + hi) / 2;
    const best = R(slope);
    return { ratio: best.ratio, intercept: best.value, slope, x0 };
}

/**
 * The ladder for a region given on a grid: the minimax constant and line, each with whether it fits inside.
 *
 * @param {{grid: Float64Array, center: Float64Array, half: Float64Array}} region  (band.js)
 * @returns {LadderRung[]}  constant first, then line
 */
export function ladder(region) {
    const { grid, center, half } = region;
    const rung = (shape, { ratio, intercept, slope, x0 }) => {
        const g = Float64Array.from(grid, x => intercept + slope * (x - x0));
        let argMax = 0;
        let worst = -1;
        for (let j = 0; j < g.length; j++) {
            const r = Math.abs(g[j] - center[j]) / half[j];
            if (r > worst) {
                worst = r;
                argMax = j;
            }
        }
        return { shape, ratio, fits: ratio <= 1 + CHECK_TOL, intercept, slope, x0, g, argMax };
    };
    const k = minimaxConstant(center, half);
    const x0 = (grid[0] + grid[grid.length - 1]) / 2;
    return [
        rung('constant', { ratio: k.ratio, intercept: k.value, slope: 0, x0 }),
        rung('line', minimaxLine(grid, center, half)),
    ];
}
