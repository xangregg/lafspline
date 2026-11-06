// Phase 2 (PLAN §1.3): a P-spline g = Bβ on a check grid, penalized severely for leaving the region [c − h, c + h],
// slightly for distance from its center, and for ambition:
//
//     S(β) = κ·Σ_j dist(g_j, [l_j, u_j])² / h_j²  +  ε·Σ_j ((g_j − c_j) / h_j)²  +  λ₁‖D₁β‖²/s²  +  λ₂‖D₂β‖²/s²
//
// with s = median h. Dividing the penalties by s² makes every term unitless, so κ, ε and λ mean the same whatever
// the units of y. Internally everything is in units of s: c' = c/s, h' = h/s, β' = β/s. The ambition knob is
// normalized further (PLAN open question 1): ambitionLambdas makes it independent of x units, segments and grid
// density, and leavingAmbition finds a*, the largest ambition before the fit leaves the region, which the UI's
// knob is measured from.
//
// Two solvers for the same convex objective:
// - fitIterated: fix which grid points are outside (above u or below l), solve the resulting banded penalized least
//   squares, update, repeat until the sets stop changing (Bollaerts, Eilers & Van Mechelen 2006). S is convex and
//   continuously differentiable, so when the sets stop changing the gradient is zero and the fit is the exact
//   minimizer. The iteration is not guaranteed to stop; it is capped and reports whether it converged.
// - fitQP: the same objective as a convex QP for HiGHS, with one free slack t_j per grid point and ranged rows
//   l'_j ≤ g'_j + t_j ≤ u'_j (cost κ t_j²/h'_j², so the optimal |t_j| is the distance outside). With hard: true the
//   slacks are dropped (the region as a constraint) and the problem can be infeasible.
//
// ε > 0 is required: with the dense check grid it makes S strictly convex and the iterated system positive definite
// (with rows only at the observations, PENALTY_FLOOR does).
//
// The region need not be a band on a grid (PLAN phase 4): each of its G constraints is a box for one linear
// functional of β, |r_kβ − c_k| ≤ h_k. For a band r_k is the basis at grid point k (deg + 1 nonzeros). For Aₙ it is
// the mean of the basis rows over the observations in an interval I, with c_k = mean_I(y) and h_k = c/√|I| (an.js),
// whose support grows with I; the Newton system is then banded only as far as the widest row reaches (dense for
// I = everything), still nb × nb. Specs carry such rows in `rows`; without them the rows are the basis at `grid`.

import {
    bsplineBasis, addDifferencePenalty, differenceCoefficients, choleskyBanded, solveCholesky, bandedInverse,
} from './pspline.js';

const MAX_ITERATIONS = 200; // iterated weights: cap on Newton solves
const CHECK_TOL = 1e-9; // a grid point counts as outside when |g − c|/h > 1 + CHECK_TOL
export const DEFAULT_KAPPA = 1e4; // weight on the squared distance outside the region, per grid point
export const DEFAULT_EPSILON = 1e-2; // weight on the squared distance from the center, per grid point
// Largest penalty relative to the center pull per coefficient, λ_d·nb/(ε·G). Beyond about 1e12 the Newton system
// loses too many digits: a straight line inside the region came back outside it, without a Cholesky failure.
const MAX_PENALTY_RATIO = 1e10;
// A second-difference penalty always present, relative to the strongest penalty in the system (the center pull per
// coefficient, or the largest λ_d·w_r). Aₙ's rows sit only at the observations, so a coefficient whose support holds
// none is free without ambition, or in a sparse fit's start and its pinned refits: the system was singular
// (Cholesky failed on chess recall). Relative to the center pull alone, the floor was still swamped by pinned
// differences 1e22 times stronger (fibers, 5 distinct x for 8 coefficients). This keeps the condition number
// near 1/PENALTY_FLOOR whatever the rows cover, and carries the fit straight across such gaps. Elsewhere it is
// negligible: 1e-12 of a penalty already present (the effective dimension without ambition is K − 9e-9).
const PENALTY_FLOOR = 1e-12;
export const LEAVE_TOL = 0.01; // the fit has left the region when its largest |g − c|/h exceeds 1 + LEAVE_TOL
const SCAN_DECADES = 16; // a* search: ambitions from the cap down this many decades
const SCAN_STEP = 1; // decades between scanned ambitions
const BISECT_TOL = 0.02; // decades: a* search stops when the bracket is this narrow
const MAX_REWEIGHTS = 100; // adaptive ridge: cap on reweightings
const MAX_PRUNE_PASSES = 3; // adaptive ridge: backward-elimination passes over the joins
const SUPPORT_STABLE = 3; // adaptive ridge stops after this many reweightings without a change of support
const MAX_SIGN_REFITS = 20; // shape search: refits of one support while differences still break their sign
const MAX_SHAPE_PASSES = 6; // shape search: accepted removals of extremes or inflections, at most
const SHAPE_GAIN = 1e-9; // shape search: a candidate must lower the objective by this much, relative
const SPARSE_TOP_FACTOR = 10; // sparse ambition: top of the range, relative to where the polynomial is optimal
const SPARSE_MIN_TOP = 1e-12; // sparse ambition: top when the polynomial matches the center (S_poly = 0)
const SPARSE_DELTA = 1e-4; // adaptive ridge: a step smaller than this (units of median h) counts as zero
const LEVEL_RISE = 0.05; // shaped bends: a straight piece rising less than this (median h) over its length is level
const POWER_TOL = 1e-7; // power penalty: reweighting stops when S falls by less than this, relative
// power penalty: cap on reweightings. ℓ₁ (q = 1) by reweighting is slow near zero differences: 160–520 on the
// examples, 100 left the fit up to 0.1 h from its limit; q ≤ 0.5 took 10–40.
const MAX_POWER_REWEIGHTS = 1000;
// power penalty: a bend θ (second difference of β') changes the fit by about θ·nseg across [xl, xr]; it counts as a
// bend in the readouts when that is at least this (units of median h)
const POWER_BEND = 1e-3;
// Shaped bends (fitShaped): defaults of the costs, in units of one plain bend's cost λ per free second difference.
// toFlat is neutral: it acts only through the reweighting, where it never changed a fit (PLAN §1.4b). Extremes and
// inflections act through the shape search.
export const SHAPE_DEFAULTS = Object.freeze({
    toFlat: 1, // per difference of a bend whose flatter side has zero slope (1 when both sides are equally steep)
    extremum: 2, // added once per bend whose sides slope in opposite directions (a local maximum or minimum)
    inflection: 1, // added once per change of the bending direction (sign of the second differences)
    length: 0.05, // fraction of the x range over which a bend leaves its tangent by one median half-width
});

/**
 * @typedef {object} Rows  linear functionals of the coefficients with contiguous support: row k is
 *     Σ_{r < width[k]} values[offset[k] + r]·β[first[k] + r]
 * @property {number} n           number of rows
 * @property {Int32Array} first   first coefficient of each row
 * @property {Int32Array} width   coefficients in each row
 * @property {Int32Array} offset  start of each row in values (length n + 1)
 * @property {Float64Array} values
 */

/**
 * @typedef {object} Phase2Spec
 * @property {Float64Array} [grid]  check grid, within [xl, xr]: the rows are the basis there (without rows)
 * @property {Rows} [rows]          the region's linear functionals (instead of grid)
 * @property {Float64Array} c       region center per row
 * @property {Float64Array} h       region half-width per row, > 0
 * @property {number} xl            knot range of the phase-2 basis
 * @property {number} xr
 * @property {number} nseg          segments of the phase-2 basis
 * @property {number} [deg]         degree (3)
 * @property {number} kappa         weight on the squared distance outside the region (κ)
 * @property {number} epsilon       weight on the squared distance from the center (ε > 0)
 * @property {Float64Array} [target]  what ε pulls toward, per row, inside the region (default c, its midpoint): a
 *     region that is not symmetric about its estimate (the bootstrap band) passes the estimate
 * @property {number} lambda1       weight on ‖D₁β‖²/s² (toward flat)
 * @property {number} lambda2       weight on ‖D₂β‖²/s² (toward straight)
 * @property {Float64Array} [w1]    weights on the squared first differences (adaptive ridge; 1 without)
 * @property {Float64Array} [w2]    weights on the squared second differences
 */

/**
 * @typedef {object} Phase2Fit
 * @property {Float64Array} beta     coefficients (units of y)
 * @property {Float64Array} g        the rows applied to β: the fit at the grid, or its interval means (units of y)
 * @property {number} objective      S(β) (unitless)
 * @property {number} maxRatio       max_j |g_j − c_j| / h_j
 * @property {number} outside        rows (grid points) with ratio > 1
 * @property {number} ms             wall-clock time
 */

function prepare(spec) {
    const { grid, c, h, xl, xr, nseg } = spec;
    const deg = spec.deg ?? 3;
    if (!(spec.epsilon > 0))
        throw new RangeError('phase 2: epsilon > 0 required');
    const rows = spec.rows ?? basisRows(bsplineBasis(grid, xl, xr, nseg, deg));
    if (c.length !== rows.n || h.length !== rows.n)
        throw new RangeError('phase 2: rows, c and h differ in length');
    const sorted = Float64Array.from(h).sort();
    const s = sorted[sorted.length >> 1];
    if (!(sorted[0] > 0))
        throw new RangeError('phase 2: half-widths must be positive');
    const cs = Float64Array.from(c, v => v / s);
    const hs = Float64Array.from(h, v => v / s);
    if (spec.target && spec.target.length !== rows.n)
        throw new RangeError('phase 2: target differs in length from the rows');
    const ts = spec.target ? Float64Array.from(spec.target, v => v / s) : cs;
    const nb = nseg + deg;
    let widest = 0;
    for (let k = 0; k < rows.n; k++)
        widest = Math.max(widest, rows.width[k]);
    const W = Math.min(nb, Math.max(widest, Math.max(deg, 2) + 1));
    return { spec, s, cs, hs, ts, rows, W, nb };
}

/**
 * The basis at its points as Rows.
 *
 * @param {import('./pspline.js').Basis} basis
 * @returns {Rows}
 */
export function basisRows(basis) {
    const bw = basis.deg + 1;
    return {
        n: basis.n,
        first: Int32Array.from(basis.first),
        width: new Int32Array(basis.n).fill(bw),
        offset: Int32Array.from({ length: basis.n + 1 }, (_, k) => k * bw),
        values: basis.values,
    };
}

/**
 * The mean of the basis rows over each interval of observations (basis at the sorted data x): the functional
 * β ↦ mean_{i∈I} g(x_i) of an Aₙ constraint. Rows of sorted x have nondecreasing first coefficients, so the
 * support of the mean is first[lo] .. first[hi] + deg. O(Σ|I|·deg) to build.
 *
 * @param {import('./pspline.js').Basis} basis  at the observations, x sorted
 * @param {{lo: number, hi: number}[]} intervals  0-based inclusive observation ranges
 * @returns {Rows}
 */
export function intervalRows(basis, intervals) {
    const bw = basis.deg + 1;
    const n = intervals.length;
    const first = new Int32Array(n);
    const width = new Int32Array(n);
    const offset = new Int32Array(n + 1);
    for (let k = 0; k < n; k++) {
        const { lo, hi } = intervals[k];
        first[k] = basis.first[lo];
        width[k] = basis.first[hi] + bw - first[k];
        offset[k + 1] = offset[k] + width[k];
    }
    const values = new Float64Array(offset[n]);
    for (let k = 0; k < n; k++) {
        const { lo, hi } = intervals[k];
        const inv = 1 / (hi - lo + 1);
        for (let i = lo; i <= hi; i++) {
            const at = offset[k] + basis.first[i] - first[k];
            for (let r = 0; r < bw; r++)
                values[at + r] += inv * basis.values[i * bw + r];
        }
    }
    return { n, first, width, offset, values };
}

/** The rows applied to β. */
function rowValues(rows, beta) {
    const g = new Float64Array(rows.n);
    for (let k = 0; k < rows.n; k++) {
        let v = 0;
        const f = rows.first[k];
        const off = rows.offset[k];
        for (let r = 0; r < rows.width[k]; r++)
            v += rows.values[off + r] * beta[f + r];
        g[k] = v;
    }
    return g;
}

/** RᵀWR (band storage of width W, which must cover the widest row) and RᵀWt. O(Σ width²). */
function rowCrossProducts(rows, w, t, nb, W) {
    const A = new Float64Array(nb * W);
    const rhs = new Float64Array(nb);
    for (let k = 0; k < rows.n; k++) {
        const f = rows.first[k];
        const off = rows.offset[k];
        for (let r = 0; r < rows.width[k]; r++) {
            const wv = w[k] * rows.values[off + r];
            rhs[f + r] += wv * t[k];
            for (let q = 0; q <= r; q++)
                A[(f + r) * W + (r - q)] += wv * rows.values[off + q];
        }
    }
    return { A, rhs };
}

/**
 * A spline with coefficients β (from a phase-2 fit) evaluated at points within [xl, xr].
 *
 * @param {ArrayLike<number>} beta
 * @param {ArrayLike<number>} at
 * @param {{xl: number, xr: number, nseg: number, deg?: number}} knots
 * @returns {Float64Array}
 */
export function splineValues(beta, at, { xl, xr, nseg, deg = 3 }) {
    return rowValues(basisRows(bsplineBasis(at, xl, xr, nseg, deg)), beta);
}

function addAmbition(A, p) {
    const { lambda1, lambda2, w1, w2 } = p.spec;
    addDifferencePenalty(A, p.nb, p.W, 2, PENALTY_FLOOR * strongestPenalty(p));
    if (lambda1 > 0)
        addWeightedDifferences(A, p.nb, p.W, 1, lambda1, w1);
    if (lambda2 > 0)
        addWeightedDifferences(A, p.nb, p.W, 2, lambda2, w2);
}

/** The largest weight on any squared difference, λ_d·max w, or the center pull per coefficient ε·G/nb if larger. */
function strongestPenalty(p) {
    const { lambda1, lambda2, w1, w2, epsilon } = p.spec;
    const top = (lambda, w) => (lambda > 0 ? lambda * (w ? w.reduce((a, v) => Math.max(a, v), 0) : 1) : 0);
    return Math.max(epsilon * p.rows.n / p.nb, top(lambda1, w1), top(lambda2, w2));
}

/** λ·DᵀΩD into A, Ω = diag(w) (one weight per difference), or λ·DᵀD without w. */
function addWeightedDifferences(A, nb, W, d, lambda, w) {
    if (!w) {
        addDifferencePenalty(A, nb, W, d, lambda);
        return;
    }
    const c = differenceCoefficients(d);
    for (let r = 0; r + d < nb; r++) {
        const scale = lambda * w[r];
        for (let i = 0; i <= d; i++) {
            for (let j = 0; j <= i; j++)
                A[(r + i) * W + (i - j)] += scale * c[i] * c[j];
        }
    }
}

function bandedQuadratic(M, N, W, v) {
    let q = 0;
    for (let i = 0; i < N; i++) {
        q += M[i * W] * v[i] * v[i];
        for (let k = 1; k <= Math.min(i, W - 1); k++)
            q += 2 * M[i * W + k] * v[i] * v[i - k];
    }
    return q;
}

/** S at scaled coefficients β' (grid values g' already computed). */
function objective(p, beta, g) {
    const { kappa, epsilon } = p.spec;
    let sum = 0;
    for (let j = 0; j < g.length; j++) {
        const lo = p.cs[j] - p.hs[j];
        const hi = p.cs[j] + p.hs[j];
        const out = g[j] > hi ? g[j] - hi : g[j] < lo ? lo - g[j] : 0;
        sum += (kappa * out * out + epsilon * (g[j] - p.ts[j]) ** 2) / (p.hs[j] * p.hs[j]);
    }
    const P = new Float64Array(p.nb * p.W);
    addAmbition(P, p);
    return sum + bandedQuadratic(P, p.nb, p.W, beta);
}

function finish(p, betaScaled, start) {
    const gs = rowValues(p.rows, betaScaled);
    const check = checkRegion(gs, p.cs, p.hs);
    return {
        beta: betaScaled.map(v => v * p.s),
        g: gs.map(v => v * p.s),
        objective: objective(p, betaScaled, gs),
        maxRatio: check.maxRatio,
        outside: check.outside,
        ms: performance.now() - start,
    };
}

/**
 * The phase-2 objective S at coefficients β (units of y), for checking a solver's result.
 *
 * @param {Phase2Spec} spec
 * @param {ArrayLike<number>} beta
 * @returns {number}  unitless
 */
export function phase2Objective(spec, beta) {
    const p = prepare(spec);
    const scaled = Float64Array.from(beta, v => v / p.s);
    return objective(p, scaled, rowValues(p.rows, scaled));
}

/**
 * Largest |g − c|/h and the number of points where it exceeds 1.
 *
 * @param {ArrayLike<number>} g
 * @param {ArrayLike<number>} c
 * @param {ArrayLike<number>} h
 * @returns {{maxRatio: number, outside: number}}
 */
export function checkRegion(g, c, h) {
    let maxRatio = 0;
    let outside = 0;
    for (let j = 0; j < g.length; j++) {
        const r = Math.abs(g[j] - c[j]) / h[j];
        maxRatio = Math.max(maxRatio, r);
        if (r > 1 + CHECK_TOL)
            outside++;
    }
    return { maxRatio, outside };
}

/**
 * Phase-2 fit by iterated weights: a semismooth Newton method. Each iteration fixes which grid points are outside
 * and solves the resulting banded penalized least squares (one Cholesky, O(G·deg² + K·p²)). That solve is the
 * Newton step for S at the current fit; if it keeps the same points outside, it is the exact minimizer. Otherwise
 * the fit moves along the step to the minimum of S on that line (exact line search, bisection on the derivative,
 * O(G) per probe), so S decreases strictly and the sets cannot cycle; undamped, they did (Blocks in the phase-0
 * risk check).
 *
 * @param {Phase2Spec} spec
 * @param {{maxIterations?: number, lineSearch?: boolean, side?: Int8Array}} [opts]  lineSearch: false takes every
 *     full step (the undamped iteration, kept for comparison; it can cycle); side: starting guess of which grid
 *     points are above (+1) or below (−1) the region, e.g. a previous fit's (all inside without it)
 * @returns {Phase2Fit & {ed: number, iterations: number, converged: boolean, damped: number, side: Int8Array}}
 *     ed: effective dimension; iterations: Newton solves; damped: steps shortened by the line search; side: which
 *     grid points end above (+1) or below (−1) the region
 */
export function fitIterated(spec, opts = {}) {
    const start = performance.now();
    const p = prepare(spec);
    const maxIterations = opts.maxIterations ?? MAX_ITERATIONS;
    const useLineSearch = opts.lineSearch ?? true;
    const G = p.rows.n;
    const P = new Float64Array(p.nb * p.W);
    addAmbition(P, p);
    let side = opts.side ? Int8Array.from(opts.side) : new Int8Array(G); // +1 above u, −1 below l, 0 inside
    let beta = null;
    let g = null;
    let iterations = 0;
    let damped = 0;
    let converged = false;
    while (iterations < maxIterations) {
        iterations++;
        const { A, rhs } = newtonSystem(p, side);
        const betaN = solveCholesky(choleskyBanded(A, p.nb, p.W), p.nb, p.W, rhs);
        const gN = rowValues(p.rows, betaN);
        const sideN = sidesOf(gN, p);
        if (sameSides(sideN, side)) {
            beta = betaN;
            g = gN;
            converged = true;
            break;
        }
        if (beta === null) {
            beta = betaN;
            g = gN;
        }
        else {
            const t = useLineSearch ? lineSearch(p, P, beta, g, betaN, gN) : 1;
            if (t === 0)
                break; // no descent along the Newton step: only possible through rounding
            if (t < 1)
                damped++;
            for (let k = 0; k < p.nb; k++)
                beta[k] += t * (betaN[k] - beta[k]);
            for (let j = 0; j < G; j++)
                g[j] += t * (gN[j] - g[j]);
        }
        side = sidesOf(g, p);
    }
    const finalSide = sidesOf(g, p);
    const ed = effectiveDimension(p, finalSide, P);
    return { ...finish(p, beta, start), ed, iterations, converged, damped, side: finalSide };
}

/**
 * The Newton system for fixed sides: (BᵀVB + P)β' = BᵀVt, with weight v_j = (ε + κ_j)/h'_j² and target
 * t_j = (ε c'_j + κ_j e_j)/(ε + κ_j), where κ_j = κ and e_j the violated edge for a point outside, κ_j = 0 inside.
 */
function newtonSystem(p, side) {
    const { kappa, epsilon } = p.spec;
    const G = p.rows.n;
    const weight = new Float64Array(G);
    const target = new Float64Array(G);
    for (let j = 0; j < G; j++) {
        const h2 = p.hs[j] * p.hs[j];
        const k = side[j] !== 0 ? kappa : 0;
        const edge = p.cs[j] + side[j] * p.hs[j];
        weight[j] = (epsilon + k) / h2;
        target[j] = (epsilon * p.ts[j] + k * edge) / (epsilon + k);
    }
    const { A, rhs } = rowCrossProducts(p.rows, weight, target, p.nb, p.W);
    addAmbition(A, p);
    return { A, rhs };
}

/**
 * Effective dimension of the fit, trace of the hat matrix of the final Newton system:
 * tr((BᵀVB + P)⁻¹BᵀVB) = K − tr(A⁻¹P), from the band of A⁻¹ (P lies within it). Between the order of the
 * ambition penalty (a line: 2) and K (no ambition). With fixed sides the fit is linear in the targets, so this is
 * the usual count of parameters spent; the sides themselves depend on the region, which it does not count.
 */
function effectiveDimension(p, side, P) {
    const { A } = newtonSystem(p, side);
    const Z = bandedInverse(choleskyBanded(A, p.nb, p.W), p.nb, p.W);
    let trace = 0;
    for (let i = 0; i < p.nb; i++) {
        trace += Z[i * p.W] * P[i * p.W];
        for (let k = 1; k <= Math.min(i, p.W - 1); k++)
            trace += 2 * Z[i * p.W + k] * P[i * p.W + k];
    }
    return p.nb - trace;
}

const LINE_SEARCH_STEPS = 50; // bisection halvings: step length to 2⁻⁵⁰

function sidesOf(g, p) {
    const side = new Int8Array(g.length);
    for (let j = 0; j < g.length; j++)
        side[j] = g[j] > p.cs[j] + p.hs[j] ? 1 : g[j] < p.cs[j] - p.hs[j] ? -1 : 0;
    return side;
}

function sameSides(a, b) {
    for (let j = 0; j < a.length; j++) {
        if (a[j] !== b[j])
            return false;
    }
    return true;
}

/** uᵀMv for a banded symmetric M. */
function bandedBilinear(M, N, W, u, v) {
    let q = 0;
    for (let i = 0; i < N; i++) {
        q += M[i * W] * u[i] * v[i];
        for (let k = 1; k <= Math.min(i, W - 1); k++)
            q += M[i * W + k] * (u[i] * v[i - k] + u[i - k] * v[i]);
    }
    return q;
}

/**
 * The t ∈ [0, 1] minimizing φ(t) = S(β + t·d), d = βN − β. φ is convex and piecewise quadratic, so its derivative
 * φ'(t) = Σ_j 2[κ·r_j(t) + ε(g_j(t) − c_j)]·gd_j/h_j² + 2(βᵀPd + t·dᵀPd), with r_j the signed distance outside,
 * is nondecreasing and bisection finds its zero.
 */
function lineSearch(p, P, beta, g, betaN, gN) {
    const { kappa, epsilon } = p.spec;
    const d = Float64Array.from(betaN, (v, k) => v - beta[k]);
    const gd = Float64Array.from(gN, (v, j) => v - g[j]);
    const bPd = bandedBilinear(P, p.nb, p.W, beta, d);
    const dPd = bandedBilinear(P, p.nb, p.W, d, d);
    const slope = t => {
        let s = 2 * (bPd + t * dPd);
        for (let j = 0; j < g.length; j++) {
            const gj = g[j] + t * gd[j];
            const hi = p.cs[j] + p.hs[j];
            const lo = p.cs[j] - p.hs[j];
            const r = gj > hi ? gj - hi : gj < lo ? gj - lo : 0;
            s += 2 * (kappa * r + epsilon * (gj - p.ts[j])) * gd[j] / (p.hs[j] * p.hs[j]);
        }
        return s;
    };
    if (slope(1) <= 0)
        return 1;
    if (slope(0) >= 0)
        return 0;
    let a = 0;
    let b = 1;
    for (let i = 0; i < LINE_SEARCH_STEPS; i++) {
        const m = (a + b) / 2;
        if (slope(m) > 0)
            b = m;
        else
            a = m;
    }
    return (a + b) / 2;
}

/**
 * The phase-2 objective as HiGHS ModelData (a convex QP; HiGHS minimizes c'x + ½x'Qx + offset). Columns: the K
 * scaled coefficients β', then (soft region) one slack per grid point. Pure: builds arrays only.
 *
 * @param {Phase2Spec} spec
 * @param {{hard?: boolean}} [opts]  hard: the region as constraints, no slacks
 * @returns {{model: object, scale: number, nb: number}}  scale s: β = s·β'
 */
export function buildQP(spec, opts = {}) {
    const p = prepare(spec);
    const hard = opts.hard ?? false;
    const { kappa, epsilon } = spec;
    const { rows, nb, W, cs, hs, ts } = p;
    const G = rows.n;
    const numCols = hard ? nb : nb + G;
    const invH2 = Float64Array.from(hs, v => 1 / (v * v));
    const { A, rhs } = rowCrossProducts(rows, invH2, ts, nb, W);
    for (let i = 0; i < A.length; i++)
        A[i] *= epsilon;
    addAmbition(A, p);

    // Hessian Q = 2·(εBᵀH'B + P) on β', 2κ/h'² on the slacks; lower triangle by columns
    const starts = new Int32Array(numCols + 1);
    const indices = [];
    const values = [];
    for (let j = 0; j < nb; j++) {
        starts[j] = indices.length;
        for (let k = 0; k < W && j + k < nb; k++) {
            const v = A[(j + k) * W + k];
            if (v !== 0) {
                indices.push(j + k);
                values.push(2 * v);
            }
        }
    }
    if (!hard) {
        for (let j = 0; j < G; j++) {
            starts[nb + j] = indices.length;
            indices.push(nb + j);
            values.push(2 * kappa * invH2[j]);
        }
    }
    starts[numCols] = indices.length;

    const colCost = new Float64Array(numCols);
    for (let k = 0; k < nb; k++)
        colCost[k] = -2 * epsilon * rhs[k];
    let offset = 0;
    for (let j = 0; j < G; j++)
        offset += epsilon * ts[j] * ts[j] * invH2[j];

    // rows: l'_j ≤ (Bβ')_j (+ t_j) ≤ u'_j
    const rowStarts = new Int32Array(G + 1);
    const rowIndices = new Int32Array(rows.offset[G] + (hard ? 0 : G));
    const rowEntries = new Float64Array(rowIndices.length);
    let e = 0;
    for (let j = 0; j < G; j++) {
        rowStarts[j] = e;
        for (let r = 0; r < rows.width[j]; r++) {
            rowIndices[e] = rows.first[j] + r;
            rowEntries[e] = rows.values[rows.offset[j] + r];
            e++;
        }
        if (!hard) {
            rowIndices[e] = nb + j;
            rowEntries[e] = 1;
            e++;
        }
    }
    rowStarts[G] = e;

    const model = {
        numCols,
        numRows: G,
        offset,
        colCost,
        colLower: new Float64Array(numCols).fill(-Infinity),
        colUpper: new Float64Array(numCols).fill(Infinity),
        rowLower: Float64Array.from(cs, (v, j) => v - hs[j]),
        rowUpper: Float64Array.from(cs, (v, j) => v + hs[j]),
        matrix: { format: 'csr', numRows: G, numCols, starts: rowStarts, indices: rowIndices, values: rowEntries },
        hessian: { format: 'triangular', dimension: numCols, starts, indices: Int32Array.from(indices),
            values: Float64Array.from(values) },
    };
    return { model, scale: p.s, nb };
}

/**
 * Turn a QP solution (from solveModel here or in the worker) into a phase-2 fit.
 *
 * @param {Phase2Spec} spec
 * @param {{status: string, x: Float64Array|null, ms: number}} solution
 * @returns {(Phase2Fit & {status: string})|{status: string, ms: number}}  without a fit unless optimal
 */
export function qpResult(spec, solution) {
    if (!solution.x)
        return { status: solution.status, ms: solution.ms };
    const p = prepare(spec);
    const fit = finish(p, solution.x.slice(0, p.nb), performance.now());
    return { ...fit, status: solution.status, ms: solution.ms };
}

/**
 * λ₁ and λ₂ for an ambition a and a share θ of it toward flat (PLAN open question 1). With x mapped to [0, 1] and
 * y in units of s, a difference of order d over K = nseg + 3 equally spaced coefficients approximates
 * δ^d·g⁽ᵈ⁾ with δ = 1/nseg, so ‖D_dβ'‖² ≈ nseg^(1−2d)·∫g'⁽ᵈ⁾², and the region terms, summed over G grid points,
 * ≈ G·∫(…). Scaling λ_d by G·nseg^(2d−1) turns S/G into the functional
 *
 *     κ∫dist²/h² + ε∫((g − c)/h)² + a·θ·∫g'² + a·(1 − θ)·∫g''²
 *
 * on [0, 1], so a means the same whatever the number of segments, the check-grid density and the units of x and
 * y (the integrals are in grid-point measure, which follows the data where the data are dense).
 *
 * @param {{ambition: number, flat?: number, G: number, nseg: number}} opts  ambition a ≥ 0; flat θ ∈ [0, 1]
 *     (0: toward straight only); G grid points; nseg segments of the phase-2 basis
 * @returns {{lambda1: number, lambda2: number}}
 */
export function ambitionLambdas({ ambition, flat = 0, G, nseg }) {
    return {
        lambda1: ambition * flat * G * nseg,
        lambda2: ambition * (1 - flat) * G * nseg ** 3,
    };
}

/**
 * A phase-2 spec from a region: a phase-1 band (band.js: the check grid, center and half-width, and the band's
 * knot range; the phase-2 basis has the band's segments unless nseg is given) or Aₙ's boxes (an.js, which carry
 * their rows, built for their own segments).
 *
 * @param {{grid: Float64Array, center: Float64Array, half: Float64Array, rows?: Rows, params: {xl: number,
 *     xr: number, nseg: number}}} region
 * @param {{ambition: number, flat?: number, kappa?: number, epsilon?: number, nseg?: number}} opts
 * @returns {Phase2Spec & {ambition: number, flat: number}}
 */
export function regionSpec(region, opts) {
    if (region.rows && opts.nseg !== undefined && opts.nseg !== region.params.nseg)
        throw new RangeError('regionSpec: the region\'s rows are for its own segments');
    const nseg = opts.nseg ?? region.params.nseg;
    const flat = opts.flat ?? 0;
    return {
        grid: region.grid, rows: region.rows, c: region.center, h: region.half, target: region.target,
        xl: region.params.xl, xr: region.params.xr, nseg, deg: 3,
        kappa: opts.kappa ?? DEFAULT_KAPPA,
        epsilon: opts.epsilon ?? DEFAULT_EPSILON,
        ambition: opts.ambition, flat,
        ...ambitionLambdas({ ambition: opts.ambition, flat, G: region.center.length, nseg }),
    };
}

/**
 * The largest ambition the iteration computes reliably: the larger of λ₁, λ₂ at most MAX_PENALTY_RATIO times
 * the center pull per coefficient, ε·G/nb (h' is 1 at the median).
 *
 * @param {{epsilon: number, flat?: number, nseg: number, deg?: number}} spec
 * @returns {number}
 */
export function maxAmbition(spec) {
    const flat = spec.flat ?? 0;
    const nb = spec.nseg + (spec.deg ?? 3);
    const perUnit = Math.max(flat * spec.nseg, (1 - flat) * spec.nseg ** 3); // λ_d / (a·G)
    return MAX_PENALTY_RATIO * spec.epsilon / (nb * perUnit);
}

/**
 * a*, the largest ambition before the fit first leaves the region: the ambition knob's zero (PLAN open question
 * 1), so the knob means the same on every data set. "Leaves" is a largest |g − c|/h above 1 + LEAVE_TOL; a soft
 * region is never met exactly, its violations shrink like 1/κ. Scans upward from SCAN_DECADES below the cap one
 * decade at a time and stops at the first ambition whose fit leaves, then bisects on log a. The ratio is not
 * proven monotone in a, so this is the first crossing found, not necessarily the only one. When no ambition up to
 * the cap leaves, a* is the cap: the region holds the limiting polynomial (a line, for D₂).
 *
 * @param {Phase2Spec & {flat?: number}} spec  its λ₁, λ₂ and ambition are ignored
 * @param {{sparse?: boolean, shaped?: boolean, power?: boolean}} [opts]  sparse: the adaptive-ridge ambition
 *     (fitSparse, sparseLambdas, sparseMaxAmbition) instead of the quadratic one; shaped: shaped bends (fitShaped,
 *     shapedLambdas, shapedMaxAmbition; spec.shape holds the costs); power: Σ|D_dβ'|^q (fitPower, powerLambdas,
 *     powerMaxAmbition; spec.q holds q, spec.flat the share on first differences)
 * @returns {{ambition: number, capped: boolean, unreachable: boolean, top: number, fit: object, fits: number,
 *     ms: number}}  fit: at a* (from fitIterated or fitSparse); unreachable: even the smallest ambition scanned
 *     leaves the region, so no a* exists and ambition and fit are that smallest one's; top: the cap (maxAmbition
 *     or sparseMaxAmbition or shapedMaxAmbition); fits: phase-2 fits computed
 */
export function leavingAmbition(spec, opts = {}) {
    const start = performance.now();
    const G = spec.c.length;
    const flat = spec.flat ?? 0;
    const sparse = opts.sparse ?? false;
    const shaped = opts.shaped ?? false;
    const power = opts.power ?? false;
    const screenCache = new Map();
    const at = logA => {
        const ambition = 10 ** logA;
        if (power)
            return fitPower({ ...spec, ...powerLambdas({ ambition, q: spec.q, flat, G, nseg: spec.nseg }) });
        if (shaped)
            return fitShaped({ ...spec, ...shapedLambdas({ ambition, G }) }, { screenCache });
        return sparse
            ? fitSparse({ ...spec, ...sparseLambdas({ ambition, flat, G }) })
            : fitIterated({ ...spec, ...ambitionLambdas({ ambition, flat, G, nseg: spec.nseg }) });
    };
    const leaves = fit => fit.maxRatio > 1 + LEAVE_TOL;
    let top = Math.log10(maxAmbition(spec));
    if (power)
        top = Math.log10(powerMaxAmbition(spec));
    else if (shaped)
        top = Math.log10(shapedMaxAmbition(spec));
    else if (sparse)
        top = Math.log10(sparseMaxAmbition(spec));
    let fits = 0;
    let lo = null;
    let loFit = null;
    let hi = null;
    let hiFit = null;
    const steps = Math.round(SCAN_DECADES / SCAN_STEP);
    for (let i = steps; i >= 0; i--) {
        const logA = top - i * SCAN_STEP;
        const fit = at(logA);
        fits++;
        if (leaves(fit)) {
            hi = logA;
            hiFit = fit;
            break;
        }
        lo = logA;
        loFit = fit;
    }
    const result = (ambition, capped, fit) => ({ ambition, capped, unreachable: false, top: 10 ** top, fit, fits,
        ms: performance.now() - start });
    if (hi === null)
        return result(10 ** top, true, loFit);
    if (lo === null) {
        // Leaves already at the smallest ambition scanned: no fit in this basis is inside the region (Aₙ's shortest
        // intervals can ask for more than the basis can follow; a band's center is inside by construction).
        return { ...result(10 ** hi, false, hiFit), unreachable: true };
    }
    while (hi - lo > BISECT_TOL) {
        const mid = (lo + hi) / 2;
        const fit = at(mid);
        fits++;
        if (leaves(fit)) {
            hi = mid;
        }
        else {
            lo = mid;
            loFit = fit;
        }
    }
    return result(10 ** lo, false, loFit);
}

/**
 * The top of the sparse ambition's range, and the polynomial fit there. With every difference pinned, the fit is a
 * polynomial (a line for D₂, a constant once D₁ is penalized) whose region terms are S_poly. Above
 * a = S_poly/G, one free difference costs λ_d = a·G > S_poly, more than all the region terms of the polynomial,
 * so the L0 optimum is the polynomial itself; SPARSE_TOP_FACTOR times that leaves room for the share θ. Not a
 * numerical limit (the weights are capped in fitSparse), but where the knob stops having an effect.
 *
 * @param {Phase2Spec & {flat?: number}} spec  its λ₁, λ₂ are ignored
 * @returns {number}
 */
export function sparseMaxAmbition(spec) {
    const flat = spec.flat ?? 0;
    const nb = spec.nseg + (spec.deg ?? 3);
    const G = spec.c.length;
    const pin = MAX_PENALTY_RATIO * spec.epsilon * G / nb; // λ_d·w_r of a pinned difference (fitSparse)
    const poly = fitIterated({
        ...spec, w1: undefined, w2: undefined, lambda1: flat > 0 ? pin : 0, lambda2: flat < 1 ? pin : 0,
    });
    const { s } = prepare(spec);
    const share = Math.min(...[flat, 1 - flat].filter(v => v > 0));
    return Math.max(SPARSE_TOP_FACTOR * regionObjective(spec, s, poly.g) / (G * share), SPARSE_MIN_TOP);
}

/** d-th differences of v. */
function differences(v, d) {
    let out = Float64Array.from(v);
    for (let k = 0; k < d; k++)
        out = out.subarray(1).map((x, i) => x - out[i]);
    return out;
}

/**
 * Sparse ambition (PLAN §1.4): the adaptive ridge of Frommlet & Nuel (2016), which approximates an L0 penalty,
 * λ_d times the number of nonzero d-th differences of β, by reweighting the squared differences,
 * w_r = 1/(θ_r² + δ_d²) with θ = D_dβ' at the previous fit. A difference much larger than δ_d then costs about λ_d
 * whatever its size, and a smaller one is driven to zero, so D₂ gives straight pieces joined by short bends and D₁
 * flat pieces joined by steps. Each reweighting is one fitIterated (the region terms are unchanged), started from
 * the fit without ambition (about the center) and then from the previous fit's sides.
 *
 * The support (which differences exceed δ_d) settles within a few reweightings, but the nonzero differences then
 * drift toward their limit only linearly (about 1e-5 of h per step after 100 steps on a sine). So the reweighting
 * stops when the support has not changed for SUPPORT_STABLE steps, and a final refit pins the differences outside
 * the support to zero and leaves those in it free: the exact minimizer of the region terms plus λ·(joins) for
 * that support. The L0 problem is not convex, so the support is a local optimum found from that start, not a
 * proven one.
 *
 * Pinning weights are capped where λ_d·w_r reaches MAX_PENALTY_RATIO times the center pull per coefficient (as
 * maxAmbition), so a pinned difference holds no harder than the quadratic penalty at its cap.
 *
 * @param {Phase2Spec} spec  lambda1, lambda2: cost per nonzero first or second difference (unitless; see
 *     sparseLambdas); w1 and w2 are ignored
 * @param {{maxReweights?: number, prunePasses?: number}} [opts]  prunePasses: backward-elimination passes
 *     (MAX_PRUNE_PASSES; 0 skips it, kept for comparison)
 * @returns {Phase2Fit & {ed: number, iterations: number, converged: boolean, side: Int8Array, reweights: number,
 *     settled: boolean, pruned: number, pieces: SparsePieces}}  iterations: Newton solves in all; converged: the
 *     final refit's iteration converged; settled: the support stopped changing before maxReweights; pruned: joins
 *     dropped by backward elimination
 */
export function fitSparse(spec, opts = {}) {
    const orders = [[1, spec.lambda1], [2, spec.lambda2]].filter(([, lambda]) => lambda > 0);
    return adaptiveRidge(spec, opts, orders, countModel(orders));
}

/** fitSparse's costs: λ_d per free d-th difference, whatever its size or place. */
function countModel(orders) {
    return {
        weights: (i, t, delta, cap) => t.map(v => Math.min(1 / (v * v + delta * delta), cap)),
        pins: (i, m, cap) => Float64Array.from(m, v => (v ? 0 : cap)),
        cost: (scaled, support) => orders.reduce((sum, [, lambda], i) => sum + lambda
            * support[i].reduce((a, v) => a + v, 0), 0),
    };
}

/**
 * The adaptive-ridge search shared by fitSparse and fitShaped: reweight until the support settles, refit with the
 * support pinned, then backward elimination of joins. The model gives the reweighting's weights for order i from
 * its differences t of β' (weights(i, t, delta, cap)), the pinned refit's weights for a support mask
 * (pins(i, mask, cap)), and the ambition part of the objective for a support at the refit's β' (cost).
 *
 * opts.signs (fitShaped's shape search) asks first or second differences to keep a sign: signs[d][r] = ±1 for a
 * difference that must not have the opposite sign, 0 where free. As in Bollaerts, Eilers & Van Mechelen (2006),
 * a difference found on the wrong side gets the pinning weight, and stays in that active set; the reweighting adds
 * to the set as it goes, and the refit repeats until no new difference violates its sign. Without signs nothing
 * changes. The result's objective is the L0 objective of the final refit.
 */
function adaptiveRidge(spec, opts, orders, model) {
    const start = performance.now();
    const maxReweights = opts.maxReweights ?? MAX_REWEIGHTS;
    const prunePasses = opts.prunePasses ?? MAX_PRUNE_PASSES;
    const { s, nb } = prepare(spec);
    const G = spec.c.length;
    const caps = orders.map(([, lambda]) => MAX_PENALTY_RATIO * spec.epsilon * G / (nb * lambda));
    // The cap on w makes the reweighting behave as if δ were at least 1/√cap, so that is also where a difference
    // counts as zero; with δ_d alone, differences the capped weights could not push below δ_d counted as free.
    const deltas = orders.map(([d], i) => Math.max(sparseDelta(d, spec.nseg), 1 / Math.sqrt(caps[i])));
    const base = { ...spec, w1: undefined, w2: undefined };
    const supportOf = beta => orders.map(([d], i) => differences(beta.map(v => v / s), d)
        .map(t => (Math.abs(t) > deltas[i] ? 1 : 0)));
    const signs = opts.signs ?? null;
    const signOrders = signs ? [1, 2].filter(d => signs[d]) : [];
    // the sign constraints' pinning weight, relative to the (first) order's λ, which also scales a D₁ constraint
    const signCap = caps[0];
    const newActive = () => Object.fromEntries(signOrders.map(d => [d, new Uint8Array(signs[d].length)]));
    const violate = (scaled, active) => signViolations(scaled, signs, active, spec.nseg);
    /** The active set's pinning weights, added to weights for the orders' differences. */
    const constrain = (weights, active) => {
        const out = { ...weights };
        if (active[1]) {
            out.lambda1 = orders[0][1];
            out.w1 = Float64Array.from(active[1], v => (v ? signCap : 0));
        }
        if (active[2])
            out.w2 = weights.w2.map((w, r) => (active[2][r] ? Math.max(w, signCap) : w));
        return out;
    };
    let active = newActive();
    let fit = fitIterated({ ...base, lambda1: 0, lambda2: 0 });
    let iterations = fit.iterations;
    let support = supportOf(fit.beta);
    let stable = 0;
    let reweights = 0;
    while (orders.length > 0 && stable < SUPPORT_STABLE && reweights < maxReweights) {
        reweights++;
        const weights = {};
        const scaled = fit.beta.map(v => v / s);
        orders.forEach(([d], i) => {
            weights[`w${d}`] = model.weights(i, differences(scaled, d), deltas[i], caps[i], scaled, reweights);
        });
        const grew = signs ? violate(scaled, active) : false;
        fit = fitIterated({ ...base, ...(signs ? constrain(weights, active) : weights) }, { side: fit.side });
        iterations += fit.iterations;
        const next = supportOf(fit.beta);
        const same = next.every((m, i) => m.every((v, r) => v === support[i][r])) && !grew;
        stable = same ? stable + 1 : 0;
        support = next;
    }
    const settled = stable >= SUPPORT_STABLE || orders.length === 0;
    let pruned = 0;
    let objective = NaN;
    if (orders.length > 0) {
        // The pinned refit for a support, and its L0 objective: region terms plus the model's ambition cost.
        const refit = (m, side) => {
            const pins = {};
            orders.forEach(([d], i) => {
                pins[`w${d}`] = model.pins(i, m[i], caps[i]);
            });
            let f = fitIterated({ ...base, ...(signs ? constrain(pins, active) : pins) }, { side });
            iterations += f.iterations;
            for (let k = 0; signs && k < MAX_SIGN_REFITS && violate(f.beta.map(v => v / s), active); k++) {
                f = fitIterated({ ...base, ...constrain(pins, active) }, { side: f.side });
                iterations += f.iterations;
            }
            const l0 = regionObjective(spec, s, f.g) + model.cost(f.beta.map(v => v / s), m);
            return { fit: f, l0 };
        };
        let best = refit(support, fit.side);
        // Backward elimination: the reweighting can settle on a support with a join the region does not need
        // (a ramp: four straight pieces where three fit). Drop each join in turn and keep the drop when it lowers
        // the L0 objective; repeat while a pass drops one, at most prunePasses times.
        for (let pass = 0; pass < prunePasses; pass++) {
            let dropped = false;
            orders.forEach((_, i) => {
                for (const run of supportRuns(support[i])) {
                    const trial = support.map(m => Uint8Array.from(m));
                    trial[i].fill(0, run.from, run.to + 1);
                    const t = refit(trial, best.fit.side);
                    if (t.l0 < best.l0) {
                        best = t;
                        support = trial;
                        pruned++;
                        dropped = true;
                    }
                }
            });
            if (!dropped)
                break;
        }
        fit = best.fit;
        objective = best.l0;
    }
    return {
        ...fit, iterations, reweights, settled, pruned, support, l0: objective,
        pieces: sparsePieces(support, orders.map(([d]) => d), spec),
        ms: performance.now() - start,
    };
}

/**
 * Add the differences of β' on the wrong side of their required sign to the active set (signs and active by order
 * d = 1, 2); true when any was added. A difference within δ_d of zero is on neither side.
 */
function signViolations(scaled, signs, active, nseg) {
    let added = false;
    for (const d of [1, 2]) {
        if (!signs[d])
            continue;
        const t = differences(scaled, d);
        const tol = sparseDelta(d, nseg);
        for (let r = 0; r < t.length; r++) {
            if (signs[d][r] !== 0 && !active[d][r] && t[r] * signs[d][r] < -tol) {
                active[d][r] = 1;
                added = true;
            }
        }
    }
    return added;
}

/**
 * Shaped bends: the sparse ambition toward straight (fitSparse with D₂ only), with each bend's cost depending on its
 * shape. The plain count charges λ per free second difference, so the cheapest way to turn is in one difference:
 * the top of a sine becomes a sharp corner. Here a bend, a run of free second differences t (of β', units of the
 * median half-width), costs λ times
 *
 *     m·(toFlat + (1 − toFlat)·ρ)  +  extremum·[slopes on its two sides of opposite sign]  +  inflection·(changes
 *     of the sign of t at its differences, counted from the previous bend)  +  q·Σ t²
 *
 * with m its differences and ρ = min/max of |slope| on its two sides: a bend that levels off (or starts from
 * level) costs toFlat per difference, one between equally steep pieces 1. The bend is judged as a whole, by the
 * pieces it joins, so a symmetric peak costs the same read from either end. The quadratic term spreads a turn:
 * a turn T over m differences costs about λ(c·m + q·T²/m), least at a constant curvature √(c/q), so a bend's
 * width grows with its turn while a small turn stays in one difference. q = (L·nseg)⁴/4 puts that curvature
 * where, without extremum or toFlat costs (c = 1), the bend leaves its tangent by one median half-width over a
 * length L of the x range (shape.length), whatever the segments. An extremum on a level piece (rise, flat, fall)
 * counts at the bend that completes it.
 *
 * The search is fitSparse's (reweighting, pinned refit, backward elimination), with weights
 * w_r = c_r/(t_r² + δ²) + q, where c_r spreads its bend's cost over the bend's differences (shapes from the
 * previous fit; the first reweighting uses c = 1), and the pinned refit leaves free differences under q.
 * Backward elimination compares supports by the exact cost above at their refits. With toFlat = 1 and the other
 * costs 0 this is fitSparse with D₂ (test).
 *
 * The refit sets the signs of the free differences from the region alone, so the search above cannot trade an
 * extremum or inflection away. The shape search can: it reruns the search under sign constraints on the
 * differences (shapeCandidates: a stretch kept monotone, or bending one way) and keeps a rerun that lowers the
 * objective. A candidate is screened first by its fit with no ambition: if that leaves the region, or its region
 * terms are already no lower than the objective, the rerun cannot win and is skipped. Greedy, first improvement:
 * a local optimum, like the rest.
 *
 * @param {Phase2Spec & {shape?: {toFlat?: number, extremum?: number, inflection?: number, length?: number}}} spec
 *     lambda2 > 0: cost of one free second difference of a plain bend (shapedLambdas); lambda1, w1, w2 ignored;
 *     shape: the costs (SHAPE_DEFAULTS), toFlat in (0, 1], extremum and inflection ≥ 0, length ≥ 0
 * @param {{maxReweights?: number, prunePasses?: number, shapePasses?: number, screenCache?: Map}} [opts]
 *     shapePasses: accepted removals at most (MAX_SHAPE_PASSES; 0 skips the shape search); screenCache: the
 *     screen's results by candidate, to share between fits of the same spec at other ambitions
 * @returns {ReturnType<typeof fitSparse> & {bends: Bend[], extremes: number, inflections: number,
 *     shapeTried: number, shapeAccepted: number}}  shapeTried: candidates searched (past the screen); shapeAccepted:
 *     kept
 */
export function fitShaped(spec, opts = {}) {
    if (!(spec.lambda2 > 0))
        throw new RangeError('fitShaped: lambda2 > 0 required');
    const start = performance.now();
    const shape = shapeOptions(spec.shape);
    const q = (shape.length * spec.nseg) ** 4 / 4;
    const lambda = spec.lambda2;
    const model = {
        weights: (i, t, delta, cap, scaled, reweights) => {
            const c = reweights === 1 ? null : differenceCosts(bendShapes(scaled, t, delta, shape), t.length);
            return t.map((v, r) => Math.min((c ? c[r] : 1) / (v * v + delta * delta) + q, cap));
        },
        pins: (i, m, cap) => Float64Array.from(m, v => (v ? q : cap)),
        cost: (scaled, support) => {
            const t = differences(scaled, 2);
            let sum = 0;
            for (const b of bendShapes(scaled, t, null, shape, support[0]))
                sum += b.cost;
            for (let r = 0; r < t.length; r++) {
                if (support[0][r])
                    sum += q * t[r] * t[r];
            }
            return lambda * sum;
        },
    };
    const { s, nb } = prepare(spec);
    const search = signs => adaptiveRidge({ ...spec, lambda1: 0 }, { ...opts, signs }, [[2, lambda]], model);
    let fit = search(null);
    let signs = null;
    let tried = 0;
    let accepted = 0;
    if (shape.extremum > 0 || shape.inflection > 0) {
        // Shape search: try each way of removing extremes (making a stretch monotone) or inflections (keeping the
        // bending direction over a stretch) as sign constraints on the coefficients' differences, rerun the search
        // under them, and keep the first that lowers the objective; repeat. A candidate whose fit with no ambition
        // already leaves the region is skipped: the region forces that extremum or inflection.
        // The screen does not depend on the ambition (λ only scales its pinning weights), so a* reuses it. Its fit
        // has about the least region terms the constraints allow, a lower bound on a candidate's objective: one
        // whose bound is no better than the current objective is skipped without a search.
        const cache = opts.screenCache ?? new Map();
        const screen = cand => {
            if (!cache.has(cand.key)) {
                const f = constrainedCenterFit(spec, cand, lambda, s, nb);
                cache.set(cand.key, f.maxRatio <= 1 + LEAVE_TOL ? regionObjective(spec, s, f.g) : Infinity);
            }
            return cache.get(cand.key) < fit.l0;
        };
        for (let pass = 0; pass < (opts.shapePasses ?? MAX_SHAPE_PASSES); pass++) {
            const scaledNow = fit.beta.map(v => v / s);
            let improved = false;
            for (const cand of shapeCandidates(scaledNow, fit.support[0], shape, signs)) {
                if (!screen(cand))
                    continue;
                tried++;
                const f = search(cand);
                if (f.l0 < fit.l0 - SHAPE_GAIN * Math.abs(fit.l0)) {
                    fit = f;
                    signs = cand;
                    accepted++;
                    improved = true;
                    break;
                }
            }
            if (!improved)
                break;
        }
    }
    const scaled = fit.beta.map(v => v / s);
    const bends = bendShapes(scaled, differences(scaled, 2), null, shape, fit.support[0]);
    const step = (spec.xr - spec.xl) / spec.nseg;
    const deg = spec.deg ?? 3;
    for (const b of bends)
        b.x = spec.xl + ((b.from + b.to + 2) / 2 - (deg - 1) / 2) * step;
    return {
        ...fit, bends, shapeTried: tried, shapeAccepted: accepted, ms: performance.now() - start,
        extremes: bends.reduce((a, b) => a + (b.extremum ? 1 : 0), 0),
        inflections: bends.reduce((a, b) => a + b.inflections, 0),
    };
}

/**
 * The fit with no ambition (a floor on the second differences only) under sign constraints: whether the region
 * admits them at all. Iterates the active set as adaptiveRidge does, and stops once the fit is outside the region
 * by more than LEAVE_TOL (more constraints keep it out), so a rejected candidate's fit is not converged.
 */
function constrainedCenterFit(spec, signs, lambda, s, nb) {
    const cap = MAX_PENALTY_RATIO * spec.epsilon * spec.c.length / (nb * lambda);
    const active = { 1: signs[1] ? new Uint8Array(signs[1].length) : null,
        2: signs[2] ? new Uint8Array(signs[2].length) : null };
    const weights = () => ({
        lambda1: active[1] ? lambda : 0, w1: active[1] ? Float64Array.from(active[1], v => (v ? cap : 0)) : undefined,
        lambda2: active[2] ? lambda : 0, w2: active[2] ? Float64Array.from(active[2], v => (v ? cap : 0)) : undefined,
    });
    let fit = fitIterated({ ...spec, ...weights() });
    for (let k = 0; k < MAX_SIGN_REFITS && signViolations(fit.beta.map(v => v / s), signs, active, spec.nseg); k++) {
        fit = fitIterated({ ...spec, ...weights() }, { side: fit.side });
        // the active set only grows, so a fit already outside stays outside: stop there
        if (fit.maxRatio > 1 + LEAVE_TOL)
            break;
    }
    return fit;
}

/**
 * Sign constraints that would remove extremes or inflections of β' (support: its free second differences), each
 * added to the constraints already in force (current). The signs of the first differences (slopes; a level
 * straight piece takes its neighbors' sign) or of the free second differences (bending direction) form segments
 * of one sign; merging three neighboring segments into the first one's sign removes two sign changes (a maximum
 * and a minimum, or two inflections), and merging the first two or the last two removes one at an end.
 *
 * @returns {{1?: Int8Array, 2?: Int8Array}[]}
 */
function shapeCandidates(scaled, support, shape, current) {
    const u = differences(scaled, 1);
    const t = differences(scaled, 2);
    const candidates = [];
    const add = (d, segs) => {
        const K = segs.length;
        const merges = [];
        if (K >= 2) {
            // the first segment joins the second, or the last the one before (both ways when there are two)
            merges.push([0, 1, segs[1].sign], [K - 2, K - 1, segs[K - 2].sign]);
        }
        for (let k = 1; k + 1 < K; k++)
            merges.push([k - 1, k + 1, segs[k - 1].sign]);
        for (const [a, b, sign] of merges) {
            const cand = { 1: current?.[1] ? Int8Array.from(current[1]) : null,
                2: current?.[2] ? Int8Array.from(current[2]) : null };
            cand[d] ??= new Int8Array(d === 1 ? u.length : t.length);
            cand[d].fill(sign, segs[a].from, segs[b].to + 1);
            cand.key = `${current?.key ?? ''}|${d}:${segs[a].from}-${segs[b].to}:${sign}`;
            candidates.push(cand);
        }
    };
    if (shape.extremum > 0) {
        const runs = supportRuns(support);
        const sign = new Int8Array(u.length);
        for (let r = 0; r < u.length; r++)
            sign[r] = Math.abs(u[r]) > SPARSE_DELTA ? Math.sign(u[r]) : 0;
        // straight pieces that rise less than LEVEL_RISE over their length are level (as bendShapes)
        let from = 0;
        for (let k = 0; k <= runs.length; k++) {
            const to = k < runs.length ? runs[k].from : u.length - 1;
            if (Math.abs(u[to]) * Math.max(to - from + 1, 1) < LEVEL_RISE)
                sign.fill(0, from, to + 1);
            if (k < runs.length)
                from = runs[k].to + 1;
        }
        add(1, signSegments(sign));
    }
    if (shape.inflection > 0)
        add(2, signSegments(Int8Array.from(t, (v, r) => (support[r] ? Math.sign(v) : 0))));
    return candidates;
}

/** Maximal runs of one sign in a sequence of −1, 0, +1, zeros joining the run before them (the first, at the start). */
function signSegments(sign) {
    const segs = [];
    for (let r = 0; r < sign.length; r++) {
        if (sign[r] === 0)
            continue;
        if (segs.length === 0)
            segs.push({ from: 0, to: r, sign: sign[r] });
        else if (sign[r] === segs[segs.length - 1].sign) {
            segs[segs.length - 1].to = r;
        }
        else {
            segs[segs.length - 1].to = r - 1;
            segs.push({ from: r, to: r, sign: sign[r] });
        }
    }
    if (segs.length > 0)
        segs[segs.length - 1].to = sign.length - 1;
    return segs;
}

function shapeOptions(shape = {}) {
    const o = { ...SHAPE_DEFAULTS, ...shape };
    if (!(o.toFlat > 0 && o.toFlat <= 1) || !(o.extremum >= 0) || !(o.inflection >= 0) || !(o.length >= 0))
        throw new RangeError('fitShaped: toFlat in (0, 1], extremum, inflection and length ≥ 0 required');
    return o;
}

/**
 * @typedef {object} Bend  a run of free second differences, by index (inclusive), and what it costs (in units of
 *     λ, without the quadratic term)
 * @property {number} from
 * @property {number} to
 * @property {number} before   slope of β' (first difference) on its left
 * @property {number} after    on its right
 * @property {boolean} extremum  the straight pieces on its two sides slope in opposite directions (the last
 *     piece that was not level, before a level one); a piece is level when it rises less than LEVEL_RISE median
 *     half-widths over its length
 * @property {number} inflections  changes of the sign of the second differences at its differences
 * @property {number} cost
 * @property {number} [x]      its middle (fitShaped's result)
 */

/**
 * The bends of β' (scaled coefficients) with second differences t: the runs of the support (|t| > delta, or the
 * mask), their shapes and costs (fitShaped).
 *
 * @returns {Bend[]}
 */
function bendShapes(scaled, t, delta, shape, mask) {
    const m = mask ?? Uint8Array.from(t, v => (Math.abs(v) > delta ? 1 : 0));
    const u = differences(scaled, 1);
    const runs = supportRuns(m);
    // A straight piece is level when it rises less than LEVEL_RISE over its whole length: a slight tilt the region
    // allows (5e-4 median half-widths per segment after a bend that levels off) is no extremum.
    const level = (slope, segments) => Math.abs(slope) * Math.max(segments, 1) < LEVEL_RISE;
    const bends = [];
    let lastSign = 0; // of the last straight piece that was not level
    let lastBend = 0; // sign of the last free second difference
    runs.forEach(({ from, to }, k) => {
        // first differences u[a..b] are equal along a straight piece: before this bend from the previous one's end
        const beforeLength = from - (k > 0 ? runs[k - 1].to + 1 : 0) + 1;
        const afterLength = (k + 1 < runs.length ? runs[k + 1].from : u.length - 1) - (to + 1) + 1;
        const before = u[from];
        const after = u[to + 1];
        if (!level(before, beforeLength))
            lastSign = Math.sign(before);
        const steepAfter = !level(after, afterLength);
        const extremum = steepAfter && lastSign !== 0 && Math.sign(after) !== lastSign;
        if (steepAfter)
            lastSign = Math.sign(after);
        let inflections = 0;
        for (let r = from; r <= to; r++) {
            const sg = Math.sign(t[r]);
            if (sg !== 0 && lastBend !== 0 && sg !== lastBend)
                inflections++;
            if (sg !== 0)
                lastBend = sg;
        }
        const steep = Math.max(Math.abs(before), Math.abs(after));
        const rho = steep > 0 ? Math.min(Math.abs(before), Math.abs(after)) / steep : 1;
        const cost = (to - from + 1) * (shape.toFlat + (1 - shape.toFlat) * rho)
            + (extremum ? shape.extremum : 0) + shape.inflection * inflections;
        bends.push({ from, to, before, after, extremum, inflections, cost });
    });
    return bends;
}

/** Each bend's cost spread evenly over its differences (1 outside the bends), for the reweighting. */
function differenceCosts(bends, n) {
    const c = new Float64Array(n).fill(1);
    for (const b of bends)
        c.fill(b.cost / (b.to - b.from + 1), b.from, b.to + 1);
    return c;
}

/**
 * λ₂ for shaped bends: a·G, as sparseLambdas toward straight (S/G is the region functional plus a per plain bend
 * difference).
 *
 * @param {{ambition: number, G: number}} opts
 * @returns {{lambda1: number, lambda2: number}}
 */
export function shapedLambdas({ ambition, G }) {
    return { lambda1: 0, lambda2: ambition * G };
}

/**
 * The top of the shaped ambition's range: sparseMaxAmbition toward straight, divided by toFlat, since the cheapest
 * free difference costs toFlat·λ.
 *
 * @param {Phase2Spec & {shape?: object}} spec
 * @returns {number}
 */
export function shapedMaxAmbition(spec) {
    return sparseMaxAmbition({ ...spec, flat: 0 }) / shapeOptions(spec.shape).toFlat;
}

/** Maximal runs of ones in a 0/1 mask: [{from, to}], inclusive. */
function supportRuns(m) {
    const runs = [];
    let r = 0;
    while (r < m.length) {
        if (!m[r]) {
            r++;
            continue;
        }
        const from = r;
        while (r < m.length && m[r])
            r++;
        runs.push({ from, to: r - 1 });
    }
    return runs;
}

/**
 * The region terms of S, κ·Σdist²/h² + ε·Σ((g − t)/h)², for g in units of y (s: the median half-width; t the
 * target, the center by default).
 */
function regionObjective(spec, s, g) {
    const target = spec.target ?? spec.c;
    let sum = 0;
    for (let j = 0; j < g.length; j++) {
        const h = spec.h[j];
        const out = Math.max(0, Math.abs(g[j] - spec.c[j]) - h);
        const dev = g[j] - target[j];
        sum += (spec.kappa * out * out + spec.epsilon * dev * dev) / (h * h);
    }
    return sum;
}

/**
 * λ₁ and λ₂ for the sparse ambition: a·θ·G and a·(1 − θ)·G, so S/G is the region functional plus a·θ per flat
 * join and a·(1 − θ) per bend, whatever the grid density. A join's count does not grow with the segments, so,
 * unlike ambitionLambdas, there is no nseg factor.
 *
 * @param {{ambition: number, flat?: number, G: number}} opts
 * @returns {{lambda1: number, lambda2: number}}
 */
export function sparseLambdas({ ambition, flat = 0, G }) {
    return { lambda1: ambition * flat * G, lambda2: ambition * (1 - flat) * G };
}

/**
 * λ₁ and λ₂ for the power penalty a·(θ·∫|g'|^q + (1 − θ)·∫|g''|^q): a d-th difference of β' over nseg segments on
 * [0, 1] approximates g⁽ᵈ⁾/nseg^d, so Σ|D_dβ'|^q ≈ nseg^(1−dq)·∫|g⁽ᵈ⁾|^q, and λ_d = a·share·G·nseg^(dq−1) makes
 * S/G the region functional plus that penalty. At q = 2 these are ambitionLambdas'. θ = 0 (the default) is the
 * power penalty on second differences alone, toward straight. For q < 1 the integral favors ever sharper bends (or
 * steps), so the discrete penalty's sharpest join (one difference) leaves a dependence on nseg; a* absorbs it.
 *
 * @param {{ambition: number, q: number, flat?: number, G: number, nseg: number}} opts  flat θ ∈ [0, 1], the share
 *     on first differences (toward flat)
 * @returns {{lambda1: number, lambda2: number}}
 */
export function powerLambdas({ ambition, q, flat = 0, G, nseg }) {
    return {
        lambda1: ambition * flat * G * nseg ** (q - 1),
        lambda2: ambition * (1 - flat) * G * nseg ** (2 * q - 1),
    };
}

/**
 * Power (bridge) ambition: λ₂·Σ|D₂β'|^q, 0 < q ≤ 2, toward straight, plus optionally λ₁·Σ|D₁β'|^q toward flat
 * (powerLambdas' share θ; λ₁ = 0 by default). q = 2 is the squared differences; q = 1 is ℓ₁ trend filtering (Kim,
 * Koh, Boyd & Gorinevsky 2009; Tibshirani 2014), straight pieces with kinks, and on first differences the fused
 * lasso's total variation, flat pieces with steps; q → 0 approaches the count of bends (or steps). In terms of the
 * turning's distribution along x, smaller q concentrates it into fewer bends (lower entropy).
 *
 * Solved by majorize-minimize (Hunter & Li 2005, the local quadratic approximation of Fan & Li 2001): |t|^q is
 * concave in t² for q ≤ 2, so it lies below its tangent in t², (q/2)(t₀² + δ²)^((q−2)/2)·t² + const, and each step
 * is one fitIterated with those weights on each penalized order. S with |t|^q smoothed to (t² + δ²)^(q/2) never
 * increases. For q ≥ 1 the objective is convex and this approaches its minimizer; for q < 1 it is not, and the
 * result is a local minimum from the start, the fit without ambition (as fitSparse). Weights are capped as in
 * fitSparse, per order.
 *
 * @param {Phase2Spec & {q: number}} spec  lambda1, lambda2 from powerLambdas; w1 and w2 are ignored
 * @param {{maxReweights?: number}} [opts]
 * @returns {Phase2Fit & {ed: number, iterations: number, converged: boolean, side: Int8Array, reweights: number,
 *     settled: boolean, pieces: SparsePieces, support: Uint8Array[]}}  settled: S stopped falling (POWER_TOL)
 *     before maxReweights; pieces: bends (or steps) of at least POWER_BEND, as fitSparse's joins, for each
 *     penalized order
 */
export function fitPower(spec, opts = {}) {
    const start = performance.now();
    const q = spec.q;
    if (!(q > 0 && q <= 2))
        throw new RangeError('fitPower: 0 < q ≤ 2 required');
    const maxReweights = opts.maxReweights ?? MAX_POWER_REWEIGHTS;
    const { s, nb } = prepare(spec);
    const G = spec.c.length;
    const orders = [[1, spec.lambda1 ?? 0], [2, spec.lambda2 ?? 0]].filter(([, lambda]) => lambda > 0);
    const base = { ...spec, w1: undefined, w2: undefined };
    let fit = fitIterated({ ...base, lambda1: 0, lambda2: 0 });
    let iterations = fit.iterations;
    let reweights = 0;
    let settled = orders.length === 0;
    const caps = orders.map(([, lambda]) => MAX_PENALTY_RATIO * spec.epsilon * G / (nb * lambda));
    const deltas = orders.map(([d]) => sparseDelta(d, spec.nseg));
    const penalty = (beta, size) => {
        const scaled = beta.map(v => v / s);
        return orders.reduce((sum, [d, lambda], i) => sum + lambda * differences(scaled, d)
            .reduce((a, t) => a + size(t, deltas[i]), 0), 0);
    };
    if (orders.length) {
        const smoothed = (beta, g) => regionObjective(spec, s, g) + penalty(beta, (t, delta) => (t * t + delta * delta)
            ** (q / 2));
        let previous = smoothed(fit.beta, fit.g);
        while (reweights < maxReweights) {
            reweights++;
            const scaled = fit.beta.map(v => v / s);
            const weights = {};
            orders.forEach(([d], i) => {
                weights[`w${d}`] = differences(scaled, d)
                    .map(v => Math.min((q / 2) * (v * v + deltas[i] * deltas[i]) ** ((q - 2) / 2), caps[i]));
            });
            fit = fitIterated({ ...base, ...weights }, { side: fit.side });
            iterations += fit.iterations;
            const current = smoothed(fit.beta, fit.g);
            const fell = previous - current;
            previous = current;
            if (fell <= POWER_TOL * Math.abs(current)) {
                settled = true;
                break;
            }
        }
    }
    const scaled = fit.beta.map(v => v / s);
    // a d-th difference t changes the fit by about |t|·nseg^(d−1) across the range (units of median h)
    const support = orders.map(([d]) => Uint8Array.from(differences(scaled, d),
        v => (Math.abs(v) * spec.nseg ** (d - 1) >= POWER_BEND ? 1 : 0)));
    const objective = regionObjective(spec, s, fit.g) + penalty(fit.beta, t => Math.abs(t) ** q);
    return {
        ...fit, objective, iterations, reweights, settled, support,
        pieces: sparsePieces(support, orders.map(([d]) => d), spec),
        ms: performance.now() - start,
    };
}

/**
 * The top of the power ambition's range, as maxAmbition's (which it equals at q = 2): the ambition at which the
 * reweighted penalty on the sharpest d-th difference of the fit without ambition, λ_d·(q/2)·|t_max|^(q−2), reaches
 * the cap on any difference's weight, MAX_PENALTY_RATIO times the center pull per coefficient, for the first
 * penalized order to get there. Smaller differences get larger weights (q < 2), so above it every difference of
 * that order is held at the cap and the fit is as straight (or flat) as the numbers allow.
 *
 * @param {Phase2Spec & {q: number, flat?: number}} spec  its λ₁, λ₂ are ignored; flat as powerLambdas
 * @returns {number}
 */
export function powerMaxAmbition(spec) {
    const q = spec.q;
    const flat = spec.flat ?? 0;
    const nb = spec.nseg + (spec.deg ?? 3);
    const G = spec.c.length;
    const pin = MAX_PENALTY_RATIO * spec.epsilon * G / nb;
    const free = fitIterated({ ...spec, w1: undefined, w2: undefined, lambda1: 0, lambda2: 0 });
    const { s } = prepare(spec);
    const scaled = free.beta.map(v => v / s);
    const unit = powerLambdas({ ambition: 1, q, flat, G, nseg: spec.nseg });
    const tops = [[1, unit.lambda1], [2, unit.lambda2]].filter(([, lambda]) => lambda > 0).map(([d, lambda]) => {
        const sharpest = differences(scaled, d).reduce((a, v) => Math.max(a, Math.abs(v)), 0);
        return pin / (lambda * (q / 2) * Math.max(sharpest, sparseDelta(d, spec.nseg)) ** (q - 2));
    });
    return Math.min(...tops);
}

/**
 * δ_d, the size below which a d-th difference of β' counts as zero: SPARSE_DELTA (in units of the median
 * half-width) for a step (d = 1); for d = 2 a bend θ changes the fit by about θ·nseg across [xl, xr], hence
 * SPARSE_DELTA/nseg.
 */
function sparseDelta(d, nseg) {
    return SPARSE_DELTA * nseg ** (1 - d);
}

/**
 * @typedef {object} SparsePieces
 * @property {{order: number, joins: {from: number, to: number, x: number}[], pieces: number, atEnds: number}[]}
 *     orders  for each penalized order d: the joins (maximal runs of nonzero d-th differences, by coefficient
 *     index, with x at their middle) and the pieces between them (flat for d = 1, straight for d = 2); a single
 *     piece is a constant or a line. A join that reaches the first or last coefficient (atEnds of them, 0–2)
 *     curves the fit's end rather than separating two pieces.
 */

/**
 * Pieces and joins from the support of a sparse fit (which d-th differences are free). Coefficient k of a cubic
 * B-spline basis peaks at xl + (k − 1)·(xr − xl)/nseg, so a run of differences r..r' (spanning coefficients
 * r..r' + d) is placed at the coefficient in its middle.
 *
 * @param {Uint8Array[]|Float64Array[]} support  per penalized order, 1 where the difference is free
 * @param {number[]} orders
 * @param {Phase2Spec} spec
 * @returns {SparsePieces}
 */
function sparsePieces(support, orders, spec) {
    const deg = spec.deg ?? 3;
    const step = (spec.xr - spec.xl) / spec.nseg;
    return {
        orders: orders.map((d, i) => {
            const m = support[i];
            const joins = supportRuns(m).map(({ from, to }) => {
                const mid = (from + to + d) / 2;
                return { from, to, x: spec.xl + (mid - (deg - 1) / 2) * step };
            });
            const atStart = joins.length > 0 && joins[0].from === 0 ? 1 : 0;
            const atEnd = joins.length > 0 && joins[joins.length - 1].to === m.length - 1 ? 1 : 0;
            return { order: d, joins, pieces: joins.length + 1 - atStart - atEnd, atEnds: atStart + atEnd };
        }),
    };
}
