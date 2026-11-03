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
const SPARSE_TOP_FACTOR = 10; // sparse ambition: top of the range, relative to where the polynomial is optimal
const SPARSE_MIN_TOP = 1e-12; // sparse ambition: top when the polynomial matches the center (S_poly = 0)
const SPARSE_DELTA = 1e-4; // adaptive ridge: a step smaller than this (units of median h) counts as zero

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
    const nb = nseg + deg;
    let widest = 0;
    for (let k = 0; k < rows.n; k++)
        widest = Math.max(widest, rows.width[k]);
    const W = Math.min(nb, Math.max(widest, Math.max(deg, 2) + 1));
    return { spec, s, cs, hs, rows, W, nb };
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
        sum += (kappa * out * out + epsilon * (g[j] - p.cs[j]) ** 2) / (p.hs[j] * p.hs[j]);
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
        target[j] = (epsilon * p.cs[j] + k * edge) / (epsilon + k);
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
            s += 2 * (kappa * r + epsilon * (gj - p.cs[j])) * gd[j] / (p.hs[j] * p.hs[j]);
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
    const { rows, nb, W, cs, hs } = p;
    const G = rows.n;
    const numCols = hard ? nb : nb + G;
    const invH2 = Float64Array.from(hs, v => 1 / (v * v));
    const { A, rhs } = rowCrossProducts(rows, invH2, cs, nb, W);
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
        offset += epsilon * cs[j] * cs[j] * invH2[j];

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
        grid: region.grid, rows: region.rows, c: region.center, h: region.half,
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
 * @param {{sparse?: boolean}} [opts]  sparse: the adaptive-ridge ambition (fitSparse, sparseLambdas,
 *     sparseMaxAmbition) instead of the quadratic one
 * @returns {{ambition: number, capped: boolean, unreachable: boolean, top: number, fit: object, fits: number,
 *     ms: number}}  fit: at a* (from fitIterated or fitSparse); unreachable: even the smallest ambition scanned
 *     leaves the region, so no a* exists and ambition and fit are that smallest one's; top: the cap (maxAmbition
 *     or sparseMaxAmbition); fits: phase-2 fits computed
 */
export function leavingAmbition(spec, opts = {}) {
    const start = performance.now();
    const G = spec.c.length;
    const flat = spec.flat ?? 0;
    const sparse = opts.sparse ?? false;
    const at = logA => {
        const ambition = 10 ** logA;
        return sparse
            ? fitSparse({ ...spec, ...sparseLambdas({ ambition, flat, G }) })
            : fitIterated({ ...spec, ...ambitionLambdas({ ambition, flat, G, nseg: spec.nseg }) });
    };
    const leaves = fit => fit.maxRatio > 1 + LEAVE_TOL;
    const top = Math.log10(sparse ? sparseMaxAmbition(spec) : maxAmbition(spec));
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
    const start = performance.now();
    const maxReweights = opts.maxReweights ?? MAX_REWEIGHTS;
    const prunePasses = opts.prunePasses ?? MAX_PRUNE_PASSES;
    const { s, nb } = prepare(spec);
    const G = spec.c.length;
    const orders = [[1, spec.lambda1], [2, spec.lambda2]].filter(([, lambda]) => lambda > 0);
    const caps = orders.map(([, lambda]) => MAX_PENALTY_RATIO * spec.epsilon * G / (nb * lambda));
    // The cap on w makes the reweighting behave as if δ were at least 1/√cap, so that is also where a difference
    // counts as zero; with δ_d alone, differences the capped weights could not push below δ_d counted as free.
    const deltas = orders.map(([d], i) => Math.max(sparseDelta(d, spec.nseg), 1 / Math.sqrt(caps[i])));
    const base = { ...spec, w1: undefined, w2: undefined };
    const supportOf = beta => orders.map(([d], i) => differences(beta.map(v => v / s), d)
        .map(t => (Math.abs(t) > deltas[i] ? 1 : 0)));
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
            const delta = deltas[i];
            weights[`w${d}`] = differences(scaled, d).map(t => Math.min(1 / (t * t + delta * delta), caps[i]));
        });
        fit = fitIterated({ ...base, ...weights }, { side: fit.side });
        iterations += fit.iterations;
        const next = supportOf(fit.beta);
        stable = next.every((m, i) => m.every((v, r) => v === support[i][r])) ? stable + 1 : 0;
        support = next;
    }
    const settled = stable >= SUPPORT_STABLE || orders.length === 0;
    let pruned = 0;
    if (orders.length > 0) {
        // The pinned refit for a support, and its L0 objective: region terms plus λ_d per free difference.
        const refit = (m, side) => {
            const pins = {};
            orders.forEach(([d], i) => {
                pins[`w${d}`] = Float64Array.from(m[i], v => (v ? 0 : caps[i]));
            });
            const f = fitIterated({ ...base, ...pins }, { side });
            iterations += f.iterations;
            let l0 = regionObjective(spec, s, f.g);
            orders.forEach(([, lambda], i) => {
                l0 += lambda * m[i].reduce((a, v) => a + v, 0);
            });
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
    }
    return {
        ...fit, iterations, reweights, settled, pruned,
        pieces: sparsePieces(support, orders.map(([d]) => d), spec),
        ms: performance.now() - start,
    };
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

/** The region terms of S, κ·Σdist²/h² + ε·Σ((g − c)/h)², for g in units of y (s: the median half-width). */
function regionObjective(spec, s, g) {
    let sum = 0;
    for (let j = 0; j < g.length; j++) {
        const h = spec.h[j];
        const dev = g[j] - spec.c[j];
        const out = Math.max(0, Math.abs(dev) - h);
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
