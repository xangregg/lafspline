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
// ε > 0 is required: it makes S strictly convex (the grid is dense in the basis) and the iterated system positive
// definite.

import { bsplineBasis, addDifferencePenalty, choleskyBanded, solveCholesky, bandedInverse, weightedCrossProducts }
    from './pspline.js';

const MAX_ITERATIONS = 200; // iterated weights: cap on Newton solves
const CHECK_TOL = 1e-9; // a grid point counts as outside when |g − c|/h > 1 + CHECK_TOL
export const DEFAULT_KAPPA = 1e4; // weight on the squared distance outside the region, per grid point
export const DEFAULT_EPSILON = 1e-2; // weight on the squared distance from the center, per grid point
// Largest penalty relative to the center pull per coefficient, λ_d·nb/(ε·G). Beyond about 1e12 the Newton system
// loses too many digits: a straight line inside the region came back outside it, without a Cholesky failure.
const MAX_PENALTY_RATIO = 1e10;
export const LEAVE_TOL = 0.01; // the fit has left the region when its largest |g − c|/h exceeds 1 + LEAVE_TOL
const SCAN_DECADES = 16; // a* search: ambitions from the cap down this many decades
const SCAN_STEP = 1; // decades between scanned ambitions
const BISECT_TOL = 0.02; // decades: a* search stops when the bracket is this narrow

/**
 * @typedef {object} Phase2Spec
 * @property {Float64Array} grid    check grid, within [xl, xr]
 * @property {Float64Array} c       region center at the grid
 * @property {Float64Array} h       region half-width at the grid, > 0
 * @property {number} xl            knot range of the phase-2 basis
 * @property {number} xr
 * @property {number} nseg          segments of the phase-2 basis
 * @property {number} [deg]         degree (3)
 * @property {number} kappa         weight on the squared distance outside the region (κ)
 * @property {number} epsilon       weight on the squared distance from the center (ε > 0)
 * @property {number} lambda1       weight on ‖D₁β‖²/s² (toward flat)
 * @property {number} lambda2       weight on ‖D₂β‖²/s² (toward straight)
 */

/**
 * @typedef {object} Phase2Fit
 * @property {Float64Array} beta     coefficients (units of y)
 * @property {Float64Array} g        fit at the grid (units of y)
 * @property {number} objective      S(β) (unitless)
 * @property {number} maxRatio       max_j |g_j − c_j| / h_j
 * @property {number} outside        grid points with ratio > 1
 * @property {number} ms             wall-clock time
 */

function prepare(spec) {
    const { grid, c, h, xl, xr, nseg } = spec;
    const deg = spec.deg ?? 3;
    if (!(spec.epsilon > 0))
        throw new RangeError('phase 2: epsilon > 0 required');
    if (c.length !== grid.length || h.length !== grid.length)
        throw new RangeError('phase 2: grid, c and h differ in length');
    const sorted = Float64Array.from(h).sort();
    const s = sorted[sorted.length >> 1];
    if (!(sorted[0] > 0))
        throw new RangeError('phase 2: half-widths must be positive');
    const cs = Float64Array.from(c, v => v / s);
    const hs = Float64Array.from(h, v => v / s);
    const basis = bsplineBasis(grid, xl, xr, nseg, deg);
    const W = Math.max(deg, 2) + 1;
    return { spec, s, cs, hs, basis, W, nb: basis.nb };
}

function addAmbition(A, p) {
    const { lambda1, lambda2 } = p.spec;
    if (lambda1 > 0)
        addDifferencePenalty(A, p.nb, p.W, 1, lambda1);
    if (lambda2 > 0)
        addDifferencePenalty(A, p.nb, p.W, 2, lambda2);
}

function gridValues(basis, beta) {
    const bw = basis.deg + 1;
    const g = new Float64Array(basis.n);
    for (let j = 0; j < basis.n; j++) {
        let v = 0;
        for (let r = 0; r < bw; r++)
            v += basis.values[j * bw + r] * beta[basis.first[j] + r];
        g[j] = v;
    }
    return g;
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
    const gs = gridValues(p.basis, betaScaled);
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
    return objective(p, scaled, gridValues(p.basis, scaled));
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
 * @param {{maxIterations?: number, lineSearch?: boolean}} [opts]  lineSearch: false takes every full step (the
 *     undamped iteration, kept for comparison; it can cycle)
 * @returns {Phase2Fit & {ed: number, iterations: number, converged: boolean, damped: number}}  ed: effective
 *     dimension; iterations: Newton solves; damped: steps shortened by the line search
 */
export function fitIterated(spec, opts = {}) {
    const start = performance.now();
    const p = prepare(spec);
    const maxIterations = opts.maxIterations ?? MAX_ITERATIONS;
    const useLineSearch = opts.lineSearch ?? true;
    const G = p.basis.n;
    const P = new Float64Array(p.nb * p.W);
    addAmbition(P, p);
    let side = new Int8Array(G); // +1 above u, −1 below l, 0 inside; all inside to start
    let beta = null;
    let g = null;
    let iterations = 0;
    let damped = 0;
    let converged = false;
    while (iterations < maxIterations) {
        iterations++;
        const { A, rhs } = newtonSystem(p, side);
        const betaN = solveCholesky(choleskyBanded(A, p.nb, p.W), p.nb, p.W, rhs);
        const gN = gridValues(p.basis, betaN);
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
    const ed = effectiveDimension(p, sidesOf(g, p), P);
    return { ...finish(p, beta, start), ed, iterations, converged, damped };
}

/**
 * The Newton system for fixed sides: (BᵀVB + P)β' = BᵀVt, with weight v_j = (ε + κ_j)/h'_j² and target
 * t_j = (ε c'_j + κ_j e_j)/(ε + κ_j), where κ_j = κ and e_j the violated edge for a point outside, κ_j = 0 inside.
 */
function newtonSystem(p, side) {
    const { kappa, epsilon } = p.spec;
    const G = p.basis.n;
    const weight = new Float64Array(G);
    const target = new Float64Array(G);
    for (let j = 0; j < G; j++) {
        const h2 = p.hs[j] * p.hs[j];
        const k = side[j] !== 0 ? kappa : 0;
        const edge = p.cs[j] + side[j] * p.hs[j];
        weight[j] = (epsilon + k) / h2;
        target[j] = (epsilon * p.cs[j] + k * edge) / (epsilon + k);
    }
    const { G: A, rhs } = weightedCrossProducts(p.basis, weight, target, p.W);
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
    const { basis, nb, W, cs, hs } = p;
    const G = basis.n;
    const numCols = hard ? nb : nb + G;
    const invH2 = Float64Array.from(hs, v => 1 / (v * v));
    const { G: A, rhs } = weightedCrossProducts(basis, invH2, cs, W);
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
    const bw = basis.deg + 1;
    const rowStarts = new Int32Array(G + 1);
    const rowIndices = new Int32Array(G * (bw + (hard ? 0 : 1)));
    const rowValues = new Float64Array(rowIndices.length);
    let e = 0;
    for (let j = 0; j < G; j++) {
        rowStarts[j] = e;
        for (let r = 0; r < bw; r++) {
            rowIndices[e] = basis.first[j] + r;
            rowValues[e] = basis.values[j * bw + r];
            e++;
        }
        if (!hard) {
            rowIndices[e] = nb + j;
            rowValues[e] = 1;
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
        matrix: { format: 'csr', numRows: G, numCols, starts: rowStarts, indices: rowIndices, values: rowValues },
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
 * A phase-2 spec from a phase-1 band region (band.js): the check grid, center and half-width, and the band's knot
 * range; the phase-2 basis has the band's segments unless nseg is given.
 *
 * @param {{grid: Float64Array, center: Float64Array, half: Float64Array, params: {xl: number, xr: number,
 *     nseg: number}}} region
 * @param {{ambition: number, flat?: number, kappa?: number, epsilon?: number, nseg?: number}} opts
 * @returns {Phase2Spec & {ambition: number, flat: number}}
 */
export function regionSpec(region, opts) {
    const nseg = opts.nseg ?? region.params.nseg;
    const flat = opts.flat ?? 0;
    return {
        grid: region.grid, c: region.center, h: region.half,
        xl: region.params.xl, xr: region.params.xr, nseg, deg: 3,
        kappa: opts.kappa ?? DEFAULT_KAPPA,
        epsilon: opts.epsilon ?? DEFAULT_EPSILON,
        ambition: opts.ambition, flat,
        ...ambitionLambdas({ ambition: opts.ambition, flat, G: region.grid.length, nseg }),
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
 * @returns {{ambition: number, capped: boolean, fit: Phase2Fit, fits: number, ms: number}}  fit: at a*;
 *     fits: Newton fits computed
 */
export function leavingAmbition(spec) {
    const start = performance.now();
    const G = spec.grid.length;
    const flat = spec.flat ?? 0;
    const at = logA => fitIterated({ ...spec, ...ambitionLambdas({ ambition: 10 ** logA, flat, G, nseg: spec.nseg }) });
    const leaves = fit => fit.maxRatio > 1 + LEAVE_TOL;
    const top = Math.log10(maxAmbition(spec));
    let fits = 0;
    let lo = null;
    let loFit = null;
    let hi = null;
    const steps = Math.round(SCAN_DECADES / SCAN_STEP);
    for (let i = steps; i >= 0; i--) {
        const logA = top - i * SCAN_STEP;
        const fit = at(logA);
        fits++;
        if (leaves(fit)) {
            hi = logA;
            break;
        }
        lo = logA;
        loFit = fit;
    }
    if (hi === null)
        return { ambition: 10 ** top, capped: true, fit: loFit, fits, ms: performance.now() - start };
    if (lo === null) {
        // leaves already at the smallest ambition scanned: the center itself is outside (not from band.js)
        lo = hi - SCAN_STEP;
        loFit = at(lo);
        fits++;
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
    return { ambition: 10 ** lo, capped: false, fit: loFit, fits, ms: performance.now() - start };
}
