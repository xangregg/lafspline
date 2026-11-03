// P-splines (Eilers & Marx 1996; Eilers & Marx 2021, "Practical Smoothing", chapters 2–3): B-splines on equally
// spaced knots with a difference penalty on the coefficients. The penalized normal equations
//
//     (BᵀWB + λ D_dᵀD_d) α = BᵀWy
//
// are banded with half-bandwidth p = max(deg, d), so everything here is O(n·deg² + nb·p²): forming BᵀWB, a banded
// Cholesky, and the band of the inverse (Hutchinson & de Hoog 1985 / Takahashi recurrence), which is all the
// effective dimension, the hat diagonal and the Bayesian standard errors need.
//
// Indexing: the book numbers basis functions and coefficients 1..nb; here they are 0..nb−1. Basis function j
// has support on knots j..j+deg+1, where knot k sits at xl + (k − deg)·dx, so the segment s = 0..nseg−1 (the
// interval [xl + s·dx, xl + (s+1)·dx]) is covered by basis functions s..s+deg.
//
// Banded symmetric matrices are stored by their lower band, row by row: a[i·W + k] = A[i][i − k] for
// k = 0..min(i, p), W = p + 1. Entries with i − k < 0 are unused.

export const DEFAULT_DEGREE = 3;
export const DEFAULT_DIFF = 2;
const MAX_SEGMENTS = 40;
const MIN_SEGMENTS = 5;
// Default segments per distinct x below MAX_SEGMENTS: keeps the basis smaller than the data, so the small-λ end
// of a GCV search stays well conditioned. With a penalty the exact number matters little (Eilers & Marx 1996 §9).
const SEGMENTS_PER_DISTINCT_X = 0.5;
const RANGE_TOL = 1e-10; // relative to xr − xl: x this far outside [xl, xr] still counts as inside (rounding)

// λ search (natural log): a grid, then golden-section refinement around the best grid point.
const LOG_LAMBDA_MIN = -10;
const LOG_LAMBDA_MAX = 18;
const LOG_LAMBDA_STEP = 0.25;
const LOG_LAMBDA_TOL = 1e-4; // golden-section stopping width in log λ
const GOLDEN = (Math.sqrt(5) - 1) / 2;

/**
 * @typedef {object} Basis  B-spline basis evaluated at points, deg + 1 nonzeros per point
 * @property {number} xl          left end of the knot range (data units)
 * @property {number} xr          right end
 * @property {number} nseg        number of equal segments on [xl, xr]
 * @property {number} deg         degree of the B-splines
 * @property {number} nb          number of basis functions, nseg + deg
 * @property {number} n           number of points
 * @property {Int32Array} first   index of the first nonzero basis function at each point
 * @property {Float64Array} values  the deg + 1 nonzero values at each point, row-major (n × (deg + 1))
 */

/**
 * Default number of segments for m distinct x values.
 *
 * @param {number} m  distinct x
 * @returns {number}
 */
export function defaultSegments(m) {
    return Math.max(MIN_SEGMENTS, Math.min(MAX_SEGMENTS, Math.ceil(m * SEGMENTS_PER_DISTINCT_X)));
}

/**
 * B-spline basis of degree deg on nseg equal segments over [xl, xr], evaluated at x. Uses the Cox–de Boor
 * recursion in the form of Piegl & Tiller's algorithm A2.2, with knots in units of the segment width.
 *
 * @param {ArrayLike<number>} x  points in [xl, xr]
 * @param {number} xl
 * @param {number} xr           > xl
 * @param {number} nseg         ≥ 1
 * @param {number} [deg]        ≥ 0
 * @returns {Basis}
 * @throws {RangeError} if a point is outside [xl, xr]
 */
export function bsplineBasis(x, xl, xr, nseg, deg = DEFAULT_DEGREE) {
    if (!(xr > xl))
        throw new RangeError(`bsplineBasis: need xr > xl (got ${xl}, ${xr})`);
    if (!(Number.isInteger(nseg) && nseg >= 1 && Number.isInteger(deg) && deg >= 0))
        throw new RangeError(`bsplineBasis: bad nseg ${nseg} or deg ${deg}`);
    const n = x.length;
    const w = deg + 1;
    const dx = (xr - xl) / nseg;
    const tol = RANGE_TOL * (xr - xl);
    const first = new Int32Array(n);
    const values = new Float64Array(n * w);
    const left = new Float64Array(w);
    const right = new Float64Array(w);
    const N = new Float64Array(w);
    for (let i = 0; i < n; i++) {
        const xi = x[i];
        if (!(xi >= xl - tol && xi <= xr + tol))
            throw new RangeError(`bsplineBasis: x = ${xi} outside [${xl}, ${xr}]`);
        const u = (xi - xl) / dx; // position in segment units; knot k is at k − deg
        const s = Math.min(nseg - 1, Math.max(0, Math.floor(u)));
        // Piegl & Tiller A2.2 with span index s + deg and knot U[k] = k − deg, so U[s + deg + j] − u = s + j − u
        // and u − U[s + deg + 1 − j] = u − (s + 1 − j).
        N[0] = 1;
        for (let j = 1; j <= deg; j++) {
            left[j] = u - (s + 1 - j);
            right[j] = s + j - u;
            let saved = 0;
            for (let r = 0; r < j; r++) {
                const temp = N[r] / (right[r + 1] + left[j - r]);
                N[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            N[j] = saved;
        }
        first[i] = s;
        for (let r = 0; r < w; r++)
            values[i * w + r] = N[r];
    }
    return { xl, xr, nseg, deg, nb: nseg + deg, n, first, values };
}

/**
 * Coefficients of the d-th order difference operator: (Δ^d a)_r = Σ_k c_k a_{r+k}, c_k = (−1)^(d−k) C(d, k).
 *
 * @param {number} d  ≥ 0
 * @returns {Float64Array}  length d + 1
 */
export function differenceCoefficients(d) {
    const c = new Float64Array(d + 1);
    let binom = 1;
    for (let k = 0; k <= d; k++) {
        c[k] = ((d - k) % 2 === 0 ? 1 : -1) * binom;
        binom = binom * (d - k) / (k + 1);
    }
    return c;
}

/**
 * Add scale·D_dᵀD_d (nb × nb) to a banded matrix in place.
 *
 * @param {Float64Array} a  lower band storage, width W ≥ d + 1
 * @param {number} nb
 * @param {number} W
 * @param {number} d        difference order
 * @param {number} scale
 */
export function addDifferencePenalty(a, nb, W, d, scale) {
    const c = differenceCoefficients(d);
    for (let r = 0; r + d < nb; r++) {
        for (let i = 0; i <= d; i++) {
            for (let j = 0; j <= i; j++)
                a[(r + i) * W + (i - j)] += scale * c[i] * c[j];
        }
    }
}

/**
 * Cholesky factor L (A = LLᵀ) of a symmetric positive definite banded matrix. O(N·p²).
 *
 * @param {Float64Array} a  lower band storage (N × W)
 * @param {number} N
 * @param {number} W        p + 1
 * @returns {Float64Array}  L in the same storage
 * @throws {RangeError} if A is not numerically positive definite
 */
export function choleskyBanded(a, N, W) {
    const p = W - 1;
    const L = new Float64Array(N * W);
    for (let i = 0; i < N; i++) {
        const lo = Math.max(0, i - p);
        for (let j = lo; j <= i; j++) {
            let s = a[i * W + (i - j)];
            for (let l = lo; l < j; l++)
                s -= L[i * W + (i - l)] * L[j * W + (j - l)];
            if (j === i) {
                if (!(s > 0))
                    throw new RangeError(`choleskyBanded: not positive definite at row ${i}`);
                L[i * W] = Math.sqrt(s);
            }
            else {
                L[i * W + (i - j)] = s / L[j * W];
            }
        }
    }
    return L;
}

/**
 * Solve LLᵀx = b with a banded Cholesky factor. O(N·p).
 *
 * @param {Float64Array} L
 * @param {number} N
 * @param {number} W
 * @param {ArrayLike<number>} b
 * @returns {Float64Array}
 */
export function solveCholesky(L, N, W, b) {
    const p = W - 1;
    const z = Float64Array.from(b);
    for (let i = 0; i < N; i++) {
        let s = z[i];
        for (let l = Math.max(0, i - p); l < i; l++)
            s -= L[i * W + (i - l)] * z[l];
        z[i] = s / L[i * W];
    }
    for (let i = N - 1; i >= 0; i--) {
        let s = z[i];
        for (let k = i + 1; k <= Math.min(N - 1, i + p); k++)
            s -= L[k * W + (k - i)] * z[k];
        z[i] = s / L[i * W];
    }
    return z;
}

/**
 * The band of A⁻¹ (same half-bandwidth as A) from A's Cholesky factor, by the Takahashi recurrence: from
 * LᵀZ = L⁻¹, Z_ij = (δ_ij / L_ii − Σ_{k=i+1}^{i+p} L_ki Z_kj) / L_ii for j ≥ i, rows from the last up. Every Z_kj
 * it needs lies in the band. O(N·p²).
 *
 * @param {Float64Array} L
 * @param {number} N
 * @param {number} W
 * @returns {Float64Array}  band of A⁻¹, lower band storage
 */
export function bandedInverse(L, N, W) {
    const p = W - 1;
    const Z = new Float64Array(N * W);
    for (let i = N - 1; i >= 0; i--) {
        const hi = Math.min(N - 1, i + p);
        const lii = L[i * W];
        for (let j = hi; j >= i; j--) {
            let s = j === i ? 1 / lii : 0;
            for (let k = i + 1; k <= hi; k++)
                s -= L[k * W + (k - i)] * bandAt(Z, W, k, j);
            Z[j * W + (j - i)] = s / lii;
        }
    }
    return Z;
}

function bandAt(a, W, r, c) {
    return r >= c ? a[r * W + (r - c)] : a[c * W + (c - r)];
}

/** bᵀMb for one basis row (first index f, values at v[off..off+deg]) and a banded symmetric M. */
function rowQuadratic(M, W, f, v, off, w) {
    let s = 0;
    for (let r = 0; r < w; r++) {
        s += v[off + r] * v[off + r] * M[(f + r) * W];
        for (let q = 0; q < r; q++)
            s += 2 * v[off + r] * v[off + q] * M[(f + r) * W + (r - q)];
    }
    return s;
}

/**
 * @typedef {object} PSplineSystem  the λ-independent part of a P-spline fit
 * @property {Float64Array} x
 * @property {Float64Array} y
 * @property {Float64Array} w     prior weights (1 by default)
 * @property {number} n           observations with w > 0
 * @property {number} diff        penalty order d
 * @property {number} W           band storage width, max(deg, d) + 1
 * @property {Basis} basis        basis at the data x
 * @property {Float64Array} G     BᵀWB, lower band storage
 * @property {Float64Array} rhs   BᵀWy
 */

/**
 * Build the λ-independent part of a P-spline fit. O(n·deg²).
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @param {{w?: ArrayLike<number>, nseg?: number, deg?: number, diff?: number, xl?: number, xr?: number}} [opts]
 *     w: prior (precision) weights ≥ 0; nseg: segments (default from the number of distinct x); deg: B-spline
 *     degree (3); diff: penalty order d (2); xl, xr: knot range (default the range of x)
 * @returns {PSplineSystem}
 */
export function pSplineSystem(x, y, opts = {}) {
    const n0 = x.length;
    if (y.length !== n0)
        throw new RangeError('pSplineSystem: x and y differ in length');
    const xs = Float64Array.from(x);
    const ys = Float64Array.from(y);
    const w = opts.w ? Float64Array.from(opts.w) : new Float64Array(n0).fill(1);
    let xmin = Infinity;
    let xmax = -Infinity;
    let n = 0;
    for (let i = 0; i < n0; i++) {
        if (!(Number.isFinite(xs[i]) && Number.isFinite(ys[i]) && w[i] >= 0))
            throw new RangeError(`pSplineSystem: bad value at ${i}`);
        xmin = Math.min(xmin, xs[i]);
        xmax = Math.max(xmax, xs[i]);
        if (w[i] > 0)
            n++;
    }
    const deg = opts.deg ?? DEFAULT_DEGREE;
    const diff = opts.diff ?? DEFAULT_DIFF;
    const nseg = opts.nseg ?? defaultSegments(new Set(xs).size);
    const basis = bsplineBasis(xs, opts.xl ?? xmin, opts.xr ?? xmax, nseg, deg);
    const W = Math.max(deg, diff) + 1;
    const { G, rhs } = weightedCrossProducts(basis, w, ys, W);
    return { x: xs, y: ys, w, n, diff, W, basis, G, rhs };
}

/**
 * BᵀWB (banded) and BᵀWy for a basis and weights. O(n·deg²).
 *
 * @param {Basis} basis
 * @param {ArrayLike<number>} w  weight per point
 * @param {ArrayLike<number>} y  value per point
 * @param {number} W             band storage width, ≥ deg + 1
 * @returns {{G: Float64Array, rhs: Float64Array}}
 */
export function weightedCrossProducts(basis, w, y, W) {
    const nb = basis.nb;
    const G = new Float64Array(nb * W);
    const rhs = new Float64Array(nb);
    const bw = basis.deg + 1;
    for (let i = 0; i < basis.n; i++) {
        const f = basis.first[i];
        const off = i * bw;
        for (let r = 0; r < bw; r++) {
            const wv = w[i] * basis.values[off + r];
            rhs[f + r] += wv * y[i];
            for (let q = 0; q <= r; q++)
                G[(f + r) * W + (r - q)] += wv * basis.values[off + q];
        }
    }
    return { G, rhs };
}

/**
 * @typedef {object} PSplineFit
 * @property {PSplineSystem} system
 * @property {number} lambda      penalty weight λ
 * @property {Float64Array} alpha coefficients, length nb
 * @property {Float64Array} fitted  Bα at the data x
 * @property {number} ed          effective dimension trace(H) = trace((BᵀWB + λDᵀD)⁻¹BᵀWB)
 * @property {number} rss         Σ w (y − fitted)²
 * @property {number} sigma       σ̂ = √(RSS / (n − ED)) (NaN if n ≤ ED)
 * @property {number} gcv         n·RSS / (n − ED)²  (as mgcv's GCV score, γ = 1)
 * @property {number} aic         n·log(RSS / n) + 2·ED  (Gaussian AIC up to a constant)
 * @property {Float64Array} hat   diagonal of the hat matrix, w_i b_iᵀ(BᵀWB + λDᵀD)⁻¹b_i
 * @property {Float64Array} L     Cholesky factor of BᵀWB + λDᵀD (band storage)
 * @property {Float64Array} Z     band of (BᵀWB + λDᵀD)⁻¹
 */

/**
 * Fit a prepared P-spline system at one λ. O(n·deg² + nb·p²).
 *
 * @param {PSplineSystem} system
 * @param {number} lambda  ≥ 0 (0 only if BᵀWB is positive definite)
 * @returns {PSplineFit}
 */
export function fitSystem(system, lambda) {
    const { basis, W, G, rhs, x, y, w, n } = system;
    const nb = basis.nb;
    const A = Float64Array.from(G);
    addDifferencePenalty(A, nb, W, system.diff, lambda);
    const L = choleskyBanded(A, nb, W);
    const alpha = solveCholesky(L, nb, W, rhs);
    const Z = bandedInverse(L, nb, W);
    const bw = basis.deg + 1;
    const fitted = new Float64Array(x.length);
    const hat = new Float64Array(x.length);
    let rss = 0;
    for (let i = 0; i < x.length; i++) {
        const f = basis.first[i];
        let s = 0;
        for (let r = 0; r < bw; r++)
            s += basis.values[i * bw + r] * alpha[f + r];
        fitted[i] = s;
        rss += w[i] * (y[i] - s) ** 2;
        hat[i] = w[i] * rowQuadratic(Z, W, f, basis.values, i * bw, bw);
    }
    let ed = 0;
    for (let i = 0; i < nb; i++) {
        ed += Z[i * W] * G[i * W];
        for (let k = 1; k <= Math.min(i, W - 1); k++)
            ed += 2 * Z[i * W + k] * G[i * W + k];
    }
    const sigma = n > ed ? Math.sqrt(rss / (n - ed)) : NaN;
    const gcv = n * rss / (n - ed) ** 2;
    const aic = n * Math.log(rss / n) + 2 * ed;
    return { system, lambda, alpha, fitted, ed, rss, sigma, gcv, aic, hat, L, Z };
}

/**
 * Fit a P-spline at a given λ.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @param {object} opts  pSplineSystem options plus `lambda` (required)
 * @returns {PSplineFit}
 */
export function fitPSpline(x, y, opts) {
    if (!(opts?.lambda >= 0))
        throw new RangeError('fitPSpline: lambda ≥ 0 required');
    return fitSystem(pSplineSystem(x, y, opts), opts.lambda);
}

/**
 * Evaluate a fit at points inside its knot range.
 *
 * @param {PSplineFit} fit
 * @param {ArrayLike<number>|Basis} at  points, or a basis already evaluated with the fit's knots
 * @returns {Float64Array}
 */
export function evaluate(fit, at) {
    const b = basisFor(fit, at);
    const bw = b.deg + 1;
    const g = new Float64Array(b.n);
    for (let i = 0; i < b.n; i++) {
        let s = 0;
        for (let r = 0; r < bw; r++)
            s += b.values[i * bw + r] * fit.alpha[b.first[i] + r];
        g[i] = s;
    }
    return g;
}

function basisFor(fit, at) {
    const sb = fit.system.basis;
    if (at.values && at.first) {
        if (at.xl !== sb.xl || at.xr !== sb.xr || at.nseg !== sb.nseg || at.deg !== sb.deg)
            throw new RangeError('basis does not match the fit\'s knots');
        return at;
    }
    return bsplineBasis(at, sb.xl, sb.xr, sb.nseg, sb.deg);
}

/**
 * Pointwise standard errors of the fit at points.
 *
 * - 'bayes' (default): from V = σ̂²(BᵀWB + λDᵀD)⁻¹ (Wahba 1983; Nychka 1988), which allows on average for the
 *   smoothing bias. Uses only the band of the inverse: O(G·deg²).
 * - 'sandwich': from σ̂²(BᵀWB + λDᵀD)⁻¹BᵀWB(BᵀWB + λDᵀD)⁻¹, the variance of the fit about its own mean (it
 *   ignores the bias). One banded solve per point: O(G·nb·p).
 *
 * @param {PSplineFit} fit
 * @param {ArrayLike<number>|Basis} at
 * @param {'bayes'|'sandwich'} [type]
 * @returns {Float64Array}  same units as y
 */
export function standardErrors(fit, at, type = 'bayes') {
    const b = basisFor(fit, at);
    const { W, G } = fit.system;
    const nb = b.nb;
    const bw = b.deg + 1;
    const se = new Float64Array(b.n);
    for (let i = 0; i < b.n; i++) {
        const f = b.first[i];
        let q;
        if (type === 'bayes') {
            q = rowQuadratic(fit.Z, W, f, b.values, i * bw, bw);
        }
        else if (type === 'sandwich') {
            const e = new Float64Array(nb);
            for (let r = 0; r < bw; r++)
                e[f + r] = b.values[i * bw + r];
            const v = solveCholesky(fit.L, nb, W, e);
            q = bandedQuadratic(G, nb, W, v);
        }
        else {
            throw new RangeError(`standardErrors: unknown type ${type}`);
        }
        se[i] = fit.sigma * Math.sqrt(Math.max(0, q));
    }
    return se;
}

/** vᵀMv for a banded symmetric M. */
function bandedQuadratic(M, N, W, v) {
    let s = 0;
    for (let i = 0; i < N; i++) {
        s += M[i * W] * v[i] * v[i];
        for (let k = 1; k <= Math.min(i, W - 1); k++)
            s += 2 * M[i * W + k] * v[i] * v[i - k];
    }
    return s;
}

/**
 * @typedef {object} LambdaChoice
 * @property {'gcv'|'aic'} criterion
 * @property {number} lambda        the chosen λ
 * @property {PSplineFit} fit       the fit at that λ
 * @property {boolean} atEdge       the best grid point was an end of the grid (the true optimum may lie beyond)
 * @property {{logLambda: Float64Array, score: Float64Array, ed: Float64Array}} grid  the search grid
 */

/**
 * Choose λ by GCV or AIC: a grid over log λ, then golden-section refinement between the best grid point's
 * neighbors. The criterion can have several local minima; the grid picks the basin. Grid points where the
 * system is not positive definite score +∞.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @param {object} [opts]  pSplineSystem options plus criterion ('gcv' default or 'aic') and logLambdaMin,
 *     logLambdaMax, logLambdaStep (natural log)
 * @returns {LambdaChoice}
 */
export function selectLambda(x, y, opts = {}) {
    const criterion = opts.criterion ?? 'gcv';
    if (criterion !== 'gcv' && criterion !== 'aic')
        throw new RangeError(`selectLambda: unknown criterion ${criterion}`);
    const system = pSplineSystem(x, y, opts);
    const lo = opts.logLambdaMin ?? LOG_LAMBDA_MIN;
    const hi = opts.logLambdaMax ?? LOG_LAMBDA_MAX;
    const step = opts.logLambdaStep ?? LOG_LAMBDA_STEP;
    const count = Math.floor((hi - lo) / step + 1e-9) + 1;
    const logLambda = new Float64Array(count);
    const score = new Float64Array(count);
    const ed = new Float64Array(count);
    const at = t => {
        try {
            const fit = fitSystem(system, Math.exp(t));
            const s = fit[criterion];
            return { fit, s: Number.isFinite(s) ? s : Infinity };
        }
        catch (err) {
            if (err instanceof RangeError)
                return { fit: null, s: Infinity };
            throw err;
        }
    };
    let best = -1;
    for (let k = 0; k < count; k++) {
        logLambda[k] = lo + k * step;
        const r = at(logLambda[k]);
        score[k] = r.s;
        ed[k] = r.fit ? r.fit.ed : NaN;
        if (best < 0 || r.s < score[best])
            best = k;
    }
    if (!Number.isFinite(score[best]))
        throw new RangeError('selectLambda: no λ on the grid gives a finite criterion');
    // golden section on [t(best − 1), t(best + 1)]
    let a = logLambda[Math.max(0, best - 1)];
    let b = logLambda[Math.min(count - 1, best + 1)];
    let c = b - GOLDEN * (b - a);
    let d = a + GOLDEN * (b - a);
    let fc = at(c).s;
    let fd = at(d).s;
    while (b - a > LOG_LAMBDA_TOL) {
        if (fc <= fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - GOLDEN * (b - a);
            fc = at(c).s;
        }
        else {
            a = c;
            c = d;
            fc = fd;
            d = a + GOLDEN * (b - a);
            fd = at(d).s;
        }
    }
    let t = (a + b) / 2;
    let r = at(t);
    if (!(r.s <= score[best])) {
        t = logLambda[best];
        r = at(t);
    }
    return { criterion, lambda: Math.exp(t), fit: r.fit, atEdge: best === 0 || best === count - 1,
        grid: { logLambda, score, ed } };
}
