import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    bsplineBasis, differenceCoefficients, addDifferencePenalty, choleskyBanded, solveCholesky, bandedInverse,
    pSplineSystem, fitSystem, fitPSpline, evaluate, standardErrors, selectLambda, defaultSegments,
} from '../src/core/pspline.js';
import { createRng, normals } from '../src/core/rng.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'pspline-mgcv.json');

/** max |a − b| / max(1, max |b|) */
function relErr(a, b) {
    let err = 0;
    let scale = 1;
    for (let i = 0; i < b.length; i++) {
        err = Math.max(err, Math.abs(a[i] - b[i]));
        scale = Math.max(scale, Math.abs(b[i]));
    }
    return err / scale;
}

function assertClose(a, b, tol, what) {
    const e = Array.isArray(a) || ArrayBuffer.isView(a) ? relErr(a, b) : relErr([a], [b]);
    assert.ok(e <= tol, `${what}: relative error ${e} > ${tol}`);
}

// --- dense reference implementation (O(n³), tests only)

function denseBasis(b) {
    const B = Array.from({ length: b.n }, () => new Float64Array(b.nb));
    for (let i = 0; i < b.n; i++) {
        for (let r = 0; r <= b.deg; r++)
            B[i][b.first[i] + r] = b.values[i * (b.deg + 1) + r];
    }
    return B;
}

function invert(A) {
    const N = A.length;
    const M = A.map((row, i) => {
        const r = new Float64Array(2 * N);
        r.set(row);
        r[N + i] = 1;
        return r;
    });
    for (let c = 0; c < N; c++) {
        let piv = c;
        for (let r = c + 1; r < N; r++) {
            if (Math.abs(M[r][c]) > Math.abs(M[piv][c]))
                piv = r;
        }
        [M[c], M[piv]] = [M[piv], M[c]];
        const d = M[c][c];
        for (let k = 0; k < 2 * N; k++)
            M[c][k] /= d;
        for (let r = 0; r < N; r++) {
            if (r === c)
                continue;
            const f = M[r][c];
            for (let k = 0; k < 2 * N; k++)
                M[r][k] -= f * M[c][k];
        }
    }
    return M.map(row => row.slice(N));
}

function matVec(M, v) {
    return Float64Array.from(M, row => row.reduce((s, m, j) => s + m * v[j], 0));
}

function denseFit(x, y, w, nseg, deg, d, lambda, grid) {
    const xl = Math.min(...x);
    const xr = Math.max(...x);
    const B = denseBasis(bsplineBasis(x, xl, xr, nseg, deg));
    const nb = nseg + deg;
    const c = differenceCoefficients(d);
    const A = Array.from({ length: nb }, () => new Float64Array(nb));
    const G = Array.from({ length: nb }, () => new Float64Array(nb));
    const rhs = new Float64Array(nb);
    for (let i = 0; i < x.length; i++) {
        for (let j = 0; j < nb; j++) {
            rhs[j] += w[i] * B[i][j] * y[i];
            for (let k = 0; k < nb; k++)
                G[j][k] += w[i] * B[i][j] * B[i][k];
        }
    }
    for (let j = 0; j < nb; j++)
        A[j].set(G[j]);
    for (let r = 0; r + d < nb; r++) {
        for (let i = 0; i <= d; i++) {
            for (let j = 0; j <= d; j++)
                A[r + i][r + j] += lambda * c[i] * c[j];
        }
    }
    const Ainv = invert(A);
    const alpha = matVec(Ainv, rhs);
    const fitted = Float64Array.from(B, row => row.reduce((s, b, j) => s + b * alpha[j], 0));
    const hat = Float64Array.from(B, (row, i) => w[i] * row.reduce((s, b, j) => s + b * matVec(Ainv, row)[j], 0));
    let ed = 0;
    for (let i = 0; i < nb; i++) {
        for (let j = 0; j < nb; j++)
            ed += Ainv[i][j] * G[j][i];
    }
    let rss = 0;
    for (let i = 0; i < x.length; i++)
        rss += w[i] * (y[i] - fitted[i]) ** 2;
    const n = x.length;
    const sigma = Math.sqrt(rss / (n - ed));
    const Bg = denseBasis(bsplineBasis(grid, xl, xr, nseg, deg));
    const seBayes = Float64Array.from(Bg, row => sigma * Math.sqrt(row.reduce((s, b, j) => s + b * matVec(Ainv, row)[j], 0)));
    const seSandwich = Float64Array.from(Bg, row => {
        const v = matVec(Ainv, row);
        const Gv = matVec(G, v);
        return sigma * Math.sqrt(v.reduce((s, vi, j) => s + vi * Gv[j], 0));
    });
    return { alpha, fitted, hat, ed, rss, sigma, seBayes, seSandwich, Ainv };
}

function randomData(seed, n, { ties = false } = {}) {
    const rng = createRng(seed);
    const x = Float64Array.from({ length: n }, () => ties ? Math.round(rng.uniform() * 20) / 2 : rng.uniform() * 10);
    x.sort();
    const e = normals(n, rng);
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + 0.5 * e[i]);
    const w = Float64Array.from({ length: n }, () => 1 + Math.floor(rng.uniform() * 3));
    return { x, y, w };
}

// --- basis

test('B-splines are nonnegative and sum to one, for degrees 0–5', () => {
    const TOL = 1e-13; // absolute
    const rng = createRng(1);
    const x = Float64Array.from({ length: 500 }, () => -2 + 7 * rng.uniform());
    x[0] = -2;
    x[1] = 5;
    x[2] = -2 + 7 * 3 / 11; // a knot
    for (let deg = 0; deg <= 5; deg++) {
        const b = bsplineBasis(x, -2, 5, 11, deg);
        assert.equal(b.nb, 11 + deg);
        for (let i = 0; i < b.n; i++) {
            let s = 0;
            for (let r = 0; r <= deg; r++) {
                const v = b.values[i * (deg + 1) + r];
                assert.ok(v >= -TOL, `deg ${deg}: negative value ${v}`);
                s += v;
            }
            assert.ok(Math.abs(s - 1) < TOL, `deg ${deg}, x = ${x[i]}: sum ${s}`);
            assert.ok(b.first[i] >= 0 && b.first[i] + deg < b.nb);
        }
    }
});

test('cubic B-spline values match the closed form on a segment', () => {
    // uniform cubic B-spline pieces at local position t ∈ [0, 1): (1−t)³/6, (3t³ − 6t² + 4)/6,
    // (−3t³ + 3t² + 3t + 1)/6, t³/6
    const TOL = 1e-14;
    const t = 0.37;
    const b = bsplineBasis([2 + t], 0, 10, 10, 3);
    assert.equal(b.first[0], 2);
    const expect = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6];
    for (let r = 0; r < 4; r++)
        assert.ok(Math.abs(b.values[r] - expect[r]) < TOL);
});

test('basis matches R splines::splineDesign (fixture grids)', () => {
    const TOL = 1e-13; // absolute
    const { cases } = JSON.parse(readFileSync(FIXTURES, 'utf8'));
    for (const c of cases) {
        const b = bsplineBasis(c.grid, Math.min(...c.x), Math.max(...c.x), c.nseg, c.deg);
        const B = denseBasis(b);
        for (let i = 0; i < c.grid.length; i++) {
            for (let j = 0; j < b.nb; j++)
                assert.ok(Math.abs(B[i][j] - c.basisGrid[i][j]) < TOL, `${c.name} row ${i} col ${j}`);
        }
    }
});

test('points outside the knot range are rejected', () => {
    assert.throws(() => bsplineBasis([0, 1.01], 0, 1, 4), RangeError);
    assert.throws(() => bsplineBasis([-0.01], 0, 1, 4), RangeError);
    assert.doesNotThrow(() => bsplineBasis([1 + 1e-14], 0, 1, 4));
});

test('difference coefficients', () => {
    assert.deepEqual([...differenceCoefficients(0)], [1]);
    assert.deepEqual([...differenceCoefficients(1)], [-1, 1]);
    assert.deepEqual([...differenceCoefficients(2)], [1, -2, 1]);
    assert.deepEqual([...differenceCoefficients(3)], [-1, 3, -3, 1]);
});

// --- banded linear algebra

test('banded Cholesky, solve and inverse band agree with a dense inverse', () => {
    const TOL = 1e-12; // relative
    const N = 30;
    const W = 4;
    const rng = createRng(7);
    const a = new Float64Array(N * W);
    for (let i = 0; i < N; i++) {
        a[i * W] = 10 + rng.uniform();
        for (let k = 1; k < W && k <= i; k++)
            a[i * W + k] = rng.uniform() - 0.5;
    }
    addDifferencePenalty(a, N, W, 3, 2.5);
    const dense = Array.from({ length: N }, (_, i) => Float64Array.from({ length: N }, (_, j) =>
        Math.abs(i - j) < W ? (i >= j ? a[i * W + (i - j)] : a[j * W + (j - i)]) : 0));
    const inv = invert(dense);
    const L = choleskyBanded(a, N, W);
    const b = Float64Array.from({ length: N }, () => rng.uniform());
    assertClose(solveCholesky(L, N, W, b), matVec(inv, b), TOL, 'solve');
    const Z = bandedInverse(L, N, W);
    for (let i = 0; i < N; i++) {
        for (let k = 0; k < W && k <= i; k++)
            assert.ok(Math.abs(Z[i * W + k] - inv[i][i - k]) < TOL, `Z[${i}][${i - k}]`);
    }
});

test('Cholesky rejects a matrix that is not positive definite', () => {
    const W = 2;
    const a = Float64Array.from([1, 0, 2, 3]); // [[1, 2], [2, 3]] has a negative eigenvalue
    assert.throws(() => choleskyBanded(a, 2, W), RangeError);
});

// --- fits

test('fit agrees with a dense solve: coefficients, hat diagonal, ED, σ̂, both standard errors', () => {
    const TOL = 1e-9; // relative
    for (const [seed, nseg, deg, d, lambda, ties] of [[1, 12, 3, 2, 3, false], [2, 8, 2, 1, 0.2, true],
        [3, 15, 3, 3, 50, true], [4, 10, 1, 2, 1, false], [5, 6, 2, 4, 10, false]]) {
        const { x, y, w } = randomData(seed, 60, { ties });
        const grid = Float64Array.from({ length: 21 }, (_, i) => x[0] + (x[x.length - 1] - x[0]) * i / 20);
        const ref = denseFit(x, y, w, nseg, deg, d, lambda, grid);
        const fit = fitPSpline(x, y, { w, nseg, deg, diff: d, lambda });
        const what = `seed ${seed} (deg ${deg}, d ${d})`;
        assertClose(fit.alpha, ref.alpha, TOL, `${what} alpha`);
        assertClose(fit.fitted, ref.fitted, TOL, `${what} fitted`);
        assertClose(fit.hat, ref.hat, TOL, `${what} hat`);
        assertClose(fit.ed, ref.ed, TOL, `${what} ed`);
        assertClose(fit.rss, ref.rss, TOL, `${what} rss`);
        assertClose(fit.sigma, ref.sigma, TOL, `${what} sigma`);
        assertClose(evaluate(fit, x), ref.fitted, TOL, `${what} evaluate`);
        assertClose(standardErrors(fit, grid), ref.seBayes, TOL, `${what} se bayes`);
        assertClose(standardErrors(fit, grid, 'sandwich'), ref.seSandwich, TOL, `${what} se sandwich`);
        let sumHat = 0;
        for (const h of fit.hat)
            sumHat += h;
        assertClose(sumHat, fit.ed, TOL, `${what} trace = Σ hat`);
    }
});

test('large λ gives the least-squares polynomial of degree d − 1', () => {
    // The penalty's null space is coefficients polynomial of degree < d in their index, which on equal knots are
    // B-spline fits polynomial of degree < d in x (for d ≤ deg + 1). The deviation shrinks like 1/λ.
    const LAMBDA = 1e9;
    const TOL = 1e-5; // relative to max |fitted|
    const { x, y, w } = randomData(11, 50);
    for (const d of [1, 2, 3, 4]) {
        const fit = fitPSpline(x, y, { w, nseg: 20, deg: 3, diff: d, lambda: LAMBDA });
        // weighted LS polynomial of degree d − 1 by dense normal equations on centered x
        const xc = x.map(v => (v - 5) / 5);
        const XtX = Array.from({ length: d }, () => new Float64Array(d));
        const Xty = new Float64Array(d);
        for (let i = 0; i < x.length; i++) {
            for (let j = 0; j < d; j++) {
                Xty[j] += w[i] * xc[i] ** j * y[i];
                for (let k = 0; k < d; k++)
                    XtX[j][k] += w[i] * xc[i] ** (j + k);
            }
        }
        const beta = matVec(invert(XtX), Xty);
        const poly = Float64Array.from(xc, v => beta.reduce((s, b, j) => s + b * v ** j, 0));
        assertClose(fit.fitted, poly, TOL, `d = ${d}`);
        assert.ok(Math.abs(fit.ed - d) < 1e-3, `d = ${d}: ED ${fit.ed}`);
    }
});

test('a polynomial of degree < d is reproduced exactly at any λ', () => {
    const TOL = 1e-9; // relative
    const x = Float64Array.from({ length: 40 }, (_, i) => i * 0.25);
    const y = Float64Array.from(x, v => 2 - 0.5 * v + 0.1 * v * v);
    for (const lambda of [0.01, 1, 1e4]) {
        const fit = fitPSpline(x, y, { nseg: 10, diff: 3, lambda });
        assertClose(fit.fitted, y, TOL, `λ = ${lambda}`);
    }
});

test('fits match mgcv (paraPen with the same basis and penalty)', () => {
    // mgcv works to its own convergence tolerances; the fixed-λ fits are direct solves, so agreement is close.
    const TOL = 1e-8; // relative
    const { cases } = JSON.parse(readFileSync(FIXTURES, 'utf8'));
    for (const c of cases) {
        const system = pSplineSystem(c.x, c.y, { w: c.w, nseg: c.nseg, deg: c.deg, diff: c.diff });
        for (const ref of c.fits) {
            const fit = fitSystem(system, ref.lambda);
            const what = `${c.name} λ = ${ref.lambda}`;
            assertClose(fit.alpha, ref.alpha, TOL, `${what} alpha`);
            assertClose(fit.fitted, ref.fitted, TOL, `${what} fitted`);
            assertClose(fit.ed, ref.ed, TOL, `${what} ed`);
            assertClose(fit.rss, ref.rss, TOL, `${what} rss`);
            assertClose(fit.sigma, ref.sigma, TOL, `${what} sigma`);
            assertClose(fit.gcv, ref.gcv, TOL, `${what} gcv`);
            assertClose(fit.hat, ref.hat, TOL, `${what} hat`);
            assertClose(standardErrors(fit, c.grid), ref.seBayes, TOL, `${what} se bayes`);
            assertClose(standardErrors(fit, c.grid, 'sandwich'), ref.seSandwich, TOL, `${what} se sandwich`);
        }
    }
});

test('GCV choice of λ matches mgcv\'s', () => {
    // mgcv minimizes by Newton steps on log λ; ours is a grid plus golden section. Both find the same minimum to
    // within their tolerances; the score is flat there, so λ agrees less closely than the score.
    const LOG_LAMBDA_TOL = 0.02;
    const SCORE_TOL = 1e-7; // relative
    const ED_TOL = 1e-2; // absolute
    const { cases } = JSON.parse(readFileSync(FIXTURES, 'utf8'));
    for (const c of cases) {
        const r = selectLambda(c.x, c.y, { w: c.w, nseg: c.nseg, deg: c.deg, diff: c.diff });
        const ref = c.gcvChoice;
        assert.equal(r.atEdge, false, c.name);
        assert.ok(Math.abs(Math.log(r.lambda / ref.lambda)) < LOG_LAMBDA_TOL,
            `${c.name}: λ ${r.lambda} vs mgcv ${ref.lambda}`);
        assertClose(r.fit.gcv, ref.gcv, SCORE_TOL, `${c.name} gcv`);
        assert.ok(r.fit.gcv <= ref.gcv * (1 + SCORE_TOL), `${c.name}: score worse than mgcv's`);
        assert.ok(Math.abs(r.fit.ed - ref.ed) < ED_TOL, `${c.name}: ED ${r.fit.ed} vs mgcv ${ref.ed}`);
    }
});

test('AIC choice minimizes AIC over a fine grid', () => {
    const { x, y } = randomData(21, 80);
    const r = selectLambda(x, y, { criterion: 'aic', nseg: 20 });
    const system = pSplineSystem(x, y, { nseg: 20 });
    for (let t = -10; t <= 18; t += 0.05)
        assert.ok(r.fit.aic <= fitSystem(system, Math.exp(t)).aic + 1e-9, `log λ = ${t}`);
    assert.equal(r.criterion, 'aic');
});

test('a grid that stops short of the optimum is flagged as at its edge', () => {
    // Noisy straight line. This GCV curve (seed 3) falls toward the line (ED → 2) at large λ, with a shallow
    // local minimum near log λ = 0 (ED ≈ 6) on the way; a grid capped at log λ = −2 is still on the slope.
    const x = Float64Array.from({ length: 60 }, (_, i) => i / 6);
    const e = normals(60, createRng(3));
    const y = Float64Array.from(x, (v, i) => 1 + 0.1 * v + e[i]);
    const full = selectLambda(x, y, { nseg: 10 });
    assert.ok(full.fit.ed < 2.001, `full grid: ED ${full.fit.ed}`);
    assert.equal(full.atEdge, true);
    const capped = selectLambda(x, y, { nseg: 10, logLambdaMax: -2 });
    assert.equal(capped.atEdge, true);
    assert.ok(Math.log(capped.lambda) <= -2 + 1e-9);
    assert.equal(full.grid.logLambda.length, full.grid.score.length);
});

test('default segments', () => {
    assert.equal(defaultSegments(4), 5);
    assert.equal(defaultSegments(25), 13);
    assert.equal(defaultSegments(1000), 40);
});

test('evaluate rejects a basis built on other knots', () => {
    const { x, y } = randomData(1, 30);
    const fit = fitPSpline(x, y, { nseg: 10, lambda: 1 });
    assert.throws(() => evaluate(fit, bsplineBasis(x, x[0], x[29], 11)), RangeError);
});
