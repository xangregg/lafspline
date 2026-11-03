import { test } from 'node:test';
import assert from 'node:assert/strict';
import loadHighs from 'highs';
import { fitIterated, buildQP, qpResult, phase2Objective, checkRegion } from '../src/core/fit.js';
import { solveModel } from '../src/core/lp-solve.js';
import { selectLambda, evaluate, standardErrors } from '../src/core/pspline.js';
import { simulate } from '../src/core/signals.js';
import { createRng, normals } from '../src/core/rng.js';
import { benchCases } from '../tools/bench-qp-cases.js';

const Q_CRIT = 3; // stand-in critical value: these tests need a region, not its coverage
const SEED = 20261102;

/** A region from a GCV P-spline fit, and a phase-2 spec on the same knots unless nseg is given. */
function makeSpec(signal, sigma, { n = 200, G = 200, nseg, ...rest } = {}) {
    const { x, y } = simulate(signal, n, sigma, SEED);
    const phase1 = selectLambda(x, y, { nseg: 20 });
    const xl = x[0];
    const xr = x[n - 1];
    const grid = Float64Array.from({ length: G }, (_, j) => xl + (xr - xl) * j / (G - 1));
    return {
        grid, xl, xr, nseg: nseg ?? 20, deg: 3,
        c: evaluate(phase1.fit, grid),
        h: standardErrors(phase1.fit, grid).map(v => Q_CRIT * v),
        kappa: 1e4, epsilon: 1e-2, lambda1: 0, lambda2: 1,
        ...rest,
    };
}

function maxDiffOverH(a, b, h) {
    let m = 0;
    for (let j = 0; j < a.length; j++)
        m = Math.max(m, Math.abs(a[j] - b[j]) / h[j]);
    return m;
}

test('iterated weights and the HiGHS QP find the same fit', async () => {
    // HiGHS's QP stops at its own tolerance; the iteration's answer is exact up to rounding.
    const G_TOL = 1e-4; // max |Δg|/h
    const highs = await loadHighs();
    for (const [signal, sigma, lambda1, lambda2] of [['heavisine', 1, 0, 1], ['blocks', 1, 0, 0.01],
        ['sine', 0.2, 100, 0], ['sine', 0.2, 0, 1e4]]) {
        const spec = makeSpec(signal, sigma, { nseg: 40, lambda1, lambda2 });
        const it = fitIterated(spec);
        const qp = qpResult(spec, solveModel(highs, buildQP(spec).model));
        assert.equal(it.converged, true, signal);
        assert.equal(qp.status, 'optimal', signal);
        assert.ok(maxDiffOverH(it.g, qp.g, spec.h) < G_TOL, `${signal}: ${maxDiffOverH(it.g, qp.g, spec.h)}`);
        assert.ok(it.objective <= qp.objective * (1 + 1e-9), `${signal}: S ${it.objective} vs QP ${qp.objective}`);
    }
});

test('the iteration converges where undamped active-set updates cycled (Blocks, low ambition)', () => {
    // Regression: the phase-0 risk-check case where full steps cycle until the iteration cap.
    const [{ spec }] = benchCases({ sizes: [[43, 200]], signals: ['blocks'], ambitions: [[0, 0.01]] });
    const undamped = fitIterated(spec, { lineSearch: false });
    assert.equal(undamped.converged, false);
    const it = fitIterated(spec);
    assert.equal(it.converged, true);
    assert.ok(it.damped > 0);
});

test('the converged fit is a minimum: no small move in any direction lowers S', () => {
    const TOL = 1e-10; // relative
    const DELTA = 1e-4; // step, relative to the coefficient scale
    const spec = makeSpec('heavisine', 1, { lambda2: 10 });
    const it = fitIterated(spec);
    const s0 = phase2Objective(spec, it.beta);
    assert.ok(Math.abs(s0 - it.objective) <= TOL * s0);
    const rng = createRng(5);
    const scale = Math.max(...it.beta.map(Math.abs));
    for (let trial = 0; trial < 50; trial++) {
        const v = normals(it.beta.length, rng);
        for (const sign of [1, -1]) {
            const moved = it.beta.map((b, k) => b + sign * DELTA * scale * v[k]);
            assert.ok(phase2Objective(spec, moved) >= s0 * (1 - TOL), `trial ${trial}`);
        }
    }
});

test('no ambition gives the center when the center is in the basis', () => {
    const G_TOL = 1e-6; // max |g − c|/h
    const spec = makeSpec('heavisine', 1, { lambda2: 0 });
    const it = fitIterated(spec);
    assert.ok(maxDiffOverH(it.g, spec.c, spec.h) < G_TOL);
    assert.equal(it.iterations, 1);
});

test('great ambition toward straight gives a straight line when one fits inside', () => {
    const LINE_TOL = 1e-6; // max |second difference of g| / median h
    const spec = makeSpec('line', 0.25, { lambda2: 1e10 });
    const it = fitIterated(spec);
    assert.equal(it.outside, 0);
    const sorted = Float64Array.from(spec.h).sort();
    const hMed = sorted[sorted.length >> 1];
    for (let j = 1; j + 1 < it.g.length; j++)
        assert.ok(Math.abs(it.g[j - 1] - 2 * it.g[j] + it.g[j + 1]) / hMed < LINE_TOL, `j = ${j}`);
});

test('a larger κ pulls the fit toward the region', () => {
    const ratios = [1e2, 1e4, 1e6, 1e8].map(kappa =>
        fitIterated(makeSpec('blocks', 1, { kappa, lambda2: 1e4 })).maxRatio);
    for (let i = 1; i < ratios.length; i++)
        assert.ok(ratios[i] < ratios[i - 1], ratios.join(', '));
    assert.ok(ratios[ratios.length - 1] < 1.001, `κ = 1e8: ${ratios[ratios.length - 1]}`);
});

test('checkRegion counts points outside', () => {
    const r = checkRegion([0, 2, -3, 1], [0, 0, 0, 0], [1, 1, 1, 1]);
    assert.equal(r.maxRatio, 3);
    assert.equal(r.outside, 2);
});

test('ε must be positive', () => {
    assert.throws(() => fitIterated(makeSpec('sine', 0.2, { epsilon: 0 })), RangeError);
});
