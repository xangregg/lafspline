import { test } from 'node:test';
import assert from 'node:assert/strict';
import loadHighs from 'highs';
import {
    fitIterated, buildQP, qpResult, phase2Objective, checkRegion, ambitionLambdas, regionSpec, leavingAmbition,
    maxAmbition, LEAVE_TOL, fitSparse, sparseLambdas, sparseMaxAmbition, fitShaped, shapedLambdas,
} from '../src/core/fit.js';
import { bandRegion } from '../src/core/band.js';
import { solveModel } from '../src/core/lp-solve.js';
import { selectLambda, evaluate, standardErrors, bsplineBasis } from '../src/core/pspline.js';
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

/** A phase-1 band region (Monte Carlo kept small: these tests need a region, not its exact coverage). */
function makeRegion(signal, sigma, opts = {}) {
    const { x, y } = simulate(signal, opts.n ?? 200, sigma, SEED);
    return bandRegion(x, y, { B: 200, ...opts });
}

/** Weighted least squares of c on the polynomials of degree < order, weights 1/h², evaluated at the grid. */
function weightedPolynomial(grid, c, h, order) {
    // normal equations in t = (x − mid)/half-range, order ≤ 2
    const mid = (grid[0] + grid[grid.length - 1]) / 2;
    const half = (grid[grid.length - 1] - grid[0]) / 2;
    const M = [[0, 0], [0, 0]];
    const r = [0, 0];
    for (let j = 0; j < grid.length; j++) {
        const w = 1 / (h[j] * h[j]);
        const f = [1, (grid[j] - mid) / half];
        for (let a = 0; a < order; a++) {
            r[a] += w * f[a] * c[j];
            for (let b = 0; b < order; b++)
                M[a][b] += w * f[a] * f[b];
        }
    }
    let coef;
    if (order === 1) {
        coef = [r[0] / M[0][0], 0];
    }
    else {
        const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
        coef = [(r[0] * M[1][1] - r[1] * M[0][1]) / det, (M[0][0] * r[1] - M[1][0] * r[0]) / det];
    }
    return grid.map(x => coef[0] + coef[1] * (x - mid) / half);
}

test('ambition is invariant to the units of x and y', () => {
    const TOL = 1e-9; // max |Δg|/h, rounding only
    const X_SCALE = 1000;
    const X_SHIFT = 1950;
    const Y_SCALE = 1e-3;
    const spec = makeSpec('heavisine', 1, { lambda2: 0 });
    const a = 1e-4;
    const base = fitIterated({ ...spec, ...ambitionLambdas({ ambition: a, G: spec.grid.length, nseg: spec.nseg }) });
    const scaled = {
        ...spec,
        grid: spec.grid.map(v => X_SHIFT + X_SCALE * v),
        xl: X_SHIFT + X_SCALE * spec.xl,
        xr: X_SHIFT + X_SCALE * spec.xr,
        c: spec.c.map(v => Y_SCALE * v), h: spec.h.map(v => Y_SCALE * v),
    };
    const moved = fitIterated({ ...scaled, ...ambitionLambdas({ ambition: a, G: spec.grid.length, nseg: spec.nseg }) });
    assert.ok(maxDiffOverH(base.g, moved.g.map(v => v / Y_SCALE), spec.h) < TOL);
});

test('ambition means about the same fit for any number of segments and any check-grid density', () => {
    // Measured (sine, n = 200, a = a*·10^k for k = −6…0): ≤ 0.053 h from 20 to 80 segments, ≤ 0.009 h for a grid
    // 4 times denser.
    const SEGMENT_TOL = 0.1; // max |Δg|/h
    const GRID_TOL = 0.03;
    const region = makeRegion('sine', 0.2, { nseg: 20 });
    const dense = makeRegion('sine', 0.2, { nseg: 20, gridPerSegment: 16 });
    const star = leavingAmbition(regionSpec(region, { ambition: 0 })).ambition;
    const gridBasis = bsplineBasis(region.grid, region.params.xl, region.params.xr, 20, 3);
    for (const k of [-6, -3, 0]) {
        const a = star * 10 ** k;
        const base = fitIterated(regionSpec(region, { ambition: a }));
        const finer = fitIterated(regionSpec(region, { ambition: a, nseg: 80 }));
        assert.ok(maxDiffOverH(base.g, finer.g, region.half) < SEGMENT_TOL, `k = ${k}, 80 segments`);
        const onDense = fitIterated(regionSpec(dense, { ambition: a }));
        const g = Float64Array.from(region.grid, (_, j) => {
            let v = 0;
            for (let r = 0; r <= 3; r++)
                v += gridBasis.values[j * 4 + r] * onDense.beta[gridBasis.first[j] + r];
            return v;
        });
        assert.ok(maxDiffOverH(base.g, g, region.half) < GRID_TOL, `k = ${k}, dense grid`);
    }
});

test('a*: the fit stays inside up to it and leaves just above it', () => {
    for (const [signal, sigma] of [['sine', 0.2], ['blocks', 1], ['ramp', 0.2]]) {
        const region = makeRegion(signal, sigma);
        const star = leavingAmbition(regionSpec(region, { ambition: 0 }));
        assert.equal(star.capped, false, signal);
        assert.ok(star.fit.maxRatio <= 1 + LEAVE_TOL, signal);
        const at = fitIterated(regionSpec(region, { ambition: star.ambition }));
        assert.ok(at.maxRatio <= 1 + LEAVE_TOL, `${signal}: ${at.maxRatio}`);
        const above = fitIterated(regionSpec(region, { ambition: star.ambition * 10 ** 0.05 }));
        assert.ok(above.maxRatio > 1 + LEAVE_TOL, `${signal}: ${above.maxRatio}`);
    }
});

test('great ambition gives the weighted least-squares line or constant toward the center, when it fits', () => {
    // At the ambition cap λ is finite, so the fit is the limit only approximately; measured 2.5e-5 h (line) and
    // 7e-8 h (constant).
    const LIMIT_TOL = 1e-3; // max |g − p|/h
    const ED_TOL = 1e-3;
    for (const [signal, flat, order] of [['line', 0, 2], ['noise', 1, 1]]) {
        const region = makeRegion(signal, 0.25);
        const spec = regionSpec(region, { ambition: 0, flat });
        const star = leavingAmbition(spec);
        assert.equal(star.capped, true, signal);
        assert.ok(Math.abs(star.ambition / maxAmbition(spec) - 1) < 1e-12, signal);
        const fit = fitIterated(regionSpec(region, { ambition: star.ambition, flat }));
        assert.equal(fit.converged, true, signal);
        assert.equal(fit.outside, 0, signal);
        const p = weightedPolynomial(region.grid, region.center, region.half, order);
        assert.ok(maxDiffOverH(fit.g, p, region.half) < LIMIT_TOL, `${signal}: ${maxDiffOverH(fit.g, p, region.half)}`);
        assert.ok(Math.abs(fit.ed - order) < ED_TOL, `${signal}: ED ${fit.ed}`);
    }
});

test('effective dimension: K with no ambition, falling toward the penalty order while the fit stays inside', () => {
    // Once points leave, their weight κ raises the trace, so ED is monotone in a only while the sides stay fixed.
    // Without ambition only fit.js's penalty floor remains: measured K − 8.6e-9.
    const TOL = 1e-7;
    const spec = makeSpec('line', 0.25, { lambda2: 0 });
    assert.ok(Math.abs(fitIterated(spec).ed - (spec.nseg + 3)) < TOL);
    let previous = Infinity;
    for (const a of [1e-10, 1e-8, 1e-6, 1e-4]) {
        const fit = fitIterated({ ...spec, ...ambitionLambdas({ ambition: a, G: spec.grid.length, nseg: spec.nseg }) });
        assert.equal(fit.outside, 0);
        assert.ok(fit.ed < previous && fit.ed > 2, `a = ${a}: ED ${fit.ed}`);
        previous = fit.ed;
    }
});

/** A region given directly on a uniform grid over [0, 1]: center cf(x), constant half-width h. */
function syntheticSpec(cf, h, { G = 400, nseg = 40, flat = 0 } = {}) {
    const grid = Float64Array.from({ length: G }, (_, j) => j / (G - 1));
    return {
        grid, xl: 0, xr: 1, nseg, deg: 3, c: grid.map(cf), h: new Float64Array(G).fill(h),
        kappa: 1e4, epsilon: 1e-2, lambda1: 0, lambda2: 0, flat,
    };
}

const logistic = (x, at, width) => 1 / (1 + Math.exp(-(x - at) / width));

test('sparse ambition finds the bends of a piecewise-linear center and the steps of a stepped one', () => {
    // A join is placed at the coefficient in its middle, so its x is good to about half a segment (1/80 here).
    // Regression: with the zero threshold at δ_d alone, differences the capped weights could not push below it
    // under κ's pull all counted as free, and a* came back as one join of 41 differences (the center).
    const X_TOL = 1 / 80;
    const ED_TOL = 1e-3; // a pinned difference holds at the weight cap, not exactly
    const cases = [
        { cf: x => 3 * Math.abs(x - 0.3) - 5 * Math.max(0, x - 0.7), h: 0.05, flat: 0, order: 2, base: 2 },
        { cf: x => logistic(x, 0.3, 0.02) - logistic(x, 0.7, 0.02), h: 0.1, flat: 1, order: 1, base: 1 },
    ];
    for (const { cf, h, flat, order, base } of cases) {
        const spec = syntheticSpec(cf, h, { flat });
        const star = leavingAmbition(spec, { sparse: true });
        assert.equal(star.capped, false);
        const { fit } = star;
        assert.ok(fit.maxRatio <= 1 + LEAVE_TOL, `order ${order}: ${fit.maxRatio}`);
        assert.equal(fit.settled, true);
        const [pieces] = fit.pieces.orders;
        assert.equal(pieces.order, order);
        assert.equal(pieces.pieces, 3, `order ${order}`);
        assert.deepEqual(pieces.joins.map(j => Math.abs(j.x - 0.3) < X_TOL || Math.abs(j.x - 0.7) < X_TOL),
            [true, true], pieces.joins.map(j => j.x).join(', '));
        // the fit's parameters: the polynomial's, plus one per free difference
        const free = pieces.joins.reduce((a, j) => a + j.to - j.from + 1, 0);
        assert.ok(Math.abs(fit.ed - (base + free)) < ED_TOL, `order ${order}: ED ${fit.ed}, ${free} free`);
    }
});

test('sparse ambition: a line or a constant when one fits; capped then, and a* is the top', () => {
    for (const [cf, flat, ed] of [[x => 2 * x - 1, 0, 2], [() => 3, 1, 1]]) {
        const spec = syntheticSpec(cf, 0.1, { flat });
        const star = leavingAmbition(spec, { sparse: true });
        assert.equal(star.capped, true);
        assert.equal(star.ambition, star.top);
        assert.equal(star.fit.pieces.orders[0].pieces, 1);
        assert.equal(star.fit.pieces.orders[0].joins.length, 0);
        assert.ok(Math.abs(star.fit.ed - ed) < 1e-3, `ED ${star.fit.ed}`);
    }
});

test('sparse ambition at the top of its range gives the polynomial, even far outside the region', () => {
    // sparseMaxAmbition's claim: above it the polynomial is the L0 optimum.
    const spec = syntheticSpec(x => Math.sin(6 * x), 0.05);
    const top = fitSparse({ ...spec, ...sparseLambdas({ ambition: sparseMaxAmbition(spec), G: spec.grid.length }) });
    assert.equal(top.pieces.orders[0].joins.length, 0);
    assert.ok(top.maxRatio > 2, `max ratio ${top.maxRatio}`);
});

test('sparse ambition drops a join the region does not need (backward elimination)', () => {
    // Regression: on this ramp the reweighting settles on four straight pieces at every ambition from 1e-2 to
    // 10^1.5; three fit inside.
    const region = makeRegion('ramp', 0.1);
    const spec = { ...regionSpec(region, { ambition: 0 }), ...sparseLambdas({ ambition: 1, G: region.grid.length }) };
    const unpruned = fitSparse(spec, { prunePasses: 0 });
    const fit = fitSparse(spec);
    assert.equal(unpruned.pieces.orders[0].pieces, 4);
    assert.equal(fit.pieces.orders[0].pieces, 3);
    assert.equal(fit.pruned, 1);
    assert.ok(fit.maxRatio <= 1 + LEAVE_TOL, `${fit.maxRatio}`);
});

test('sparse fits stay inside up to a* and leave just above it', () => {
    for (const [signal, sigma, flat] of [['sine', 0.2, 0], ['blocks', 1, 1]]) {
        const region = makeRegion(signal, sigma);
        const spec = regionSpec(region, { ambition: 0, flat });
        const star = leavingAmbition(spec, { sparse: true });
        assert.equal(star.capped, false, signal);
        const G = region.grid.length;
        const at = fitSparse({ ...spec, ...sparseLambdas({ ambition: star.ambition, flat, G }) });
        assert.ok(at.maxRatio <= 1 + LEAVE_TOL, `${signal}: ${at.maxRatio}`);
        const above = fitSparse({ ...spec, ...sparseLambdas({ ambition: star.ambition * 10 ** 0.05, flat, G }) });
        assert.ok(above.maxRatio > 1 + LEAVE_TOL, `${signal}: ${above.maxRatio}`);
    }
});

const NEUTRAL_SHAPE = { toFlat: 1, extremum: 0, inflection: 0, length: 0 };

test('rounded bends with neutral costs and no length are the sparse fit toward straight, exactly', () => {
    const region = makeRegion('sine', 0.2);
    const spec = regionSpec(region, { ambition: 0 });
    const G = region.grid.length;
    for (const ambition of [1, 100, 1e4]) {
        const plain = fitSparse({ ...spec, ...sparseLambdas({ ambition, G }) });
        const shaped = fitShaped({ ...spec, ...shapedLambdas({ ambition, G }), shape: NEUTRAL_SHAPE });
        assert.deepEqual(shaped.beta, plain.beta, `ambition ${ambition}`);
    }
});

test('a bend length spreads the sharp tops of a sine into wider, gentler bends', () => {
    // Measured (n = 200, 40 segments): at a*, the plain count's 5 bends are one second difference each, the largest
    // |D₂β| 2.48 median half-widths; with length 0.05, bends up to 2 differences wide and the largest 0.97.
    const SOFTER = 0.6; // largest |D₂β| with length 0.05 ÷ the plain count's, at most
    const region = makeRegion('sine', 0.2);
    const spec = regionSpec(region, { ambition: 0 });
    const s = Float64Array.from(region.half).sort()[region.half.length >> 1];
    const sharpest = beta => {
        let m = 0;
        for (let i = 2; i < beta.length; i++)
            m = Math.max(m, Math.abs(beta[i] - 2 * beta[i - 1] + beta[i - 2]) / s);
        return m;
    };
    const plain = leavingAmbition(spec, { sparse: true }).fit;
    const star = leavingAmbition({ ...spec, shape: { ...NEUTRAL_SHAPE, length: 0.05 } }, { shaped: true });
    const fit = star.fit;
    assert.equal(star.capped, false);
    assert.ok(fit.maxRatio <= 1 + LEAVE_TOL, `${fit.maxRatio}`);
    assert.ok(Math.max(...plain.pieces.orders[0].joins.map(j => j.to - j.from + 1)) === 1);
    assert.ok(Math.max(...fit.bends.map(b => b.to - b.from + 1)) >= 2);
    assert.ok(sharpest(fit.beta) < SOFTER * sharpest(plain.beta), `${sharpest(fit.beta)} vs ${sharpest(plain.beta)}`);
    assert.equal(fit.extremes, 4);
});

test('rounded bends report extremes and inflections; a bend that levels off is not an extremum', () => {
    const cases = [
        // slopes −3, 3, −2: a minimum and a maximum, and one change of bending direction between them
        { cf: x => 3 * Math.abs(x - 0.3) - 5 * Math.max(0, x - 0.7), bends: 2, extremes: 2, inflections: 1 },
        // rising, then level
        { cf: x => Math.min(2 * x, 0.8), bends: 1, extremes: 0, inflections: 0 },
    ];
    for (const { cf, bends, extremes, inflections } of cases) {
        const spec = { ...syntheticSpec(cf, 0.05), shape: NEUTRAL_SHAPE };
        const { fit } = leavingAmbition(spec, { shaped: true });
        assert.equal(fit.bends.length, bends);
        assert.equal(fit.extremes, extremes);
        assert.equal(fit.inflections, inflections);
    }
});

test('rounded bends need a straight penalty and valid costs', () => {
    const spec = syntheticSpec(x => x, 0.1);
    assert.throws(() => fitShaped({ ...spec, lambda2: 0 }), RangeError);
    for (const shape of [{ toFlat: 0 }, { toFlat: 1.5 }, { extremum: -1 }, { length: -0.1 }])
        assert.throws(() => fitShaped({ ...spec, lambda2: 1, shape }), RangeError, JSON.stringify(shape));
});

test('an extremum cost makes a fit level off where the region allows, instead of peaking and falling back', () => {
    // A rise to 0.8 at x = 0.4 that then sags by 0.18: with h = 0.1 a fit that rises and stays level fits inside
    const spec = syntheticSpec(x => (x < 0.4 ? 2 * x : 0.8 - 0.3 * (x - 0.4)), 0.1);
    const plain = leavingAmbition({ ...spec, shape: NEUTRAL_SHAPE }, { shaped: true }).fit;
    const { fit } = leavingAmbition({ ...spec, shape: { ...NEUTRAL_SHAPE, extremum: 2 } }, { shaped: true });
    assert.equal(plain.extremes, 1);
    assert.equal(fit.extremes, 0);
    assert.equal(fit.bends.length, 1);
    assert.equal(fit.shapeAccepted, 1);
    assert.ok(fit.maxRatio <= 1 + LEAVE_TOL, `${fit.maxRatio}`);
});

test('extremum and inflection costs leave a forced shape alone, and remove what the region does not force', () => {
    // A rising line with a wiggle of amplitude 0.06 in a region of half-width 0.05: the wiggles are forced, but
    // the fit need not fall between them. Measured: extremes 6 → 0 with both costs, inflections 7 either way.
    const spec = syntheticSpec(x => x + 0.06 * Math.sin(10 * Math.PI * x), 0.05);
    const plain = leavingAmbition({ ...spec, shape: NEUTRAL_SHAPE }, { shaped: true }).fit;
    const { fit } = leavingAmbition({ ...spec, shape: { ...NEUTRAL_SHAPE, extremum: 2, inflection: 1 } },
        { shaped: true });
    assert.ok(plain.extremes >= 4, `plain: ${plain.extremes}`);
    assert.equal(fit.extremes, 0);
    assert.ok(fit.inflections >= plain.inflections - 1, `${fit.inflections} vs ${plain.inflections}`);
    assert.ok(fit.maxRatio <= 1 + LEAVE_TOL, `${fit.maxRatio}`);
});
