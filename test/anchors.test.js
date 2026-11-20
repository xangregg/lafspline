import { test } from 'node:test';
import assert from 'node:assert/strict';
import loadHighs from 'highs';
import { makeAnchors, boundBand } from '../src/core/anchors.js';
import {
    fitIterated, buildQP, qpResult, regionSpec, leavingAmbition, fitShaped, shapedLambdas, fitSparse, sparseLambdas,
    splineValues, maxAmbition, ambitionLambdas, pointRows,
} from '../src/core/fit.js';
import { bandRegion } from '../src/core/band.js';
import { selectLambda, evaluate, standardErrors, bsplineBasis } from '../src/core/pspline.js';
import { simulate } from '../src/core/signals.js';
import { solveModel } from '../src/core/lp-solve.js';
import { createRng, normals } from '../src/core/rng.js';

const SEED = 20261120;
const Q_CRIT = 3; // stand-in critical value: these tests need a region, not its coverage
const ANCHOR_TOL = 1e-9; // |g(x_a) − y_a| relative to the median half-width: the equalities are exact

/** A region from a GCV P-spline fit, as regionSpec takes it. */
function makeRegion(signal, sigma, { n = 200, G = 200, nseg = 20 } = {}) {
    const { x, y } = simulate(signal, n, sigma, SEED);
    const phase1 = selectLambda(x, y, { nseg });
    const xl = x[0];
    const xr = x[n - 1];
    const grid = Float64Array.from({ length: G }, (_, j) => xl + (xr - xl) * j / (G - 1));
    const center = evaluate(phase1.fit, grid);
    const half = standardErrors(phase1.fit, grid).map(v => Q_CRIT * v);
    return { grid, center, half, params: { xl, xr, nseg } };
}

function median(v) {
    const s = Float64Array.from(v).sort();
    return s[s.length >> 1];
}

test('makeAnchors sorts by x and rejects repeated x, non-finite values and too many points', () => {
    const a = makeAnchors([[3, 1], [-1, 2]]);
    assert.deepEqual(Array.from(a.x), [-1, 3]);
    assert.deepEqual(Array.from(a.y), [2, 1]);
    assert.equal(makeAnchors([]), null);
    assert.throws(() => makeAnchors([[1, 0], [1, 2]]), /two anchors/);
    assert.throws(() => makeAnchors([[NaN, 0]]), RangeError);
    assert.throws(() => makeAnchors(Array.from({ length: 21 }, (_, k) => [k, 0])), RangeError);
});

test('beyond the knot range a spline continues as its tangent line at the nearer end', () => {
    // a cubic on 7 segments over [1, 3]: value and slope at each end against a central difference inside
    const knots = { xl: 1, xr: 3, nseg: 7, deg: 3 };
    const beta = Float64Array.from({ length: 10 }, (_, k) => Math.sin(1.3 * k) + 0.1 * k * k);
    const h = 1e-6;
    for (const [end, out] of [[1, -0.5], [3, 4.25]]) {
        const inner = end === 1 ? end + h : end - h;
        const [g, gIn] = splineValues(beta, [end, inner], knots);
        const slope = (g - gIn) / (end - inner);
        const [far] = splineValues(beta, [out], knots);
        const want = g + (out - end) * slope;
        assert.ok(Math.abs(far - want) < 1e-5 * (1 + Math.abs(want)), `${end}: ${far} vs ${want}`);
    }
    // inside, the rows are the basis
    const inside = pointRows([1, 2.2, 3], knots);
    assert.deepEqual(Array.from(inside.width), [4, 4, 4]);
    // a line in the coefficients is a line in x, and its tangent is itself
    const line = Float64Array.from({ length: 10 }, (_, k) => 2 - 0.5 * k);
    const [a, b, c] = splineValues(line, [-2, 2, 6], knots);
    assert.ok(Math.abs((b - a) - (c - b)) < 1e-12);
});

test('boundBand intersects each box with the bounds and clips the target into it', () => {
    const band = {
        grid: Float64Array.of(0, 1, 2),
        center: Float64Array.of(0.1, 0.5, 0.95),
        half: Float64Array.of(0.2, 0.1, 0.1),
        target: Float64Array.of(-0.05, 0.5, 1.02),
    };
    const b = boundBand(band, { lower: 0, upper: 1 });
    assert.equal(b.clipped, 2);
    const close = (got, want) => got.forEach((v, j) => assert.ok(Math.abs(v - want[j]) < 1e-12, `${got} vs ${want}`));
    close(b.lower, [0, 0.4, 0.85]);
    close(b.upper, [0.3, 0.6, 1]);
    close(b.center, [0.15, 0.5, 0.925]);
    close(b.half, [0.15, 0.1, 0.075]);
    close(b.target, [0, 0.5, 1]);
    // one-sided
    assert.equal(boundBand(band, { lower: 0, upper: null }).clipped, 1);
    assert.equal(boundBand(band, {}).clipped, 0);
});

test('boundBand refuses bounds the region lies outside of', () => {
    const band = { grid: Float64Array.of(0, 1), center: Float64Array.of(0.5, 1.5), half: Float64Array.of(0.1, 0.2) };
    assert.throws(() => boundBand(band, { upper: 1 }), /above the upper bound 1 at x = 1/);
    assert.throws(() => boundBand(band, { lower: 0.7 }), /below the lower bound 0.7 at x = 0/);
    assert.throws(() => boundBand(band, { lower: 1, upper: 0 }), RangeError);
    // touching the bound in one point leaves no width to fit in
    assert.throws(() => boundBand(band, { upper: 1.3 + 1e-6 }), /barely/);
});

test('an anchored fit passes through its anchors exactly, and the iteration and the HiGHS QP agree', async () => {
    const G_TOL = 1e-4; // max |Δg|/h: HiGHS's own tolerance
    const highs = await loadHighs();
    const region = makeRegion('sine', 0.2);
    const s = median(region.half);
    for (const pairs of [[[0.5, 0.1]], [[0.2, 0.5], [0.8, -0.5]], [[region.params.xl, 0], [region.params.xr, 0]]]) {
        const anchors = makeAnchors(pairs);
        const spec = regionSpec(region, { ambition: 1e-3, anchors });
        const it = fitIterated(spec);
        assert.equal(it.converged, true);
        assert.ok(it.anchorGap < ANCHOR_TOL * s, `gap ${it.anchorGap}`);
        const at = splineValues(it.beta, anchors.x, spec);
        at.forEach((v, a) => assert.ok(Math.abs(v - anchors.y[a]) < ANCHOR_TOL * s));
        const qp = qpResult(spec, solveQP(highs, spec));
        assert.equal(qp.status, 'optimal');
        let worst = 0;
        for (let j = 0; j < it.g.length; j++)
            worst = Math.max(worst, Math.abs(it.g[j] - qp.g[j]) / region.half[j]);
        assert.ok(worst < G_TOL, `iteration vs QP ${worst}`);
        assert.ok(it.objective <= qp.objective * (1 + 1e-9));
    }
});

function solveQP(highs, spec) {
    return solveModel(highs, buildQP(spec).model);
}

test('great ambition toward straight with one anchor: the weighted least-squares line through it', () => {
    // With D₂ at the top of the range the fit is a line through the anchor; the center pull ε·Σ((g − c)/h)²
    // then picks its slope: b = Σ w(x − x_a)(c − y_a) / Σ w(x − x_a)², w = 1/h².
    const LINE_TOL = 1e-4; // relative to the median half-width
    const region = makeRegion('line', 0.2);
    const { grid, center, half } = region;
    const xa = 0.3;
    const ya = center[Math.round(0.3 * (grid.length - 1))] + 0.5 * half[0];
    const anchors = makeAnchors([[xa, ya]]);
    const base = regionSpec(region, { ambition: 0, anchors });
    const top = maxAmbition(base);
    const fit = fitIterated({ ...base, ...ambitionLambdas({ ambition: top, G: grid.length, nseg: base.nseg }) });
    let num = 0;
    let den = 0;
    for (let j = 0; j < grid.length; j++) {
        const w = 1 / (half[j] * half[j]);
        num += w * (grid[j] - xa) * (center[j] - ya);
        den += w * (grid[j] - xa) ** 2;
    }
    const b = num / den;
    const s = median(half);
    let worst = 0;
    for (let j = 0; j < grid.length; j++)
        worst = Math.max(worst, Math.abs(fit.g[j] - (ya + b * (grid[j] - xa))) / s);
    assert.ok(worst < LINE_TOL, `off the line by ${worst}`);
    // a line through one fixed point has one free parameter
    assert.ok(Math.abs(fit.ed - 1) < 0.01, `ED ${fit.ed}`);
    const free = fitIterated({ ...regionSpec(region, { ambition: 0 }),
        ...ambitionLambdas({ ambition: top, G: grid.length, nseg: base.nseg }) });
    assert.ok(Math.abs(free.ed - 2) < 0.01, `ED without the anchor ${free.ed}`);
});

test('each anchor spends one effective dimension at no ambition', () => {
    const region = makeRegion('sine', 0.2);
    const plain = fitIterated(regionSpec(region, { ambition: 0 }));
    const two = fitIterated(regionSpec(region, { ambition: 0, anchors: makeAnchors([[0.25, 1], [0.75, -1]]) }));
    assert.ok(Math.abs(plain.ed - (two.ed + 2)) < 1e-6, `${plain.ed} vs ${two.ed}`);
});

test('an anchor beyond the data ties the end\'s tangent line, and the basis is unchanged', () => {
    const region = makeRegion('line', 0.2);
    const { xl, xr, nseg } = region.params;
    const xa = xl - 0.3 * (xr - xl);
    const anchors = makeAnchors([[xa, 0]]);
    const spec = regionSpec(region, { ambition: 1e-2, anchors });
    assert.equal(spec.nseg, nseg);
    assert.equal(spec.xl, xl);
    const fit = fitIterated(spec);
    const s = median(region.half);
    assert.ok(fit.anchorGap < ANCHOR_TOL * s);
    assert.ok(Math.abs(splineValues(fit.beta, [xa], spec)[0]) < ANCHOR_TOL * s);
});

test('anchors far beyond the data: the counting penalties\' unpenalized solves stay well posed', () => {
    // Regression: a basis widened to reach anchors 10 x ranges away lost positive definiteness (shaped bends,
    // the a* scan).
    const region = makeRegion('sine', 0.2);
    const anchors = makeAnchors([[-10, 0], [11, 0]]);
    const spec = { ...regionSpec(region, { ambition: 0, anchors }), shape: {} };
    const star = leavingAmbition(spec, { shaped: true });
    assert.ok(star.fit.anchorGap < ANCHOR_TOL * median(region.half), `gap ${star.fit.anchorGap}`);
});

test('the ambition-counting fits keep their anchors through reweighting and pinned refits', () => {
    const region = makeRegion('sine', 0.2);
    const anchors = makeAnchors([[0, 0], [1, 0]]);
    const s = median(region.half);
    const base = regionSpec(region, { ambition: 0, anchors });
    const G = region.grid.length;
    const sparse = fitSparse({ ...base, ...sparseLambdas({ ambition: 1e-3, G }) });
    assert.ok(sparse.anchorGap < ANCHOR_TOL * s, `sparse gap ${sparse.anchorGap}`);
    const shaped = fitShaped({ ...base, ...shapedLambdas({ ambition: 1e-3, G }) });
    assert.ok(shaped.anchorGap < ANCHOR_TOL * s, `shaped gap ${shaped.anchorGap}`);
});

test('an anchor the region does not allow: no ambition keeps the fit inside, and a* says so', () => {
    const region = makeRegion('line', 0.2);
    const j = region.grid.length >> 1;
    const anchors = makeAnchors([[region.grid[j], region.center[j] + 3 * region.half[j]]]);
    const star = leavingAmbition(regionSpec(region, { ambition: 0, anchors }));
    assert.equal(star.unreachable, true);
    assert.ok(star.fit.maxRatio > 1);
});

test('too many anchors for the basis, or anchors too close together, are refused', () => {
    const region = makeRegion('line', 0.2, { nseg: 5 });
    // within the data, so the basis keeps its 5 + 3 coefficients
    const many = makeAnchors(Array.from({ length: 8 }, (_, k) => [0.2 + 0.6 * k / 7, 0]));
    assert.throws(() => fitIterated(regionSpec(region, { ambition: 0, anchors: many })), /8 anchors for 8/);
    const close = makeAnchors([[0.5, 0], [0.5 + 1e-13, 1]]);
    assert.throws(() => fitIterated(regionSpec(region, { ambition: 0, anchors: close })), /too close/);
});

test('a bounded band: the fit stays within the bounds at the check grid', () => {
    // a rise from 0 to 1 whose band pokes past both ends
    const x = Float64Array.from({ length: 120 }, (_, i) => i / 119);
    const noise = normals(x.length, createRng(SEED));
    const y = x.map((v, i) => 1 / (1 + Math.exp(-12 * (v - 0.5))) + 0.08 * noise[i]);
    const band = bandRegion(x, y, { seed: SEED });
    assert.ok(band.upper.some(v => v > 1) && band.lower.some(v => v < 0), 'the test needs a band past both bounds');
    const bounded = { ...band, ...boundBand(band, { lower: 0, upper: 1 }) };
    const fit = fitIterated(regionSpec(bounded, { ambition: 1e-6 }));
    const slack = 1e-3; // the soft region: violations shrink like 1/κ
    assert.ok(fit.g.every(v => v >= -slack && v <= 1 + slack), `range ${Math.min(...fit.g)} – ${Math.max(...fit.g)}`);
    assert.ok(fit.maxRatio < 1 + 1e-3);
});
