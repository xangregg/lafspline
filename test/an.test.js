import { test } from 'node:test';
import assert from 'node:assert/strict';
import loadHighs from 'highs';
import { anBoxes } from '../src/core/an.js';
import { anRegion, checkFit } from '../src/core/regions.js';
import {
    regionSpec, fitIterated, leavingAmbition, splineValues, buildQP, qpResult, checkRegion, LEAVE_TOL,
} from '../src/core/fit.js';
import { ladder } from '../src/core/ladder.js';
import { solveModel } from '../src/core/lp-solve.js';
import { simulate } from '../src/core/signals.js';
import { createRng, normals } from '../src/core/rng.js';

const SEED = 20261103;
const REL_TOL = 1e-12; // relative, on quantities computed two ways in floating point
const NSEG = 20;

function setup(signal, n, sigma, opts = {}) {
    const sim = simulate(signal, n, sigma, SEED);
    const x = opts.x ?? sim.x;
    const region = anRegion(x, sim.y, { family: opts.family });
    return { x, y: sim.y, region, boxes: anBoxes(region, x, { nseg: opts.nseg ?? NSEG }) };
}

function maxRelDiff(a, b) {
    let m = 0;
    for (let k = 0; k < a.length; k++)
        m = Math.max(m, Math.abs(a[k] - b[k]) / Math.max(1, Math.abs(b[k])));
    return m;
}

test('Aₙ boxes give Aₙ\'s own ratios |w(I)|/c for any spline (blocks, overlapping, ties)', () => {
    const tiedX = Float64Array.from({ length: 120 }, (_, i) => Math.floor(i / 3) / 40);
    const cases = [
        setup('sine', 150, 0.3),
        setup('ramp', 150, 0.2, { family: { kind: 'overlapping', lambda: 2, offsets: 4 } }),
        setup('ramp', 120, 0.2, { x: tiedX }),
    ];
    const rng = createRng(1);
    for (const { x, region, boxes } of cases) {
        for (let trial = 0; trial < 3; trial++) {
            const beta = normals(NSEG + 3, rng);
            const g = splineValues(beta, x, boxes.params);
            const ours = checkRegion(boxRowValues(boxes, beta), boxes.center, boxes.half);
            const theirs = checkFit(region, g, 0);
            assert.ok(Math.abs(ours.maxRatio - theirs.maxRatio) <= REL_TOL * theirs.maxRatio,
                `${ours.maxRatio} vs ${theirs.maxRatio}`);
            assert.equal(ours.outside, theirs.violations);
        }
    }
});

/** The boxes' rows applied to β: mean_I of the spline, per interval. */
function boxRowValues(boxes, beta) {
    const { rows } = boxes;
    const g = new Float64Array(rows.n);
    for (let k = 0; k < rows.n; k++) {
        for (let r = 0; r < rows.width[k]; r++)
            g[k] += rows.values[rows.offset[k] + r] * beta[rows.first[k] + r];
    }
    return g;
}

test('the interval rows are means of the spline over the interval\'s observations', () => {
    const { x, region, boxes } = setup('sine', 100, 0.3);
    const beta = normals(NSEG + 3, createRng(2));
    const g = splineValues(beta, x, boxes.params);
    const means = Float64Array.from(region.lo, (lo, k) => {
        let sum = 0;
        for (let i = lo; i <= region.hi[k]; i++)
            sum += g[i];
        return sum / (region.hi[k] - lo + 1);
    });
    assert.ok(maxRelDiff(boxRowValues(boxes, beta), means) <= REL_TOL);
});

test('phase 2 in Aₙ: iterated weights and the HiGHS QP find the same fit', async () => {
    const G_TOL = 1e-4; // max |Δ mean_I g|/h_I: HiGHS's QP stops at its own tolerance
    const highs = await loadHighs();
    const { boxes } = setup('heavisine', 120, 1);
    const spec = regionSpec(boxes, { ambition: 1e-4 });
    const it = fitIterated(spec);
    const qp = qpResult(spec, solveModel(highs, buildQP(spec).model));
    assert.equal(it.converged, true);
    assert.equal(qp.status, 'optimal');
    let worst = 0;
    for (let k = 0; k < it.g.length; k++)
        worst = Math.max(worst, Math.abs(it.g[k] - qp.g[k]) / boxes.half[k]);
    assert.ok(worst < G_TOL, `${worst}`);
});

test('a* in Aₙ: inside to 1% there, outside just above; checked at the data as Aₙ checks', () => {
    const { x, region, boxes } = setup('sine', 300, 0.3);
    const star = leavingAmbition(regionSpec(boxes, { ambition: 0 }));
    assert.equal(star.capped, false);
    assert.equal(star.unreachable, false);
    const atData = checkFit(region, splineValues(star.fit.beta, x, boxes.params), 0);
    assert.ok(atData.maxRatio <= 1 + LEAVE_TOL, `${atData.maxRatio}`);
    assert.ok(Math.abs(atData.maxRatio - star.fit.maxRatio) <= REL_TOL * atData.maxRatio);
    const above = fitIterated(regionSpec(boxes, { ambition: star.ambition * 10 ** 0.05 }));
    assert.ok(above.maxRatio > 1 + LEAVE_TOL, `${above.maxRatio}`);
});

test('in Aₙ the ladder and phase 2 agree: a line fits a noisy line\'s Aₙ, not a sine\'s', () => {
    const line = setup('line', 200, 0.5);
    const [, lineRung] = ladder(line.boxes);
    assert.equal(lineRung.fits, true, `${lineRung.ratio}`);
    // the ladder's ratio is Aₙ's ratio for that line at the data
    const g = Float64Array.from(line.x, v => lineRung.intercept + lineRung.slope * (v - lineRung.x0));
    assert.ok(Math.abs(checkFit(line.region, g, 0).maxRatio - lineRung.ratio) <= 1e-9 * lineRung.ratio);
    assert.equal(leavingAmbition(regionSpec(line.boxes, { ambition: 0 })).capped, true);

    const sine = setup('sine', 200, 0.3);
    assert.ok(ladder(sine.boxes).every(r => !r.fits));
});

test('when no fit in the basis gets inside Aₙ, a* says so instead of a value', () => {
    // Blocks' jumps are sharper than a cubic spline on 20 segments can follow at Aₙ's shortest intervals.
    const { boxes } = setup('blocks', 512, 1);
    const star = leavingAmbition(regionSpec(boxes, { ambition: 0 }));
    assert.equal(star.unreachable, true);
    assert.ok(star.fit.maxRatio > 1 + LEAVE_TOL);
    assert.equal(star.fits, 1);
});

test('regionSpec refuses other segments for Aₙ\'s rows', () => {
    const { boxes } = setup('sine', 100, 0.3);
    assert.throws(() => regionSpec(boxes, { ambition: 0, nseg: NSEG + 1 }), RangeError);
});

test('Aₙ over data with a gap in x: no singular system without ambition or in a sparse fit', () => {
    // Regression: a coefficient whose support holds no observation was free, and Cholesky failed (chess recall).
    const x = Float64Array.from({ length: 140 }, (_, i) => (i < 70 ? i / 70 * 0.35 : 0.65 + (i - 70) / 69 * 0.35));
    const { y } = simulate('sine', 140, 0.3, SEED);
    const region = anRegion(x, y);
    const boxes = anBoxes(region, x, { nseg: NSEG });
    const spec = regionSpec(boxes, { ambition: 0 });
    const free = fitIterated(spec);
    assert.ok(free.beta.every(Number.isFinite));
    const sparse = leavingAmbition(spec, { sparse: true });
    assert.ok(sparse.fit.beta.every(Number.isFinite));
    assert.ok(sparse.fit.maxRatio <= 1 + LEAVE_TOL || sparse.unreachable);
});

test('Aₙ with few distinct x: sparse fits never meet a singular system (pins swamping the floor)', () => {
    // Regression: fibers distance, 5 distinct x for 8 coefficients. Pinned differences, 1e22 times the penalty floor
    // when it was relative to the center pull alone, made a pruning refit's system numerically singular.
    const levels = [16, 24, 41, 67, 115];
    const x = Float64Array.from({ length: 45 }, (_, i) => levels[Math.floor(i / 9)]);
    const rng = createRng(3);
    const y = Float64Array.from(x, v => 30 - 0.1 * v + rng.normal());
    const region = anRegion(x, y);
    const boxes = anBoxes(region, x, { nseg: 5 });
    for (const flat of [0, 1]) {
        const star = leavingAmbition(regionSpec(boxes, { ambition: 0, flat }), { sparse: true });
        assert.ok(star.fit.beta.every(Number.isFinite), `flat ${flat}`);
    }
});
