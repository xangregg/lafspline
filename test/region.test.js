import { test } from 'node:test';
import assert from 'node:assert/strict';
import { threshold, checkRegion, constraintBoxes } from '../src/core/region.js';
import { multiresolution, allIntervals } from '../src/core/intervals.js';
import { sigmaMAD } from '../src/core/sigma.js';
import { simulate } from '../src/core/signals.js';
import { normals, createRng } from '../src/core/rng.js';

const TAU = 3; // paper default

function constantMean(y) {
    const m = y.reduce((a, b) => a + b, 0) / y.length;
    return new Float64Array(y.length).fill(m);
}

test('threshold c = σ√(τ log n)', () => {
    assert.ok(Math.abs(threshold(2, 3, 500) - 2 * Math.sqrt(3 * Math.log(500))) < 1e-12);
});

test('g = y is always in Aₙ', () => {
    for (const name of ['blocks', 'doppler', 'noise']) {
        const { y } = simulate(name, 300, 1, 9);
        const r = checkRegion(y, y, multiresolution(300), threshold(sigmaMAD(y), TAU, 300));
        assert.ok(r.ok);
        assert.equal(r.maxRatio, 0);
    }
});

test('w(I) and ratios match a direct computation (tol 1e-12)', () => {
    const rng = createRng(4);
    const n = 40;
    const y = normals(n, rng);
    const g = normals(n, rng);
    const intervals = allIntervals(n);
    const c = 1.7;
    const r = checkRegion(y, g, intervals, c);
    let maxRatio = 0;
    intervals.forEach(({ lo, hi }, k) => {
        let s = 0;
        for (let i = lo; i <= hi; i++)
            s += y[i] - g[i];
        const w = s / Math.sqrt(hi - lo + 1);
        assert.ok(Math.abs(r.w[k] - w) < 1e-12);
        assert.ok(Math.abs(r.ratios[k] - Math.abs(w) / c) < 1e-12);
        maxRatio = Math.max(maxRatio, Math.abs(w) / c);
    });
    assert.ok(Math.abs(r.maxRatio - maxRatio) < 1e-12);
    assert.equal(r.ratios[r.argMax], r.maxRatio);
    assert.equal(r.violations, Array.from(r.ratios).filter(v => v > 1).length);
});

test('a constant misses Blocks (σ = 1, n = 1024): the region detects signal', () => {
    const { y } = simulate('blocks', 1024, 1, 1);
    const r = checkRegion(y, constantMean(y), multiresolution(1024), threshold(sigmaMAD(y), TAU, 1024));
    assert.ok(!r.ok);
    assert.ok(r.violations > 10, `${r.violations} violations`);
});

test('tol admits solutions marginally outside the boundary', () => {
    const y = Float64Array.from([1, 0]);
    const g = Float64Array.from([0, 0]);
    // singleton [0,0] has |w| = 1, so c = 1 sits exactly on the boundary
    assert.ok(checkRegion(y, g, [{ lo: 0, hi: 0 }], 1 - 1e-9).ok === false);
    assert.ok(checkRegion(y, g, [{ lo: 0, hi: 0 }], 1 - 1e-9, 1e-7).ok);
});

test('constant mean(y) is in Aₙ on pure noise for ≥ 95% of seeds (n = 500, τ = 3, σ̂)', () => {
    const n = 500;
    const seeds = 400;
    const intervals = multiresolution(n);
    let covered = 0;
    for (let seed = 1; seed <= seeds; seed++) {
        const { y } = simulate('noise', n, 1, seed);
        if (checkRegion(y, constantMean(y), intervals, threshold(sigmaMAD(y), TAU, n)).ok)
            covered++;
    }
    const rate = covered / seeds;
    console.log(`# constant-in-Aₙ rate on pure noise: ${(100 * rate).toFixed(1)}% of ${seeds} seeds`);
    assert.ok(rate >= 0.95, `rate ${rate}`);
});

test('constraint boxes: g’s mean is inside a box exactly when that constraint holds', () => {
    const rng = createRng(21);
    const n = 64;
    const y = normals(n, rng);
    const g = normals(n, rng).map(v => 0.3 * v);
    const intervals = multiresolution(n);
    const c = 1.1;
    const boxes = constraintBoxes(y, g, intervals, c);
    const check = checkRegion(y, g, intervals, c);
    intervals.forEach(({ lo, hi }, k) => {
        let sy = 0;
        let sg = 0;
        for (let i = lo; i <= hi; i++) {
            sy += y[i];
            sg += g[i];
        }
        const len = hi - lo + 1;
        assert.ok(Math.abs(boxes.yMean[k] - sy / len) < 1e-12);
        assert.ok(Math.abs(boxes.gMean[k] - sg / len) < 1e-12);
        assert.ok(Math.abs(boxes.half[k] - c / Math.sqrt(len)) < 1e-15);
        assert.ok(Math.abs(boxes.ratios[k] - check.ratios[k]) < 1e-12);
        const inside = Math.abs(boxes.gMean[k] - boxes.yMean[k]) <= boxes.half[k] * (1 + 1e-12);
        assert.equal(inside, check.ratios[k] <= 1 + 1e-12, `interval ${k}`);
    });
});
