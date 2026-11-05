import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { noiseStatistics, checkNoise } from '../src/core/noisecheck.js';
import { selectLambda, fitPSpline, defaultSegments } from '../src/core/pspline.js';
import { readDataset } from '../src/core/datasets.js';
import { simulate } from '../src/core/signals.js';

const UNDERSMOOTH = 0.25; // as the app's bands

function example(id) {
    return readDataset(readFileSync(new URL(`../data/${id}.csv`, import.meta.url), 'utf8'));
}

// The band's fit as the app makes it: λ by GCV, undersmoothed.
function check(x, y, nseg) {
    const lambda = selectLambda(x, y, { nseg }).lambda * UNDERSMOOTH;
    const fit = fitPSpline(x, y, { nseg, lambda });
    return checkNoise(x, y, fit.fitted, fit.sigma);
}

test('noiseStatistics: lag-1 autocorrelation of the residuals\' means at distinct x', () => {
    const x = [0, 0, 1, 2, 3];
    const y = [1, 3, -1, 2, 0];
    const fitted = [0, 0, 0, 0, 0];
    // x-group means of the residuals: 2, −1, 2, 0 → Σ m_k m_{k−1} = −2 − 2 + 0 = −4, Σ m_k² = 9
    const s = noiseStatistics(x, y, fitted, 1);
    assert.equal(s.groups, 4);
    assert.ok(Math.abs(s.lag1 - -4 / 9) < 1e-12, `lag1 ${s.lag1}`);
    assert.equal(s.ratio, 1 / s.reference);
});

test('checkNoise: fewer than 10 distinct x leaves r₁ unjudged', () => {
    const d = example('fibers-distance'); // 5 distinct x
    const c = check(d.x, d.y, defaultSegments(d.m));
    assert.equal(c.lagLimits, null);
    assert.equal(c.verdict, 'ok');
});

test('checkNoise: smooth curves with independent noise pass', () => {
    for (const name of ['sine', 'ramp', 'line', 'noise']) {
        const s = simulate(name, 500, 0.2, 1);
        const c = check(s.x, s.y, defaultSegments(500));
        assert.equal(c.verdict, 'ok', `${name}: ratio ${c.ratio}, r1 ${c.lag1}`);
        assert.ok(Math.abs(c.ratio - 1) < 0.1, `${name}: ratio ${c.ratio} within 0.1 of 1`);
    }
});

test('checkNoise: GCV interpolating Tukey coal at 80 segments (49 points) is caught', () => {
    const d = example('tukey-coal');
    const c = check(d.x, d.y, 80);
    assert.equal(c.verdict, 'follows');
    assert.ok(c.ratio < 0.1, `ratio ${c.ratio}`);
    // at its default segments it passes
    assert.equal(check(d.x, d.y, defaultSegments(d.m)).verdict, 'ok');
});

test('checkNoise: a basis too coarse for jumps leaves residuals in runs (Blocks, HeaviSine)', () => {
    const s = simulate('blocks', 1024, 1, 1);
    const c = check(s.x, s.y, 40);
    assert.equal(c.verdict, 'runs');
    assert.ok(c.ratio > 2 && c.lag1 > 0.5, `ratio ${c.ratio}, r1 ${c.lag1}`);
    // HeaviSine's two jumps, once the noise is small enough to show them (measured: ratio 1.89, r₁ 0.58)
    const h = simulate('heavisine', 500, 0.2, 1);
    assert.equal(check(h.x, h.y, 40).verdict, 'runs');
});

test('checkNoise: ozone\'s replicates agree more closely than its years do', () => {
    const d = example('ozone');
    const c = check(d.x, d.y, defaultSegments(d.m));
    assert.equal(c.verdict, 'between');
    // measured: ratio 1.85, r₁ −0.75
    assert.ok(c.ratio > 1.5 && c.lag1 < -0.6, `ratio ${c.ratio}, r1 ${c.lag1}`);
});

test('checkNoise: larger residuals without runs read as spread (dog ages)', () => {
    const d = example('dog-ages');
    const c = check(d.x, d.y, defaultSegments(d.m));
    assert.equal(c.verdict, 'spread');
    // measured: ratio 1.59 (limit 1.51), r₁ 0.04
    assert.ok(c.ratio > c.ratioLimits[1] && Math.abs(c.lag1) < 0.2, `ratio ${c.ratio}, r1 ${c.lag1}`);
});
