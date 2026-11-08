import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { noiseStatistics, checkNoise } from '../src/core/noisecheck.js';
import { selectLambda, fitPSpline, defaultSegments } from '../src/core/pspline.js';
import { readDataset } from '../src/core/datasets.js';
import { simulate } from '../src/core/signals.js';
import { createRng } from '../src/core/rng.js';

const UNDERSMOOTH = 0.25; // as the app's bands
const YEARS = 47; // distinct x of the replicated example
const REPLICATES = 25; // observations at each

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
    // 5 distinct x with 9 replicates each (the shape of the fibers data once bundled), a gentle rise, small noise
    const levels = [16, 24, 41, 67, 115];
    const rng = createRng(2);
    const x = Array.from({ length: 45 }, (_, i) => levels[Math.floor(i / 9)]);
    const y = x.map(v => 3.4 + 0.35 * (1 - Math.exp(-v / 30)) + 0.02 * rng.normal());
    const c = check(x, y, defaultSegments(levels.length));
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

test('checkNoise: replicates that agree more closely than their x values do read as between', () => {
    // 47 "years" of 25 replicates each, every year with its own offset (sd 1) on top of replicate noise (sd 0.5),
    // as daily values recorded at the year's x: the pattern of the daily Antarctic ozone set, since removed
    // (measured there: ratio 1.85, r₁ −0.75)
    const rng = createRng(1);
    const x = [];
    const y = [];
    for (let k = 0; k < YEARS; k++) {
        const offset = rng.normal();
        for (let r = 0; r < REPLICATES; r++) {
            x.push(k);
            y.push(3 * Math.sin(k / 8) + offset + 0.5 * rng.normal());
        }
    }
    const c = check(x, y, defaultSegments(YEARS));
    assert.equal(c.verdict, 'between');
    // measured: ratio 1.69, r₁ −0.80
    assert.ok(c.ratio > 1.5 && c.lag1 < -0.6, `ratio ${c.ratio}, r1 ${c.lag1}`);
});

test('checkNoise: larger residuals without runs read as spread (Gapminder\'s low outliers)', () => {
    // dog ages served here until its data were digitized anew (now ratio 1.47, within the limit 1.54: ok)
    const d = example('gapminder-2007');
    const c = check(d.x, d.y, defaultSegments(d.m));
    assert.equal(c.verdict, 'spread');
    // measured: ratio 1.64 (limit 1.40), r₁ 0.03
    assert.ok(c.ratio > c.ratioLimits[1] && Math.abs(c.lag1) < 0.2, `ratio ${c.ratio}, r1 ${c.lag1}`);
});
