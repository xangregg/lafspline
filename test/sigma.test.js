import { test } from 'node:test';
import assert from 'node:assert/strict';
import { median, sigmaMAD, sigmaGSJS, weightedMedian, replicateInfo } from '../src/core/sigma.js';
import { normals, createRng } from '../src/core/rng.js';
import { simulate } from '../src/core/signals.js';

test('median of odd and even arrays, input unchanged', () => {
    const v = [5, 1, 3];
    assert.equal(median(v), 3);
    assert.deepEqual(v, [5, 1, 3]);
    assert.equal(median([4, 1, 3, 2]), 2.5);
});

// At n = 10⁴ the relative sd of σ̂ is ≈ 1.4% (measured), so a single draw lands within 3% only ≈ 97%
// of the time. Check the bias on the average and the 3% band on most seeds.
test('σ̂ ≈ σ on pure noise, n = 10⁴: mean over 50 seeds within 0.5%, ≥ 90% of seeds within 3%', () => {
    const sigma = 2.5;
    const seeds = 50;
    let sumRel = 0;
    let within = 0;
    for (let seed = 1; seed <= seeds; seed++) {
        const y = normals(10000, createRng(seed)).map(z => sigma * z);
        const rel = sigmaMAD(y) / sigma - 1;
        sumRel += rel;
        if (Math.abs(rel) < 0.03)
            within++;
    }
    assert.ok(Math.abs(sumRel / seeds) < 0.005, `mean relative error ${sumRel / seeds}`);
    assert.ok(within / seeds >= 0.9, `${within}/${seeds} within 3%`);
});

test('σ̂ is unaffected by adding a constant and scales with y', () => {
    const y = normals(1000, createRng(5));
    const base = sigmaMAD(y);
    assert.ok(Math.abs(sigmaMAD(y.map(v => v + 100)) - base) < 1e-9);
    assert.ok(Math.abs(sigmaMAD(y.map(v => 3 * v)) - 3 * base) < 1e-12);
});

test('σ̂ is only mildly biased upward by a smooth signal (sin(4πt), σ = 0.2, n = 500)', () => {
    const { y } = simulate('sine', 500, 0.2, 11);
    const est = sigmaMAD(y);
    assert.ok(est > 0.17 && est < 0.25, `σ̂ = ${est}`);
});

test('σ̂ rejects n < 2', () => {
    assert.throws(() => sigmaMAD([1]), RangeError);
});

function unevenX(n, rng) {
    const x = new Float64Array(n);
    for (let i = 1; i < n; i++)
        x[i] = x[i - 1] + 0.1 + 3 * rng.uniform() ** 2;
    return x;
}

test('GSJS σ̂ ≈ σ on pure noise, equal and unequal spacing (mean of 50 seeds within 1.5%)', () => {
    for (const uneven of [false, true]) {
        let sum = 0;
        for (let seed = 1; seed <= 50; seed++) {
            const rng = createRng(seed);
            const x = uneven ? unevenX(2000, rng) : Float64Array.from({ length: 2000 }, (_, i) => i);
            const y = normals(2000, rng).map(v => 0.7 * v);
            sum += sigmaGSJS(x, y) / 0.7;
        }
        assert.ok(Math.abs(sum / 50 - 1) < 0.015, `uneven ${uneven}: mean ratio ${sum / 50}`);
    }
});

test('GSJS σ̂ is unchanged by adding any linear trend a + b·x (tol 1e-9), at unequal spacing', () => {
    const rng = createRng(4);
    const x = unevenX(300, rng);
    const y = normals(300, rng);
    const base = sigmaGSJS(x, y);
    const trended = y.map((v, i) => v + 5 - 40 * x[i]);
    assert.ok(Math.abs(sigmaGSJS(x, trended) - base) < 1e-9 * Math.max(1, base));
    assert.ok(Math.abs(sigmaGSJS(x, y.map(v => 3 * v)) - 3 * base) < 1e-12);
});

test('a steep smooth trend inflates eq. (10) but not GSJS', () => {
    const rng = createRng(6);
    const x = unevenX(200, rng);
    const y = Float64Array.from(x, v => 2 * v + 0.05 * v * v + 0.5 * rng.normal());
    assert.ok(sigmaMAD(y) > 2, `eq. 10 gave ${sigmaMAD(y)}`);
    const g = sigmaGSJS(x, y);
    assert.ok(g > 0.35 && g < 0.7, `GSJS gave ${g}`);
});

test('GSJS needs replicates or three distinct x; replicates alone give a pure-error estimate', () => {
    assert.ok(Number.isFinite(sigmaGSJS([0, 0, 0, 1, 2], [1, 2, 3, 4, 6])));
    assert.throws(() => sigmaGSJS([1, 2], [1, 2]), RangeError);
    // three replicates at one x: pure error only, sd-scaled residuals (−1, 0, 1)·√(3/2)
    const expected = weightedMedian([Math.sqrt(1.5), 0, Math.sqrt(1.5)], [1, 1, 1]) / 0.6744897501960817;
    assert.ok(Math.abs(sigmaGSJS([1, 1, 1], [1, 2, 3]) - expected) < 1e-12);
});

test('weighted median', () => {
    assert.equal(weightedMedian([3, 1, 2], [1, 1, 1]), 2);
    assert.equal(weightedMedian([1, 2, 3], [1, 1, 5]), 3);
    assert.equal(weightedMedian([1, 2, 3, 4], [1, 1, 1, 1]), 2); // lower median
});

// Tied x: order-free, uses replicates, and reduces to the usual estimators without ties.
function tiedDesign(rng, values, maxRep, trend, sigma) {
    const xs = [];
    const ys = [];
    for (let v = 0; v < values; v++) {
        const k = 1 + Math.floor(rng.uniform() * maxRep);
        for (let q = 0; q < k; q++) {
            xs.push(v);
            ys.push(trend(v) + sigma * rng.normal());
        }
    }
    return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

test('without ties, passing x changes nothing (bitwise), for eq. 10', () => {
    const rng = createRng(3);
    const x = unevenX(500, rng);
    const y = normals(500, rng);
    assert.equal(sigmaMAD(y, x), sigmaMAD(y));
});

test('with ties, both estimators are invariant to the order of tied rows', () => {
    const rng = createRng(17);
    const { x, y } = tiedDesign(rng, 60, 5, v => Math.sin(v / 8), 0.5);
    const a = [sigmaMAD(y, x), sigmaGSJS(x, y)];
    const shuffled = Float64Array.from(y);
    for (let i = 1; i < y.length; i++) {
        if (x[i] === x[i - 1] && rng.uniform() < 0.5)
            [shuffled[i], shuffled[i - 1]] = [shuffled[i - 1], shuffled[i]];
    }
    assert.notDeepEqual(shuffled, y);
    assert.equal(sigmaMAD(shuffled, x), a[0]);
    assert.equal(sigmaGSJS(x, shuffled), a[1]);
});

test('with replicates, a steep trend no longer inflates σ̂ (mean of 100 designs within 3% of σ)', () => {
    const rng = createRng(19);
    let mad = 0;
    let gsjs = 0;
    let plain = 0;
    for (let r = 0; r < 100; r++) {
        const { x, y } = tiedDesign(rng, 100, 4, v => 3 * v, 1);
        mad += sigmaMAD(y, x) / 100;
        gsjs += sigmaGSJS(x, y) / 100;
        plain += sigmaMAD(y) / 100;
    }
    assert.ok(plain > 1.4, `plain eq. 10 ${plain} (row order) should be inflated`);
    assert.ok(Math.abs(mad - 1) < 0.03, `tie-aware eq. 10 ${mad}`);
    assert.ok(Math.abs(gsjs - 1) < 0.03, `tie-aware GSJS ${gsjs}`);
});

test('with only a few ties the between-x terms are kept (≈ eq. 10 on pure noise, within 3%)', () => {
    const rng = createRng(23);
    let mean = 0;
    for (let r = 0; r < 100; r++) {
        const xs = [];
        for (let v = 0; v < 300; v++) {
            xs.push(v);
            if (rng.uniform() < 0.05)
                xs.push(v);
        }
        const x = Float64Array.from(xs);
        const y = normals(x.length, rng);
        assert.equal(replicateInfo(x).pureErrorOnly, false);
        mean += sigmaMAD(y, x) / 100;
    }
    assert.ok(Math.abs(mean - 1) < 0.03, `mean ${mean}`);
});


test('0/1 outcomes: when most neighbors are equal the median is 0, and σ̂ falls back to the RMS (Bernoulli sd)', () => {
    // Regression: exam-pass-simulated, a logistic P(1) from 0 to 1; median |Δy| was 0, so Aₙ had σ̂ = 0 and refused,
    // and the noise check divided by 0. Here P(1) = 0.2 everywhere, so the Bernoulli sd is 0.4 (tol 0.04, 4000 draws).
    const rng = createRng(17);
    const n = 4000;
    const x = Float64Array.from({ length: n }, (_, i) => i / n);
    const y = Float64Array.from({ length: n }, () => (rng.uniform() < 0.2 ? 1 : 0));
    for (const [name, s] of [['eq. 10', sigmaMAD(y, x)], ['GSJS', sigmaGSJS(x, y)]])
        assert.ok(Math.abs(s - 0.4) < 0.04, `${name}: ${s}`);
    // with tied x (pairs of replicates), the pooled pseudo-residuals take the same fallback
    const xt = Float64Array.from({ length: n }, (_, i) => Math.floor(i / 2));
    for (const [name, s] of [['eq. 10', sigmaMAD(y, xt)], ['GSJS', sigmaGSJS(xt, y)]])
        assert.ok(Math.abs(s - 0.4) < 0.04, `${name} tied: ${s}`);
    // a constant y still has σ̂ = 0
    assert.equal(sigmaMAD(new Float64Array(10).fill(1)), 0);
    assert.equal(sigmaGSJS(Float64Array.from({ length: 10 }, (_, i) => i), new Float64Array(10).fill(1)), 0);
});
