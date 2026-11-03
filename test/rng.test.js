import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng, normals } from '../src/core/rng.js';

const N = 200000; // draws for moment checks

test('same seed gives the same stream; different seeds differ', () => {
    const a = createRng(42);
    const b = createRng(42);
    const c = createRng(43);
    const sa = Array.from({ length: 10 }, () => a.uniform());
    const sb = Array.from({ length: 10 }, () => b.uniform());
    const sc = Array.from({ length: 10 }, () => c.uniform());
    assert.deepEqual(sa, sb);
    assert.notDeepEqual(sa, sc);
});

test('uniform is in [0, 1) with mean 1/2 and variance 1/12 (tol 0.005)', () => {
    const rng = createRng(1);
    let sum = 0;
    let ss = 0;
    for (let i = 0; i < N; i++) {
        const u = rng.uniform();
        assert.ok(u >= 0 && u < 1);
        sum += u;
        ss += u * u;
    }
    const mean = sum / N;
    assert.ok(Math.abs(mean - 0.5) < 0.005, `mean ${mean}`);
    assert.ok(Math.abs(ss / N - mean * mean - 1 / 12) < 0.005);
});

test('normal has mean 0, sd 1 (tol 0.01) and plausible tails', () => {
    const z = normals(N, createRng(7));
    let sum = 0;
    let ss = 0;
    let beyond2 = 0;
    for (const v of z) {
        sum += v;
        ss += v * v;
        if (Math.abs(v) > 2)
            beyond2++;
    }
    const mean = sum / N;
    const sd = Math.sqrt(ss / N - mean * mean);
    assert.ok(Math.abs(mean) < 0.01, `mean ${mean}`);
    assert.ok(Math.abs(sd - 1) < 0.01, `sd ${sd}`);
    // P(|Z| > 2) = 0.0455; binomial sd ≈ 0.00047 at this N
    assert.ok(Math.abs(beyond2 / N - 0.0455) < 0.002, `tail ${beyond2 / N}`);
});
