import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SIGNALS, designPoints, signalValues, signalFunction, simulate } from '../src/core/signals.js';

function sd(v) {
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
}

test('design points are i/n, i = 1..n', () => {
    const t = designPoints(4);
    assert.deepEqual(Array.from(t), [0.25, 0.5, 0.75, 1]);
});

test('Donoho–Johnstone signals are rescaled to sd 7 (tol 1e-9)', () => {
    for (const name of ['blocks', 'bumps', 'heavisine', 'doppler'])
        assert.ok(Math.abs(sd(signalValues(name, 1024)) - 7) < 1e-9, name);
});

test('unscaled signals evaluate the formula directly', () => {
    const t = designPoints(500);
    const f = signalValues('sine', 500);
    for (let i = 0; i < 500; i++)
        assert.ok(Math.abs(f[i] - Math.sin(4 * Math.PI * t[i])) < 1e-15);
    const box = signalValues('box', 1000);
    // b((t − 1/2)/0.01) = 1 for t ∈ [0.49, 0.51]: points 490..510 (1-based) → 21 points
    assert.equal(box.reduce((a, b) => a + b, 0), 21);
});

test('Blocks has the documented jump at t = 0.10 (raw scale)', () => {
    const f = signalValues('blocks', 1000, null);
    // 0-based index 98 is t = 0.099 (before), index 100 is t = 0.101 (after)
    assert.ok(Math.abs(f[100] - f[98] - 4) < 1e-12);
});

test('simulate is deterministic in the seed and noise has the requested sd (tol 5%)', () => {
    const a = simulate('noise', 4000, 0.3, 17);
    const b = simulate('noise', 4000, 0.3, 17);
    const c = simulate('noise', 4000, 0.3, 18);
    assert.deepEqual(a.y, b.y);
    assert.notDeepEqual(a.y, c.y);
    assert.ok(Math.abs(sd(a.y) / 0.3 - 1) < 0.05);
});

test('every built-in signal simulates finite data at its defaults', () => {
    for (const [name, spec] of Object.entries(SIGNALS)) {
        const { x, f, y } = simulate(name, spec.n, spec.sigma, 1);
        assert.equal(y.length, spec.n);
        assert.ok(y.every(Number.isFinite) && f.every(Number.isFinite) && x.every(Number.isFinite), name);
    }
});

test('signalFunction matches signalValues at the design points, with and without rescaling', () => {
    const TOL = 1e-12; // relative
    for (const name of ['sine', 'blocks', 'heavisine', 'ramp']) {
        const n = 300;
        const f = signalFunction(name, n);
        const values = signalValues(name, n);
        const t = designPoints(n);
        for (let i = 0; i < n; i++)
            assert.ok(Math.abs(f(t[i]) - values[i]) <= TOL * Math.max(1, Math.abs(values[i])), `${name} at ${i}`);
    }
});
