import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tieGroups, differenceRows, derivative, totalVariation, supNorm } from '../src/core/derivatives.js';
import { createRng } from '../src/core/rng.js';

test('tie groups', () => {
    const { group, xs, count } = tieGroups([1, 1, 2, 3, 3, 3]);
    assert.deepEqual(Array.from(group), [0, 0, 1, 2, 2, 2]);
    assert.deepEqual(Array.from(xs), [1, 2, 3]);
    assert.deepEqual(Array.from(count), [2, 1, 3]);
});

test('divided differences are exact for polynomials at unequal spacing (tol 1e-9)', () => {
    const rng = createRng(5);
    const t = [0];
    for (let i = 1; i < 30; i++)
        t.push(t[i - 1] + 0.1 + rng.uniform());
    const poly = v => 3 - 2 * v + 0.5 * v * v; // g′ = −2 + v, g″ = 1
    const g = t.map(poly);
    const d2 = derivative(t, g, 2);
    assert.ok(d2.every(v => Math.abs(v - 1) < 1e-9), `D² ${d2[0]}`);
    const cubic = t.map(v => v ** 3); // g‴ = 6
    assert.ok(derivative(t, cubic, 3).every(v => Math.abs(v - 6) < 1e-7));
});

test('equal spacing h = 1/n reproduces the paper Δ⁽¹⁾ = n(g_i − g_{i−1}) iterated', () => {
    const n = 50;
    const t = Array.from({ length: n }, (_, i) => (i + 1) / n);
    const g = t.map(v => Math.sin(5 * v));
    const d2 = derivative(t, g, 2);
    for (let j = 0; j < d2.length; j++) {
        const paper = n * (n * (g[j + 2] - g[j + 1]) - n * (g[j + 1] - g[j]));
        assert.ok(Math.abs(d2[j] - paper) < 1e-8 * Math.abs(paper) + 1e-8);
    }
});

test('differenceRows agree with derivative()', () => {
    const t = [0, 0.3, 1.1, 1.5, 2.6, 3.0];
    const g = [1, -2, 0.5, 4, 3, -1];
    for (const k of [0, 1, 2, 3]) {
        const rows = differenceRows(t, k);
        const direct = derivative(t, g, k);
        rows.forEach((row, j) => {
            const v = row.reduce((s, c, r) => s + c * g[j + r], 0);
            assert.ok(Math.abs(v - direct[j]) < 1e-12);
        });
    }
});

test('TV and sup norm', () => {
    const t = [0, 1, 2, 3];
    assert.equal(totalVariation(t, [0, 2, 1, 1], 0), 3);
    assert.equal(supNorm(t, [0, 2, 1, 1], 1), 2);
});
