import { test } from 'node:test';
import assert from 'node:assert/strict';
import loadHighs from 'highs';
import { minimaxConstant, minimaxLine, ladder } from '../src/core/ladder.js';
import { bandRegion } from '../src/core/band.js';
import { regionSpec, leavingAmbition } from '../src/core/fit.js';
import { solveModel } from '../src/core/lp-solve.js';
import { simulate } from '../src/core/signals.js';
import { createRng, normals } from '../src/core/rng.js';

const SEED = 20261103;

function randomRegion(G, seed) {
    const rng = createRng(seed);
    const x = Float64Array.from({ length: G }, (_, j) => j / (G - 1) + 0.3 * rng.uniform() / G).sort();
    const z = normals(G, rng);
    const c = Float64Array.from(x, (v, j) => 2 * v + Math.sin(6 * v) + 0.3 * z[j]);
    const h = Float64Array.from({ length: G }, () => 0.2 + rng.uniform());
    return { x, c, h };
}

/** max over j of |g_j − c_j|/h_j. */
function maxRatio(g, c, h) {
    let m = 0;
    for (let j = 0; j < g.length; j++)
        m = Math.max(m, Math.abs(g[j] - c[j]) / h[j]);
    return m;
}

test('minimax constant: the pairwise bound max (c_i − c_j)/(h_i + h_j), attained', () => {
    const TOL = 1e-12; // relative
    for (const seed of [1, 2, 3]) {
        const { c, h } = randomRegion(60, seed);
        let brute = 0;
        for (let i = 0; i < c.length; i++) {
            for (let j = 0; j < c.length; j++)
                brute = Math.max(brute, (c[i] - c[j]) / (h[i] + h[j]));
        }
        const k = minimaxConstant(c, h);
        assert.ok(Math.abs(k.ratio - brute) <= TOL * brute, `seed ${seed}: ${k.ratio} vs ${brute}`);
        const g = new Float64Array(c.length).fill(k.value);
        assert.ok(Math.abs(maxRatio(g, c, h) - k.ratio) <= TOL * brute, `seed ${seed}: not attained`);
    }
});

test('minimax line agrees with the HiGHS LP', async () => {
    // HiGHS stops at its own feasibility and optimality tolerances (1e-7); the search is exact to about 1e-12.
    const TOL = 1e-6; // relative, on the ratio
    const highs = await loadHighs();
    for (const seed of [4, 5, 6]) {
        const { x, c, h } = randomRegion(80, seed);
        const line = minimaxLine(x, c, h);
        const G = x.length;
        // columns a, b, r; rows a + b·(x − x0) − r·h ≤ c and a + b·(x − x0) + r·h ≥ c
        const starts = [];
        const indices = [];
        const values = [];
        const rowLower = [];
        const rowUpper = [];
        for (let j = 0; j < G; j++) {
            for (const sign of [-1, 1]) {
                starts.push(indices.length);
                indices.push(0, 1, 2);
                values.push(1, x[j] - line.x0, sign * h[j]);
                rowLower.push(sign < 0 ? -highs.infinity : c[j]);
                rowUpper.push(sign < 0 ? c[j] : highs.infinity);
            }
        }
        starts.push(indices.length);
        const model = {
            numCols: 3, numRows: 2 * G,
            colCost: [0, 0, 1],
            colLower: [-highs.infinity, -highs.infinity, 0],
            colUpper: [highs.infinity, highs.infinity, highs.infinity],
            rowLower, rowUpper,
            matrix: { format: 'csr', numRows: 2 * G, numCols: 3, starts, indices, values },
        };
        const lp = solveModel(highs, model);
        assert.equal(lp.status, 'optimal');
        assert.ok(Math.abs(line.ratio - lp.objective) <= TOL * lp.objective,
            `seed ${seed}: search ${line.ratio}, LP ${lp.objective}`);
        const g = Float64Array.from(x, v => line.intercept + line.slope * (v - line.x0));
        assert.ok(Math.abs(maxRatio(g, c, h) - line.ratio) <= 1e-12 * line.ratio, `seed ${seed}: not attained`);
    }
});

test('minimax shapes in exact cases', () => {
    const TOL = 1e-12; // absolute (ratios of order 1)
    const x = Float64Array.from({ length: 101 }, (_, j) => j / 100);
    const h = new Float64Array(101).fill(1);
    // c rising from 0 to 1: the best constant is 1/2, a ratio of 1/2; a line matches c exactly
    const c = Float64Array.from(x);
    assert.ok(Math.abs(minimaxConstant(c, h).ratio - 0.5) < TOL);
    assert.ok(minimaxLine(x, c, h).ratio < TOL);
    // c = x²: the minimax line is the Chebyshev one, error 1/8
    const sq = Float64Array.from(x, v => v * v);
    assert.ok(Math.abs(minimaxLine(x, sq, h).ratio - 0.125) < TOL);
});

test('ladder on bands: a line fits a noisy line\'s band and not a sine\'s; it agrees with phase 2', () => {
    const lineData = simulate('line', 200, 0.5, SEED);
    const lineRegion = bandRegion(lineData.x, lineData.y, { B: 300 });
    const [constant, line] = ladder(lineRegion);
    assert.equal(constant.shape, 'constant');
    assert.equal(line.shape, 'line');
    assert.equal(line.fits, true, `line ratio ${line.ratio}`);
    assert.equal(constant.fits, false, `constant ratio ${constant.ratio}`);
    // when the line fits, the quadratic ambition never leaves the region: a* is the cap
    assert.equal(leavingAmbition(regionSpec(lineRegion, { ambition: 0 })).capped, true);

    const sineData = simulate('sine', 200, 0.3, SEED);
    const sineRegion = bandRegion(sineData.x, sineData.y, { B: 300 });
    const rungs = ladder(sineRegion);
    assert.ok(rungs.every(r => !r.fits && r.ratio > 1), rungs.map(r => r.ratio).join(', '));
    assert.ok(rungs[1].ratio <= rungs[0].ratio, 'a line does at least as well as a constant');
    assert.equal(leavingAmbition(regionSpec(sineRegion, { ambition: 0 })).capped, false);
});
