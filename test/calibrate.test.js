import { test } from 'node:test';
import assert from 'node:assert/strict';
import { maxStatistic, simulateMaxStatistics, tauFromSample, coverageOfTau } from '../src/core/calibrate.js';
import { multiresolution, allIntervals } from '../src/core/intervals.js';
import { sigmaMAD } from '../src/core/sigma.js';
import { createRng, normals } from '../src/core/rng.js';

test('maxStatistic matches brute force over all intervals, with and without σ̂', () => {
    const z = normals(30, createRng(1));
    let max = 0;
    for (let lo = 0; lo < 30; lo++) {
        for (let hi = lo; hi < 30; hi++) {
            let s = 0;
            for (let i = lo; i <= hi; i++)
                s += z[i];
            max = Math.max(max, Math.abs(s) / Math.sqrt(hi - lo + 1));
        }
    }
    assert.ok(Math.abs(maxStatistic(z, allIntervals(30), false) - max) < 1e-12);
    assert.ok(Math.abs(maxStatistic(z, allIntervals(30), true) - max / sigmaMAD(z)) < 1e-12);
});

test('simulation is deterministic in the seed and sorted', () => {
    const fam = multiresolution(64);
    const a = simulateMaxStatistics(64, fam, { B: 200, seed: 3 });
    const b = simulateMaxStatistics(64, fam, { B: 200, seed: 3 });
    assert.deepEqual(a, b);
    for (let i = 1; i < a.length; i++)
        assert.ok(a[i] >= a[i - 1]);
});

test('tauFromSample: order statistic, interval around it, monotone in α', () => {
    const sorted = Float64Array.from({ length: 1000 }, (_, i) => 1 + i / 1000);
    const n = 100;
    const t = tauFromSample(sorted, 0.95, n);
    assert.equal(t.q, sorted[949]); // ⌈0.95 · 1000⌉ = 950th (1-based)
    assert.ok(Math.abs(t.tau - t.q ** 2 / Math.log(n)) < 1e-12);
    assert.ok(t.tauLo < t.tau && t.tau < t.tauHi);
    assert.ok(tauFromSample(sorted, 0.9, n).tau < t.tau);
    assert.ok(coverageOfTau(sorted, t.tau, n) >= 0.95);
});

test('calibrated τ gives the nominal coverage on independent simulations (dyadic, n = 200, ±2.5%)', () => {
    const fam = multiresolution(200);
    const train = simulateMaxStatistics(200, fam, { B: 2000, seed: 1 });
    const testSample = simulateMaxStatistics(200, fam, { B: 2000, seed: 2 });
    for (const alpha of [0.9, 0.95]) {
        const { tau } = tauFromSample(train, alpha, 200);
        const cov = coverageOfTau(testSample, tau, 200);
        assert.ok(Math.abs(cov - alpha) < 0.025, `α = ${alpha}: out-of-sample coverage ${cov}`);
    }
});

// PLAN.md Known issue 13: paper eq. (11) says τ = 3 with σ̂ covers ≥ 0.95 for n ≥ 500. Measured (B = 4000):
// dyadic family 97.7%; all intervals 93.1% ± 0.8 (n = 500), 94.8% ± 0.7 (n = 1000).
test('τ = 3 with σ̂ at n = 500: ≥ 95% for the dyadic family, below 95% over all intervals (Known issue 13)', () => {
    const dyadic = simulateMaxStatistics(500, multiresolution(500), { B: 2000, seed: 11 });
    const all = simulateMaxStatistics(500, allIntervals(500), { B: 2000, seed: 11 });
    const covDyadic = coverageOfTau(dyadic, 3, 500);
    const covAll = coverageOfTau(all, 3, 500);
    console.log(`# coverage of τ = 3, n = 500, σ̂: dyadic ${(100 * covDyadic).toFixed(1)}%, all intervals ${(100 * covAll).toFixed(1)}%`);
    assert.ok(covDyadic >= 0.95);
    assert.ok(covAll < 0.95);
});
