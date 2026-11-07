import { test } from 'node:test';
import assert from 'node:assert/strict';
import { anRegion, checkFit, CHECK_TOL, calibrateTau, nullSample } from '../src/core/regions.js';
import { checkRegion, threshold } from '../src/core/region.js';
import { xIntervalFamily, familyIntervals } from '../src/core/intervals.js';
import { coverageOfTau } from '../src/core/calibrate.js';
import { simulate } from '../src/core/signals.js';
import { sigmaMAD } from '../src/core/sigma.js';
import { createRng } from '../src/core/rng.js';

const REL_TOL = 1e-12; // relative, on quantities computed two ways in floating point

function assertRelClose(actual, expected, tol, label) {
    const scale = Math.max(1, Math.abs(expected));
    assert.ok(Math.abs(actual - expected) <= tol * scale, `${label}: ${actual} vs ${expected}`);
}

function tiedData() {
    const x = Float64Array.from({ length: 40 }, (_, i) => Math.floor(i / 2) / 20);
    const { y } = simulate('ramp', 40, 0.1, 5);
    return { x, y };
}

test('Aₙ region: one constraint per interval with center Σ_I y, width √|I| and scale c', () => {
    const { x, y } = simulate('sine', 64, 0.2, 1);
    const region = anRegion(x, y);
    const intervals = xIntervalFamily(x, 2);
    const sigma = sigmaMAD(y, x);
    assert.equal(region.kind, 'An');
    assert.equal(region.lo.length, intervals.length);
    assertRelClose(region.scale, threshold(sigma, 3, 64), REL_TOL, 'scale');
    intervals.forEach((iv, k) => {
        assert.equal(region.lo[k], iv.lo);
        assert.equal(region.hi[k], iv.hi);
        let sum = 0;
        for (let i = iv.lo; i <= iv.hi; i++)
            sum += y[i];
        assertRelClose(region.center[k], sum, REL_TOL, `center ${k}`);
        assert.equal(region.width[k], Math.sqrt(iv.hi - iv.lo + 1));
    });
    assert.deepEqual(region.params, { n: 64, sigma, sigmaSource: 'estimated', tau: 3, lambda: 2,
        family: { kind: 'blocks', lambda: 2 }, calibration: null });
});

test('Aₙ constraints never split tied x', () => {
    const { x, y } = tiedData();
    const region = anRegion(x, y);
    for (let k = 0; k < region.lo.length; k++) {
        assert.ok(region.lo[k] === 0 || x[region.lo[k] - 1] !== x[region.lo[k]], `constraint ${k} starts inside a tie`);
        assert.ok(region.hi[k] === x.length - 1 || x[region.hi[k] + 1] !== x[region.hi[k]], `constraint ${k} ends inside a tie`);
    }
});

test('checkFit agrees with tautfit checkRegion on Aₙ', () => {
    const { x, y } = simulate('ramp', 50, 0.2, 2);
    const region = anRegion(x, y, { sigma: 0.2 });
    const intervals = xIntervalFamily(x, 2);
    const rng = createRng(3);
    const outcomes = [];
    for (const spread of [0.05, 0.2, 0.5]) {
        const g = Float64Array.from(y, v => v + spread * rng.normal());
        const ours = checkFit(region, g, CHECK_TOL);
        const theirs = checkRegion(y, g, intervals, region.scale, CHECK_TOL);
        assert.equal(ours.violations, theirs.violations);
        assert.equal(ours.ok, theirs.ok);
        assertRelClose(ours.maxRatio, theirs.maxRatio, REL_TOL, 'maxRatio');
        ours.ratios.forEach((r, k) => assertRelClose(r, theirs.ratios[k], REL_TOL, `ratio ${k}`));
        outcomes.push(ours.ok);
    }
    assert.ok(outcomes.includes(true) && outcomes.includes(false), `need fits inside and outside: ${outcomes}`);
});

test('the claim states the noise assumption and how τ was chosen', () => {
    const { x, y } = simulate('line', 30, 0.25, 7);
    const estimated = anRegion(x, y).claim;
    assert.match(estimated, /about 95% for large n/);
    assert.match(estimated, /σ estimated from the data/);
    assert.match(estimated, /No smoothness is assumed/);

    assert.match(anRegion(x, y, { sigma: 0.25 }).claim, /with the given σ/);
    assert.match(anRegion(x, y, { tau: 2.5 }).claim, /Coverage of τ = 2.5 is not known/);
    const calibrated = anRegion(x, y, { tau: 2.81, calibration: { alpha: 0.95, B: 1000, family: 'chosen' } });
    assert.match(calibrated.claim, /probability ≈ 0.95 \(τ = 2.81 calibrated by Monte Carlo, B = 1000/);
    assert.equal(calibrated.label, 'Aₙ (τ = 2.81, λ = 2)');
    // the τ = 3 numbers are for the paper's blocks; denser families cover less
    assert.match(anRegion(x, y, { family: { kind: 'all' } }).claim, /Coverage of τ = 3 with these intervals is not known/);
});

test('with a shortest interval the claim says what is not checked and cites the simulations for it', () => {
    const { x, y } = simulate('line', 30, 0.25, 7);
    const calibration = { alpha: 0.95, B: 1000, family: 'chosen' };
    const at5 = anRegion(x, y, { tau: 2.3, calibration, family: { kind: 'blocks', lambda: 2, minShare: 0.05 } });
    assert.match(at5.claim, /Every interval spans at least 5% of the x range \(shorter ones are merged or widened\)/);
    assert.match(at5.claim, /these intervals it held f in 92.8–99.3% of samples under Gaussian noise, but in 16–56%/);
    assert.match(at5.label, /spanning at least 5% of x/);
    const all = anRegion(x, y, { tau: 2.3, calibration }).claim;
    assert.match(all, /with these intervals it held f in 96–99.8% .* but in 0–2.5%/);
    assert.doesNotMatch(all, /Every interval spans/);
    assert.match(anRegion(x, y, { tau: 2.3, calibration, family: { kind: 'blocks', lambda: 2, minShare: 0.1 } }).claim,
        /In simulations \(not this shortest interval\)/);
    assert.match(anRegion(x, y, { tau: 2.3, calibration, family: { kind: 'all', minShare: 0.05 } }).claim,
        /with the paper's blocks \(not these intervals\)/);
    // τ = 3's measured numbers are for every one of the paper's blocks
    assert.match(anRegion(x, y, { family: { kind: 'blocks', lambda: 2, minShare: 0.05 } }).claim,
        /Coverage of τ = 3 with these intervals is not known: higher/);
});

test('with σ = 0 the check gives ratio 0 or ∞, never NaN (regression: pasted data with exact replicates)', () => {
    const x = Float64Array.from({ length: 12 }, (_, i) => Math.floor(i / 2));
    const y = Float64Array.from(x, v => 1 + 0.1 * v); // replicates agree exactly, so σ̂ = 0
    const region = anRegion(x, y);
    assert.equal(region.params.sigma, 0);
    const exact = checkFit(region, y);
    assert.equal(exact.ok, true);
    assert.equal(exact.maxRatio, 0);
    const off = Float64Array.from(y, (v, i) => (i === 5 ? v + 0.01 : v));
    const check = checkFit(region, off);
    assert.equal(check.ok, false);
    assert.equal(check.maxRatio, Infinity);
    assert.ok(check.ratios.every(r => !Number.isNaN(r)));
});

test('calibrated τ grows with the family (same draws: a superset can only raise each maximum) and gives the stated coverage', () => {
    const { x } = simulate('sine', 60, 0.3, 4);
    const B = 400; // the property is exact for any B; small keeps the test fast
    const taus = [{ kind: 'blocks', lambda: 2 }, { kind: 'overlapping', lambda: 2, offsets: Infinity }, { kind: 'all' }]
        .map(family => calibrateTau(x, familyIntervals(x, family), { alpha: 0.9, B }));
    assert.ok(taus[0].tau <= taus[2].tau && taus[1].tau <= taus[2].tau, taus.map(t => t.tau).join(', '));
    for (const t of taus) {
        assert.ok(coverageOfTau(t.sample, t.tau, x.length) >= 0.9);
        assert.ok(t.tauLo <= t.tau && t.tau <= t.tauHi);
        assert.deepEqual(t.calibration, { alpha: 0.9, B, family: 'chosen', tauLo: t.tauLo, tauHi: t.tauHi });
    }
});

test('an Aₙ region over a family names it, and calibrateTau draws the sample nullSample would', () => {
    const { x, y } = simulate('sine', 50, 0.3, 5);
    const family = { kind: 'overlapping', lambda: 2, offsets: 4 };
    const intervals = familyIntervals(x, family);
    const cal = calibrateTau(x, intervals);
    const region = anRegion(x, y, { family, tau: cal.tau, calibration: cal.calibration });
    assert.equal(region.lo.length, intervals.length);
    assert.match(region.label, /1\/4-width steps/);
    assert.match(region.claim, /calibrated by Monte Carlo/);
    assert.deepEqual(region.params.family, family);
    assert.deepEqual(nullSample(region, x), cal.sample);
    assert.deepEqual(anRegion(x, y, { lambda: 2 }).params.family, { kind: 'blocks', lambda: 2 });
});
