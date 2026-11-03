import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bandRegion, checkBand, checkGrid, maxTSample, quantile } from '../src/core/band.js';
import { fitPSpline, bsplineBasis, standardErrors, evaluate } from '../src/core/pspline.js';
import { simulate } from '../src/core/signals.js';

test('check grid: the distinct data x and a uniform grid, sorted, without duplicates', () => {
    const x = [0.3, 0.3, 1, 0.55, 2];
    const g = checkGrid(x, 0, 2, 4, 2);
    assert.deepEqual([...g], [0, 0.25, 0.3, 0.5, 0.55, 0.75, 1, 1.25, 1.5, 1.75, 2]);
});

test('quantile is the ⌈αB⌉-th order statistic', () => {
    const s = Float64Array.from({ length: 100 }, (_, i) => i + 1);
    assert.equal(quantile(s, 0.95), 95);
    assert.equal(quantile(s, 0.951), 96);
});

// With a single grid point the statistic is |z|/s: |N(0, 1)| with σ known, |t_ν| with σ estimated. This checks
// that both covariance draws are scaled to their standard errors.
test('one grid point: the critical value is the normal or t quantile', () => {
    const B = 40000;
    const TOL = 0.04; // Monte Carlo sd of the 95% quantile at B = 40000 is about 0.013
    const { x, y } = simulate('sine', 80, 0.2, 3);
    const fit = fitPSpline(x, y, { nseg: 15, lambda: 2 });
    const at = bsplineBasis([0.37], fit.system.basis.xl, fit.system.basis.xr, 15, 3);
    for (const covariance of ['bayes', 'sandwich']) {
        const s = standardErrors(fit, at, covariance).map(v => v / fit.sigma);
        const known = quantile(maxTSample(fit, at, s, { covariance, B, seed: 2 }), 0.95);
        assert.ok(Math.abs(known - 1.960) < TOL, `${covariance}, σ known: ${known}`);
        const t5 = quantile(maxTSample(fit, at, s, { covariance, B, seed: 3, df: 5 }), 0.95);
        assert.ok(Math.abs(t5 - 2.571) < 2 * TOL, `${covariance}, ν = 5: ${t5}`);
    }
});

test('at a fixed λ the sandwich band covers its own target, the P-spline mean, about 95% of the time', () => {
    // 300 replications: binomial sd of the coverage ≈ 1.3 points; seeds fixed, so the outcome is deterministic.
    const REPS = 300;
    const LAMBDA = 1;
    let covered = 0;
    for (let r = 0; r < REPS; r++) {
        const { x, f, y } = simulate('sine', 100, 0.2, 1000 + r);
        const band = bandRegion(x, y, { covariance: 'sandwich', lambda: LAMBDA, nseg: 20, B: 500, seed: r });
        const mean = evaluate(fitPSpline(x, f, { nseg: 20, lambda: LAMBDA }), band.grid);
        if (checkBand(band, mean).ok)
            covered++;
    }
    const coverage = covered / REPS;
    assert.ok(coverage > 0.91 && coverage < 0.99, `coverage ${coverage}`);
});

test('band region: center is the fit, half-widths are q·se, the center passes its own check', () => {
    const { x, y } = simulate('sine', 120, 0.2, 9);
    const band = bandRegion(x, y, { B: 200 });
    const p = band.params;
    assert.equal(p.undersmooth, 1 / 16);
    const fit = fitPSpline(x, y, { nseg: p.nseg, lambda: p.lambda });
    const se = standardErrors(fit, band.grid);
    for (let j = 0; j < band.grid.length; j++) {
        assert.ok(Math.abs(band.half[j] - p.q * se[j]) < 1e-9 * band.half[j]);
        assert.ok(Math.abs(band.upper[j] - band.lower[j] - 2 * band.half[j]) < 1e-9 * band.half[j]);
    }
    for (const xi of x)
        assert.ok(band.grid.includes(xi), `data x ${xi} in the grid`);
    const self = checkBand(band, band.center);
    assert.equal(self.ok, true);
    assert.equal(self.maxRatio, 0);
    assert.equal(checkBand(band, band.upper).ok, true);
    const shifted = band.center.map((c, j) => c + 1.01 * band.half[j] * (j === 5 ? 1 : 0));
    const r = checkBand(band, shifted);
    assert.equal(r.outside, 1);
    assert.equal(r.argMax, 5);
    assert.equal(p.lambdaSource, 'gcv');
    assert.match(band.claim, /Bayesian covariance/);
    assert.match(band.claim, /GCV/);
});

test('undersmoothing divides λ and widens the band; the claim says so', () => {
    const { x, y } = simulate('ramp', 150, 0.2, 4);
    const plain = bandRegion(x, y, { B: 200, undersmooth: 1 });
    const under = bandRegion(x, y, { B: 200, undersmooth: 1 / 16, covariance: 'sandwich' });
    assert.ok(Math.abs(under.params.lambda - plain.params.lambda / 16) < 1e-9 * plain.params.lambda);
    assert.ok(under.params.ed > plain.params.ed);
    assert.match(under.claim, /λ chosen by GCV × 0\.0625 \(undersmoothing\)/);
    assert.doesNotMatch(plain.claim, /\(undersmoothing\)/);
    assert.match(under.claim, /P-spline's mean/);
});

test('a given λ is used as is', () => {
    const { x, y } = simulate('sine', 60, 0.2, 5);
    const band = bandRegion(x, y, { lambda: 3, B: 100 });
    assert.equal(band.params.lambda, 3);
    assert.equal(band.params.lambdaSource, 'given');
    assert.match(band.claim, /λ given/);
});

test('a fit that leaves too few degrees of freedom is refused with a reason', () => {
    // 100 segments for 100 points: GCV chooses near-interpolation (found in the coverage experiments)
    const { x, y } = simulate('blocks', 100, 1, 10000);
    assert.throws(() => bandRegion(x, y, { nseg: 100, undersmooth: 1, B: 50 }), /too few to estimate σ/);
});
