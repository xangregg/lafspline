import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    bootstrapBand, dirichletWeights, matchedWeights, refitChange, scaledResiduals,
} from '../src/core/bootstrap.js';
import { fitPSpline, solveCholesky } from '../src/core/pspline.js';
import { checkBand, bandRegion } from '../src/core/band.js';
import { readDataset } from '../src/core/datasets.js';
import { readFileSync } from 'node:fs';
import { simulate } from '../src/core/signals.js';
import { createRng } from '../src/core/rng.js';

const SEED = 20261103;
const B = 300; // replicates in the band tests: enough for their checks, a few hundred milliseconds

function moments(draw) {
    const rng = createRng(SEED);
    const N_PER = 1000;
    const REPS = 200;
    const m = { min: Infinity, maxSumError: 0, c2: 0, c3: 0 };
    for (let r = 0; r < REPS; r++) {
        const w = draw(N_PER, rng);
        let sum = 0;
        for (const v of w) {
            sum += v;
            m.min = Math.min(m.min, v);
            m.c2 += (v - 1) ** 2;
            m.c3 += (v - 1) ** 3;
        }
        m.maxSumError = Math.max(m.maxSumError, Math.abs(sum - N_PER) / N_PER);
    }
    m.c2 /= N_PER * REPS;
    m.c3 /= N_PER * REPS;
    return m;
}

test('fractional weights are positive, sum to n, and have a count\'s variance; matched ones its skewness', () => {
    const SUM_TOL = 1e-12; // relative
    const MOMENT_TOL = 0.05; // on variance and third central moment, Monte Carlo over 200 000 weights
    const matched = moments(matchedWeights);
    const dirichlet = moments(dirichletWeights);
    for (const m of [matched, dirichlet]) {
        assert.ok(m.min > 0);
        assert.ok(m.maxSumError < SUM_TOL);
        assert.ok(Math.abs(m.c2 - 1) < MOMENT_TOL, `variance ${m.c2}`);
    }
    // a resampled count's third central moment is 1 − 3/n + 2/n²; Exp(1)'s is 2
    assert.ok(Math.abs(matched.c3 - 1) < 2 * MOMENT_TOL, `matched ${matched.c3}`);
    assert.ok(Math.abs(dirichlet.c3 - 2) < 4 * MOMENT_TOL, `dirichlet ${dirichlet.c3}`);
});

test('refitChange with the fit\'s residuals is exactly the reweighted refit minus the fit', () => {
    const TOL = 1e-9; // relative to the largest coefficient
    const { x, y } = simulate('sine', 150, 0.3, SEED);
    const opts = { nseg: 15, lambda: 0.5 };
    const fit = fitPSpline(x, y, opts);
    const wb = matchedWeights(x.length, createRng(1));
    const refit = fitPSpline(x, y, { ...opts, w: wb });
    const r = Float64Array.from(y, (v, i) => v - fit.fitted[i]);
    const delta = refitChange(fit, wb, r);
    const scale = Math.max(...fit.alpha.map(Math.abs));
    for (let k = 0; k < delta.length; k++)
        assert.ok(Math.abs(fit.alpha[k] + delta[k] - refit.alpha[k]) <= TOL * scale, `coefficient ${k}`);
});

test('scaledResiduals uses the exact variance factor of each residual, diag (I − H)(I − H)ᵀ', () => {
    const TOL = 1e-10;
    const { x, y } = simulate('ramp', 40, 0.2, SEED);
    const fit = fitPSpline(x, y, { nseg: 8, lambda: 0.1 });
    const { basis, W } = fit.system;
    const n = x.length;
    const bw = basis.deg + 1;
    // column k of H: B A⁻¹ b_k (unit weights)
    const H = Array.from({ length: n }, () => new Float64Array(n));
    for (let k = 0; k < n; k++) {
        const e = new Float64Array(basis.nb);
        for (let r = 0; r < bw; r++)
            e[basis.first[k] + r] = basis.values[k * bw + r];
        const a = solveCholesky(fit.L, basis.nb, W, e);
        for (let i = 0; i < n; i++) {
            for (let r = 0; r < bw; r++)
                H[i][k] += basis.values[i * bw + r] * a[basis.first[i] + r];
        }
    }
    const { v } = scaledResiduals(fit);
    for (let i = 0; i < n; i++) {
        let exact = 0;
        for (let k = 0; k < n; k++)
            exact += ((i === k ? 1 : 0) - H[i][k]) ** 2;
        assert.ok(Math.abs(v[i] - exact) < TOL, `${i}: ${v[i]} vs ${exact}`);
    }
});

test('the bootstrap band contains its fit, is reproducible, and is nearly symmetric for Gaussian noise', () => {
    const SYMMETRY = [0.75, 1.33]; // range allowed for (reach above)/(reach below)
    const { x, y } = simulate('sine', 300, 0.3, SEED);
    const band = bootstrapBand(x, y, { B });
    const p = band.params;
    for (let j = 0; j < band.grid.length; j++)
        assert.ok(band.lower[j] < p.fit[j] && p.fit[j] < band.upper[j]);
    assert.equal(bootstrapBand(x, y, { B }).params.q, p.q);
    const ratios = p.up.map((u, j) => u / p.down[j]);
    assert.ok(Math.min(...ratios) > SYMMETRY[0] && Math.max(...ratios) < SYMMETRY[1],
        `${Math.min(...ratios)} – ${Math.max(...ratios)}`);
    assert.equal(checkBand(band, p.fit).ok, true);
});

test('with right-skewed noise the band reaches further below the fit; with growing noise it widens', () => {
    // mean reach below / mean reach above, at least. Measured on 4 seeds: 1.08–1.18 with the default refits (first
    // order in the weights), 1.24–1.30 with exact refits
    const SKEW_RATIO = 1.05;
    const WIDEN = 2.5; // mean half-width over the last fifth of x / the first fifth, at least (noise sd ratio 7)
    const n = 400;
    const sim = simulate('sine', n, 0.3, SEED);
    const rng = createRng(SEED);
    const skewed = Float64Array.from(sim.f, f => f + 0.3 * (-Math.log(1 - rng.uniform()) - 1));
    const sk = bootstrapBand(sim.x, skewed, { B }).params;
    const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
    assert.ok(mean(sk.down) > SKEW_RATIO * mean(sk.up), `${mean(sk.down)} vs ${mean(sk.up)}`);

    const growing = Float64Array.from(sim.f, (f, i) => f + 0.3 * (0.25 + 1.5 * i / (n - 1)) * rng.normal());
    const band = bootstrapBand(sim.x, growing, { B });
    const G = band.grid.length;
    const first = mean(band.half.slice(0, Math.floor(G / 5)));
    const last = mean(band.half.slice(G - Math.floor(G / 5)));
    assert.ok(last > WIDEN * first, `${last} vs ${first}`);
});

test('as λ → ∞ the bootstrap band becomes the least-squares line\'s simultaneous band', () => {
    // With the penalty toward lines, a huge λ makes the P-spline the OLS line. The band's σ̂ and standard errors are
    // then OLS's, and its half-width the Working–Hotelling band's ±√(2F₂,ᵥ;α)·s·√(1/n + (x − x̄)²/Sxx) up to the
    // bootstrap's own estimate of the noise. Measured over 8 seeds (n = 100, B = 2000): ED 2.00016, fit within 5e-5 s,
    // σ̂ and se within 4e-6, half-width 0.99–1.05 of WH on average and 0.89–1.12 pointwise, above ÷ below 0.89–1.13.
    // WH holds for all x, so on the data's range it is about 1.5% wider than the exact band.
    const LAMBDA = 1e8;
    const ED_TOL = 1e-3;
    const FIT_TOL = 1e-3; // in units of OLS's s
    const SE_TOL = 1e-4; // relative, for σ̂ and the standard errors
    const MEAN_RATIO = [0.95, 1.08]; // mean half-width ÷ WH's
    const POINT_RATIO = [0.85, 1.2]; // half-width ÷ WH's at each grid point
    const SYMMETRY = [0.8, 1.25]; // reach above ÷ reach below
    const ALPHA = 0.95;
    const n = 100;
    const { x, y } = simulate('line', n, 0.25, SEED);
    const band = bootstrapBand(x, y, { lambda: LAMBDA, alpha: ALPHA, B: 2000 });
    const p = band.params;

    const xm = x.reduce((a, b) => a + b) / n;
    const ym = y.reduce((a, b) => a + b) / n;
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < n; i++) {
        sxx += (x[i] - xm) ** 2;
        sxy += (x[i] - xm) * (y[i] - ym);
    }
    const b1 = sxy / sxx;
    const b0 = ym - b1 * xm;
    let rss = 0;
    for (let i = 0; i < n; i++)
        rss += (y[i] - b0 - b1 * x[i]) ** 2;
    const nu = n - 2;
    const s = Math.sqrt(rss / nu);
    // F₂,ᵥ has CDF 1 − (1 + 2f/ν)^(−ν/2), so 2F₂,ᵥ;α = ν((1 − α)^(−2/ν) − 1)
    const wh = Math.sqrt(nu * ((1 - ALPHA) ** (-2 / nu) - 1));

    assert.ok(Math.abs(p.ed - 2) < ED_TOL, `ED ${p.ed}`);
    assert.ok(Math.abs(p.sigma / s - 1) < SE_TOL, `σ̂ ${p.sigma} vs ${s}`);
    const ratios = [];
    for (let j = 0; j < band.grid.length; j++) {
        const g = band.grid[j];
        const se = s * Math.sqrt(1 / n + (g - xm) ** 2 / sxx);
        assert.ok(Math.abs(p.fit[j] - (b0 + b1 * g)) < FIT_TOL * s, `fit at ${g}`);
        assert.ok(Math.abs(p.se[j] / se - 1) < SE_TOL, `se at ${g}: ${p.se[j]} vs ${se}`);
        ratios.push(band.half[j] / (wh * se));
    }
    const meanRatio = ratios.reduce((a, b) => a + b) / ratios.length;
    assert.ok(meanRatio > MEAN_RATIO[0] && meanRatio < MEAN_RATIO[1], `mean ÷ WH ${meanRatio}`);
    assert.ok(Math.min(...ratios) > POINT_RATIO[0] && Math.max(...ratios) < POINT_RATIO[1],
        `÷ WH ${Math.min(...ratios)} – ${Math.max(...ratios)}`);
    const reach = p.up.map((u, j) => u / p.down[j]);
    assert.ok(Math.min(...reach) > SYMMETRY[0] && Math.max(...reach) < SYMMETRY[1],
        `${Math.min(...reach)} – ${Math.max(...reach)}`);
});

test('data on a line, with the penalty toward lines, leave no spread for a bootstrap band', () => {
    const x = Float64Array.from({ length: 30 }, (_, i) => i);
    const y = x.map(v => 2 + 0.5 * v);
    assert.throws(() => bootstrapBand(x, y, { B: 10, lambda: 1e6 }), RangeError);
});

test('a sparse stretch with a small λ does not make the band swing: Tukey coal stays near the sandwich band', () => {
    // Regression: with exact refits, ED 25 of n = 49 (two points per segment) gave a half-width 15 times the
    // sandwich band's at 1920 and 3.5 times near 1950. Measured with the default refits: 0.33–1.76 times, mean 1.05.
    const MAX_RATIO = 2.5;
    const text = readFileSync(new URL('../data/tukey-coal.csv', import.meta.url), 'utf8');
    const { x, y } = readDataset(text);
    const opts = { undersmooth: 1 / 16 };
    const band = bootstrapBand(x, y, { ...opts, B: 1000 });
    const sw = bandRegion(x, y, { ...opts, covariance: 'sandwich', B: 1000 });
    const ratios = band.half.map((h, j) => h / sw.half[j]);
    assert.ok(Math.max(...ratios) < MAX_RATIO, `${Math.max(...ratios)}`);
    const exact = bootstrapBand(x, y, { ...opts, B: 1000, refit: 'exact' });
    assert.ok(Math.max(...exact.half.map((h, j) => h / sw.half[j])) > 4 * MAX_RATIO);
});

test('the band\'s target is the replicates\' median: inside the band, at the fit when the noise is symmetric', () => {
    const TOL = 0.1; // of the half-width, for Gaussian noise
    const { x, y } = simulate('sine', 300, 0.3, SEED);
    const band = bootstrapBand(x, y, { B });
    for (let j = 0; j < band.grid.length; j++) {
        assert.ok(band.lower[j] < band.target[j] && band.target[j] < band.upper[j]);
        assert.ok(Math.abs(band.target[j] - band.params.fit[j]) < TOL * band.half[j], `${band.grid[j]}`);
    }
    assert.throws(() => bootstrapBand(x, y, { B: 10, refit: 'quadratic' }), RangeError);
});
