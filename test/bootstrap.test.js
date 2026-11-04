import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    bootstrapBand, dirichletWeights, matchedWeights, refitChange, scaledResiduals,
} from '../src/core/bootstrap.js';
import { fitPSpline, solveCholesky } from '../src/core/pspline.js';
import { checkBand } from '../src/core/band.js';
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
    const SKEW_RATIO = 1.15; // mean reach below / mean reach above, at least (measured: about 1.4)
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

test('data on a line, with the penalty toward lines, leave no spread for a bootstrap band', () => {
    const x = Float64Array.from({ length: 30 }, (_, i) => i);
    const y = x.map(v => 2 + 0.5 * v);
    assert.throws(() => bootstrapBand(x, y, { B: 10, lambda: 1e6 }), RangeError);
});
