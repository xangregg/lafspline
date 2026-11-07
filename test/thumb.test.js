import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    silvermanWidth, ruleOfThumbWidth, localLinearTrace, ruleLambda, lambdaSourceName, scaleLambda, lambdaScale, sineGain,
    MIN_POINTS_PER_PERIOD,
} from '../src/core/thumb.js';
import { fitPSpline, pSplineSystem, LOG_LAMBDA_RANGE } from '../src/core/pspline.js';
import { bandRegion } from '../src/core/band.js';
import { createRng, normals } from '../src/core/rng.js';

const grid = n => Float64Array.from({ length: n }, (_, i) => i / (n - 1));

test('Silverman\'s width by hand: x = 1..10', () => {
    const x = Float64Array.from({ length: 10 }, (_, i) => i + 1);
    const TOL = 1e-12; // relative, rounding only
    const sd = Math.sqrt(110 / 12); // Σ(x − 5.5)² = 82.5 over 9
    const iqr = 7.75 - 3.25; // R type 7 quartiles
    const want = 0.9 * Math.min(sd, iqr / 1.34) * 10 ** (-1 / 5);
    const got = silvermanWidth(x);
    assert.ok(Math.abs(got.sd - sd) < TOL * sd, `sd ${got.sd}`);
    assert.ok(Math.abs(got.iqr - iqr) < TOL * iqr, `IQR ${got.iqr}`);
    assert.ok(Math.abs(got.h - want) < TOL * want, `h ${got.h} vs ${want}`);
    assert.throws(() => silvermanWidth([2, 2, 2]), RangeError);
});

test('Fan & Gijbels\' width: the pilot quartic\'s curvature and σ, and the width from them', () => {
    const n = 400;
    const x = grid(n);
    const sigma = 0.1;
    const e = normals(n, createRng(7));
    const y = x.map((v, i) => v * v + sigma * e[i]);
    const exact = ruleOfThumbWidth(x, x.map(v => v * v));
    assert.ok(Math.abs(exact.curvature - 2) < 1e-9, `curvature ${exact.curvature}`); // m'' = 2, rounding only
    const r = ruleOfThumbWidth(x, y);
    assert.ok(Math.abs(r.sigma / sigma - 1) < 0.1, `σ̃ ${r.sigma}`); // sampling error of σ̂ at n = 400: 3.5%
    // the quartic's curvature from noisy data is itself noisy (measured 1.72 here): the rule's weak point
    // the formula from the returned inputs, with R(K) = 1/(2√π), μ₂ = 1, b − a = 1 and Σ m̃''² = n·curvature²
    const want = (r.sigma ** 2 / (2 * Math.sqrt(Math.PI) * n * r.curvature ** 2)) ** (1 / 5);
    assert.ok(Math.abs(r.h - want) < 1e-12 * want, `h ${r.h} vs ${want}`);
    assert.throws(() => ruleOfThumbWidth([0, 1, 2, 3, 4], [0, 1, 0, 1, 0]), RangeError);
});

test('local linear trace: 2 for an infinite width, the distinct x for a vanishing one', () => {
    const x = Float64Array.from([0, 0.1, 0.1, 0.3, 0.6, 0.65, 1]);
    const TOL = 1e-9; // rounding
    assert.ok(Math.abs(localLinearTrace(x, Infinity) - 2) < TOL, `${localLinearTrace(x, Infinity)}`);
    // each distinct x alone: the fit there is the mean of its observations, which counts 1 per distinct x
    assert.ok(Math.abs(localLinearTrace(x, 1e-6) - 6) < TOL, `${localLinearTrace(x, 1e-6)}`);
    const wide = localLinearTrace(x, 0.5);
    const narrow = localLinearTrace(x, 0.05);
    assert.ok(wide > 2 && wide < narrow && narrow < 6, `${wide}, ${narrow}`);
});

test('rule λ: the P-spline\'s ED matches the rule\'s local linear trace', () => {
    const n = 300;
    const x = grid(n);
    const e = normals(n, createRng(3));
    const y = x.map((v, i) => Math.sin(6 * v) + 0.2 * e[i]);
    const ED_TOL = 1e-3; // relative: bisection to 1e-4 in log λ
    for (const rule of ['silverman', 'rot']) {
        const choice = ruleLambda(x, y, { rule, nseg: 40 });
        assert.equal(choice.atEdge, false, rule);
        assert.ok(Math.abs(choice.fit.ed / choice.rule.ed - 1) < ED_TOL,
            `${rule}: ${choice.fit.ed} vs ${choice.rule.ed}`);
        const refit = fitPSpline(x, y, { nseg: 40, lambda: choice.lambda });
        assert.ok(Math.abs(refit.ed - choice.fit.ed) < 1e-9, rule);
    }
    // measured: Silverman's h = 0.083 (from x alone) asks for ED 6.2, Fan & Gijbels' h = 0.032 for ED 14.0
    assert.throws(() => ruleLambda(x, y, { rule: 'scott' }), RangeError);
});

test('rule λ at the end of the range: a trace beyond the basis takes the smallest λ', () => {
    // Silverman at n = 5000 asks for ED ≈ 10 (h ≈ 0.04), more than 5 segments' 8 coefficients
    const x = grid(5000);
    const y = x.map(v => Math.sin(6 * v));
    const choice = ruleLambda(x, y, { rule: 'silverman', nseg: 5 });
    assert.ok(choice.rule.ed > 8, `${choice.rule.ed}`);
    assert.equal(choice.atEdge, true);
    assert.equal(Math.log(choice.lambda), LOG_LAMBDA_RANGE[0]);
});

test('a band with λ by a rule of thumb names the rule and keeps its width', () => {
    const x = grid(200);
    const e = normals(200, createRng(11));
    const y = x.map((v, i) => Math.sin(4 * v) + 0.2 * e[i]);
    const region = bandRegion(x, y, { criterion: 'silverman', B: 200 });
    assert.equal(region.params.lambdaSource, 'silverman');
    assert.ok(region.params.rule.h > 0);
    assert.ok(region.claim.includes(lambdaSourceName('silverman')), region.claim);
    assert.equal(bandRegion(x, y, { B: 200 }).params.rule, null);
});

test('feature scale: λ keeps half a sine of that period, and the scale λ implies is the one asked for', () => {
    const n = 400;
    const x = grid(n);
    const e = normals(n, createRng(5));
    const y = x.map((v, i) => Math.sin(6 * v) + 0.3 * e[i]);
    const SCALE_TOL = 0.01; // relative: bisections to 1e-4 in log λ and log period
    let prev = null;
    for (const scale of [0.1, 0.2, 0.4]) {
        const choice = scaleLambda(x, y, { scale, nseg: 40 });
        assert.equal(choice.atEdge, false, `${scale}`);
        assert.equal(choice.rule.period, scale); // x spans [0, 1]
        const gain = sineGain(pSplineSystem(x, y, { nseg: 40 }), choice.lambda, scale);
        assert.ok(Math.abs(gain - 0.5) < 1e-3, `${scale}: gain ${gain}`);
        const back = lambdaScale(x, choice.lambda, { nseg: 40 });
        assert.ok(Math.abs(back / scale - 1) < SCALE_TOL, `${scale}: ${back}`);
        // a longer scale: a larger λ and a smoother fit
        if (prev) {
            assert.ok(choice.lambda > prev.lambda && choice.fit.ed < prev.fit.ed, `${scale}`);
        }
        prev = choice;
    }
    // a long sine passes, a short one does not
    const system = pSplineSystem(x, y, { nseg: 40 });
    assert.ok(sineGain(system, prev.lambda, 1) > 0.9 && sineGain(system, prev.lambda, 0.1) < 0.05);
    assert.throws(() => scaleLambda(x, y, { scale: 0 }), RangeError);
    assert.throws(() => scaleLambda(x, y, { scale: 1.5 }), RangeError);
});

test('feature scale: λ from x alone, in any units; finer than the basis takes the smallest λ', () => {
    const x = Float64Array.from({ length: 150 }, (_, i) => (i / 149) ** 1.5); // uneven spacing
    const e = normals(150, createRng(8));
    const a = scaleLambda(x, x.map((_, i) => e[i]), { scale: 0.15, nseg: 30 });
    const b = scaleLambda(x, x.map(v => 5 * Math.cos(9 * v)), { scale: 0.15, nseg: 30 });
    assert.equal(a.lambda, b.lambda); // y does not enter
    const years = scaleLambda(x.map(v => 1900 + 120 * v), x.map(v => v), { scale: 0.15, nseg: 30 });
    assert.ok(Math.abs(Math.log(years.lambda / a.lambda)) < 1e-9, `${years.lambda} vs ${a.lambda}`);
    // 30 segments: knots 3.3% of the range apart, so 1% is beyond the basis (no floor by n, which would raise it)
    const fine = scaleLambda(x, x, { scale: 0.01, nseg: 30, minPoints: 0 });
    assert.equal(fine.atEdge, true);
    assert.equal(Math.log(fine.lambda), LOG_LAMBDA_RANGE[0]);
});

test('feature scale: a period spans at least MIN_POINTS_PER_PERIOD observations', () => {
    const n = 46;
    const x = grid(n);
    const e = normals(n, createRng(3));
    const y = x.map((v, i) => 1 - v + 0.2 * e[i]);
    const floor = MIN_POINTS_PER_PERIOD / n;
    const low = scaleLambda(x, y, { scale: 0.1 });
    assert.equal(low.rule.asked, 0.1);
    assert.equal(low.rule.scale, floor);
    assert.equal(low.lambda, scaleLambda(x, y, { scale: floor }).lambda);
    // above the floor the scale is used as asked
    const high = scaleLambda(x, y, { scale: 0.3 });
    assert.equal(high.rule.scale, 0.3);
    // fewer observations than a period needs: the whole range
    assert.equal(scaleLambda(x.slice(0, 8), y.slice(0, 8), { scale: 0.1, nseg: 5 }).rule.scale, 1);
    // the band's claim says the scale was raised
    const region = bandRegion(x, y, { criterion: 'scale', scale: 0.1, B: 200 });
    assert.ok(region.claim.includes(`raised from 10% so that a period spans ${MIN_POINTS_PER_PERIOD} observations`),
        region.claim);
    assert.ok(region.claim.includes(`features longer than ${+(100 * floor).toPrecision(3)}%`), region.claim);
});

test('a band with λ by feature scale says so, with no undersmoothing', () => {
    const x = grid(200);
    const e = normals(200, createRng(12));
    const y = x.map((v, i) => Math.sin(4 * v) + 0.2 * e[i]);
    const region = bandRegion(x, y, { criterion: 'scale', scale: 0.15, B: 200, undersmooth: 0.25 });
    assert.equal(region.params.lambdaSource, 'scale');
    assert.equal(region.params.undersmooth, 1);
    assert.equal(region.params.lambda, scaleLambda(x, y, { scale: 0.15 }).lambda);
    assert.ok(region.claim.includes('features longer than 15% of the x range'), region.claim);
});
