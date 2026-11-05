// Null distribution of the noise check's statistics (src/core/noisecheck.js): smooth curves plus independent
// Gaussian noise, fitted as the bands fit them (λ by GCV, times an undersmoothing factor). Prints quantiles of
//   z = log(σ̂_fit / σ̂_GSJS)·√n   and   √m·r₁ (lag-1 autocorrelation of the residuals' x-group means)
// by n, which set RATIO_Z, LAG_LOW and LAG_HIGH there.
//
// Usage: node tools/noise-check-null.mjs [undersmooth = 0.25] [runs per curve and n = 200]

import { selectLambda, fitPSpline, defaultSegments } from '../src/core/pspline.js';
import { createRng } from '../src/core/rng.js';
import { noiseStatistics } from '../src/core/noisecheck.js';

const undersmooth = Number(process.argv[2] ?? 0.25);
const RUNS = Number(process.argv[3] ?? 200);
const SIGMA = 0.2;
const SIZES = [25, 50, 100, 200, 500, 1000];
const CURVES = {
    line: t => t,
    sine: t => Math.sin(4 * Math.PI * t),
    bump: t => Math.exp(-(((t - 0.5) / 0.15) ** 2)),
};

function quantile(a, p) {
    const s = Float64Array.from(a).sort();
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

const fmt = v => v.toFixed(2).padStart(6);
console.log(`undersmooth ${undersmooth}, ${RUNS} runs per curve and n; quantiles 0.1% 0.5% 2.5% | 97.5% 99.5% 99.9%`);
for (const n of SIZES) {
    const z = [];
    const r = [];
    for (const f of Object.values(CURVES)) {
        for (let run = 1; run <= RUNS; run++) {
            const rng = createRng(run * 7919 + n);
            const x = Float64Array.from({ length: n }, (_, i) => i / (n - 1));
            const y = Float64Array.from(x, t => f(t) + SIGMA * rng.normal());
            const nseg = defaultSegments(n);
            const lambda = selectLambda(x, y, { nseg }).lambda * undersmooth;
            const fit = fitPSpline(x, y, { nseg, lambda });
            const s = noiseStatistics(x, y, fit.fitted, fit.sigma);
            z.push(Math.log(s.ratio) * Math.sqrt(n));
            r.push(s.lag1 * Math.sqrt(s.groups));
        }
    }
    const qs = a => [0.001, 0.005, 0.025, 0.975, 0.995, 0.999].map(p => fmt(quantile(a, p)));
    const zq = qs(z);
    const rq = qs(r);
    console.log(`n ${String(n).padStart(4)}: z ${zq.slice(0, 3).join(' ')} |${zq.slice(3).join(' ')}`
        + `   √m·r₁ ${rq.slice(0, 3).join(' ')} |${rq.slice(3).join(' ')}`);
}
