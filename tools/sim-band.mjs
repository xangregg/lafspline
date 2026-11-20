// Coverage simulation for the phase-1 band (PLAN.md phase 1). For each signal, n
// and configuration (covariance, undersmoothing factor on λ), R replications y = f + σε (seeded); on each, GCV picks
// λ, the band uses undersmooth × λ (src/core/band.js), and is checked on its own grid (data x plus 4 points per
// segment). Reported: the fraction of replications in which the band contains f on the whole grid, and at the data
// x only (f between two data points that straddle a jump is not identified by the data); the
// fraction in which it contains
// the P-spline's mean E f̂ (f smoothed with the same λ: the sandwich band's own target); the mean half-width in
// units of the true σ; the mean effective dimension. σ is estimated in every case, as the app does.
//
// Configurations are covariance:factor, where covariance is bayes or sandwich (band.js) or the bootstrap band
// (bootstrap.js) as boot[-weights[-window[-refit]]], e.g. boot, boot-dirichlet, boot-matched-Infinity,
// boot-matched-0.1-linear, and factor is the undersmoothing factor on GCV's λ or s<scale> for λ by feature scale
// (thumb.js scaleLambda, with its floor by n), e.g. bayes:s0.1.
// --noise sets the noise ε: gauss (the signals' own), skew (Exp(1) − 1: mean 0, sd 1, right-skewed) or hetero
// (Gaussian with sd growing from 0.25 to 1.75 times σ across x, average 1).
// --design sets the x: even (the signals' own design points), uneven (x = t + 0.8·sin(2πt)/2π: nine times as dense
// in the middle as at the ends) or gap (the design points with x in (0.6, 0.8) left out, so n is 80% of the given).
// A bootstrap window ending in p is a share of the observations (windowBy 'points'), e.g. boot-matched-0.1p.
//
// Usage: node tools/sim-band.mjs [--reps 400] [--B 1000] [--n 100,500] [--signals line,sine,ramp,heavisine,blocks]
//                                [--configs bayes:1,bayes:0.0625,sandwich:1,sandwich:0.0625] [--noise gauss]
//                                [--design even]

import { bandRegion, checkBand, CHECK_TOL } from '../src/core/band.js';
import { fitPSpline, evaluate } from '../src/core/pspline.js';
import { simulate, signalFunction, SIGNALS, TEST_SIGNALS } from '../src/core/signals.js';
import { bootstrapBand } from '../src/core/bootstrap.js';
import { createRng } from '../src/core/rng.js';

const ALPHA = 0.95;
const SEED_BASE = 10000; // replication r uses seed SEED_BASE + r
const CALIBRATION_SEED = 1;

function parseConfigs(text) {
    return text.split(',').map(item => {
        const [covariance, factor] = item.split(':');
        if (factor.startsWith('s'))
            return { covariance, scale: Number(factor.slice(1)), label: factor };
        return { covariance, undersmooth: Number(factor), label: factor };
    });
}

function parseArgs(argv) {
    const opts = { reps: 400, B: 1000, sizes: [100, 500], signals: ['line', 'sine', 'ramp', 'heavisine', 'blocks'],
        configs: parseConfigs('bayes:1,bayes:0.0625,sandwich:1,sandwich:0.0625'), noise: 'gauss',
        design: 'even' };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--reps')
            opts.reps = Number(argv[++i]);
        else if (argv[i] === '--B')
            opts.B = Number(argv[++i]);
        else if (argv[i] === '--n')
            opts.sizes = argv[++i].split(',').map(Number);
        else if (argv[i] === '--signals')
            opts.signals = argv[++i].split(',');
        else if (argv[i] === '--configs')
            opts.configs = parseConfigs(argv[++i]);
        else if (argv[i] === '--noise')
            opts.noise = argv[++i];
        else if (argv[i] === '--design')
            opts.design = argv[++i];
        else
            throw new Error(`unknown argument ${argv[i]}`);
    }
    return opts;
}

/** Is every ratio at a data x within 1 (+ the check tolerance)? The grid is sorted and contains every x. */
function atData(band, x, ratios) {
    const index = new Map();
    band.grid.forEach((v, j) => index.set(v, j));
    for (const xi of x) {
        if (ratios[index.get(xi)] > 1 + CHECK_TOL)
            return false;
    }
    return true;
}

const UNEVEN = 0.8; // the uneven design's amplitude: density 1/(1 − 0.8) = 5 times the mean in the middle, 1/1.8 at the ends
const GAP = [0.6, 0.8]; // the gap design's empty stretch of x

/** The x design: the signal's own points, moved or thinned. */
function design(x, kind) {
    if (kind === 'even')
        return x;
    if (kind === 'uneven')
        return x.map(t => t + UNEVEN * Math.sin(2 * Math.PI * t) / (2 * Math.PI));
    if (kind === 'gap')
        return x.filter(t => t <= GAP[0] || t >= GAP[1]);
    throw new Error(`unknown design ${kind}`);
}

/** y = f + σε with the chosen noise and design; gauss on the even design is the signal's own simulation. */
function sample(name, n, sigma, seed, noise, kind, fAt) {
    let sim = simulate(name, n, sigma, seed);
    if (kind !== 'even') {
        const x = design(sim.x, kind);
        const rng = createRng(seed + 5555555);
        const f = x.map(fAt);
        sim = { x, f, y: f.map(v => v + sigma * rng.normal()) };
    }
    if (noise === 'gauss')
        return sim;
    const rng = createRng(seed + 7777777);
    const y = Float64Array.from(sim.f, (f, i) => {
        if (noise === 'skew')
            return f + sigma * (-Math.log(1 - rng.uniform()) - 1);
        if (noise === 'hetero')
            return f + sigma * (0.25 + 1.5 * i / (sim.f.length - 1)) * rng.normal();
        throw new Error(`unknown noise ${noise}`);
    });
    return { ...sim, y };
}

function makeBand(x, y, config, B) {
    const lambda = config.scale === undefined ? { undersmooth: config.undersmooth }
        : { criterion: 'scale', scale: config.scale };
    const common = { alpha: ALPHA, ...lambda, B, seed: CALIBRATION_SEED };
    if (config.covariance.startsWith('boot')) {
        // boot-<weights>-<window>-<refit>, each optional: boot, boot-dirichlet, boot-matched-0.2-linear
        const [, weights, window, refit] = config.covariance.split('-');
        const byPoints = window?.endsWith('p');
        return bootstrapBand(x, y, { ...common, weights, refit,
            window: window === undefined ? undefined : Number(byPoints ? window.slice(0, -1) : window),
            windowBy: byPoints ? 'points' : undefined });
    }
    return bandRegion(x, y, { ...common, covariance: config.covariance });
}

const pct = (count, reps) => `${(100 * count / reps).toFixed(1)}%`;

function main() {
    const opts = parseArgs(process.argv.slice(2));
    console.log(`α = ${ALPHA}, ${opts.reps} replications, B = ${opts.B}; σ estimated; λ by GCV × factor or by `
        + 'feature scale (s); '
        + `noise ${opts.noise}, design ${opts.design}`);
    console.log('signal | n | config | ∋ f | ∋ f at data x | ∋ E f̂ | half-width/σ | ED');
    for (const name of opts.signals) {
        const sigma = (SIGNALS[name] ?? TEST_SIGNALS[name]).sigma;
        for (const n of opts.sizes) {
            const fAt = signalFunction(name, n);
            for (const config of opts.configs) {
                let coverF = 0;
                let coverData = 0;
                let coverMean = 0;
                let halfSum = 0;
                let edSum = 0;
                for (let r = 0; r < opts.reps; r++) {
                    const { x, f, y } = sample(name, n, sigma, SEED_BASE + r, opts.noise, opts.design, fAt);
                    const band = makeBand(x, y, config, opts.B);
                    const p = band.params;
                    const check = checkBand(band, band.grid.map(fAt));
                    if (check.ok)
                        coverF++;
                    if (atData(band, x, check.ratios))
                        coverData++;
                    const mean = evaluate(fitPSpline(x, f, { nseg: p.nseg, lambda: p.lambda }), band.grid);
                    if (checkBand(band, mean).ok)
                        coverMean++;
                    halfSum += band.half.reduce((a, b) => a + b) / band.half.length / sigma;
                    edSum += p.ed;
                }
                console.log(`${name} | ${n} | ${config.covariance} ${config.label} | ${pct(coverF, opts.reps)} | ${pct(coverData, opts.reps)} | `
                    + `${pct(coverMean, opts.reps)} | ${(halfSum / opts.reps).toFixed(2)} | `
                    + `${(edSum / opts.reps).toFixed(1)}`);
            }
        }
    }
}

main();
