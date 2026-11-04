// Coverage simulation for the phase-1 band (PLAN.md phase 1), as the earlier project's tools/sim-band.mjs. For each signal, n
// and configuration (covariance, undersmoothing factor on λ), R replications y = f + σε (seeded); on each, GCV picks
// λ, the band uses undersmooth × λ (src/core/band.js), and is checked on its own grid (data x plus 4 points per
// segment). Reported: the fraction of replications in which the band contains f on the whole grid, and at the data
// x only (as the earlier project measured; f between two data points that straddle a jump is not identified by the data); the
// fraction in which it contains
// the P-spline's mean E f̂ (f smoothed with the same λ: the sandwich band's own target); the mean half-width in
// units of the true σ; the mean effective dimension. σ is estimated in every case, as the app does.
//
// Configurations are covariance:factor, where covariance is bayes or sandwich (band.js) or the bootstrap band
// (bootstrap.js) as boot[-weights[-window]], e.g. boot, boot-dirichlet, boot-matched-Infinity.
// --noise sets the noise ε: gauss (the signals' own), skew (Exp(1) − 1: mean 0, sd 1, right-skewed) or hetero
// (Gaussian with sd growing from 0.25 to 1.75 times σ across x, average 1).
//
// Usage: node tools/sim-band.mjs [--reps 400] [--B 1000] [--n 100,500] [--signals line,sine,ramp,heavisine,blocks]
//                                [--configs bayes:1,bayes:0.0625,sandwich:1,sandwich:0.0625] [--noise gauss]

import { bandRegion, checkBand, CHECK_TOL } from '../src/core/band.js';
import { fitPSpline, evaluate } from '../src/core/pspline.js';
import { simulate, signalFunction, SIGNALS } from '../src/core/signals.js';
import { bootstrapBand } from '../src/core/bootstrap.js';
import { createRng } from '../src/core/rng.js';

const ALPHA = 0.95;
const SEED_BASE = 10000; // replication r uses seed SEED_BASE + r
const CALIBRATION_SEED = 1;

function parseConfigs(text) {
    return text.split(',').map(item => {
        const [covariance, factor] = item.split(':');
        return { covariance, undersmooth: Number(factor) };
    });
}

function parseArgs(argv) {
    const opts = { reps: 400, B: 1000, sizes: [100, 500], signals: ['line', 'sine', 'ramp', 'heavisine', 'blocks'],
        configs: parseConfigs('bayes:1,bayes:0.0625,sandwich:1,sandwich:0.0625'), noise: 'gauss' };
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

/** y = f + σε with the chosen noise; gauss is the signal's own simulation. */
function sample(name, n, sigma, seed, noise) {
    const sim = simulate(name, n, sigma, seed);
    if (noise === 'gauss')
        return sim;
    const rng = createRng(seed + 7777777);
    const y = Float64Array.from(sim.f, (f, i) => {
        if (noise === 'skew')
            return f + sigma * (-Math.log(1 - rng.uniform()) - 1);
        if (noise === 'hetero')
            return f + sigma * (0.25 + 1.5 * i / (n - 1)) * rng.normal();
        throw new Error(`unknown noise ${noise}`);
    });
    return { ...sim, y };
}

function makeBand(x, y, config, B) {
    const common = { alpha: ALPHA, undersmooth: config.undersmooth, B, seed: CALIBRATION_SEED };
    if (config.covariance.startsWith('boot')) {
        // boot-<weights>-<window>, each optional: boot, boot-dirichlet, boot-matched-0.2
        const [, weights, window] = config.covariance.split('-');
        return bootstrapBand(x, y, { ...common, weights, window: window === undefined ? undefined : Number(window) });
    }
    return bandRegion(x, y, { ...common, covariance: config.covariance });
}

const pct = (count, reps) => `${(100 * count / reps).toFixed(1)}%`;

function main() {
    const opts = parseArgs(process.argv.slice(2));
    console.log(`α = ${ALPHA}, ${opts.reps} replications, B = ${opts.B}; σ estimated; λ by GCV × factor; `
        + `noise ${opts.noise}`);
    console.log('signal | n | config | ∋ f | ∋ f at data x | ∋ E f̂ | half-width/σ | ED');
    for (const name of opts.signals) {
        const sigma = SIGNALS[name].sigma;
        for (const n of opts.sizes) {
            const fAt = signalFunction(name, n);
            for (const config of opts.configs) {
                let coverF = 0;
                let coverData = 0;
                let coverMean = 0;
                let halfSum = 0;
                let edSum = 0;
                for (let r = 0; r < opts.reps; r++) {
                    const { x, f, y } = sample(name, n, sigma, SEED_BASE + r, opts.noise);
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
                console.log(`${name} | ${n} | ${config.covariance} ${config.undersmooth} | ${pct(coverF, opts.reps)} | ${pct(coverData, opts.reps)} | `
                    + `${pct(coverMean, opts.reps)} | ${(halfSum / opts.reps).toFixed(2)} | `
                    + `${(edSum / opts.reps).toFixed(1)}`);
            }
        }
    }
}

main();
