// Coverage simulation for the Aₙ region (src/core/regions.js anRegion) over interval families with a shortest span
// (intervals.js familyLevels, minShare). For each signal, n, design and configuration, R replications y = f + σε
// (seeded); τ is calibrated once per design and family by Monte Carlo (B draws, as the app does), σ is estimated
// from the data. Reported: the fraction of replications in which Aₙ contains f; the fraction in which it rejects
// the least-squares line and the mean (Aₙ's power to see that f is not straight or not flat; for the line signal
// the first is a false alarm); τ, the threshold c in units of the true σ, and the number of intervals.
//
// Configurations are family:share, family one of blocks (the paper's) or quarter (overlapping, ¼-width steps), and
// share the shortest span as a share of the x range (0 keeps every interval), e.g. blocks:0.05.
// --design even uses the signals' own design (i/n); uneven uses x = (i/n)², dense near 0 and sparse near 1.
// --noise as in tools/sim-band.mjs: gauss, skew (Exp(1) − 1) or hetero (sd from 0.25 to 1.75 times σ across x).
//
// Usage: node tools/sim-an.mjs [--reps 400] [--B 1000] [--n 100,500] [--signals line,sine,ramp,heavisine]
//                              [--configs blocks:0,blocks:0.05] [--design even] [--noise gauss]

import { anRegion, calibrateTau, checkFit } from '../src/core/regions.js';
import { familyIntervals } from '../src/core/intervals.js';
import { designPoints, signalFunction, SIGNALS, TEST_SIGNALS } from '../src/core/signals.js';
import { createRng } from '../src/core/rng.js';

const ALPHA = 0.95;
const AN_LAMBDA = 2; // interval lengths grow by this factor per level, as in the app
const SEED_BASE = 10000; // replication r uses seed SEED_BASE + r
const CALIBRATION_SEED = 1;
const NOISE_SEED_OFFSET = 7777777; // the noise stream of replication r is seeded SEED_BASE + r + this

function parseConfigs(text) {
    return text.split(',').map(item => {
        const [kind, share] = item.split(':');
        return { kind, share: Number(share ?? 0), label: item };
    });
}

function parseArgs(argv) {
    const opts = { reps: 400, B: 1000, sizes: [100, 500], signals: ['line', 'sine', 'ramp', 'heavisine'],
        configs: parseConfigs('blocks:0,blocks:0.05'), design: 'even', noise: 'gauss' };
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
        else if (argv[i] === '--design')
            opts.design = argv[++i];
        else if (argv[i] === '--noise')
            opts.noise = argv[++i];
        else
            throw new Error(`unknown argument ${argv[i]}`);
    }
    return opts;
}

function familyOf(config) {
    const family = config.kind === 'quarter' ? { kind: 'overlapping', lambda: AN_LAMBDA, offsets: 4 }
        : { kind: config.kind, lambda: AN_LAMBDA };
    if (config.share > 0)
        family.minShare = config.share;
    return family;
}

function design(n, kind) {
    const t = designPoints(n);
    if (kind === 'even')
        return t;
    if (kind === 'uneven')
        return t.map(v => v * v);
    throw new Error(`unknown design ${kind}`);
}

/** y = f + σε with the chosen noise. */
function sample(f, sigma, seed, noise) {
    const n = f.length;
    const rng = createRng(seed + NOISE_SEED_OFFSET);
    return Float64Array.from(f, (v, i) => {
        if (noise === 'gauss')
            return v + sigma * rng.normal();
        if (noise === 'skew')
            return v + sigma * (-Math.log(1 - rng.uniform()) - 1);
        if (noise === 'hetero')
            return v + sigma * (0.25 + 1.5 * i / (n - 1)) * rng.normal();
        throw new Error(`unknown noise ${noise}`);
    });
}

/** The least-squares line at x, and the mean as a constant. */
function lineAndMean(x, y) {
    const n = x.length;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
        mx += x[i] / n;
        my += y[i] / n;
    }
    let sxy = 0;
    let sxx = 0;
    for (let i = 0; i < n; i++) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) ** 2;
    }
    const b = sxy / sxx;
    return { line: x.map(v => my + b * (v - mx)), mean: new Float64Array(n).fill(my) };
}

const pct = (count, reps) => `${(100 * count / reps).toFixed(1)}%`;

function main() {
    const opts = parseArgs(process.argv.slice(2));
    console.log(`α = ${ALPHA}, ${opts.reps} replications, τ calibrated with B = ${opts.B}; σ estimated; `
        + `design ${opts.design}; noise ${opts.noise}`);
    console.log('signal | n | config | ∋ f | rejects line | rejects mean | τ | c/σ | intervals');
    for (const name of opts.signals) {
        const sigma = (SIGNALS[name] ?? TEST_SIGNALS[name]).sigma;
        for (const n of opts.sizes) {
            const x = design(n, opts.design);
            const fAt = signalFunction(name, n);
            const f = x.map(fAt);
            for (const config of opts.configs) {
                const family = familyOf(config);
                const intervals = familyIntervals(x, family);
                const cal = calibrateTau(x, intervals, { alpha: ALPHA, B: opts.B, seed: CALIBRATION_SEED });
                let cover = 0;
                let line = 0;
                let mean = 0;
                let cSum = 0;
                for (let r = 0; r < opts.reps; r++) {
                    const y = sample(f, sigma, SEED_BASE + r, opts.noise);
                    const region = anRegion(x, y, { family, intervals, tau: cal.tau, calibration: cal.calibration });
                    if (checkFit(region, f, 0).ok)
                        cover++;
                    const fits = lineAndMean(x, y);
                    if (!checkFit(region, fits.line, 0).ok)
                        line++;
                    if (!checkFit(region, fits.mean, 0).ok)
                        mean++;
                    cSum += region.scale / sigma;
                }
                console.log(`${name} | ${n} | ${config.label} | ${pct(cover, opts.reps)} | ${pct(line, opts.reps)} | `
                    + `${pct(mean, opts.reps)} | ${cal.tau.toFixed(2)} | ${(cSum / opts.reps).toFixed(2)} | `
                    + `${intervals.length}`);
            }
        }
    }
}

main();
