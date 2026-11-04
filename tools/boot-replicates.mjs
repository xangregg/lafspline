// Monte Carlo variability of the bootstrap band (src/core/bootstrap.js) across seeds and replicate counts B, against
// a reference band at a large B. It is the study behind the app's B = 2000 for n ≤ 1000 and 1000 above (PLAN.md §1.2,
// "Replicates (B)"). Data: three bundled data sets and a simulated sine with Gaussian and with right-skewed noise.
//
// For each data set and B, over seeds 1..S, it reports:
// - the mean time per band;
// - the seed-to-seed sd of the mean critical values below and above;
// - the mean half-width as a ratio to the reference's (mean ± sd over seeds);
// - the worst edge gap anywhere on the grid, |edge − reference edge| / reference half-width, as its median and 90th
//   percentile over seeds;
// - the range of the above ÷ below ratio over the grid, as the mean (sd) of its low and high ends.
//
// Usage: node tools/boot-replicates.mjs [--seeds 20] [--B 250,500,1000,2000] [--ref 8000]

import { readFileSync } from 'node:fs';
import { bootstrapBand } from '../src/core/bootstrap.js';
import { readDataset } from '../src/core/datasets.js';
import { simulate } from '../src/core/signals.js';
import { createRng } from '../src/core/rng.js';

const DATA_SETS = ['neuroimage', 'dog-ages', 'ozone'];
const SINE_SIGMA = 0.3; // noise sd of the simulated sine, units of y
const SIM_SEED = 1; // seed of the simulated data
const SKEW_SEED = 5; // seed of the skewed noise
const REF_SEED = 99999; // seed of the reference band, outside the seeds studied

function parseArgs(argv) {
    const opts = { seeds: 20, Bs: [250, 500, 1000, 2000], ref: 8000 };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--seeds')
            opts.seeds = Number(argv[++i]);
        else if (argv[i] === '--B')
            opts.Bs = argv[++i].split(',').map(Number);
        else if (argv[i] === '--ref')
            opts.ref = Number(argv[++i]);
        else
            throw new Error(`unknown argument ${argv[i]}`);
    }
    return opts;
}

function dataSets() {
    const sets = [];
    for (const id of DATA_SETS) {
        const d = readDataset(readFileSync(new URL(`../data/${id}.csv`, import.meta.url), 'utf8'));
        sets.push({ name: `${id} (n=${d.n})`, x: d.x, y: d.y });
    }
    const gauss = simulate('sine', 200, SINE_SIGMA, SIM_SEED);
    sets.push({ name: 'sine n=200 gauss', x: gauss.x, y: gauss.y });
    // Exp(1) − 1: mean 0, sd 1, right-skewed (as tools/sim-band.mjs --noise skew)
    const skew = simulate('sine', 400, SINE_SIGMA, SIM_SEED);
    const rng = createRng(SKEW_SEED);
    const y = Float64Array.from(skew.f, f => f + SINE_SIGMA * (-Math.log(1 - rng.uniform()) - 1));
    sets.push({ name: 'sine n=400 skew', x: skew.x, y });
    return sets;
}

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

function sd(a) {
    const m = mean(a);
    return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}

function percentile(a, p) {
    const s = a.slice().sort((u, v) => u - v);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

const f2 = v => v.toFixed(2);
const f3 = v => v.toFixed(3);

function ratioRange(p) {
    const r = p.up.map((u, j) => u / p.down[j]);
    return [Math.min(...r), Math.max(...r)];
}

/** The largest gap between a band's edges and the reference's, in units of the reference's half-width. */
function worstEdgeGap(band, ref) {
    let e = 0;
    for (let j = 0; j < band.grid.length; j++) {
        e = Math.max(e, Math.abs(band.upper[j] - ref.upper[j]) / ref.half[j],
            Math.abs(band.lower[j] - ref.lower[j]) / ref.half[j]);
    }
    return e;
}

function study(set, opts) {
    const ref = bootstrapBand(set.x, set.y, { B: opts.ref, seed: REF_SEED });
    const rp = ref.params;
    const [refLo, refHi] = ratioRange(rp);
    console.log(`\n${set.name}: grid ${ref.grid.length}; reference B=${opts.ref}: q below/above `
        + `${f2(mean(rp.qBelow))}/${f2(mean(rp.qAbove))}, above÷below ${f2(refLo)}–${f2(refHi)}, `
        + `tail level ${rp.tailLevel.toExponential(1)}`);
    console.log('B | ms | sd q below | sd q above | mean half-width vs ref | edge error/half: median, 90% | '
        + 'ratio range: mean (sd) lo – hi');
    for (const B of opts.Bs) {
        const qb = [];
        const qa = [];
        const hw = [];
        const edge = [];
        const lo = [];
        const hi = [];
        const ms = [];
        for (let seed = 1; seed <= opts.seeds; seed++) {
            const t = performance.now();
            const band = bootstrapBand(set.x, set.y, { B, seed });
            ms.push(performance.now() - t);
            const p = band.params;
            qb.push(mean(p.qBelow));
            qa.push(mean(p.qAbove));
            hw.push(mean(band.half) / mean(ref.half));
            edge.push(worstEdgeGap(band, ref));
            const [l, h] = ratioRange(p);
            lo.push(l);
            hi.push(h);
        }
        console.log(`${B} | ${mean(ms).toFixed(0)} | ${f3(sd(qb))} | ${f3(sd(qa))} | ${f3(mean(hw))} ± ${f3(sd(hw))} | `
            + `${f3(percentile(edge, 0.5))}, ${f3(percentile(edge, 0.9))} | `
            + `${f2(mean(lo))} (${f2(sd(lo))}) – ${f2(mean(hi))} (${f2(sd(hi))})`);
    }
}

const opts = parseArgs(process.argv.slice(2));
for (const set of dataSets())
    study(set, opts);
