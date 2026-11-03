// Phase-0 QP risk check (PLAN §3, phase 0): the phase-2 objective solved by iterated weights (src/core/fit.js
// fitIterated) and as a HiGHS QP (buildQP, soft region; and the hard region), on the cases of bench-qp-cases.js.
// Reports, per case: iterations and time of the iteration; QP status, iterations and time; the largest difference
// between the two fits relative to the half-width; the relative objective difference.
//
// Usage: node tools/bench-qp.mjs [--sizes '[[43,200],[103,800]]'] [--time-limit seconds] [--no-hard]

import loadHighs from 'highs';
import { benchCases } from './bench-qp-cases.js';
import { fitIterated, buildQP, qpResult } from '../src/core/fit.js';
import { solveModel } from '../src/core/lp-solve.js';

const DEFAULT_TIME_LIMIT = 30; // s per QP solve

function parseArgs(argv) {
    const opts = { timeLimit: DEFAULT_TIME_LIMIT, hard: true };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--sizes')
            opts.sizes = JSON.parse(argv[++i]);
        else if (argv[i] === '--time-limit')
            opts.timeLimit = Number(argv[++i]);
        else if (argv[i] === '--no-hard')
            opts.hard = false;
    }
    return opts;
}

/**
 * Compare one case's iterated fit with a QP result.
 *
 * @returns {{maxDiff: number, objDiff: number}}  max_j |Δg_j|/h_j and |ΔS|/max(1, S)
 */
export function compare(spec, it, qp) {
    if (!qp.g)
        return { maxDiff: NaN, objDiff: NaN };
    let maxDiff = 0;
    for (let j = 0; j < qp.g.length; j++)
        maxDiff = Math.max(maxDiff, Math.abs(qp.g[j] - it.g[j]) / spec.h[j]);
    return { maxDiff, objDiff: Math.abs(qp.objective - it.objective) / Math.max(1, it.objective) };
}

const fmtMs = ms => ms < 10 ? ms.toFixed(1) : ms.toFixed(0);

async function main() {
    const opts = parseArgs(process.argv.slice(2));
    const highs = await loadHighs();
    const cases = benchCases({ sizes: opts.sizes });
    const summary = { cases: 0, maxIterations: 0, notConverged: 0, maxIterMs: 0, softFailures: 0, maxDiff: 0,
        maxObjDiff: 0, softMs: [], hardFailures: 0, hardFeasibleFailures: 0 };
    console.log('case | iter: n, ms, outside, max ratio | soft QP: status, its, ms, max|Δg|/h, |ΔS| | hard QP: status, ms');
    for (const c of cases) {
        const it = fitIterated(c.spec);
        const soft = qpResult(c.spec, solveModel(highs, buildQP(c.spec).model, { time_limit: opts.timeLimit },
            ['qp_iteration_count']));
        const softRaw = soft;
        const d = compare(c.spec, it, soft);
        let line = `${c.name} | ${it.iterations}${it.damped ? ` (${it.damped} damped)` : ""}${it.converged ? '' : ' (not converged)'}, ${fmtMs(it.ms)} ms, `
            + `${it.outside}, ${it.maxRatio.toFixed(3)} | ${soft.status}, ${fmtMs(soft.ms)} ms, `
            + `${d.maxDiff.toExponential(1)}, ${d.objDiff.toExponential(1)}`;
        summary.cases++;
        summary.maxIterations = Math.max(summary.maxIterations, it.iterations);
        summary.notConverged += it.converged ? 0 : 1;
        summary.maxIterMs = Math.max(summary.maxIterMs, it.ms);
        if (softRaw.status !== 'optimal')
            summary.softFailures++;
        else {
            summary.maxDiff = Math.max(summary.maxDiff, d.maxDiff);
            summary.maxObjDiff = Math.max(summary.maxObjDiff, d.objDiff);
            summary.softMs.push({ K: c.K, G: c.G, ms: soft.ms });
        }
        if (opts.hard) {
            const hard = qpResult(c.spec, solveModel(highs, buildQP(c.spec, { hard: true }).model,
                { time_limit: opts.timeLimit }));
            line += ` | ${hard.status}, ${fmtMs(hard.ms)} ms`;
            if (hard.status !== 'optimal') {
                summary.hardFailures++;
                // the soft fit inside the region proves the hard problem feasible
                if (it.outside === 0)
                    summary.hardFeasibleFailures++;
            }
        }
        console.log(line);
    }
    console.log('\nsummary');
    console.log(`  cases: ${summary.cases}`);
    console.log(`  iterated weights: max ${summary.maxIterations} iterations, ${summary.notConverged} not converged, `
        + `slowest ${fmtMs(summary.maxIterMs)} ms`);
    console.log(`  soft QP: ${summary.softFailures} not optimal; vs iteration max |Δg|/h ${summary.maxDiff.toExponential(2)}, `
        + `max |ΔS|/S ${summary.maxObjDiff.toExponential(2)}`);
    const bySize = new Map();
    for (const { K, G, ms } of summary.softMs) {
        const key = `K=${K} G=${G}`;
        bySize.set(key, [...(bySize.get(key) ?? []), ms]);
    }
    for (const [key, list] of bySize) {
        list.sort((a, b) => a - b);
        console.log(`    ${key}: median ${fmtMs(list[list.length >> 1])} ms, max ${fmtMs(list[list.length - 1])} ms`);
    }
    if (opts.hard)
        console.log(`  hard QP: ${summary.hardFailures} not optimal, ${summary.hardFeasibleFailures} of them provably feasible`);
}

main();
