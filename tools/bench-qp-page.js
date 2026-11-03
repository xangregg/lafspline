// Browser half of the phase-0 QP risk check (see tools/bench-qp.mjs): the same cases, the iteration on the main
// thread and the soft-region QP in the solver worker, so the timings include the worker round trip.

import { benchCases } from './bench-qp-cases.js';
import { fitIterated, buildQP, qpResult } from '../src/core/fit.js';
import { solveModelInWorker } from '../src/ui/lp.js';

const SIZES = [[43, 200], [103, 800]];
const LARGE = { sizes: [[203, 2000]], signals: ['heavisine'], ambitions: [[0, 1], [0, 1e4]] };
const TIME_LIMIT = 60; // s per QP

const out = document.getElementById('out');

async function run() {
    const cases = [...benchCases({ sizes: SIZES }), ...benchCases(LARGE)];
    const rows = [];
    const lines = [];
    for (const c of cases) {
        const it = fitIterated(c.spec);
        const start = performance.now();
        const sol = await solveModelInWorker(buildQP(c.spec).model, { time_limit: TIME_LIMIT });
        const roundTrip = performance.now() - start;
        const qp = qpResult(c.spec, sol);
        let maxDiff = NaN;
        if (qp.g) {
            maxDiff = 0;
            for (let j = 0; j < qp.g.length; j++)
                maxDiff = Math.max(maxDiff, Math.abs(qp.g[j] - it.g[j]) / c.spec.h[j]);
        }
        rows.push({ name: c.name, K: c.K, G: c.G, iterations: it.iterations, converged: it.converged,
            iterMs: it.ms, qpStatus: qp.status, qpMs: sol.ms, roundTripMs: roundTrip, maxDiff });
        lines.push(`${c.name} | iter ${it.iterations}${it.converged ? '' : ' NOT CONVERGED'} ${it.ms.toFixed(1)} ms`
            + ` | QP ${qp.status} ${sol.ms.toFixed(0)} ms (round trip ${roundTrip.toFixed(0)} ms)`
            + ` | max |Δg|/h ${maxDiff.toExponential(1)}`);
        out.textContent = lines.join('\n') + '\n…';
    }
    const ok = rows.filter(r => r.qpStatus === 'optimal');
    const summary = {
        cases: rows.length,
        notConverged: rows.filter(r => !r.converged).length,
        maxIterations: Math.max(...rows.map(r => r.iterations)),
        maxIterMs: Math.max(...rows.map(r => r.iterMs)),
        qpNotOptimal: rows.length - ok.length,
        maxDiff: Math.max(...ok.map(r => r.maxDiff)),
        qpMsBySize: Object.fromEntries([...new Set(rows.map(r => `K=${r.K} G=${r.G}`))].map(key => {
            const ms = ok.filter(r => `K=${r.K} G=${r.G}` === key).map(r => r.qpMs).sort((a, b) => a - b);
            return [key, { median: ms[ms.length >> 1], max: ms[ms.length - 1] }];
        })),
    };
    out.textContent = lines.join('\n') + '\n\nsummary ' + JSON.stringify(summary, null, 2);
    window.benchResult = { summary, rows };
    document.body.dataset.ready = 'true';
}

run().catch(err => {
    out.textContent = `Error: ${err.message}`;
    console.error(err);
});
