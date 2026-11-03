// Cases for the phase-0 QP risk check (tools/bench-qp.mjs in Node, tools/bench-qp.html in the browser worker):
// a phase-1 region from a GCV P-spline fit on a simulated signal, then the phase-2 objective at several sizes and
// ambitions. The region here is a stand-in (center ± Q_CRIT Bayesian standard errors, no Monte Carlo critical
// value); the solvers only need some region of realistic shape. Pure: runs unchanged in Node and the browser.

import { simulate } from '../src/core/signals.js';
import { selectLambda, evaluate, standardErrors } from '../src/core/pspline.js';

const Q_CRIT = 3; // stand-in critical value for the band half-width
const SEED = 20261102;
const N = 400; // observations per simulated data set
const DEG = 3;
// noise sd per signal (units of f); the Donoho–Johnstone signals are rescaled to sd 7
const NOISE = { sine: 0.2, heavisine: 1, blocks: 1, line: 0.25 };
export const KAPPA = 1e4;
export const EPSILON = 1e-2;
export const SIZES = [[43, 200], [103, 800], [203, 2000]]; // [K̃ basis functions, G grid points]
export const SIGNALS = ['sine', 'heavisine', 'blocks', 'line'];
export const AMBITIONS = [[0, 1e-2], [0, 1], [0, 1e2], [0, 1e4], [0, 1e6], [1e2, 0]]; // [λ₁, λ₂]

/**
 * The region for one signal on a G-point uniform grid.
 *
 * @param {string} signal
 * @param {number} G
 * @returns {{grid: Float64Array, c: Float64Array, h: Float64Array, xl: number, xr: number}}
 */
export function benchRegion(signal, G) {
    const { x, y } = simulate(signal, N, NOISE[signal], SEED);
    const phase1 = selectLambda(x, y);
    const xl = x[0];
    const xr = x[x.length - 1];
    const grid = Float64Array.from({ length: G }, (_, j) => xl + (xr - xl) * j / (G - 1));
    const c = evaluate(phase1.fit, grid);
    const h = standardErrors(phase1.fit, grid).map(v => Q_CRIT * v);
    return { grid, c, h, xl, xr };
}

/**
 * All benchmark cases.
 *
 * @param {{sizes?: number[][], signals?: string[], ambitions?: number[][]}} [opts]
 * @returns {{name: string, K: number, G: number, signal: string, lambda1: number, lambda2: number,
 *     spec: import('../src/core/fit.js').Phase2Spec}[]}
 */
export function benchCases(opts = {}) {
    const cases = [];
    for (const [K, G] of opts.sizes ?? SIZES) {
        for (const signal of opts.signals ?? SIGNALS) {
            const region = benchRegion(signal, G);
            for (const [lambda1, lambda2] of opts.ambitions ?? AMBITIONS) {
                cases.push({
                    name: `${signal} K=${K} G=${G} λ1=${lambda1} λ2=${lambda2}`,
                    K, G, signal, lambda1, lambda2,
                    spec: { ...region, nseg: K - DEG, deg: DEG, kappa: KAPPA, epsilon: EPSILON, lambda1, lambda2 },
                });
            }
        }
    }
    return cases;
}
