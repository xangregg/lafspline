// Built-in test signals on t ∈ [0, 1] and simulated data y = f(tᵢ) + σ·Zᵢ with tᵢ = i/n (paper §1).
// Donoho–Johnstone constants as in WaveLab's MakeSignal. SIGNALS are the app's menu; Blocks, Bumps and Doppler
// (TEST_SIGNALS) are not offered there (lafspline: academic tests of jump and spike detection, closer to tautfit's
// subject) but simulate as before, for the tests and tools that use them.

import { createRng } from './rng.js';

const DJ_POS = [0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81];
const BLOCKS_HGT = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2];
const BUMPS_HGT = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2];
const BUMPS_WTH = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005];
const DJ_SD = 7; // conventional sd(f) for the Donoho–Johnstone signals with σ = 1
const BOX_CENTER = 0.5; // eq. (18)
const BOX_HALF_WIDTH = 0.01; // eq. (18): b((t − 1/2)/0.01), b = 1 on [−1, 1]
const BOX_EDGE_TOL = 1e-9; // relative; keeps design points on the closed edges (e.g. t = 0.49) inside
const RAMP_CENTER = 0.5;
const RAMP_SCALE = 0.04; // logistic ramp steepness, units of t
const STEP_EDGES = [1 / 3, 2 / 3]; // three equal flat spans, at the STEP_LEVELS
const STEP_LEVELS = [0, 1, 0.5];
const STEP_EDGE_TOL = 1e-9; // relative; t on an edge (n a multiple of 3) belongs to the span it starts

function blocks(t) {
    let f = 0;
    for (let j = 0; j < DJ_POS.length; j++)
        f += BLOCKS_HGT[j] * (1 + Math.sign(t - DJ_POS[j])) / 2;
    return f;
}

function bumps(t) {
    let f = 0;
    for (let j = 0; j < DJ_POS.length; j++)
        f += BUMPS_HGT[j] / Math.pow(1 + Math.abs((t - DJ_POS[j]) / BUMPS_WTH[j]), 4);
    return f;
}

function heavisine(t) {
    return 4 * Math.sin(4 * Math.PI * t) - Math.sign(t - 0.3) - Math.sign(0.72 - t);
}

function doppler(t) {
    return Math.sqrt(t * (1 - t)) * Math.sin(2 * Math.PI * 1.05 / (t + 0.05));
}

function step(t) {
    let k = 0;
    while (k < STEP_EDGES.length && t >= STEP_EDGES[k] * (1 - STEP_EDGE_TOL))
        k++;
    return STEP_LEVELS[k];
}

/**
 * @typedef {object} SignalSpec
 * @property {string} label
 * @property {function(number): number} f   signal on t ∈ [0, 1]
 * @property {number} n        default sample size
 * @property {number} sigma    default noise sd
 * @property {number|null} sd  if set, f is rescaled so its sd over the design points equals this
 */

/** @type {Record<string, SignalSpec>} the app's simulated signals */
export const SIGNALS = {
    heavisine: { label: 'HeaviSine', f: heavisine, n: 1024, sigma: 1, sd: DJ_SD },
    exp5: { label: 'exp(5t) (paper Fig. 3)', f: t => Math.exp(5 * t), n: 1000, sigma: 5, sd: null },
    sine: { label: 'sin(4πt) (paper Fig. 6)', f: t => Math.sin(4 * Math.PI * t), n: 500, sigma: 0.2, sd: null },
    box: {
        label: 'Narrow box (paper eq. 18)',
        f: t => (Math.abs((t - BOX_CENTER) / BOX_HALF_WIDTH) <= 1 + BOX_EDGE_TOL ? 1 : 0),
        n: 2000, sigma: 0.25, sd: null,
    },
    step: { label: 'Step (thirds at 0, 1, ½)', f: step, n: 500, sigma: 0.25, sd: null },
    noise: { label: 'Pure noise', f: () => 0, n: 500, sigma: 1, sd: null },
    line: { label: 'Straight line', f: t => t, n: 500, sigma: 0.25, sd: null },
    ramp: {
        label: 'Logistic ramp',
        f: t => 1 / (1 + Math.exp(-(t - RAMP_CENTER) / RAMP_SCALE)),
        n: 500, sigma: 0.2, sd: null,
    },
};

/** @type {Record<string, SignalSpec>} signals for tests and tools only, not in the app's menu */
export const TEST_SIGNALS = {
    blocks: { label: 'Blocks', f: blocks, n: 1024, sigma: 1, sd: DJ_SD },
    bumps: { label: 'Bumps', f: bumps, n: 1024, sigma: 1, sd: DJ_SD },
    doppler: { label: 'Doppler', f: doppler, n: 1024, sigma: 1, sd: DJ_SD },
};

function specOf(name) {
    return SIGNALS[name] ?? TEST_SIGNALS[name];
}

/**
 * Design points tᵢ = i/n, i = 1..n (paper §1); 0-based index i holds (i + 1)/n.
 *
 * @param {number} n
 * @returns {Float64Array}
 */
export function designPoints(n) {
    const t = new Float64Array(n);
    for (let i = 0; i < n; i++)
        t[i] = (i + 1) / n;
    return t;
}

/**
 * Evaluate a built-in signal at the design points, rescaled to sd `sd` if given.
 *
 * @param {string} name  key of SIGNALS or TEST_SIGNALS
 * @param {number} n
 * @param {number|null} [sd]  target sd over the design points; defaults to the signal's spec
 * @returns {Float64Array}
 */
export function signalValues(name, n, sd = specOf(name)?.sd ?? null) {
    const spec = specOf(name);
    if (!spec)
        throw new RangeError(`unknown signal '${name}'`);
    const t = designPoints(n);
    const f = new Float64Array(n);
    for (let i = 0; i < n; i++)
        f[i] = spec.f(t[i]);
    if (sd !== null) {
        let mean = 0;
        for (let i = 0; i < n; i++)
            mean += f[i];
        mean /= n;
        let ss = 0;
        for (let i = 0; i < n; i++)
            ss += (f[i] - mean) ** 2;
        const current = Math.sqrt(ss / (n - 1));
        // Scale about zero, not the mean, so features keep their meaning (e.g. Bumps stays ≥ 0).
        if (current > 0) {
            for (let i = 0; i < n; i++)
                f[i] *= sd / current;
        }
    }
    return f;
}

/**
 * A built-in signal as a function of t, with the same rescaling signalValues applies for sample size n, so it can be
 * evaluated between the design points (lafspline: true f on a check grid).
 *
 * @param {string} name  key of SIGNALS or TEST_SIGNALS
 * @param {number} n     the sample size whose rescaling to use
 * @returns {function(number): number}
 */
export function signalFunction(name, n) {
    const spec = specOf(name);
    if (!spec)
        throw new RangeError(`unknown signal '${name}'`);
    if (spec.sd === null)
        return t => spec.f(t);
    const values = signalValues(name, n);
    const t = designPoints(n);
    let k = 0;
    for (let i = 0; i < n; i++) {
        if (Math.abs(spec.f(t[i])) > Math.abs(spec.f(t[k])))
            k = i;
    }
    const scale = spec.f(t[k]) === 0 ? 1 : values[k] / spec.f(t[k]);
    return u => scale * spec.f(u);
}

/**
 * Simulate y = f(tᵢ) + σ·Zᵢ.
 *
 * @param {string} name   key of SIGNALS or TEST_SIGNALS
 * @param {number} n      sample size
 * @param {number} sigma  noise sd, units of f
 * @param {number} seed   seed for `createRng`
 * @param {{sd?: number|null}} [opts]  override the signal's sd rescaling (null = none)
 * @returns {{x: Float64Array, f: Float64Array, y: Float64Array}}  x = design points tᵢ
 */
export function simulate(name, n, sigma, seed, opts = {}) {
    const sd = 'sd' in opts ? opts.sd : specOf(name)?.sd ?? null;
    const f = signalValues(name, n, sd);
    const rng = createRng(seed);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++)
        y[i] = f[i] + sigma * rng.normal();
    return { x: designPoints(n), f, y };
}
