// App state ⇄ URL hash, for shareable links and reloads.
//
// Only settings that differ from the defaults are written, as readable key=value pairs
// (#source=sine&ambition=-2&alpha=0.9). Pasted data goes in `data`, deflate-compressed
// and base64url-encoded, when the result is short enough for a link; larger data is left out and the
// caller says so. Reading validates every value against the same ranges the controls allow and drops
// anything else, so a hand-edited or stale link can never put the app in an impossible state.
// No DOM use: runs in Node for tests (CompressionStream is global in Node ≥ 18 and in browsers).
// Copied with the field table rewritten for lafspline's state (see VENDORED.md).

import { SIGNALS } from '../core/signals.js';
import { LOG_LAMBDA_RANGE } from '../core/pspline.js';
import { WEIGHTS } from '../core/bootstrap.js';

const MAX_DATA_CHARS = 8000; // encoded data longer than this stays out of the link (chat apps truncate long URLs)
export const MAX_N = 5000; // simulated sample size cap, used by controls.js
export const MAX_SEGMENTS = 400; // phase-1 basis segments, used by controls.js
export const AMBITION_RANGE = [-8, 4]; // decades of a/a* (fit.js leavingAmbition), used by controls.js
export const LOG_KAPPA_RANGE = [1, 8]; // log₁₀ κ
export const LOG_EPSILON_RANGE = [-6, 0]; // log₁₀ ε
export const UNDERSMOOTH_VALUES = [1, 1 / 4, 1 / 16, 1 / 64];
export const PENALTIES = ['quadratic', 'sparse', 'shaped', 'power']; // ambition penalty: squared differences, their
// count, the count of bends with rounding (fit.js fitShaped), or Σ|second differences|^q (fit.js fitPower)
export const POWER_RANGE = [0.1, 2]; // q of the power penalty
export const BEND_LENGTH_RANGE = [0, 0.25]; // shaped bends' length L, fraction of the x range
export const SHAPE_COST_RANGE = [0, 20]; // shaped bends' extremum and inflection costs, in plain bends
export const FAMILIES = ['blocks', 'quarter', 'every', 'all']; // Aₙ intervals
export const BOX_VIEWS = ['tightest', 'all', 'off']; // Aₙ boxes drawn on the chart
export const AN_MAX_SEGMENTS = 100; // phase-2 basis for Aₙ: its long intervals make the Newton system dense
export const TAU_RANGE = [0.1, 20];
export const REGIONS = ['band', 'boot', 'An']; // P-spline band, bootstrap band, Aₙ

const MAX_COLUMN = 10000; // column index sanity cap
const USER = 'user';

const enumOf = values => ({
    write: v => String(v),
    read: s => (values.includes(s) ? s : undefined),
});
const num = (min, max = Infinity) => ({
    write: v => String(v),
    read: s => {
        const v = Number(s);
        return s !== '' && Number.isFinite(v) && v >= min && v <= max ? v : undefined;
    },
});
const int = (min, max) => ({
    write: v => String(v),
    read: s => {
        const v = Number(s);
        return s !== '' && Number.isInteger(v) && v >= min && v <= max ? v : undefined;
    },
});
const bool = {
    write: v => (v ? '1' : '0'),
    read: s => (s === '1' ? true : s === '0' ? false : undefined),
};

/**
 * n and σ a simulated signal starts with: its own σ, and its own n capped at MAX_N.
 *
 * @param {string} source  key of SIGNALS
 * @returns {{n: number, sigma: number}|null}  null for user data
 */
export function signalDefaults(source) {
    const spec = SIGNALS[source];
    return spec ? { n: Math.min(spec.n, MAX_N), sigma: spec.sigma } : null;
}

// bundled data sets (data/) and local ones (localdata/, this machine only); the app checks the id exists
const EXAMPLE = /^(example|local):[a-z0-9][a-z0-9-]*$/;

// Every persisted state key with its codec. Ranges match the inputs in controls.js.
const FIELDS = {
    source: {
        write: v => v,
        read: s => (SIGNALS[s] || s === USER || EXAMPLE.test(s) ? s : undefined),
    },
    n: int(3, MAX_N),
    sigma: num(0),
    seed: int(-1e9, 1e9),
    showTruth: bool,
    xCol: int(-1, MAX_COLUMN),
    yCol: int(0, MAX_COLUMN),
    region: enumOf(REGIONS),
    family: enumOf(FAMILIES),
    calibrate: bool,
    tau: num(...TAU_RANGE),
    boxes: enumOf(BOX_VIEWS),
    alpha: num(0.5, 0.999),
    covariance: enumOf(['bayes', 'sandwich']),
    bootWeights: enumOf(WEIGHTS),
    lambdaMode: enumOf(['gcv', 'aic', 'silverman', 'rot', 'given']),
    logLambda: num(...LOG_LAMBDA_RANGE),
    // written as the divisor (16 for λ × 1/16): a slash would be percent-encoded in the link
    undersmooth: {
        write: v => String(Math.round(1 / v)),
        read: s => UNDERSMOOTH_VALUES.find(v => s === String(Math.round(1 / v))),
    },
    nseg: {
        write: v => (v === null ? 'auto' : String(v)),
        read: s => {
            if (s === 'auto')
                return null;
            const v = Number(s);
            return s !== '' && Number.isInteger(v) && v >= 3 && v <= MAX_SEGMENTS ? v : undefined;
        },
    },
    diff: int(1, 3),
    penalty: enumOf(PENALTIES),
    ambition: num(...AMBITION_RANGE),
    flat: num(0, 1),
    bendLength: num(...BEND_LENGTH_RANGE),
    extremumCost: num(...SHAPE_COST_RANGE),
    inflectionCost: num(...SHAPE_COST_RANGE),
    q: num(...POWER_RANGE),
    logKappa: num(...LOG_KAPPA_RANGE),
    logEpsilon: num(...LOG_EPSILON_RANGE),
    showData: bool,
    showCenter: bool,
    showBand: bool,
    showFit: bool,
};
const USER_ONLY = new Set(['xCol', 'yCol']);
const SIMULATED_ONLY = new Set(['n', 'sigma', 'seed', 'showTruth']);

// n and σ default to the chosen signal's starting values, so a link names them only when changed.
function defaultFor(key, st, defaults) {
    const sim = signalDefaults(st.source);
    if (sim && key === 'n')
        return sim.n;
    if (sim && key === 'sigma')
        return sim.sigma;
    return defaults[key];
}

function toBase64Url(bytes) {
    let bin = '';
    for (let i = 0; i < bytes.length; i++)
        bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(s) {
    const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/'));
    return Uint8Array.from(bin, ch => ch.charCodeAt(0));
}

async function pipe(bytes, stream) {
    const out = new Blob([bytes]).stream().pipeThrough(stream);
    return new Uint8Array(await new Response(out).arrayBuffer());
}

/**
 * Compress text for a URL: deflate-raw, then base64url.
 *
 * @param {string} text
 * @returns {Promise<string>}
 */
export async function packText(text) {
    return toBase64Url(await pipe(new TextEncoder().encode(text), new CompressionStream('deflate-raw')));
}

/**
 * Inverse of packText.
 *
 * @param {string} packed
 * @returns {Promise<string>}  rejects on malformed input
 */
export async function unpackText(packed) {
    return new TextDecoder().decode(await pipe(fromBase64Url(packed), new DecompressionStream('deflate-raw')));
}

/**
 * The URL hash for a state.
 *
 * @param {object} st        app state (see main.js)
 * @param {object} defaults  the app's initial state
 * @returns {Promise<{hash: string, dataOmitted: number}>}  hash is '' or starts with '#';
 *     dataOmitted = length of pasted text left out of the link (0 if none)
 */
export async function stateToHash(st, defaults) {
    const params = new URLSearchParams();
    const isUser = st.source === USER;
    const isSimulated = Boolean(SIGNALS[st.source]);
    for (const [key, codec] of Object.entries(FIELDS)) {
        if ((USER_ONLY.has(key) && !isUser) || (SIMULATED_ONLY.has(key) && !isSimulated))
            continue;
        const value = st[key];
        if (value === undefined || (value === null && key !== 'nseg'))
            continue;
        if (value !== defaultFor(key, st, defaults))
            params.set(key, codec.write(value));
    }
    let dataOmitted = 0;
    if (isUser && st.userText) {
        const packed = await packText(st.userText);
        if (packed.length <= MAX_DATA_CHARS)
            params.set('data', packed);
        else
            dataOmitted = st.userText.length;
    }
    const s = params.toString();
    return { hash: s ? `#${s}` : '', dataOmitted };
}

/**
 * State patch from a URL hash: the defaults overridden by every valid value in the hash.
 * Invalid or unknown entries are ignored and listed in `rejected`.
 *
 * @param {string} hash      location.hash ('' or '#…')
 * @param {object} defaults  the app's initial state
 * @returns {Promise<{state: object, rejected: string[]}>}  a complete state
 */
export async function hashToState(hash, defaults) {
    const params = new URLSearchParams(hash.replace(/^#/, ''));
    const st = structuredClone(defaults);
    const rejected = [];
    const source = FIELDS.source.read(params.get('source') ?? '');
    if (source !== undefined)
        st.source = source;
    else if (params.has('source'))
        rejected.push('source');
    const sim = signalDefaults(st.source);
    if (sim) {
        st.n = sim.n;
        st.sigma = sim.sigma;
    }
    for (const [key, raw] of params) {
        if (key === 'source' || key === 'data')
            continue;
        const codec = FIELDS[key];
        const value = codec ? codec.read(raw) : undefined;
        if (value === undefined)
            rejected.push(key);
        else
            st[key] = value;
    }
    if (params.has('data')) {
        try {
            st.userText = await unpackText(params.get('data'));
        }
        catch {
            rejected.push('data');
        }
    }
    return { state: st, rejected };
}
