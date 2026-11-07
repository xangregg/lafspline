// App entry point: state (⇄ URL hash) → data → region (phase 1: a P-spline band, a bootstrap band, or Aₙ) → fit
// (phase 2) → chart and readouts.
//
// Everything runs on the main thread: the band takes milliseconds to tens of milliseconds (one λ search, one Monte
// Carlo of B = 1000 cheap draws; the bootstrap band's replicates are each a banded refit, 0.2 s for 2000 at
// n = 200 to 1.4 s for 1000 at n = 5000), a* (fit.js leavingAmbition) some 15–25 phase-2 fits of a few milliseconds each
// (sparse fits are some 10–30 iterated fits each, so a* takes up to about half a second for n in the thousands),
// and a knob change one more fit. Aₙ takes a calibration of τ (B = 1000 draws of its multiscale statistic, tens to
// hundreds of milliseconds, cached by design and interval family) and its phase-2 solves are dense (at most
// AN_MAX_SEGMENTS segments). Regions and a* are cached by data and settings; the ladder (a few milliseconds) is
// kept on the region.

import * as d3 from 'd3';
import { simulate, signalFunction, SIGNALS } from '../core/signals.js';
import { parseTable, prepareXY, rowIndex } from '../core/data.js';
import { readDataset, splitMetadata, datasetId, indexEntry } from '../core/datasets.js';
import { bandRegion, checkBand, checkGrid } from '../core/band.js';
import { bootstrapBand } from '../core/bootstrap.js';
import { defaultSegments } from '../core/pspline.js';
import {
    regionSpec, leavingAmbition, fitIterated, fitSparse, sparseLambdas, fitShaped, shapedLambdas, splineValues,
    fitPower, powerLambdas,
    LEAVE_TOL, SHAPE_DEFAULTS,
} from '../core/fit.js';
import { ladder } from '../core/ladder.js';
import { anRegion, calibrateTau, checkFit } from '../core/regions.js';
import { anBoxes } from '../core/an.js';
import { checkNoise } from '../core/noisecheck.js';
import { lambdaSourceName, lambdaScale } from '../core/thumb.js';
import { familyIntervals, familyLevels } from '../core/intervals.js';
import { tauFromSample } from '../core/calibrate.js';
import { constraintBoxes } from '../core/region.js';
import { createControls, sourceSelect } from './controls.js';
import { infoIcon } from './info.js';
import { stateToHash, hashToState, AN_MAX_SEGMENTS, MAX_N, PANE_COUNTS, FEATURE_SCALE_START } from './urlstate.js';
import { fmtExp } from './format.js';
import { createMainChart, drawLegend } from './chart.js';
import { cellEdges } from './layout.js';
import { onThemeChange } from './theme.js';
import { toCsv, fitTables } from './export.js';
import { createHistory } from './history.js';

const fmt = d3.format('.4~g');
const fmtInt = d3.format(',');
const fmtShort = d3.format('.2~g');
const fmtShort3 = d3.format('.3~g');
const CACHE_LIMIT = 24; // bands kept
const BAND_B = 1000; // Monte Carlo draws for the band's critical value
const BAND_SEED = 1;
// Bootstrap band replicates. Over 20 seeds against B = 8000, the band's edges somewhere on the grid were off by a
// median 0.11–0.17 of the half-width at B = 1000 and 0.08–0.12 at 2000, and the above ÷ below range was widened by
// noise by up to 0.09 at 1000 (PLAN §1.2). 2000 costs up to about 0.6 s (at n = 1000); above it, 1000 keeps the
// band under about 1.5 s.
const BOOT_B_SMALL = 2000;
const BOOT_B_LARGE = 1000;
const BOOT_SMALL_N = 1000; // observations, at most, for BOOT_B_SMALL
const AN_B = 1000; // Monte Carlo draws of Aₙ's multiscale statistic for a calibrated τ
const AN_LAMBDA = 2; // Aₙ's interval lengths grow by this factor per level (the paper's)
const CENTER_COLOR = 'series-3';
const FIT_COLOR = 'series-1';
const HASH_DEBOUNCE = 400; // ms after the last change before the URL is updated
const MESSAGE_MS = 4000; // toolbar message display time
const IMAGE_SCALE = 2; // PNG and JPEG export: pixels per CSS pixel of the chart
const JPEG_QUALITY = 0.92;
const IMAGE_TYPES = { png: 'image/png', jpeg: 'image/jpeg' };
const EXTENSIONS = { png: 'png', jpeg: 'jpg' };
const USER = 'user';
const EXAMPLE_PREFIX = 'example:';
const EXAMPLES_INDEX = 'data/index.json';
// Data sets kept on this machine only: localdata/ is ignored by git except its README, so it exists in every clone.
// A static site cannot list a directory. Python's server (npm run serve) can, so there the listing is used and a new
// file appears on reload; other servers (WebStorm's built-in one answers 404) get localdata/index.json, written by
// npm run data:index. The README's response names the server, so neither way requests something that is not there.
// The app asks only when served from this machine, so the published site never requests any of it.
const LOCAL_PREFIX = 'local:';
const LOCAL_DIR = 'localdata/';
const LOCAL_INDEX = 'index.json'; // in LOCAL_DIR: a JSON array of CSV file names
const LOCAL_PROBE = 'README.md'; // in LOCAL_DIR, committed, so it is always there to ask
const LISTING_SERVER = /^SimpleHTTP\//; // the Server header of Python's http.server, which lists directories
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);
// The gallery's default data sets, in order (fewer panes show the first ones): real data with different features
// (level stretches, a level shift, leveling off, a hump), then two signals.
const GALLERY_SOURCES = ['example:gistemp-global', 'example:ozone-annual', 'example:nile-flow', 'example:dog-ages',
    'example:auto-mpg-horsepower', 'example:bike-sharing-temperature', 'ramp', 'step'];
const DEFAULT_PANES = PANE_COUNTS[2];

const state = {
    view: 'single', // 'single' (one chart with its readouts) | 'gallery' (several data sets, the settings shared)
    panes: GALLERY_SOURCES.slice(0, DEFAULT_PANES), // the gallery's sources
    source: 'ramp',
    n: SIGNALS.ramp.n,
    sigma: SIGNALS.ramp.sigma,
    seed: 1,
    showTruth: true, // chart layer: the true f of simulated data (its check in the readouts is always shown)
    userText: '',
    xCol: null, // column index, −1 = row number, null = auto (first column if there are ≥ 2)
    yCol: null, // column index, null = auto (second column if there are ≥ 2, else the only one)
    region: 'boot', // phase 1: 'band' (P-spline) | 'boot' (bootstrap band) | 'An' (Davies, Kovac & Meise's)
    family: 'blocks', // Aₙ intervals: 'blocks' | 'quarter' (¼-width steps) | 'every' (every position) | 'all'
    calibrate: true, // Aₙ: τ calibrated for coverage alpha, instead of tau
    tau: 3, // Aₙ: τ when not calibrated
    boxes: 'tightest', // Aₙ boxes on the chart: 'tightest' (the level of the largest ratio) | 'all' | 'off'
    alpha: 0.95, // coverage: the band's, and Aₙ's when calibrated
    covariance: 'bayes', // 'bayes' | 'sandwich'
    bootWeights: 'matched', // bootstrap band: 'matched' (like resampled counts) | 'dirichlet' (Bayesian bootstrap)
    lambdaMode: 'gcv', // 'gcv' | 'aic' | 'silverman' | 'rot' (rules of thumb, core/thumb.js) | 'scale' | 'given'
    logLambda: 0, // natural log, when lambdaMode is 'given'
    featureScale: FEATURE_SCALE_START, // lambdaMode 'scale': the shortest features of interest, share of the x range
    undersmooth: 1 / 4, // multiplier on the chosen λ
    nseg: null, // null = auto (pspline.defaultSegments)
    diff: 2, // penalty order
    penalty: 'shaped', // phase-2 ambition: 'quadratic' (squared differences) | 'sparse' (their number) |
    // 'shaped' (the number of bends, rounded) | 'power' (Σ|second differences|^q)
    q: 1, // power penalty's exponent: 2 is 'quadratic', 1 is ℓ₁ trend filtering, toward 0 the count
    ambition: -3.4, // decades from a*, the largest ambition before the fit leaves the region
    flat: 0.15, // share of the ambition toward flat (D₁) rather than straight (D₂)
    bendLength: SHAPE_DEFAULTS.length, // rounded bends: fraction of the x range (fit.js fitShaped)
    extremumCost: SHAPE_DEFAULTS.extremum, // rounded bends: extra cost of an extremum, in plain bends
    inflectionCost: SHAPE_DEFAULTS.inflection, // and of an inflection
    logKappa: 4, // log₁₀ κ
    logEpsilon: -2, // log₁₀ ε
    showData: true, // chart layers (the readouts are unchanged): the data points,
    showCenter: true, // the band's P-spline fit (phase 1),
    showBand: true, // the band,
    showFit: true, // and the phase-2 fit
};
// Settings that change only what the chart draws: a change to them redraws without refitting.
const DISPLAY_KEYS = new Set(['showTruth', 'showData', 'showCenter', 'showBand', 'showFit', 'boxes']);
const PANE_SUMMARY_SEP = ' · ';
// What chooses the data (the source, pasted data and its columns, a signal's n, σ and seed, the gallery's panes)
// and the view: Reset, Undo and Redo leave these alone and change only the other settings.
const DATA_KEYS = new Set(['view', 'panes', 'source', 'n', 'sigma', 'seed', 'userText', 'xCol', 'yCol']);
const DEFAULTS = structuredClone(state);
const SETTING_KEYS = Object.keys(DEFAULTS).filter(key => !DATA_KEYS.has(key));

/** The settings that Reset, Undo and Redo act on. */
function settingsOf(st) {
    return Object.fromEntries(SETTING_KEYS.map(key => [key, st[key]]));
}

let tableCache = { text: null, table: null };

function getTable(text) {
    if (tableCache.text !== text)
        tableCache = { text, table: parseTable(text) };
    return tableCache.table;
}

// FNV-1a, to key pasted data without keeping the text in every cache key.
function hashText(text) {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++)
        h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
    return `${text.length}:${h.toString(16)}`;
}

function distinctCount(x) {
    let m = x.length > 0 ? 1 : 0;
    for (let i = 1; i < x.length; i++) {
        if (x[i] !== x[i - 1])
            m++;
    }
    return m;
}

/**
 * The data: observations, true f for simulated data, notes; {error} when there is nothing to fit, {pending, id}
 * while a bundled example is loading (id: its source, example:… or local:…).
 */
function deriveData(st, exampleTexts, examples) {
    let x;
    let y;
    let fAt = null;
    let columns = [];
    let dataKey;
    let label;
    const notes = [];
    if (examples.has(st.source)) {
        const id = st.source;
        const loaded = exampleTexts.get(id);
        label = examples.get(id).label;
        if (loaded === undefined)
            return { pending: true, id, label, columns };
        if (loaded.error)
            return { error: `Could not load the example: ${loaded.error}`, columns };
        const d = readDataset(loaded);
        if (d.errors.length)
            return { error: `Example ${id.slice(id.indexOf(':') + 1)}: ${d.errors.join('; ')}`, columns };
        x = d.x;
        y = d.y;
        columns = [d.xName, d.yName];
        dataKey = st.source;
        notes.push(`Source: ${d.meta.source}`);
        if (d.meta.note)
            notes.push(d.meta.note);
        notes.push(`x = ${d.xName}, y = ${d.yName}`);
        if (d.dropped)
            notes.push(`${fmtInt(d.dropped)} incomplete row(s) dropped`);
        if (d.ties)
            notes.push(`${fmtInt(d.ties)} observations share an x with another.`);
    }
    else if (st.source === USER) {
        // leading '# key: value' lines, as in the example files: skipped, and their x and y name the default columns
        const { meta, body } = splitMetadata(st.userText);
        const table = getTable(body);
        columns = table.names;
        label = meta.label ? `Your data: ${meta.label}` : 'Your data';
        if (table.columns.length === 0)
            return { error: 'Paste or drop data with at least one numeric column.', columns };
        const two = columns.length >= 2;
        const named = (name, fallback) => (name !== undefined && columns.includes(name) ? columns.indexOf(name)
            : fallback);
        const xCol = st.xCol !== null && st.xCol < columns.length ? st.xCol : named(meta.x, two ? 0 : -1);
        const yCol = st.yCol !== null && st.yCol < columns.length ? st.yCol : named(meta.y, two ? 1 : 0);
        st.xCol = xCol;
        st.yCol = yCol;
        const d = prepareXY(xCol < 0 ? rowIndex(table.rows) : table.columns[xCol], table.columns[yCol]);
        x = d.x;
        y = d.y;
        dataKey = `user|${hashText(st.userText)}|${xCol}|${yCol}`;
        if (meta.source)
            notes.push(`Source: ${meta.source}`);
        if (meta.note)
            notes.push(meta.note);
        if (d.dropped)
            notes.push(`${fmtInt(d.dropped)} incomplete row(s) dropped`);
        if (d.ties)
            notes.push(`${fmtInt(d.ties)} observations share an x with another.`);
    }
    else {
        const sim = simulate(st.source, st.n, st.sigma, st.seed);
        x = sim.x;
        y = sim.y;
        fAt = signalFunction(st.source, st.n);
        label = SIGNALS[st.source].label;
        dataKey = `${st.source}|${st.n}|${st.sigma}|${st.seed}`;
        notes.push(`tᵢ = i/n; y = f(t) + ${fmt(st.sigma)}·Z, seed ${st.seed}`);
    }
    const n = y.length;
    const m = distinctCount(x);
    if (m < 4)
        return { error: 'Need at least 4 distinct x values.', columns };
    return { columns, notes, label, x, y, fAt, n, m, dataKey, edges: cellEdges(x) };
}

function bandKey(st, data) {
    let lambda = `${st.lambdaMode}:${st.undersmooth}`;
    if (st.lambdaMode === 'given')
        lambda = `given:${st.logLambda}`;
    else if (st.lambdaMode === 'scale')
        lambda = `scale:${st.featureScale}`;
    const kind = st.region === 'boot' ? `boot:${st.bootWeights}` : st.covariance;
    return [data.dataKey, st.alpha, kind, lambda, st.nseg ?? 'auto', st.diff].join('|');
}

function familyOf(st) {
    if (st.family === 'all')
        return { kind: 'all' };
    if (st.family === 'quarter' || st.family === 'every')
        return { kind: 'overlapping', lambda: AN_LAMBDA, offsets: st.family === 'quarter' ? 4 : Infinity };
    return { kind: 'blocks', lambda: AN_LAMBDA };
}

function anSegments(st, data) {
    const wanted = st.nseg ?? defaultSegments(data.m);
    return { nseg: Math.min(wanted, AN_MAX_SEGMENTS), capped: wanted > AN_MAX_SEGMENTS };
}

function regionKey(st, data) {
    if (st.region !== 'An')
        return bandKey(st, data);
    const tau = st.calibrate ? `cal:${st.alpha}` : st.tau;
    return ['An', data.dataKey, st.family, tau, anSegments(st, data).nseg].join('|');
}

/**
 * Aₙ over the chosen intervals, as boxes for phase 2 (an.js). A calibrated τ needs the null sample of the
 * multiscale statistic, which depends on the design and the intervals, not on y or α: it is cached by both.
 */
function computeAn(st, data, nullCache) {
    const start = performance.now();
    const family = familyOf(st);
    let intervals;
    try {
        intervals = familyIntervals(data.x, family);
    }
    catch (err) {
        if (err instanceof RangeError)
            return { error: err.message };
        throw err;
    }
    let tau = st.tau;
    let calibration = null;
    let calMs = null;
    if (st.calibrate) {
        const key = `${data.dataKey}|${st.family}`;
        let cal = nullCache.get(key);
        if (!cal) {
            const t = performance.now();
            cal = { sample: calibrateTau(data.x, intervals, { alpha: st.alpha, B: AN_B }).sample };
            cal.ms = performance.now() - t;
            remember(nullCache, key, cal);
            calMs = cal.ms;
        }
        const t = tauFromSample(cal.sample, st.alpha, data.n);
        tau = t.tau;
        calibration = { alpha: st.alpha, B: AN_B, family: 'chosen', tauLo: t.tauLo, tauHi: t.tauHi };
    }
    const an = anRegion(data.x, data.y, { family, tau, calibration });
    if (!(an.params.sigma > 0)) {
        return { error: 'σ̂ = 0: neighboring observations agree exactly, so Aₙ admits only the data themselves, '
            + 'which no spline fit can match' };
    }
    const { nseg, capped } = anSegments(st, data);
    const region = anBoxes(an, data.x, { nseg });
    region.claim = an.claim;
    region.nsegCapped = capped;
    return { region, an, intervals, family, calMs, ms: performance.now() - start };
}

function remember(cache, key, value) {
    cache.set(key, value);
    if (cache.size > CACHE_LIMIT)
        cache.delete(cache.keys().next().value);
}

function computeBand(st, data) {
    const boot = st.region === 'boot';
    const opts = boot
        ? { alpha: st.alpha, weights: st.bootWeights, diff: st.diff, seed: BAND_SEED,
            B: data.n <= BOOT_SMALL_N ? BOOT_B_SMALL : BOOT_B_LARGE }
        : { alpha: st.alpha, covariance: st.covariance, diff: st.diff, B: BAND_B, seed: BAND_SEED };
    if (st.nseg !== null)
        opts.nseg = st.nseg;
    if (st.lambdaMode === 'given') {
        opts.lambda = Math.exp(st.logLambda);
    }
    else if (st.lambdaMode === 'scale') {
        opts.criterion = 'scale';
        opts.scale = st.featureScale;
    }
    else {
        opts.criterion = st.lambdaMode;
        opts.undersmooth = st.undersmooth;
    }
    const start = performance.now();
    try {
        const region = boot ? bootstrapBand(data.x, data.y, opts) : bandRegion(data.x, data.y, opts);
        let noise = null; // the check needs 3 distinct x (or replicates) for its reference σ̂
        try {
            noise = checkNoise(data.x, data.y, region.params.fitted, region.params.sigma);
        }
        catch (err) {
            if (!(err instanceof RangeError))
                throw err;
        }
        return { region, noise, ms: performance.now() - start };
    }
    catch (err) {
        if (err instanceof RangeError)
            return { error: err.message };
        throw err;
    }
}

function starKey(st, data) {
    let shape = [st.flat];
    if (st.penalty === 'shaped')
        shape = [st.bendLength, st.extremumCost, st.inflectionCost, st.flat];
    else if (st.penalty === 'power')
        shape = [st.q, st.flat];
    return [regionKey(st, data), st.penalty, ...shape, st.logKappa, st.logEpsilon].join('|');
}

function phase2Options(st) {
    const shaped = st.penalty === 'shaped';
    const power = st.penalty === 'power';
    return { flat: st.flat, kappa: 10 ** st.logKappa, epsilon: 10 ** st.logEpsilon,
        shape: shaped ? { length: st.bendLength, extremum: st.extremumCost, inflection: st.inflectionCost }
            : undefined,
        q: power ? st.q : undefined };
}

/** The positions in the (sorted, distinct) check grid of the distinct data x; the grid contains every data x. */
function dataPositions(grid, x) {
    const at = [];
    let j = 0;
    for (let i = 0; i < x.length; i++) {
        if (i > 0 && x[i] === x[i - 1])
            continue;
        while (grid[j] < x[i])
            j++;
        at.push(j);
    }
    return Int32Array.from(at);
}

/**
 * The phase-2 fit at the knob setting: a = a*·10^ambition, at most the top of the range. Re-checked against the
 * region on the whole check grid and at the data x.
 */
function computeFit(st, region, star) {
    const opts = phase2Options(st);
    const shaped = st.penalty === 'shaped';
    const sparse = st.penalty === 'sparse' || shaped;
    const power = st.penalty === 'power';
    const wanted = star.ambition * 10 ** st.ambition;
    const a = Math.min(wanted, star.top);
    const spec = { ...regionSpec(region, { ...opts, ambition: a }), shape: opts.shape, q: opts.q };
    const G = region.center.length;
    let fit;
    if (power)
        fit = fitPower({ ...spec, ...powerLambdas({ ambition: a, q: opts.q, flat: opts.flat, G,
            nseg: spec.nseg }) });
    else if (shaped)
        fit = fitShaped({ ...spec, ...shapedLambdas({ ambition: a, flat: opts.flat, G }) });
    else if (sparse)
        fit = fitSparse({ ...spec, ...sparseLambdas({ ambition: a, flat: opts.flat, G }) });
    else
        fit = fitIterated(spec);
    const ratios = fit.g.map((g, j) => Math.abs(g - region.center[j]) / region.half[j]);
    let argMax = 0;
    for (let j = 1; j < ratios.length; j++) {
        if (ratios[j] > ratios[argMax])
            argMax = j;
    }
    let atData = null; // band only: Aₙ's boxes are all at the data
    if (region.dataAt) {
        atData = 0;
        for (const j of region.dataAt) {
            if (ratios[j] > 1 + LEAVE_TOL)
                atData++;
        }
    }
    let leaving = 0;
    for (const r of ratios) {
        if (r > 1 + LEAVE_TOL)
            leaving++;
    }
    return { ...fit, a, sparse, shaped, power, q: opts.q, clamped: wanted > star.top, argMax, leaving,
        leavingAtData: atData, opts };
}

/**
 * Where box k of a region is, for readouts: a band's grid point, or an Aₙ interval of observations.
 */
function boxPlace(region, k, x) {
    if (region.kind !== 'An')
        return `at x = ${fmt(region.grid[k])}`;
    const lo = region.region.lo[k];
    const hi = region.region.hi[k];
    return lo === hi ? `at the observation x = ${fmt(x[lo])}`
        : `on the interval x = ${fmt(x[lo])} – ${fmt(x[hi])} (${fmtInt(hi - lo + 1)} observations)`;
}

function renderFitReadouts(sel, region, star, fit, st, x) {
    sel.append('h2').text('Fit');
    const an = region.kind === 'An';
    const G = region.center.length;
    const ratio = an ? '|w(I)|/c' : '|g − c|/h';
    const v = sel.append('p').attr('class', `verdict ${fit.leaving === 0 ? 'ok' : 'bad'}`);
    if (fit.outside === 0) {
        v.append('strong').text(`fit ∈ ${an ? 'Aₙ' : 'region'}`);
        v.append('span').text(` on every ${an ? 'interval' : 'check-grid point'} (largest ${ratio} = `
            + `${fmt(fit.maxRatio)}).`);
    }
    else if (fit.leaving === 0) {
        const boxes = an ? `${fit.outside === 1 ? 'box is' : 'boxes are'} touched` : `grid ${fit.outside === 1
            ? 'point touches' : 'points touch'} the edge`;
        v.append('strong').text(`fit within ${an ? 'Aₙ' : 'the region'} to 1% of its ${an ? 'threshold'
            : 'half-width'}`);
        v.append('span').text(`: largest ${ratio} = ${d3.format('.4f')(fit.maxRatio)}, `
            + `${boxPlace(region, fit.argMax, x)}; ${fmtInt(fit.outside)} ${boxes} from outside, as a soft region `
            + 'allows.');
    }
    else {
        const atData = fit.leavingAtData === null ? ''
            : ` (${fmtInt(fit.leavingAtData)} of ${fmtInt(region.dataAt.length)} data x)`;
        v.append('strong').text(`fit ∉ ${an ? 'Aₙ' : 'region'}`);
        v.append('span').text(` by more than 1% at ${fmtInt(fit.leaving)} of ${fmtInt(G)} `
            + `${an ? 'intervals' : 'check-grid points'}${atData}; largest ${ratio} = ${fmt(fit.maxRatio)}, `
            + `${boxPlace(region, fit.argMax, x)}.`);
    }
    v.node().append(infoIcon(an
        ? `Checked against the region above, ${region.label}, on its ${fmtInt(G)} intervals: the fit's mean over each `
            + 'interval of observations must lie within c/√|I| of the mean of y there. Aₙ looks only at the '
            + 'observations, so between them the ambition penalty alone shapes the fit. The region is soft: '
            + 'leaving it costs κ per squared box half-height, and the check says by how much.'
        : `Checked against the region above, the ${region.label}, on its check grid. The region is soft: leaving it `
            + 'costs κ per squared half-width, so the fit can cross its edge, and the check says by how much.',
    'the check'));
    if (star.unreachable) {
        sel.append('p').attr('class', 'warning').text(`No fit with this basis gets inside ${an ? 'Aₙ' : 'the region'}, `
            + `even with no ambition: the largest ${ratio} is ${fmt(star.fit.maxRatio)}. A cubic spline on `
            + `${region.params.nseg} equal segments cannot follow the data closely enough for the shortest `
            + 'intervals (a sharp jump or peak); more segments may help'
            + `${an ? ` (at most ${AN_MAX_SEGMENTS} here)` : ''}. The fit shown is at the smallest ambition searched.`);
    }
    if (!fit.converged) {
        sel.append('p').attr('class', 'warning').text(`The iteration stopped after ${fit.iterations} solves without `
            + 'settling which points are outside: the fit is not the exact minimizer.');
    }
    if (fit.power && !fit.settled) {
        sel.append('p').attr('class', 'warning').text(`The reweighting was still lowering the objective after `
            + `${fit.reweights} reweightings: the fit shown is where it stopped, not the minimizer.`);
    }
    if (fit.sparse && !fit.settled) {
        sel.append('p').attr('class', 'warning').text(`The bends and steps were still changing after `
            + `${fit.reweights} reweightings; the count shown is where the reweighting stopped.`);
    }
    const rows = [];
    const add = (label, value, note, tip) => rows.push({ label, value, note, tip });
    const decades = `${Math.abs(st.ambition)} decade${Math.abs(st.ambition) === 1 ? '' : 's'} `
        + `${st.ambition > 0 ? 'above' : 'below'} a*`;
    const top = fit.sparse ? 'the top of the range' : 'the largest ambition computed reliably';
    let ambitionNote;
    if (star.unreachable)
        ambitionNote = 'no a*: the fit is outside the region at every ambition searched';
    else if (star.capped)
        ambitionNote = `a*: no ambition up to ${top} takes the fit outside the region`;
    else if (fit.clamped)
        ambitionNote = `${decades} is beyond ${top}; using that`;
    else if (st.ambition === 0)
        ambitionNote = 'a*, the largest ambition before the fit leaves the region';
    else
        ambitionNote = `${decades}; a* = ${fmtExp(star.ambition)} is the largest before the fit leaves the region`;
    if (fit.sparse)
        ambitionNote += `. Each bend or step costs a per ${an ? 'interval' : 'check-grid point'}`;
    add('ambition a', fmtExp(fit.a), ambitionNote);
    const counted = st.flat === 0 ? 'bends' : st.flat === 1 ? 'steps' : 'bends and steps';
    let penaltyName = 'squared differences';
    if (fit.power && st.flat === 0)
        penaltyName = `|second differences|^q, q = ${fit.q}`;
    else if (fit.power)
        penaltyName = `|${st.flat === 1 ? 'first' : 'first and second'} differences|^q, q = ${fit.q}`;
    else if (fit.shaped && st.flat === 1)
        penaltyName = 'count of steps';
    else if (fit.shaped)
        penaltyName = st.flat === 0 ? 'count of bends, rounded' : 'count of bends, rounded, and steps';
    else if (fit.sparse)
        penaltyName = `count of ${counted}`;
    let rounding = '';
    if (fit.shaped && st.flat === 1) {
        rounding = '. Entirely toward flat there are no bends to round: this is the count of steps';
    }
    else if (fit.shaped) {
        rounding = '. Each bend also pays for the square of its second differences, which spreads a large turn over '
            + `a wider, rounder bend; that cost levels off at ${SHAPE_DEFAULTS.roundingLimit} of the center's range `
            + 'squared, so a turn the region forces to be sharp (a spike narrower than the bend length) costs at most '
            + 'about that much' + (st.flat > 0 ? '. Steps (free first differences, the share toward flat) are '
            + 'counted plainly, and the corners on either side of a step no wider than the bend length are not '
            + 'rounded; longer slopes are, at their bends' : '');
    }
    let penaltyNote = fit.sparse
        ? 'adaptive ridge (Frommlet & Nuel 2016), an approximation of the count: a local optimum, not proven fewest'
            + rounding
        : 'straighter (or flatter) everywhere; no piece is exactly straight';
    if (fit.power) {
        const integral = st.flat === 0 ? 'a·∫|g\'\'|^q' : st.flat === 1 ? 'a·∫|g\'|^q'
            : 'a·(θ·∫|g\'|^q + (1 − θ)·∫|g\'\'|^q)';
        penaltyNote = `${integral}: q = 2 is the squared differences, q = 1 ℓ₁ trend filtering (straight pieces `
            + 'and kinks; flat pieces and steps on first differences), toward 0 the count of bends (or steps); '
            + 'smaller q concentrates the turning into fewer bends. By '
            + 'majorize-minimize (iteratively reweighted squared differences): ' + (fit.q >= 1
            ? 'for q ≥ 1 the objective is convex and this is its minimizer'
            : 'for q < 1 the objective is not convex, and this is a local minimum from the fit without ambition');
    }
    add('penalty', penaltyName, penaltyNote);
    if (fit.sparse)
        addPieceRows(add, fit.pieces, region);
    if (fit.power) {
        addPieceRows(add, fit.pieces, region, 'Pieces where the fit is straight (or flat) to within 0.1% of the '
            + 'region\'s median half-width across the x range (second, or first, differences of its coefficients '
            + 'below that), and the bends (or steps) between them. Only for q ≤ 1 are pieces exactly straight (or '
            + 'flat); above, every difference is nonzero.');
    }
    if (fit.shaped && st.flat < 1)
        addShapeRows(add, fit, st);
    add('toward', st.flat === 0 ? 'straight' : st.flat === 1 ? 'flat' : `${Math.round(100 * (1 - st.flat))}% `
        + `straight, ${Math.round(100 * st.flat)}% flat`, 'second (straight) and first (flat) differences of the '
        + 'coefficients');
    add('ED', fmt(fit.ed), `effective dimension: the trace of the hat matrix, ${an ? 'the interval means of y'
        : `the center`} as data${an ? '' : ` (band: ${fmt(region.params.ed)})`}${fit.sparse
        ? '; about the polynomial\'s coefficients plus one per free difference' : ''}`);
    add('κ', fmtExp(fit.opts.kappa), 'weight on the squared distance outside the region');
    let pullNote = 'weight on the squared distance from the center';
    if (an)
        pullNote = 'weight on the squared distance from the box centers, the interval means of y';
    else if (region.target)
        pullNote = 'weight on the squared distance from the bootstrap replicates\' median (reflected as the band\'s '
            + 'edges are), not the midpoint of the edges, which a long tail on one side moves away from the estimate';
    add('ε', fmtExp(fit.opts.epsilon), pullNote);
    add('basis', `${region.params.nseg} segments`, an
        ? `cubic, equal segments over the data's x range${region.nsegCapped ? `; capped at ${AN_MAX_SEGMENTS}` : ''}`
        : 'cubic, on the band\'s knots');
    if (fit.power) {
        add('solver', `${fit.iterations} solves`, `majorize-minimize: ${fit.reweights} reweighting`
            + `${fit.reweights === 1 ? '' : 's'}; ${fit.settled ? 'the objective settled' : 'not settled'}`);
    }
    else if (fit.sparse) {
        add('solver', `${fit.iterations} solves`, `adaptive ridge: ${fit.reweights} reweighting`
            + `${fit.reweights === 1 ? '' : 's'}; ${fit.settled ? 'the joins settled' : 'the joins had not settled'}; `
            + `${fit.pruned} dropped by backward `
            + `elimination; the final fit is ${fit.converged ? 'the exact minimizer for its joins' : 'not converged'}`
            + (fit.countStart ? '; the reweighting was the plain count\'s: its joins, refit with rounded bends, '
                + 'gave a lower objective than the rounded search\'s own' : ''));
    }
    else {
        const shortened = fit.damped ? `; ${fit.damped} step${fit.damped === 1 ? '' : 's'} shortened by the line `
            + 'search' : '';
        add('solver', `${fit.iterations} solve${fit.iterations === 1 ? '' : 's'}`, `iterated weights: `
            + `${fit.converged ? 'converged, the exact minimizer' : 'not converged'}${shortened}`);
    }
    add('computed in', `${fmt(fit.ms)} ms`, `this fit; finding a* took ${star.fits} fits, ${fmt(star.ms)} ms`);
    appendRows(sel, rows);
}

/**
 * The rule of thumb behind λ: its kernel width, its inputs, and the effective dimension λ was matched to.
 * Matched before undersmoothing, so the band's ED differs from the rule's by the undersmoothing.
 */
function addRuleRow(add, p) {
    const r = p.rule;
    const inputs = p.lambdaSource === 'silverman'
        ? `0.9·min(sd, IQR/1.34)·n^(−1/5) with sd = ${fmt(r.sd)}, IQR = ${fmt(r.iqr)} of x, n = ${fmtInt(r.n)}`
        : `from a pilot quartic: its residual σ̃ = ${fmt(r.sigma)} and root-mean-square curvature `
            + `${fmt(r.curvature)} (units of y per x²)`;
    add('rule width', Number.isFinite(r.h) ? `h = ${fmt(r.h)}` : 'h = ∞', `${inputs}. A Gaussian local linear `
        + `smoother of this width has ED ${fmt(r.ed)} on these x; λ is the one whose P-spline has that ED`
        + (p.undersmooth === 1 ? '' : ', before undersmoothing'),
    'A rule of thumb gives the width of a kernel smoother without searching over λ. The P-spline λ is the one with '
        + 'the same effective dimension (trace of the hat matrix), the usual way to compare smoothers of different '
        + 'kinds.');
}

/** Rows for rounded bends: the bend length, the widest bend, and the extremes and inflections the bends make. */
function addShapeRows(add, fit, st) {
    const widest = fit.bends.reduce((a, b) => Math.max(a, b.to - b.from + 1), 0);
    add('bend length', st.bendLength.toFixed(3), `a bend leaves its tangent by one median half-width over this `
        + `fraction of the x range; the widest bend spans ${fmtInt(widest)} second `
        + `difference${widest === 1 ? '' : 's'} of the coefficients`);
    const each = cost => (cost > 0 ? `each costs ${fmt(cost)} plain bend${cost === 1 ? '' : 's'} more`
        : 'no extra cost');
    add('extremes', fmtInt(fit.extremes), `bends whose two sides slope in opposite directions (a straight piece that `
        + `rises less than 5% of the median half-width counts as level); ${each(st.extremumCost)}`,
    'Local maxima and minima of the fit, counted at its bends.');
    add('inflections', fmtInt(fit.inflections), `changes of the sign of the second differences of the coefficients; `
        + each(st.inflectionCost), 'Places where the fit changes from bending one way to the other.');
    if (st.extremumCost > 0 || st.inflectionCost > 0) {
        add('shape search', `${fmtInt(fit.shapeAccepted)} kept of ${fmtInt(fit.shapeTried)}`, 'stretches kept '
            + 'monotone or bending one way, searched (the others were ruled out: the region forces their extremes '
            + 'or inflections) and kept because the whole cost fell; greedy, a local optimum',
        'How often removing an extremum or an inflection paid.');
    }
}

/**
 * Rows for a sparse fit's pieces: straight pieces and bends (second differences), flat pieces and steps (first).
 * A join's x is its middle coefficient's peak, good to about half a segment.
 */
function addPieceRows(add, pieces, region, tip) {
    const step = (region.params.xr - region.params.xl) / region.params.nseg;
    const digits = Math.max(0, -Math.floor(Math.log10(step / 2)));
    const plural = (k, word) => `${fmtInt(k)} ${word}${k === 1 ? '' : 's'}`;
    for (const { order, joins, pieces: k, atEnds } of pieces.orders) {
        const [piece, join] = order === 2 ? ['straight piece', 'bend'] : ['flat piece', 'step'];
        const ends = atEnds ? ` (${atEnds} at an end of the range)` : '';
        const where = joins.length
            ? `${join}s at x ≈ ${joins.map(j => j.x.toFixed(digits)).join(', ')} (± ${fmtShort(step / 2)}, half a `
                + 'segment)'
            : (order === 2 ? 'a straight line' : 'a constant');
        const what = order === 2
            ? 'Pieces where the fit is exactly straight (second differences of its coefficients are zero), and the '
                + 'short bends between them.'
            : 'Pieces where the fit is exactly flat (first differences of its coefficients are zero), and the '
                + 'steps between them.';
        add(order === 2 ? 'straight' : 'flat', `${plural(k, piece)}, ${plural(joins.length, join)}${ends}`, where,
            tip ?? what);
    }
}

function addLadderRows(add, region, x) {
    const an = region.kind === 'An';
    for (const rung of region.ladder) {
        const name = rung.shape === 'line' ? 'a straight line' : 'a constant';
        add(name, `${rung.fits ? 'fits inside' : 'does not fit'} (${fmt(rung.ratio)})`,
            `the best (minimax) ${rung.shape} reaches `
            + `${an ? '|w(I)|/c' : '|g − c|/h'} = ${fmt(rung.ratio)}, ${boxPlace(region, rung.argMax, x)}; exact `
            + `${an ? 'on every interval' : 'on the check grid'}`);
    }
}

function renderAnReadouts(sel, result, truthCheck, data) {
    const { region, an } = result;
    const p = an.params;
    if (truthCheck) {
        const v = sel.append('p').attr('class', `verdict ${truthCheck.ok ? 'ok' : 'bad'}`);
        v.append('strong').text(`true f ${truthCheck.ok ? '∈' : '∉'} Aₙ`);
        v.append('span').text(truthCheck.ok
            ? ` on every interval (largest |w(I)|/c = ${fmt(truthCheck.maxRatio)}).`
            : ` on ${fmtInt(truthCheck.violations)} of ${fmtInt(an.lo.length)} intervals (largest |w(I)|/c = `
                + `${fmt(truthCheck.maxRatio)}, ${boxPlace(region, truthCheck.argMax, data.x)}).`);
    }
    appendClaim(sel, shortClaim(result), `${region.claim} Aₙ bounds the residuals' sum over each interval of the `
        + 'observations: |Σ_I (y − g)| ≤ c·√|I|.');
    const rows = [];
    const add = (label, value, note, tip) => rows.push({ label, value, note, tip });
    addLadderRows(add, region, data.x);
    add('σ̂', fmt(p.sigma), 'from first differences (Davies, Kovac & Meise eq. 10), tie-aware',
        'Estimated standard deviation of the noise, units of y. First differences of y are nearly free of the '
        + 'signal, so σ̂ needs no fit.');
    const cal = p.calibration;
    add('τ', fmt(p.tau), cal
        ? `calibrated for coverage ${fmt(cal.alpha)} by Monte Carlo (B = ${fmtInt(cal.B)}) for this n, design and `
            + `intervals; Monte Carlo 95% interval ${fmt(cal.tauLo)} – ${fmt(cal.tauHi)}`
        : 'given');
    add('threshold c', fmt(an.scale), 'σ̂·√(τ log n), units of y');
    const f = result.family;
    const lengths = `lengths ${[1, 2, 3].map(k => f.lambda ** (k - 1)).join(', ')}, …`;
    const kind = {
        blocks: `the paper's blocks, ${lengths}`,
        overlapping: `overlapping windows, ${lengths}, ${f.offsets === Infinity ? 'at every position'
            : `in steps of 1/${f.offsets} of the length`}`,
        all: 'every interval',
    };
    add('intervals', fmtInt(an.lo.length), kind[f.kind]);
    add('box half-height', `${fmt(d3.min(region.half))} – ${fmt(d3.max(region.half))}`, 'c/√|I|, units of y');
    add('computed in', `${fmt(result.ms)} ms`, result.calMs === null
        ? (cal ? 'the calibration sample was already computed for these x and intervals' : '')
        : `including ${fmt(result.calMs)} ms to simulate the calibration sample`);
    appendRows(sel, rows);
}

/** A note fragment ("from the residuals, ν = 69") as a sentence, for a tip. */
function sentence(text) {
    const t = text.charAt(0).toUpperCase() + text.slice(1);
    return /[.!?]$/.test(t) ? t : `${t}.`;
}

/**
 * Readout rows {label, value, note?, tip?}: label and value shown; the tip (what the item is) and the note (how
 * this value came about) behind an ⓘ on the label, so the readouts stay compact while every detail stays one
 * hover or tap away.
 */
function appendRows(sel, rows) {
    const dl = sel.append('dl');
    for (const r of rows) {
        const dt = dl.append('dt').text(r.label);
        const text = [r.tip, r.note ? sentence(r.note) : ''].filter(Boolean).join(' ');
        if (text)
            dt.node().append(infoIcon(text, r.label));
        dl.append('dd').text(r.value);
    }
}

/**
 * The noise check's row (noisecheck.js): the band fit's σ̂ against the noise neighboring observations show, and the
 * lag-1 autocorrelation of its residuals.
 */
function addNoiseRow(add, c) {
    const [lo, hi] = c.ratioLimits;
    const lag = c.lagLimits
        ? `residuals' lag-1 r ${fmtShort(c.lag1)} (expected ${fmtShort(c.lagLimits[0])} to ${fmtShort(c.lagLimits[1])})`
        : `lag-1 r not judged with fewer than 10 distinct x`;
    const status = { ok: 'ok', spread: 'see note' }[c.verdict] ?? 'flagged';
    add('σ̂ check', `${fmtShort(c.ratio)} (${status})`,
        `σ̂ ÷ ${fmt(c.reference)} from neighboring observations, expected ${fmtShort(lo)} – ${fmtShort(hi)}; ${lag}`,
        'Does the fit leave the noise the data show? The reference σ̂ comes from each observation\'s distance to the '
        + 'line through its neighbors (pure error between replicates at a tied x), which does not depend on λ. A '
        + 'ratio far below 1 means the fit follows the noise; above 1, it misses structure or the noise has heavy '
        + 'tails or unequal variance, told apart by whether the residuals run in streaks along x. The expected '
        + 'ranges hold 99.9% of fits to smooth curves with independent Gaussian noise. Serially correlated noise '
        + 'that the λ choice has absorbed into the fit fools both alike and is not detected.');
}

/** The noise check's finding as a sentence, or null when there is nothing to say. */
function noiseSentence(c) {
    if (!Number.isFinite(c.ratio))
        return 'The band\'s fit spends as many parameters as there are observations (ED ≥ n): it interpolates the '
            + 'data, and the band is meaningless. Fewer segments or a given λ may help.';
    const ratio = `its σ̂ is ${fmtShort(c.ratio)} times the ${fmt(c.reference)} neighboring observations show`;
    switch (c.verdict) {
    case 'follows':
        return `The band's fit follows the noise: ${ratio} (expected ${fmtShort(c.ratioLimits[0])} or more), so the `
            + 'band is too narrow. Fewer segments or a given λ may help.';
    case 'runs':
        return `The band's residuals run in streaks along x (lag-1 r ${fmtShort(c.lag1)}): the fit misses structure `
            + 'the data show (more segments may help), or the noise is positively correlated. Either way the band '
            + 'may not cover f.';
    case 'between':
        return `Observations at the same x agree more closely than the band's fit does (σ̂ ${fmt(c.reference)} within `
            + `an x, ${fmt(c.sigma)} around the fit), and the means at each x alternate around it (lag-1 r `
            + `${fmtShort(c.lag1)}): each x adds a deviation of its own, so the observations are not independent, `
            + 'as the band\'s claim assumes.';
    case 'alternates':
        return `The band's residuals alternate around its fit (lag-1 r ${fmtShort(c.lag1)}, expected `
            + `${fmtShort(c.lagLimits[0])} or more): the fit may follow part of the noise, or neighboring `
            + 'observations are negatively correlated.';
    case 'spread':
        return `The band's σ̂ is ${fmtShort(c.ratio)} times the ${fmt(c.reference)} neighboring observations show, `
            + 'without streaks: heavy tails or unequal variance, which the bootstrap band and the sandwich '
            + 'covariance allow.';
    default:
        return null;
    }
}

/** A short claim, shown, with the full statement behind an ⓘ. */
function appendClaim(sel, short, full) {
    const p = sel.append('p').attr('class', 'claim').text(short);
    p.node().append(infoIcon(full, 'the claim'));
}

/**
 * The bootstrap band's own readouts: its critical value, how far it reaches below and above the fit, and how
 * asymmetric it is (bootstrap.js).
 */
function addBootstrapRows(add, region) {
    const p = region.params;
    add('q below / above', `${fmt(d3.mean(p.qBelow))} / ${fmt(d3.mean(p.qAbove))}`, `averages over the grid; `
        + `pointwise tail level ${fmtShort(p.tailLevel)}, B = ${fmtInt(p.B)} replicates, `
        + `${p.weights === 'dirichlet' ? 'Dirichlet' : 'matched'} weights, refits to `
        + `${p.refit === 'exact' ? 'all orders' : 'first order'} in the weights`,
        'Critical values: the band reaches q·se below and above the fit, where se is the standard error for constant '
        + 'noise. They are the replicates\' upper and lower tail quantiles (standardized, pooled over ±10% of the x '
        + 'range) at the pointwise tail level that holds for a fraction α of the replicates everywhere at once. '
        + 'Reflected, as a bootstrap-t interval: when the replicates run high, f̂ itself may have run high, so the '
        + 'band reaches further below.');
    add('half-width', `${fmt(d3.min(region.half))} – ${fmt(d3.max(region.half))}`, `below f̂ ${fmt(d3.min(p.down))} – `
        + `${fmt(d3.max(p.down))}, above ${fmt(d3.min(p.up))} – ${fmt(d3.max(p.up))}; units of y`,
        'Half the band\'s width, smallest to largest over the grid; the band is centered at its midpoint, which '
        + 'differs from the phase-1 fit f̂ where the band is asymmetric.');
    const ratios = p.up.map((u, j) => u / p.down[j]);
    add('above ÷ below', `${fmt(d3.min(ratios))} – ${fmt(d3.max(ratios))}`, '1 is symmetric',
        'The band\'s reach above the fit divided by its reach below, smallest to largest over the grid. Above 1 '
        + 'the replicates ran further below the fit there (a long lower tail of the residuals). The extremes are '
        + 'Monte Carlo estimates, and noise widens them: with B = 2000 replicates they varied by about ±0.05 from '
        + 'seed to seed, with 1000 by ±0.1.');
}

/** What a region contains and under which assumptions, in one line (the region's claim has the details). */
function shortClaim(result) {
    if (result.an) {
        const p = result.an.params;
        const level = p.calibration
            ? `Contains f with probability ≈ ${fmt(p.calibration.alpha)} (τ calibrated)`
            : `τ = ${fmt(p.tau)} is given, so its coverage is not calibrated`;
        return `${level}, if the noise is independent Gaussian; no smoothness is assumed about f.`;
    }
    const p = result.region.params;
    if (p.covariance === 'bootstrap') {
        return `Contains the P-spline's mean (f smoothed), not f, at every check-grid point with probability ≈ `
            + `${p.alpha}, for independent observations; the noise need not be Gaussian or of constant variance.`;
    }
    return p.covariance === 'sandwich'
        ? `Contains the P-spline's mean (f smoothed), not f, at every check-grid point with probability ≈ ${p.alpha}, `
            + 'for independent Gaussian noise.'
        : `Contains f at every check-grid point with probability ≈ ${p.alpha}, if f is smooth and the noise `
            + 'independent Gaussian.';
}

function renderReadouts(container, data, result, truthCheck, phase2, st) {
    const sel = d3.select(container);
    sel.selectAll('*').remove();
    sel.append('h2').text('Region');
    if (result.error) {
        sel.append('p').attr('class', 'error').text(`No ${st.region === 'An' ? 'Aₙ' : 'band'}: ${result.error}`);
    }
    else if (result.an) {
        renderAnReadouts(sel, result, truthCheck, data);
    }
    else {
        const { region } = result;
        const p = region.params;
        if (truthCheck) {
            const v = sel.append('p').attr('class', `verdict ${truthCheck.ok ? 'ok' : 'bad'}`);
            v.append('strong').text(`true f ${truthCheck.ok ? '∈' : '∉'} band`);
            v.append('span').text(truthCheck.ok
                ? ` at every check-grid point (largest |f − c|/h = ${fmt(truthCheck.maxRatio)}).`
                : ` at ${fmtInt(truthCheck.outside)} of ${fmtInt(region.grid.length)} check-grid points (largest `
                    + `|f − c|/h = ${fmt(truthCheck.maxRatio)}, at x = ${fmt(region.grid[truthCheck.argMax])}).`);
        }
        appendClaim(sel, shortClaim(result), region.claim);
        if (p.atEdge && p.lambdaSource === 'scale') {
            const finer = Math.log(p.lambdaChosen) < 0;
            sel.append('p').attr('class', 'warning').text(finer
                ? `A feature scale of ${fmtShort3(100 * p.rule.scale)}% of the x range is finer than this basis `
                    + `follows (its knots are ${fmtShort3(100 / p.nseg)}% apart); λ is the smallest searched.`
                : `Even the largest λ searched keeps features of ${fmtShort3(100 * p.rule.scale)}% of the x range; `
                    + 'λ is the largest searched.');
        }
        else if (p.atEdge && p.rule) {
            const below = Math.log(p.lambdaChosen) > 0;
            sel.append('p').attr('class', 'warning').text(`${lambdaSourceName(p.lambdaSource)} asks for ED `
                + `${fmt(p.rule.ed)}, ${below ? `no more than the penalty's polynomial of degree ${p.diff - 1}`
                    : 'more than the P-spline reaches on these x'}; λ is the ${below ? 'largest' : 'smallest'} searched.`);
        }
        else if (p.atEdge) {
            const end = Math.log(p.lambdaChosen) > 0 ? 'largest' : 'smallest';
            sel.append('p').attr('class', 'warning').text(`${lambdaSourceName(p.lambdaSource)} chose the ${end} λ `
                + 'searched' + (end === 'largest'
                    ? `: the fit is close to its limit, a polynomial of degree ${p.diff - 1}.`
                    : ': the optimum may lie further out; consider fewer segments or a given λ.'));
        }
        const noiseNote = result.noise ? noiseSentence(result.noise) : null;
        if (noiseNote)
            sel.append('p').attr('class', result.noise.verdict === 'spread' ? null : 'warning').text(noiseNote);
        const rows = [];
        const add = (label, value, note, tip) => rows.push({ label, value, note, tip });
        addLadderRows(add, region, data.x);
        let lambdaNote;
        if (p.lambdaSource === 'given')
            lambdaNote = 'given';
        else if (p.lambdaSource === 'scale' && p.rule.scale > p.rule.asked) {
            lambdaNote = `for features longer than ${fmtShort3(100 * p.rule.scale)}% of the x range, raised from `
                + `${fmtShort3(100 * p.rule.asked)}% so that a period spans ${p.rule.minPoints} of the `
                + `${fmtInt(data.x.length)} observations`;
        }
        else if (p.lambdaSource === 'scale')
            lambdaNote = `for features longer than ${fmtShort3(100 * p.rule.scale)}% of the x range`;
        else if (p.undersmooth === 1)
            lambdaNote = `chosen by ${lambdaSourceName(p.lambdaSource)}`;
        else
            lambdaNote = `${lambdaSourceName(p.lambdaSource)} chose ${fmt(p.lambdaChosen)}; × ${fmt(p.undersmooth)} `
                + 'undersmoothing';
        add('λ', `${fmt(p.lambda)} (log ${p.lambda > 0 ? Math.log(p.lambda).toFixed(2) : '−∞'})`, lambdaNote,
            'The smoothing parameter of the band\'s P-spline: the weight on the squared differences of its '
            + 'coefficients. Larger is smoother, with a narrower band; log is natural.');
        if (p.rule && p.lambdaSource !== 'scale')
            addRuleRow(add, p);
        // the feature scale this λ amounts to, whatever chose it (from x alone, cached on the region)
        p.featureScale ??= lambdaScale(data.x, p.lambda, { nseg: p.nseg, deg: p.deg, diff: p.diff });
        add('feature scale', `${fmtShort3(100 * p.featureScale)}% of the x range`, 'a sine of this period keeps half '
            + 'its amplitude in the band\'s P-spline: longer features are followed, shorter ones smoothed away',
        'The length of the shortest features the band\'s λ follows, as a share of the x range. It depends on λ, '
            + 'the basis and the x values, not on y, so it compares λs across data sets. "By feature scale" (λ) '
            + 'sets λ from it.');
        rows.push({ label: 'ED', value: fmt(p.ed), tip: 'Effective dimension, the trace of the P-spline\'s hat '
            + 'matrix: the number of parameters its fit spends, between the penalty order (λ → ∞, a polynomial) and '
            + 'the number of coefficients (λ = 0).' });
        add('σ̂', fmt(p.sigma), `from the residuals, ν = n − ED = ${fmt(p.df)} degrees of freedom`,
            'Estimated standard deviation of the noise: √(RSS/ν), units of y.');
        if (result.noise)
            addNoiseRow(add, result.noise);
        if (p.covariance === 'bootstrap') {
            addBootstrapRows(add, region);
        }
        else {
            add('q', fmt(p.q), `${p.alpha} quantile of the max over the grid, Monte Carlo B = ${fmtInt(p.B)}`,
                'Critical value: the band is the center ± q standard errors. q exceeds the pointwise 1.96 because the '
                + 'band must hold at every grid point at once: it is the α quantile of the largest |error|/se over '
                + 'the grid.');
            add('half-width', `${fmt(d3.min(region.half))} – ${fmt(d3.max(region.half))}`,
                'q × standard error, units of y');
        }
        add('basis', `${p.nseg} segments`, `B-splines of degree ${p.deg}, penalty order ${p.diff}`);
        add('check grid', `${fmtInt(region.grid.length)} points`, 'the distinct data x and 4 points per segment');
        add('computed in', `${fmt(result.ms)} ms`);
        appendRows(sel, rows);
    }
    // phase 2 after phase 1, in the order they are computed (and as in the controls)
    if (phase2)
        renderFitReadouts(sel, result.region, phase2.star, phase2.fit, st, data.x);
    sel.append('h2').text('Data');
    appendRows(sel, [
        { label: 'observations', value: fmtInt(data.n), tip: 'Rows with both x and y.' },
        { label: 'distinct x', value: fmtInt(data.m), tip: 'Observations that share an x are tied: the region never '
            + 'separates them, and segments and the check grid are set from the distinct values.' },
    ]);
    if (data.notes.length)
        sel.append('ul').attr('class', 'notes').selectAll('li').data(data.notes).join('li').text(d => d);
}

async function loadExamples() {
    try {
        const response = await fetch(EXAMPLES_INDEX);
        if (!response.ok)
            throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }
    catch (err) {
        console.warn(`No example data sets: ${EXAMPLES_INDEX} could not be loaded (${err.message ?? err})`);
        return [];
    }
}

/** Names of the CSV files in localdata/: from the server's directory listing, or from localdata/index.json. */
async function localFileNames() {
    const probe = await fetch(LOCAL_DIR + LOCAL_PROBE, { method: 'HEAD', cache: 'no-store' });
    if (LISTING_SERVER.test(probe.headers.get('server') ?? '')) {
        const response = await fetch(LOCAL_DIR, { cache: 'no-store' });
        if (!response.ok)
            throw new Error(`${LOCAL_DIR} could not be listed (HTTP ${response.status})`);
        const page = new DOMParser().parseFromString(await response.text(), 'text/html');
        return [...page.querySelectorAll('a[href]')]
            .map(a => decodeURIComponent(a.getAttribute('href').split(/[?#]/)[0]))
            .filter(name => name.endsWith('.csv'));
    }
    const response = await fetch(LOCAL_DIR + LOCAL_INDEX, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`this server does not list directories and there is no ${LOCAL_DIR}${LOCAL_INDEX} `
            + `(HTTP ${response.status}); run npm run data:index`);
    }
    const names = await response.json();
    if (!Array.isArray(names) || !names.every(name => typeof name === 'string'))
        throw new Error(`${LOCAL_DIR}${LOCAL_INDEX} is not an array of file names; run npm run data:index`);
    return names.filter(name => name.endsWith('.csv'));
}

/**
 * The CSV files in localdata/, when the page is served from this machine: index entries like data/index.json's,
 * with each file's text (so it is not fetched twice). A file without '# label:' or '# source:' lines gets its name
 * and path; one that cannot be read as a data set is left out with a warning.
 */
async function loadLocalExamples() {
    if (!LOCAL_HOSTS.has(location.hostname))
        return [];
    let names;
    try {
        names = await localFileNames();
    }
    catch (err) {
        console.warn(`No local data sets: ${err.message ?? err}`);
        return [];
    }
    const entries = await Promise.all([...new Set(names)].sort().map(async name => {
        try {
            if (datasetId(name) === null)
                throw new Error('the name must be lower-case letters, digits and hyphens, ending in .csv');
            const response = await fetch(LOCAL_DIR + encodeURIComponent(name));
            if (!response.ok)
                throw new Error(`HTTP ${response.status}`);
            // the file's own label and source, when it has them, override these (later keys win)
            const text = `# label: ${name}\n# source: ${LOCAL_DIR}${name} (local)\n${await response.text()}`;
            const { entry, errors } = indexEntry(name, text);
            if (!entry)
                throw new Error(errors.join('; '));
            return { ...entry, text };
        }
        catch (err) {
            console.warn(`Local data set ${name} left out: ${err.message ?? err}`);
            return null;
        }
    }));
    return entries.filter(Boolean);
}

/**
 * The export tables' layers on the region's grid, with the grid and each data x's place in it: for a band, the
 * phase-1 fit, the band, its center when that is not the fit (the bootstrap band's midpoint, which the phase-2
 * ratio is measured from) and the phase-2 fit with its ratio |g − c|/h; for Aₙ, the phase-2 fit on its drawing
 * grid (Aₙ's region is a set of interval sums, not a curve).
 */
function exportLayers(result, phase2, data) {
    const region = result.region;
    if (region.kind === 'An') {
        return { grid: phase2.curve.x, at: dataPositions(phase2.curve.x, data.x),
            layers: [{ name: 'phase2_fit', values: phase2.curve.g }] };
    }
    const boot = region.params.covariance === 'bootstrap';
    const g = phase2.fit.g;
    const layers = [
        { name: 'phase1_fit', values: boot ? region.params.fit : region.center },
        { name: 'band_lower', values: region.lower },
        { name: 'band_upper', values: region.upper },
    ];
    if (boot)
        layers.push({ name: 'band_center', values: region.center });
    layers.push({ name: 'phase2_fit', values: g },
        { name: 'phase2_ratio', values: g.map((v, j) => Math.abs(v - region.center[j]) / region.half[j]) });
    return { grid: region.grid, at: region.dataAt, layers };
}

/** Column names for x and y from the data: the file's or the pasted table's (row number for x = row). */
function exportNames(st, data) {
    if (st.source === USER)
        return { x: st.xCol < 0 ? 'row' : data.columns[st.xCol], y: data.columns[st.yCol] };
    return data.columns.length === 2 ? { x: data.columns[0], y: data.columns[1] } : {};
}

function exportBase(st) {
    if (st.source === USER)
        return 'lafspline-data';
    return `lafspline-${st.source.replace(EXAMPLE_PREFIX, '').replace(LOCAL_PREFIX, '')}`;
}

function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** An SVG document drawn into a canvas at scale × its size, as a PNG or JPEG blob. */
async function rasterize(svg, type, scale) {
    const url = URL.createObjectURL(new Blob([svg.text], { type: 'image/svg+xml' }));
    try {
        const img = new Image();
        img.src = url;
        await img.decode();
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(svg.width * scale);
        canvas.height = Math.round(svg.height * scale);
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, type, JPEG_QUALITY));
        if (!blob)
            throw new Error('the browser could not encode the image');
        return { blob, width: canvas.width, height: canvas.height };
    }
    finally {
        URL.revokeObjectURL(url);
    }
}

/** ', 3 flat pieces, 4 straight pieces' for a sparse fit; '' for the others. */
function piecesText(fit) {
    if (!fit.sparse)
        return '';
    return fit.pieces.orders.map(o => `, ${fmtInt(o.pieces)} ${o.order === 2 ? 'straight' : 'flat'} `
        + `piece${o.pieces === 1 ? '' : 's'}`).join('');
}

/**
 * Sources for the gallery that this page has: a data file missing here (a local one, or one from another version)
 * is replaced by a default source not already shown. missing: how many were replaced.
 */
function availablePanes(list, examples) {
    const has = src => Boolean(SIGNALS[src]) || examples.has(src);
    const spare = [...GALLERY_SOURCES, ...Object.keys(SIGNALS)].filter(src => has(src) && !list.includes(src));
    const panes = list.map(src => (has(src) ? src : spare.shift()));
    return { panes, missing: list.filter(src => !has(src)).length };
}

function rejectedNote(rejected) {
    return rejected.length ? `Ignored invalid link setting(s): ${rejected.join(', ')}.` : null;
}

async function startup() {
    const [loaded, exampleList, localList] = await Promise.all([hashToState(location.hash, DEFAULTS), loadExamples(),
        loadLocalExamples()]);
    // data sets by source: bundled (data/) and local (localdata/), with the path to fetch
    const examples = new Map([
        ...exampleList.map(e => [EXAMPLE_PREFIX + e.id, { ...e, path: `data/${e.file}` }]),
        ...localList.map(e => [LOCAL_PREFIX + e.id, { ...e, path: LOCAL_DIR + e.file }]),
    ]);
    Object.assign(state, loaded.state);
    let linkNote = rejectedNote(loaded.rejected);
    const isFile = state.source.startsWith(EXAMPLE_PREFIX) || state.source.startsWith(LOCAL_PREFIX);
    if (isFile && !examples.has(state.source)) {
        const name = state.source.slice(state.source.indexOf(':') + 1);
        linkNote = state.source.startsWith(LOCAL_PREFIX)
            ? `The link's local data set '${name}' is not in ${LOCAL_DIR} here; showing the default.`
            : `The link's example data set '${name}' is not in this version; showing the default.`;
        Object.assign(state, { source: DEFAULTS.source, n: DEFAULTS.n, sigma: DEFAULTS.sigma });
    }
    const checkedPanes = availablePanes(state.panes, examples);
    state.panes = checkedPanes.panes;
    if (checkedPanes.missing) {
        linkNote = `${linkNote ? `${linkNote} ` : ''}${checkedPanes.missing} of the gallery's data sets `
            + `${checkedPanes.missing === 1 ? 'is' : 'are'} not here; showing others.`;
    }
    const exampleTexts = new Map(localList.map(e => [LOCAL_PREFIX + e.id, e.text])); // source → text, or {error}
    const status = d3.select('#status');
    const readouts = document.getElementById('readouts');
    const chart = createMainChart(document.getElementById('main-chart'));
    const mainChartEl = document.getElementById('main-chart');
    const galleryEl = document.getElementById('gallery');
    const regionCache = new Map();
    const starCache = new Map();
    const nullCache = new Map(); // Aₙ calibration samples, by design and interval family
    const fetching = new Map(); // data file id → its pending fetch
    let current = null; // {data, result, phase2, truthCheck} drawn in the single view

    const controlsEl = document.getElementById('controls');
    const settingsHistory = createHistory(settingsOf(state));
    const onChange = patch => {
        Object.assign(state, patch);
        if (Object.keys(patch).some(key => !DATA_KEYS.has(key))) {
            settingsHistory.record(settingsOf(state), performance.now());
            syncHistory();
        }
        if ((current || state.view === 'gallery') && Object.keys(patch).every(key => DISPLAY_KEYS.has(key)))
            draw();
        else
            update();
        scheduleHash();
    };
    let controls = createControls(controlsEl, state, onChange, exampleList, localList);
    function rebuildControls() {
        controlsEl.replaceChildren();
        controls = createControls(controlsEl, state, onChange, exampleList, localList);
    }

    /** A data file's text, fetched once; resolves when exampleTexts holds it (the text, or {error}). */
    function ensureText(id) {
        if (exampleTexts.get(id) !== undefined)
            return Promise.resolve();
        if (!fetching.has(id)) {
            exampleTexts.set(id, undefined);
            fetching.set(id, fetch(examples.get(id).path)
                .then(r => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
                .then(text => exampleTexts.set(id, text))
                .catch(err => exampleTexts.set(id, { error: err.message ?? String(err) }))
                .finally(() => fetching.delete(id)));
        }
        return fetching.get(id);
    }

    /**
     * Data, region, a* and fit for a state, from the caches where they hold them: {data} alone while a data file
     * loads or when there is nothing to fit, else {data, result, phase2, truthCheck} (phase2 null when the region
     * failed: result.error).
     */
    function computeView(st) {
        const data = deriveData(st, exampleTexts, examples);
        if (data.pending || data.error)
            return { data };
        const key = regionKey(st, data);
        let result = regionCache.get(key);
        if (!result) {
            result = st.region === 'An' ? computeAn(st, data, nullCache) : computeBand(st, data);
            remember(regionCache, key, result);
        }
        const region = result.region;
        const an = region?.kind === 'An';
        let phase2 = null;
        if (region) {
            if (!an)
                region.dataAt ??= dataPositions(region.grid, data.x);
            region.ladder ??= ladder(region);
            const sk = starKey(st, data);
            let star = starCache.get(sk);
            if (!star) {
                const opts = phase2Options(st);
                star = leavingAmbition({ ...regionSpec(region, { ...opts, ambition: 0 }), shape: opts.shape,
                    q: opts.q }, { sparse: st.penalty === 'sparse', shaped: st.penalty === 'shaped',
                    power: st.penalty === 'power' });
                remember(starCache, sk, star);
            }
            const fit = computeFit(st, region, star);
            // the curve drawn: a band's fit is already on its check grid; Aₙ's boxes are interval means
            let curve = { x: region.grid, g: fit.g };
            if (an) {
                region.drawGrid ??= checkGrid(data.x, region.params.xl, region.params.xr, region.params.nseg);
                curve = { x: region.drawGrid, g: splineValues(fit.beta, region.drawGrid, region.params) };
            }
            phase2 = { star, fit, curve };
        }
        let truthCheck = null;
        if (region && data.fAt) {
            truthCheck = an ? checkFit(result.an, data.x.map(data.fAt), 0)
                : checkBand(region, region.grid.map(data.fAt));
        }
        return { data, result, phase2, truthCheck };
    }

    /** The chart model and Aₙ boxes for a computed view, with the display settings' hidden layers. */
    function chartModel(view) {
        const { data, result, phase2 } = view;
        const region = result.region;
        const an = region?.kind === 'An';
        const fits = [];
        if (region && !an) {
            // a bootstrap band need not be symmetric: draw the P-spline it is built around, not its midpoint
            const boot = region.params.covariance === 'bootstrap';
            fits.push({ key: 'center', label: boot ? 'P-spline fit (phase 1)' : 'band center (phase 1)',
                g: boot ? region.params.fit : region.center, x: region.grid, colorVar: CENTER_COLOR,
                primary: !phase2 });
        }
        if (phase2) {
            fits.push({ key: 'fit', label: 'fit (phase 2)', g: phase2.curve.g, x: phase2.curve.x, colorVar: FIT_COLOR,
                primary: true });
        }
        const boxes = an && phase2 ? anBoxSpec(result, phase2, data) : null;
        const hidden = new Set();
        if (!state.showTruth)
            hidden.add('truth');
        if (!state.showData)
            hidden.add('data');
        if (!state.showCenter)
            hidden.add('center');
        if (!state.showBand)
            hidden.add('band');
        if (!state.showFit)
            hidden.add('fit');
        const model = {
            x: data.x,
            y: data.y,
            f: data.fAt ? data.x.map(data.fAt) : null,
            fits,
            extremes: [],
            edges: data.edges,
            band: region && !an ? { lb: region.lower, ub: region.upper, label: region.label, x: region.grid } : null,
            boxesLabel: boxes ? boxes.label : null,
            hidden,
        };
        return { model, boxes };
    }

    function drawInto(target, view) {
        const { model, boxes } = chartModel(view);
        target.setBoxes(null);
        target.render(model);
        target.setBoxes(boxes);
    }

    function draw() {
        if (state.view === 'gallery') {
            for (const pane of panes)
                drawPane(pane);
            drawGalleryLegend();
            return;
        }
        if (current)
            drawInto(chart, current);
    }

    /**
     * Aₙ boxes for the chart: the level of the interval with the largest ratio (the tightest), or every level in
     * outline. Each box is mean_I(y) ± c/√|I| over its interval's cells, with the fit's mean as a tick.
     */
    function anBoxSpec(result, phase2, data) {
        if (state.boxes === 'off')
            return null;
        result.levels ??= familyLevels(data.x, result.family);
        phase2.atData ??= splineValues(phase2.fit.beta, data.x, result.region.params);
        const boxesOf = level => constraintBoxes(data.y, phase2.atData, level, result.an.scale);
        const common = { limit: 1 + LEAVE_TOL, fitColorVar: FIT_COLOR };
        if (state.boxes === 'all') {
            return { ...common, faint: true, label: 'Aₙ boxes, all levels',
                levels: result.levels.map(level => ({ boxes: boxesOf(level), emphasize: -1 })) };
        }
        const tight = result.intervals[phase2.fit.argMax];
        const level = result.levels[tight.level];
        const emphasize = level.findIndex(iv => iv.lo === tight.lo && iv.hi === tight.hi);
        return { ...common, faint: false, levels: [{ boxes: boxesOf(level), emphasize }],
            label: `Aₙ boxes at the tightest level (${fmtInt(tight.hi - tight.lo + 1)}-observation intervals): the `
                + 'fit\'s mean (tick) must lie in each box' };
    }

    /** Show the single view or the gallery: the readouts and the main chart, or the panes. */
    function showView() {
        const gallery = state.view === 'gallery';
        document.body.classList.toggle('gallery', gallery);
        readouts.hidden = gallery;
        mainChartEl.hidden = gallery;
        galleryEl.hidden = !gallery;
        for (const button of document.querySelectorAll('#view-toggle [data-view]'))
            button.setAttribute('aria-pressed', String(button.dataset.view === state.view));
    }

    function update() {
        showView();
        if (state.view === 'gallery') {
            current = null;
            updateGallery();
            return;
        }
        galleryRun++; // a gallery still computing stops
        const view = computeView(state);
        const { data } = view;
        if (data.pending) {
            status.classed('error', false).text(`Loading ${data.label}…`);
            controls.sync(state, { columns: data.columns });
            ensureText(data.id).then(update);
            return;
        }
        if (data.error) {
            status.classed('error', true).text(data.error);
            controls.sync(state, { columns: data.columns });
            d3.select(readouts).selectAll('*').remove();
            return;
        }
        current = view;
        const { result, phase2, truthCheck } = view;
        const region = result.region;
        const an = region?.kind === 'An';
        controls.sync(state, {
            columns: data.columns,
            nsegAuto: defaultSegments(data.m),
            lambdaChosen: region && !an && region.params.lambdaSource !== 'given' ? region.params.lambdaChosen : null,
            lambdaUsed: region && !an ? region.params.lambda : null,
            scaleRaised: region && !an && region.params.lambdaSource === 'scale'
                && region.params.rule.scale > region.params.rule.asked
                ? { scale: region.params.rule.scale, minPoints: region.params.rule.minPoints, n: data.x.length }
                : null,
            ambition: phase2 ? { a: phase2.fit.a, star: phase2.star.ambition, capped: phase2.star.capped,
                clamped: phase2.fit.clamped, sparse: phase2.fit.sparse } : null,
        });
        draw();
        renderReadouts(readouts, data, result, truthCheck, phase2, state);
        const note = linkNote ? ` ${linkNote}` : '';
        if (result.error) {
            status.classed('error', true).text(`${data.label}: no ${an || state.region === 'An' ? 'Aₙ' : 'band'} `
                + `(${result.error})${note}`);
        }
        else {
            status.classed('error', Boolean(linkNote)).text(`${data.label}: n = ${fmtInt(data.n)}; ${region.label}; `
                + `fit ED ${fmt(phase2.fit.ed)}${piecesText(phase2.fit)}.${note}`);
        }
        document.body.dataset.ready = 'true';
    }

    // --- gallery: several data sets with the same settings, a compact chart each and one legend.
    const galleryLegend = document.getElementById('gallery-legend');
    const galleryGrid = document.getElementById('gallery-grid');
    const paneCount = document.getElementById('pane-count');
    paneCount.replaceChildren(...PANE_COUNTS.map(k => new Option(`${k} panes`, String(k))));
    paneCount.addEventListener('change', () => {
        const k = Number(paneCount.value);
        const next = state.panes.slice(0, k);
        // more panes take the default sources not shown yet, then simulated signals
        for (const s of [...GALLERY_SOURCES, ...Object.keys(SIGNALS)]) {
            if (next.length < k && !next.includes(s) && (examples.has(s) || SIGNALS[s]))
                next.push(s);
        }
        onChange({ panes: next });
    });
    let panes = []; // {source, root, chartEl, chart, summary, select, view, key}
    let galleryRun = 0;
    // Settings that do not change a pane's fit: the view, the panes, the display toggles, and the single view's data
    // (a pane has its own source, and a signal its own n and σ).
    const NOT_PANE_KEYS = new Set([...DISPLAY_KEYS, 'view', 'panes', 'source', 'n', 'sigma', 'userText', 'xCol',
        'yCol']);

    /** What a pane's fit depends on: its source and the shared settings (the seed only for a signal). */
    function paneKey(source) {
        const shared = Object.entries(state).filter(([key]) => !NOT_PANE_KEYS.has(key)
            && (key !== 'seed' || SIGNALS[source]));
        return JSON.stringify([source, shared]);
    }

    /** A pane's state: the shared settings with its source (a signal at its own n and σ). */
    function paneState(source) {
        const spec = SIGNALS[source];
        return { ...state, source, xCol: null, yCol: null,
            ...(spec ? { n: Math.min(spec.n, MAX_N), sigma: spec.sigma } : {}) };
    }

    function makePane(i) {
        const root = document.createElement('div');
        root.className = 'pane';
        const head = document.createElement('div');
        head.className = 'pane-head';
        const select = sourceSelect(exampleList, localList, { user: false });
        select.setAttribute('aria-label', `Data set in pane ${i + 1}`);
        select.addEventListener('change', () => {
            const next = [...state.panes];
            next[i] = select.value;
            onChange({ panes: next });
        });
        const open = document.createElement('button');
        open.type = 'button';
        open.className = 'pane-open';
        open.textContent = '↗';
        open.title = 'Open this data set in the single view, with its readouts';
        open.setAttribute('aria-label', `Open pane ${i + 1} in the single view`);
        open.addEventListener('click', () => openPane(i));
        head.append(select, open);
        const chartEl = document.createElement('div');
        const summary = document.createElement('p');
        summary.className = 'pane-summary';
        root.append(head, chartEl, summary);
        galleryGrid.append(root);
        return { source: null, root, chartEl, chart: createMainChart(chartEl, { compact: true }), summary, select,
            view: null, key: null };
    }

    /** As many panes as state.panes, keeping the ones there are (and what they show). */
    function buildPanes() {
        while (panes.length > state.panes.length)
            panes.pop().root.remove();
        while (panes.length < state.panes.length)
            panes.push(makePane(panes.length));
        panes.forEach((pane, i) => {
            pane.source = state.panes[i];
            pane.select.value = pane.source;
        });
        paneCount.value = String(state.panes.length);
    }

    function openPane(i) {
        const source = state.panes[i];
        const spec = SIGNALS[source];
        Object.assign(state, { view: 'single', source, xCol: null, yCol: null,
            ...(spec ? { n: Math.min(spec.n, MAX_N), sigma: spec.sigma } : {}) });
        rebuildControls();
        update();
        scheduleHash();
    }

    /**
     * One line under a pane: n, the region's size (a band's ED, Aₙ's τ), the fit's ED and pieces, and only where
     * the fit (or the true f of simulated data) leaves the region: {line, outside} (outside '' when nothing leaves).
     * The full region label and the data's notes are its tooltip.
     */
    function paneSummary(view) {
        const { data, result, phase2, truthCheck } = view;
        const region = result.region;
        const an = region.kind === 'An';
        const fit = phase2.fit;
        const unit = an ? 'intervals' : 'pts';
        const parts = [`n = ${fmtInt(data.n)}`, an ? `Aₙ τ = ${fmtShort3(result.an.params.tau)}`
            : `band ED ${fmtShort3(region.params.ed)}`, `fit ED ${fmtShort3(fit.ed)}`];
        if (fit.sparse) {
            const counts = fit.pieces.orders.filter(o => o.pieces > 0)
                .map(o => `${fmtInt(o.pieces)} ${o.order === 2 ? 'straight' : 'flat'}`);
            if (counts.length)
                parts.push(counts.join(', '));
        }
        const out = [];
        if (fit.leaving > 0)
            out.push(`fit ${fmtInt(fit.leaving)} ${unit} (${fmtShort3(fit.maxRatio)} h)`);
        if (truthCheck && !truthCheck.ok)
            out.push(`true f ${fmtInt(truthCheck.outside)} ${unit}`);
        // each phrase kept whole (no-break spaces), so a wrapped line breaks between them
        const whole = text => text.replaceAll(' ', '\u00a0');
        return { line: parts.join(PANE_SUMMARY_SEP),
            outside: out.length ? `${whole(`outside: ${out[0]}`)}${out.slice(1).map(o => `, ${whole(o)}`).join('')}`
                : '' };
    }

    function drawPane(pane) {
        const view = pane.view;
        if (!view)
            return;
        pane.root.classList.remove('stale');
        pane.select.title = view.data.label ?? '';
        const problem = view.data.error ?? (view.result.error
            ? `No ${state.region === 'An' ? 'Aₙ' : 'band'}: ${view.result.error}` : null);
        pane.chartEl.hidden = Boolean(problem);
        pane.summary.classList.toggle('error', Boolean(problem));
        if (problem) {
            pane.summary.textContent = problem;
            pane.summary.title = '';
            return;
        }
        drawInto(pane.chart, view);
        const where = view.result.region.kind === 'An' ? 'Aₙ' : 'the band';
        const { line, outside } = paneSummary(view);
        const fit = view.phase2.fit;
        // an excursion is never cut off: the line may wrap to show it
        pane.summary.replaceChildren(line);
        pane.summary.classList.toggle('outside', Boolean(outside));
        if (outside) {
            const warn = document.createElement('span');
            warn.className = 'pane-outside';
            warn.textContent = outside;
            pane.summary.append(PANE_SUMMARY_SEP, warn);
        }
        pane.summary.title = [line + (outside ? PANE_SUMMARY_SEP + outside : ''), view.result.region.label,
            fit.leaving === 0 ? `The fit stays inside ${where}.`
                : `The fit leaves ${where} at ${fmtInt(fit.leaving)} ${view.result.region.kind === 'An' ? 'intervals'
                    : 'check-grid points'}, by up to ${fmt(fit.maxRatio)} half-widths.`, ...view.data.notes].join('\n');
    }

    /**
     * One legend for the gallery: every pane's items once. A band's label names its ED and Aₙ's boxes their
     * interval length, which differ by pane: the panes' summaries give them, the legend the kind.
     */
    function drawGalleryLegend() {
        const items = new Map();
        for (const pane of panes) {
            if (!pane.view?.phase2 || pane.chartEl.hidden)
                continue;
            for (const item of pane.chart.legendItems()) {
                let label = item.label;
                if (item.box)
                    label = label.replace(/,\s*ED [^)]*\)/, ')');
                else if (item.glyph === '▭')
                    label = label.replace(/\s*\([^)]*\)/, '');
                if (!items.has(label))
                    items.set(label, { ...item, label });
            }
        }
        // the data and true f first, as in a single chart's legend, whichever pane has them
        const first = ['data', 'true f'];
        const rank = item => (first.includes(item.label) ? first.indexOf(item.label) : first.length);
        drawLegend(galleryLegend, [...items.values()].sort((a, b) => rank(a) - rank(b)));
    }

    async function updateGallery() {
        const run = ++galleryRun;
        document.body.dataset.ready = 'false';
        controls.sync(state, {});
        buildPanes();
        // only the panes whose data set or settings changed are recomputed; the others keep their fit
        const todo = panes.filter(pane => pane.key !== paneKey(pane.source));
        for (const pane of todo)
            pane.root.classList.add('stale');
        for (let i = 0; i < todo.length; i++) {
            const pane = todo[i];
            status.classed('error', false).text(`Gallery: computing ${pane.select.selectedOptions[0]?.text ?? ''} `
                + `(${i + 1} of ${todo.length})…`);
            // let the page paint the panes done so far before the next (a band can take a second)
            await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
            if (run !== galleryRun)
                return;
            const st = paneState(pane.source);
            let view = computeView(st);
            if (view.data.pending) {
                await ensureText(view.data.id);
                if (run !== galleryRun)
                    return;
                view = computeView(st);
            }
            pane.view = view;
            pane.key = paneKey(pane.source);
            drawPane(pane);
        }
        drawGalleryLegend();
        const note = linkNote ? ` ${linkNote}` : '';
        status.classed('error', Boolean(linkNote)).text(`Gallery of ${panes.length} data sets, each with the settings `
            + `on the left; hover a chart for its values, or open it (↗) for its readouts.${note}`);
        document.body.dataset.ready = 'true';
    }

    // --- URL hash: written (debounced, replaceState) after every change; read on load and on hashchange.
    let hashTimer = null;
    let writtenHash = location.hash;
    let dataOmitted = 0;
    let hashSeq = 0;
    async function writeHash() {
        clearTimeout(hashTimer);
        hashTimer = null;
        const seq = ++hashSeq;
        const result = await stateToHash(state, DEFAULTS);
        if (seq !== hashSeq)
            return;
        dataOmitted = result.dataOmitted;
        if (result.hash !== writtenHash) {
            writtenHash = result.hash;
            history.replaceState(null, '', result.hash || location.pathname + location.search);
        }
    }
    function scheduleHash() {
        linkNote = null;
        clearTimeout(hashTimer);
        hashTimer = setTimeout(writeHash, HASH_DEBOUNCE);
    }
    window.addEventListener('hashchange', async () => {
        if (location.hash === writtenHash)
            return;
        const next = await hashToState(location.hash, DEFAULTS);
        writtenHash = location.hash;
        linkNote = rejectedNote(next.rejected);
        Object.assign(state, next.state);
        state.panes = availablePanes(state.panes, examples).panes;
        settingsHistory.record(settingsOf(state), performance.now(), { coalesce: false });
        syncHistory();
        rebuildControls();
        update();
    });

    // --- toolbar
    const message = document.getElementById('toolbar-message');
    let messageTimer = null;
    function say(text) {
        message.textContent = text;
        clearTimeout(messageTimer);
        messageTimer = setTimeout(() => {
            message.textContent = '';
        }, MESSAGE_MS);
    }
    document.getElementById('copy-link').addEventListener('click', async () => {
        await writeHash();
        const omitted = dataOmitted
            ? ` without your data (${fmt(dataOmitted / 1024)} KB is too long for a link; paste it again after opening)`
            : '';
        try {
            await navigator.clipboard.writeText(location.href);
            say(`Link copied${omitted}.`);
        }
        catch {
            say(`Copy the address bar to share this view${omitted}.`);
        }
    });

    const exportMenu = document.getElementById('export-menu');
    async function exportAs(kind) {
        if (state.view === 'gallery') {
            say('Export works in the single view: open a pane there with ↗.');
            return;
        }
        if (!current?.phase2) {
            say('Nothing to export until there is a fit.');
            return;
        }
        const { data, result, phase2 } = current;
        const base = exportBase(state);
        if (kind === 'observations' || kind === 'grid') {
            const tables = fitTables({ x: data.x, y: data.y, fAt: data.fAt, names: exportNames(state, data),
                ...exportLayers(result, phase2, data) });
            const columns = tables[kind === 'grid' ? 'grid' : 'observations'];
            const name = `${base}-${kind}.csv`;
            download(new Blob([toCsv(columns)], { type: 'text/csv' }), name);
            say(`Saved ${name} (${fmtInt(columns[0].values.length)} rows).`);
            return;
        }
        const svg = chart.toSvg({ title: data.label });
        if (kind === 'svg') {
            download(new Blob([svg.text], { type: 'image/svg+xml' }), `${base}.svg`);
            say(`Saved ${base}.svg.`);
            return;
        }
        try {
            const image = await rasterize(svg, IMAGE_TYPES[kind], IMAGE_SCALE);
            const name = `${base}.${EXTENSIONS[kind]}`;
            download(image.blob, name);
            say(`Saved ${name} (${image.width} × ${image.height} px).`);
        }
        catch (err) {
            say(`Could not make the image: ${err.message}`);
        }
    }
    exportMenu.addEventListener('click', event => {
        const item = event.target.closest('[data-export]');
        if (!item)
            return;
        exportMenu.open = false;
        exportAs(item.dataset.export);
    });
    document.addEventListener('click', event => {
        if (exportMenu.open && !exportMenu.contains(event.target))
            exportMenu.open = false;
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && exportMenu.open) {
            exportMenu.open = false;
            exportMenu.querySelector('summary').focus();
        }
    });
    for (const button of document.querySelectorAll('#view-toggle [data-view]')) {
        button.addEventListener('click', () => {
            if (state.view !== button.dataset.view)
                onChange({ view: button.dataset.view });
        });
    }
    // --- settings history: Reset, Undo and Redo change the settings, never the data sets
    const resetButton = document.getElementById('settings-reset');
    const undoButton = document.getElementById('settings-undo');
    const redoButton = document.getElementById('settings-redo');
    const mac = /Mac|iPhone|iPad/.test(navigator.platform);
    const undoKeys = mac ? '⌘Z' : 'Ctrl+Z';
    const redoKeys = mac ? '⇧⌘Z' : 'Ctrl+Shift+Z or Ctrl+Y';
    resetButton.title = 'Reset the settings to their defaults (the data sets stay as they are; Undo brings them back)';
    function syncHistory() {
        undoButton.disabled = !settingsHistory.canUndo();
        redoButton.disabled = !settingsHistory.canRedo();
        resetButton.disabled = JSON.stringify(settingsOf(state)) === JSON.stringify(settingsOf(DEFAULTS));
        undoButton.title = `Undo the last change to the settings (${undoKeys})`;
        redoButton.title = `Redo (${redoKeys})`;
    }
    function applySettings(snapshot) {
        if (!snapshot)
            return;
        Object.assign(state, snapshot);
        syncHistory();
        rebuildControls();
        update();
        scheduleHash();
    }
    resetButton.addEventListener('click', () => {
        const defaults = settingsOf(DEFAULTS);
        if (settingsHistory.record(defaults, performance.now(), { coalesce: false }))
            applySettings(defaults);
    });
    undoButton.addEventListener('click', () => applySettings(settingsHistory.undo()));
    redoButton.addEventListener('click', () => applySettings(settingsHistory.redo()));
    // the keys act on the settings unless a text field has focus, which keeps its own undo
    const typing = target => target instanceof HTMLTextAreaElement || (target instanceof HTMLInputElement
        && !['range', 'checkbox', 'radio', 'button', 'file'].includes(target.type)) || target?.isContentEditable;
    document.addEventListener('keydown', event => {
        if (!(event.metaKey || event.ctrlKey) || event.altKey || typing(event.target))
            return;
        const key = event.key.toLowerCase();
        const redo = (key === 'z' && event.shiftKey) || (key === 'y' && !mac && !event.shiftKey);
        if (key === 'z' && !event.shiftKey) {
            event.preventDefault();
            applySettings(settingsHistory.undo());
        }
        else if (redo) {
            event.preventDefault();
            applySettings(settingsHistory.redo());
        }
    });
    syncHistory();
    const about = document.getElementById('about');
    document.getElementById('about-open').addEventListener('click', () => about.showModal());

    window.addEventListener('resize', draw);
    onThemeChange(draw);
    update();
}

startup().catch(err => {
    d3.select('#status').classed('error', true).text(`Error: ${err.message}`);
    console.error(err);
});
