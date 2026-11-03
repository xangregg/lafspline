// App entry point: state (⇄ URL hash) → data → region (phase 1: a P-spline band, or Aₙ) → fit (phase 2) → chart and
// readouts.
//
// Everything runs on the main thread: the band takes milliseconds to tens of milliseconds (one λ search, one Monte
// Carlo of B = 1000 cheap draws), a* (fit.js leavingAmbition) some 15–25 phase-2 fits of a few milliseconds each
// (sparse fits are some 10–30 iterated fits each, so a* takes up to about half a second for n in the thousands),
// and a knob change one more fit. Aₙ takes a calibration of τ (B = 1000 draws of its multiscale statistic, tens to
// hundreds of milliseconds, cached by design and interval family) and its phase-2 solves are dense (at most
// AN_MAX_SEGMENTS segments). Regions and a* are cached by data and settings; the ladder (a few milliseconds) is
// kept on the region.

import * as d3 from 'd3';
import { simulate, signalFunction, SIGNALS } from '../core/signals.js';
import { parseTable, prepareXY, rowIndex } from '../core/data.js';
import { readDataset } from '../core/datasets.js';
import { bandRegion, checkBand, checkGrid } from '../core/band.js';
import { defaultSegments } from '../core/pspline.js';
import {
    regionSpec, leavingAmbition, fitIterated, fitSparse, sparseLambdas, splineValues, LEAVE_TOL,
} from '../core/fit.js';
import { ladder } from '../core/ladder.js';
import { anRegion, calibrateTau, checkFit } from '../core/regions.js';
import { anBoxes } from '../core/an.js';
import { familyIntervals, familyLevels, familyText } from '../core/intervals.js';
import { tauFromSample } from '../core/calibrate.js';
import { constraintBoxes } from '../core/region.js';
import { createControls } from './controls.js';
import { stateToHash, hashToState, AN_MAX_SEGMENTS } from './urlstate.js';
import { fmtExp } from './format.js';
import { createMainChart } from './chart.js';
import { cellEdges } from './layout.js';
import { onThemeChange } from './theme.js';

const fmt = d3.format('.4~g');
const fmtInt = d3.format(',');
const fmtShort = d3.format('.2~g');
const CACHE_LIMIT = 24; // bands kept
const BAND_B = 1000; // Monte Carlo draws for the band's critical value
const BAND_SEED = 1;
const AN_B = 1000; // Monte Carlo draws of Aₙ's multiscale statistic for a calibrated τ
const AN_LAMBDA = 2; // Aₙ's interval lengths grow by this factor per level (the paper's)
const CENTER_COLOR = 'series-3';
const FIT_COLOR = 'series-1';
const HASH_DEBOUNCE = 400; // ms after the last change before the URL is updated
const MESSAGE_MS = 4000; // toolbar message display time
const USER = 'user';
const EXAMPLE_PREFIX = 'example:';
const EXAMPLES_INDEX = 'data/index.json';

const state = {
    source: 'ramp',
    n: SIGNALS.ramp.n,
    sigma: SIGNALS.ramp.sigma,
    seed: 1,
    showTruth: true,
    userText: '',
    xCol: null, // column index, −1 = row number, null = auto (first column if there are ≥ 2)
    yCol: null, // column index, null = auto (second column if there are ≥ 2, else the only one)
    region: 'band', // phase-1 region: 'band' (P-spline) | 'An' (Davies, Kovac & Meise's multiresolution region)
    family: 'blocks', // Aₙ intervals: 'blocks' | 'quarter' (¼-width steps) | 'every' (every position) | 'all'
    calibrate: true, // Aₙ: τ calibrated for coverage alpha, instead of tau
    tau: 3, // Aₙ: τ when not calibrated
    boxes: 'tightest', // Aₙ boxes on the chart: 'tightest' (the level of the largest ratio) | 'all' | 'off'
    alpha: 0.95, // coverage: the band's, and Aₙ's when calibrated
    covariance: 'bayes', // 'bayes' | 'sandwich'
    lambdaMode: 'gcv', // 'gcv' | 'aic' | 'given'
    logLambda: 0, // natural log, when lambdaMode is 'given'
    undersmooth: 1 / 16, // multiplier on the chosen λ
    nseg: null, // null = auto (pspline.defaultSegments)
    diff: 2, // penalty order
    penalty: 'quadratic', // phase-2 ambition: 'quadratic' (squared differences) | 'sparse' (their number)
    ambition: 0, // decades from a*, the largest ambition before the fit leaves the region
    flat: 0, // share of the ambition toward flat (D₁) rather than straight (D₂)
    logKappa: 4, // log₁₀ κ
    logEpsilon: -2, // log₁₀ ε
};
const DEFAULTS = structuredClone(state);

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
 * while a bundled example is loading. As the earlier project's deriveData.
 */
function deriveData(st, exampleTexts, examples) {
    let x;
    let y;
    let fAt = null;
    let columns = [];
    let dataKey;
    let label;
    const notes = [];
    if (st.source.startsWith(EXAMPLE_PREFIX)) {
        const id = st.source.slice(EXAMPLE_PREFIX.length);
        const loaded = exampleTexts.get(id);
        label = examples.get(id)?.label ?? id;
        if (loaded === undefined)
            return { pending: true, id, label, columns };
        if (loaded.error)
            return { error: `Could not load the example: ${loaded.error}`, columns };
        const d = readDataset(loaded);
        if (d.errors.length)
            return { error: `Example ${id}: ${d.errors.join('; ')}`, columns };
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
        const table = getTable(st.userText);
        columns = table.names;
        label = 'Your data';
        if (table.columns.length === 0)
            return { error: 'Paste or drop data with at least one numeric column.', columns };
        const two = columns.length >= 2;
        const xCol = st.xCol !== null && st.xCol < columns.length ? st.xCol : (two ? 0 : -1);
        const yCol = st.yCol !== null && st.yCol < columns.length ? st.yCol : (two ? 1 : 0);
        st.xCol = xCol;
        st.yCol = yCol;
        const d = prepareXY(xCol < 0 ? rowIndex(table.rows) : table.columns[xCol], table.columns[yCol]);
        x = d.x;
        y = d.y;
        dataKey = `user|${hashText(st.userText)}|${xCol}|${yCol}`;
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
    const lambda = st.lambdaMode === 'given' ? `given:${st.logLambda}` : `${st.lambdaMode}:${st.undersmooth}`;
    return [data.dataKey, st.alpha, st.covariance, lambda, st.nseg ?? 'auto', st.diff].join('|');
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
    const opts = { alpha: st.alpha, covariance: st.covariance, diff: st.diff, B: BAND_B, seed: BAND_SEED };
    if (st.nseg !== null)
        opts.nseg = st.nseg;
    if (st.lambdaMode === 'given') {
        opts.lambda = Math.exp(st.logLambda);
    }
    else {
        opts.criterion = st.lambdaMode;
        opts.undersmooth = st.undersmooth;
    }
    const start = performance.now();
    try {
        const region = bandRegion(data.x, data.y, opts);
        return { region, ms: performance.now() - start };
    }
    catch (err) {
        if (err instanceof RangeError)
            return { error: err.message };
        throw err;
    }
}

function starKey(st, data) {
    return [regionKey(st, data), st.penalty, st.flat, st.logKappa, st.logEpsilon].join('|');
}

function phase2Options(st) {
    return { flat: st.flat, kappa: 10 ** st.logKappa, epsilon: 10 ** st.logEpsilon };
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
    const sparse = st.penalty === 'sparse';
    const wanted = star.ambition * 10 ** st.ambition;
    const a = Math.min(wanted, star.top);
    const spec = regionSpec(region, { ...opts, ambition: a });
    const fit = sparse
        ? fitSparse({ ...spec, ...sparseLambdas({ ambition: a, flat: opts.flat, G: region.center.length }) })
        : fitIterated(spec);
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
    return { ...fit, a, sparse, clamped: wanted > star.top, argMax, leaving, leavingAtData: atData, opts };
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
    sel.append('p').attr('class', 'claim').text(an
        ? `Checked against the region below, ${region.label}, on its ${fmtInt(G)} intervals: the fit's mean over each `
            + 'interval of observations must lie within c/√|I| of the mean of y there. Aₙ looks only at the '
            + 'observations, so between them the ambition penalty alone shapes the fit. The region is soft: '
            + 'leaving it costs κ per squared box half-height, and the check says by how much.'
        : `Checked against the region below, the ${region.label}, on its check grid. The region is soft: leaving it `
            + 'costs κ per squared half-width, so the fit can cross its edge, and the check says by how much.');
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
    if (fit.sparse && !fit.settled) {
        sel.append('p').attr('class', 'warning').text(`The bends and steps were still changing after `
            + `${fit.reweights} reweightings; the count shown is where the reweighting stopped.`);
    }
    const rows = [];
    const add = (label, value, note) => rows.push({ label, value, note });
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
    add('penalty', fit.sparse ? 'number of bends or steps' : 'squared differences', fit.sparse
        ? 'adaptive ridge (Frommlet & Nuel 2016), an approximation of the count: a local optimum, not proven fewest'
        : 'straighter (or flatter) everywhere; no piece is exactly straight');
    if (fit.sparse)
        addPieceRows(add, fit.pieces, region);
    add('toward', st.flat === 0 ? 'straight' : st.flat === 1 ? 'flat' : `${Math.round(100 * (1 - st.flat))}% straight, `
        + `${Math.round(100 * st.flat)}% flat`, 'second (straight) and first (flat) differences of the coefficients');
    add('effective dimension', fmt(fit.ed), `trace of the hat matrix, ${an ? 'the interval means of y'
        : `the center`} as data${an ? '' : ` (band: ${fmt(region.params.ed)})`}${fit.sparse
        ? '; about the polynomial\'s coefficients plus one per free difference' : ''}`);
    add('κ', fmtExp(fit.opts.kappa), 'weight on the squared distance outside the region');
    add('ε', fmtExp(fit.opts.epsilon), an ? 'weight on the squared distance from the box centers, the interval '
        + 'means of y' : 'weight on the squared distance from the center');
    add('basis', `${region.params.nseg} segments`, an
        ? `cubic, equal segments over the data's x range${region.nsegCapped ? `; capped at ${AN_MAX_SEGMENTS}` : ''}`
        : 'cubic, on the band\'s knots');
    if (fit.sparse) {
        add('solver', `${fit.reweights} reweighting${fit.reweights === 1 ? '' : 's'}, ${fit.iterations} solves`,
            `${fit.settled ? 'the joins settled' : 'the joins had not settled'}; ${fit.pruned} dropped by backward `
            + `elimination; the final fit is ${fit.converged ? 'the exact minimizer for its joins' : 'not converged'}`);
    }
    else {
        const shortened = fit.damped ? `; ${fit.damped} step${fit.damped === 1 ? '' : 's'} shortened by the line `
            + 'search' : '';
        add('solver', `${fit.iterations} solve${fit.iterations === 1 ? '' : 's'}`, `iterated weights: `
            + `${fit.converged ? 'converged, the exact minimizer' : 'not converged'}${shortened}`);
    }
    add('computed in', `${fmt(fit.ms)} ms`, `a*: ${star.fits} fits, ${fmt(star.ms)} ms`);
    appendRows(sel, rows);
}

/**
 * Rows for a sparse fit's pieces: straight pieces and bends (second differences), flat pieces and steps (first).
 * A join's x is its middle coefficient's peak, good to about half a segment.
 */
function addPieceRows(add, pieces, region) {
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
        add(order === 2 ? 'straight' : 'flat', `${plural(k, piece)}, ${plural(joins.length, join)}${ends}`, where);
    }
}

function addLadderRows(add, region, x) {
    const an = region.kind === 'An';
    for (const rung of region.ladder) {
        const name = rung.shape === 'line' ? 'a straight line' : 'a constant';
        add(name, rung.fits ? 'fits inside' : 'does not fit', `the best (minimax) ${rung.shape} reaches `
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
    sel.append('p').attr('class', 'claim').text(`${region.claim} Aₙ bounds the residuals' sum over each interval `
        + 'of the observations: |Σ_I (y − g)| ≤ c·√|I|.');
    const rows = [];
    const add = (label, value, note) => rows.push({ label, value, note });
    addLadderRows(add, region, data.x);
    add('σ̂', fmt(p.sigma), 'from first differences (Davies, Kovac & Meise eq. 10), tie-aware');
    const cal = p.calibration;
    add('τ', fmt(p.tau), cal
        ? `calibrated for coverage ${fmt(cal.alpha)} by Monte Carlo (B = ${fmtInt(cal.B)}) for this n, design and `
            + `intervals; Monte Carlo 95% interval ${fmt(cal.tauLo)} – ${fmt(cal.tauHi)}`
        : 'given');
    add('threshold c', fmt(an.scale), 'σ̂·√(τ log n), units of y');
    const kind = { blocks: 'the paper\'s blocks', overlapping: 'overlapping windows', all: 'every interval' };
    add('intervals', fmtInt(an.lo.length), `${kind[result.family.kind]}${result.family.kind === 'all' ? ''
        : ` (${familyText(result.family)}: interval lengths grow by λ per level)`}`);
    add('box half-height', `${fmt(d3.min(region.half))} – ${fmt(d3.max(region.half))}`, 'c/√|I|, units of y');
    add('computed in', `${fmt(result.ms)} ms`, result.calMs === null
        ? (cal ? 'the calibration sample was already computed for these x and intervals' : '')
        : `including ${fmt(result.calMs)} ms to simulate the calibration sample`);
    appendRows(sel, rows);
}

function appendRows(sel, rows) {
    const dl = sel.append('dl');
    for (const r of rows) {
        dl.append('dt').text(r.label);
        dl.append('dd').text(r.value);
        if (r.note)
            dl.append('dd').attr('class', 'dd-note').text(r.note);
    }
}

function renderReadouts(container, data, result, truthCheck, phase2, st) {
    const sel = d3.select(container);
    sel.selectAll('*').remove();
    if (phase2)
        renderFitReadouts(sel, result.region, phase2.star, phase2.fit, st, data.x);
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
        sel.append('p').attr('class', 'claim').text(region.claim);
        if (p.atEdge) {
            const end = Math.log(p.lambdaChosen) > 0 ? 'largest' : 'smallest';
            sel.append('p').attr('class', 'warning').text(`${p.lambdaSource.toUpperCase()} chose the ${end} λ searched`
                + (end === 'largest'
                    ? `: the fit is close to its limit, a polynomial of degree ${p.diff - 1}.`
                    : ': the optimum may lie further out; consider fewer segments or a given λ.'));
        }
        const rows = [];
        const add = (label, value, note) => rows.push({ label, value, note });
        addLadderRows(add, region, data.x);
        let lambdaNote;
        if (p.lambdaSource === 'given')
            lambdaNote = 'given';
        else if (p.undersmooth === 1)
            lambdaNote = `chosen by ${p.lambdaSource.toUpperCase()}`;
        else
            lambdaNote = `${p.lambdaSource.toUpperCase()} chose ${fmt(p.lambdaChosen)}; × ${fmt(p.undersmooth)} undersmoothing`;
        add('λ', `${fmt(p.lambda)} (log ${p.lambda > 0 ? Math.log(p.lambda).toFixed(2) : '−∞'})`, lambdaNote);
        add('effective dimension', fmt(p.ed));
        add('σ̂', fmt(p.sigma), `from the residuals, ν = n − ED = ${fmt(p.df)} degrees of freedom`);
        add('critical value q', fmt(p.q), `${p.alpha} quantile of the max over the grid, Monte Carlo B = ${fmtInt(p.B)}`);
        add('half-width', `${fmt(d3.min(region.half))} – ${fmt(d3.max(region.half))}`, 'q × standard error, units of y');
        add('basis', `${p.nseg} segments, degree ${p.deg}`, `penalty order ${p.diff}`);
        add('check grid', `${fmtInt(region.grid.length)} points`, 'the distinct data x and 4 points per segment');
        add('computed in', `${fmt(result.ms)} ms`);
        appendRows(sel, rows);
    }
    sel.append('h2').text('Data');
    const dl = sel.append('dl');
    dl.append('dt').text('observations');
    dl.append('dd').text(fmtInt(data.n));
    dl.append('dt').text('distinct x');
    dl.append('dd').text(fmtInt(data.m));
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

function rejectedNote(rejected) {
    return rejected.length ? `Ignored invalid link setting(s): ${rejected.join(', ')}.` : null;
}

async function startup() {
    const [loaded, exampleList] = await Promise.all([hashToState(location.hash, DEFAULTS), loadExamples()]);
    const examples = new Map(exampleList.map(e => [e.id, e]));
    Object.assign(state, loaded.state);
    let linkNote = rejectedNote(loaded.rejected);
    if (state.source.startsWith(EXAMPLE_PREFIX) && !examples.has(state.source.slice(EXAMPLE_PREFIX.length))) {
        linkNote = `The link's example data set '${state.source.slice(EXAMPLE_PREFIX.length)}' is not in this version; `
            + 'showing the default.';
        Object.assign(state, { source: DEFAULTS.source, n: DEFAULTS.n, sigma: DEFAULTS.sigma });
    }
    const exampleTexts = new Map(); // id → file text, or {error}
    const status = d3.select('#status');
    const readouts = document.getElementById('readouts');
    const chart = createMainChart(document.getElementById('main-chart'));
    const regionCache = new Map();
    const starCache = new Map();
    const nullCache = new Map(); // Aₙ calibration samples, by design and interval family
    let current = null; // {data, result, phase2} drawn

    const controlsEl = document.getElementById('controls');
    const onChange = patch => {
        Object.assign(state, patch);
        update();
        scheduleHash();
    };
    let controls = createControls(controlsEl, state, onChange, exampleList);
    function rebuildControls() {
        controlsEl.replaceChildren();
        controls = createControls(controlsEl, state, onChange, exampleList);
    }

    function draw() {
        if (!current)
            return;
        const { data, result, phase2 } = current;
        const region = result.region;
        const an = region?.kind === 'An';
        const truth = data.fAt && state.showTruth;
        const fits = [];
        if (region && !an) {
            fits.push({ key: 'center', label: 'band center (phase 1)', g: region.center, x: region.grid,
                colorVar: CENTER_COLOR, primary: !phase2 });
        }
        if (phase2) {
            fits.push({ key: 'fit', label: 'fit (phase 2)', g: phase2.curve.g, x: phase2.curve.x, colorVar: FIT_COLOR,
                primary: true });
        }
        const boxes = an && phase2 ? anBoxSpec(result, phase2, data) : null;
        chart.setBoxes(null);
        chart.render({
            x: data.x,
            y: data.y,
            f: truth ? data.x.map(data.fAt) : null,
            fits,
            extremes: [],
            edges: data.edges,
            band: region && !an ? { lb: region.lower, ub: region.upper, label: region.label, x: region.grid } : null,
            boxesLabel: boxes ? boxes.label : null,
        });
        chart.setBoxes(boxes);
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

    function update() {
        const data = deriveData(state, exampleTexts, examples);
        if (data.pending) {
            status.classed('error', false).text(`Loading ${data.label}…`);
            controls.sync(state, { columns: data.columns });
            if (!exampleTexts.has(data.id)) {
                exampleTexts.set(data.id, undefined);
                fetch(`data/${examples.get(data.id)?.file ?? `${data.id}.csv`}`)
                    .then(r => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
                    .then(text => exampleTexts.set(data.id, text))
                    .catch(err => exampleTexts.set(data.id, { error: err.message ?? String(err) }))
                    .finally(update);
            }
            return;
        }
        if (data.error) {
            status.classed('error', true).text(data.error);
            controls.sync(state, { columns: data.columns });
            d3.select(readouts).selectAll('*').remove();
            return;
        }
        const key = regionKey(state, data);
        let result = regionCache.get(key);
        if (!result) {
            result = state.region === 'An' ? computeAn(state, data, nullCache) : computeBand(state, data);
            remember(regionCache, key, result);
        }
        const region = result.region;
        const an = region?.kind === 'An';
        let phase2 = null;
        if (region) {
            if (!an)
                region.dataAt ??= dataPositions(region.grid, data.x);
            region.ladder ??= ladder(region);
            const sk = starKey(state, data);
            let star = starCache.get(sk);
            if (!star) {
                star = leavingAmbition(regionSpec(region, { ...phase2Options(state), ambition: 0 }),
                    { sparse: state.penalty === 'sparse' });
                remember(starCache, sk, star);
            }
            const fit = computeFit(state, region, star);
            // the curve drawn: a band's fit is already on its check grid; Aₙ's boxes are interval means
            let curve = { x: region.grid, g: fit.g };
            if (an) {
                region.drawGrid ??= checkGrid(data.x, region.params.xl, region.params.xr, region.params.nseg);
                curve = { x: region.drawGrid, g: splineValues(fit.beta, region.drawGrid, region.params) };
            }
            phase2 = { star, fit, curve };
        }
        current = { data, result, phase2 };
        let truthCheck = null;
        if (region && data.fAt) {
            truthCheck = an ? checkFit(result.an, data.x.map(data.fAt), 0)
                : checkBand(region, region.grid.map(data.fAt));
        }
        controls.sync(state, {
            columns: data.columns,
            nsegAuto: defaultSegments(data.m),
            lambdaChosen: region && !an && region.params.lambdaSource !== 'given' ? region.params.lambdaChosen : null,
            ambition: phase2 ? { a: phase2.fit.a, star: phase2.star.ambition, capped: phase2.star.capped,
                clamped: phase2.fit.clamped, sparse: phase2.fit.sparse } : null,
        });
        draw();
        renderReadouts(readouts, data, result, state.showTruth ? truthCheck : null, phase2, state);
        const note = linkNote ? ` ${linkNote}` : '';
        if (result.error) {
            status.classed('error', true).text(`${data.label}: no ${an || state.region === 'An' ? 'Aₙ' : 'band'} `
                + `(${result.error})${note}`);
        }
        else {
            const pieces = phase2.fit.sparse
                ? phase2.fit.pieces.orders.map(o => `, ${fmtInt(o.pieces)} ${o.order === 2 ? 'straight' : 'flat'} `
                    + `piece${o.pieces === 1 ? '' : 's'}`).join('')
                : '';
            status.classed('error', Boolean(linkNote)).text(`${data.label}: n = ${fmtInt(data.n)}; ${region.label}; `
                + `fit ED ${fmt(phase2.fit.ed)}${pieces}.${note}`);
        }
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

    window.addEventListener('resize', draw);
    onThemeChange(draw);
    update();
}

startup().catch(err => {
    d3.select('#status').classed('error', true).text(`Error: ${err.message}`);
    console.error(err);
});
