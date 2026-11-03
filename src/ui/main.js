// App entry point: state (⇄ URL hash) → data → band region (phase 1) → fit (phase 2) → chart and readouts.
//
// Everything runs on the main thread: the band takes milliseconds to tens of milliseconds (one λ search, one Monte
// Carlo of B = 1000 cheap draws), a* (fit.js leavingAmbition) some 15–25 phase-2 fits of a few milliseconds each,
// and a knob change one more fit. Bands and a* are cached by data and settings.

import * as d3 from 'd3';
import { simulate, signalFunction, SIGNALS } from '../core/signals.js';
import { parseTable, prepareXY, rowIndex } from '../core/data.js';
import { readDataset } from '../core/datasets.js';
import { bandRegion, checkBand } from '../core/band.js';
import { defaultSegments } from '../core/pspline.js';
import { regionSpec, leavingAmbition, fitIterated, maxAmbition, LEAVE_TOL } from '../core/fit.js';
import { createControls } from './controls.js';
import { stateToHash, hashToState } from './urlstate.js';
import { fmtExp } from './format.js';
import { createMainChart } from './chart.js';
import { cellEdges } from './layout.js';
import { onThemeChange } from './theme.js';

const fmt = d3.format('.4~g');
const fmtInt = d3.format(',');
const CACHE_LIMIT = 24; // bands kept
const BAND_B = 1000; // Monte Carlo draws for the band's critical value
const BAND_SEED = 1;
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
    alpha: 0.95,
    covariance: 'bayes', // 'bayes' | 'sandwich'
    lambdaMode: 'gcv', // 'gcv' | 'aic' | 'given'
    logLambda: 0, // natural log, when lambdaMode is 'given'
    undersmooth: 1 / 16, // multiplier on the chosen λ
    nseg: null, // null = auto (pspline.defaultSegments)
    diff: 2, // penalty order
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
    return [bandKey(st, data), st.flat, st.logKappa, st.logEpsilon].join('|');
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
 * The phase-2 fit at the knob setting: a = a*·10^ambition, at most the cap. Re-checked against the region on the
 * whole check grid and at the data x.
 */
function computeFit(st, region, star) {
    const opts = phase2Options(st);
    const spec0 = regionSpec(region, { ...opts, ambition: 0 });
    const cap = maxAmbition(spec0);
    const wanted = star.ambition * 10 ** st.ambition;
    const a = Math.min(wanted, cap);
    const fit = fitIterated(regionSpec(region, { ...opts, ambition: a }));
    const ratios = fit.g.map((g, j) => Math.abs(g - region.center[j]) / region.half[j]);
    let argMax = 0;
    for (let j = 1; j < ratios.length; j++) {
        if (ratios[j] > ratios[argMax])
            argMax = j;
    }
    let atData = 0;
    for (const j of region.dataAt) {
        if (ratios[j] > 1 + LEAVE_TOL)
            atData++;
    }
    let leaving = 0;
    for (const r of ratios) {
        if (r > 1 + LEAVE_TOL)
            leaving++;
    }
    return { ...fit, a, clamped: wanted > cap, argMax, leaving, leavingAtData: atData, opts };
}

function renderFitReadouts(sel, region, star, fit, st) {
    sel.append('h2').text('Fit');
    const G = region.grid.length;
    const m = region.dataAt.length;
    const v = sel.append('p').attr('class', `verdict ${fit.leaving === 0 ? 'ok' : 'bad'}`);
    if (fit.outside === 0) {
        v.append('strong').text('fit ∈ region');
        v.append('span').text(` at every check-grid point (largest |g − c|/h = ${fmt(fit.maxRatio)}).`);
    }
    else if (fit.leaving === 0) {
        v.append('strong').text('fit within the region to 1% of its half-width');
        v.append('span').text(`: largest |g − c|/h = ${d3.format('.4f')(fit.maxRatio)}, at x = `
            + `${fmt(region.grid[fit.argMax])}; ${fmtInt(fit.outside)} grid points touch the edge from outside, as a `
            + 'soft region allows.');
    }
    else {
        v.append('strong').text('fit ∉ region');
        v.append('span').text(` by more than 1% of the half-width at ${fmtInt(fit.leaving)} of ${fmtInt(G)} check-grid `
            + `points (${fmtInt(fit.leavingAtData)} of ${fmtInt(m)} data x); largest |g − c|/h = `
            + `${fmt(fit.maxRatio)}, at x = ${fmt(region.grid[fit.argMax])}.`);
    }
    sel.append('p').attr('class', 'claim').text(`Checked against the region below, the ${region.label}, on its check `
        + 'grid. The region is soft: leaving it costs κ per squared half-width, so the fit can cross its edge, and '
        + 'the check says by how much.');
    if (!fit.converged) {
        sel.append('p').attr('class', 'warning').text(`The iteration stopped after ${fit.iterations} solves without `
            + 'settling which points are outside: the fit is not the exact minimizer.');
    }
    const rows = [];
    const add = (label, value, note) => rows.push({ label, value, note });
    const decades = `${Math.abs(st.ambition)} decade${Math.abs(st.ambition) === 1 ? '' : 's'} `
        + `${st.ambition > 0 ? 'above' : 'below'} a*`;
    let ambitionNote;
    if (star.capped)
        ambitionNote = 'a*: no ambition up to the largest computed reliably takes the fit outside the region';
    else if (fit.clamped)
        ambitionNote = `${decades} is beyond the largest ambition computed reliably; using that`;
    else if (st.ambition === 0)
        ambitionNote = 'a*, the largest ambition before the fit leaves the region';
    else
        ambitionNote = `${decades}; a* = ${fmtExp(star.ambition)} is the largest before the fit leaves the region`;
    add('ambition a', fmtExp(fit.a), ambitionNote);
    add('toward', st.flat === 0 ? 'straight' : st.flat === 1 ? 'flat' : `${Math.round(100 * (1 - st.flat))}% straight, `
        + `${Math.round(100 * st.flat)}% flat`, 'second (straight) and first (flat) differences of the coefficients');
    add('effective dimension', fmt(fit.ed),
        `trace of the hat matrix, the center as data (band: ${fmt(region.params.ed)})`);
    add('κ', fmtExp(fit.opts.kappa), 'weight on the squared distance outside the region');
    add('ε', fmtExp(fit.opts.epsilon), 'weight on the squared distance from the center');
    add('basis', `${region.params.nseg} segments`, 'cubic, on the band\'s knots');
    const shortened = fit.damped ? `; ${fit.damped} step${fit.damped === 1 ? '' : 's'} shortened by the line search`
        : '';
    add('solver', `${fit.iterations} solve${fit.iterations === 1 ? '' : 's'}`, `iterated weights: `
        + `${fit.converged ? 'converged, the exact minimizer' : 'not converged'}${shortened}`);
    add('computed in', `${fmt(fit.ms)} ms`, `a*: ${star.fits} fits, ${fmt(star.ms)} ms`);
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
        renderFitReadouts(sel, result.region, phase2.star, phase2.fit, st);
    sel.append('h2').text('Region');
    if (result.error) {
        sel.append('p').attr('class', 'error').text(`No band: ${result.error}`);
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
    const bandCache = new Map();
    const starCache = new Map();
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
        const truth = data.fAt && state.showTruth;
        const fits = [];
        if (region) {
            fits.push({ key: 'center', label: 'band center (phase 1)', g: region.center, x: region.grid,
                colorVar: CENTER_COLOR, primary: !phase2 });
        }
        if (phase2) {
            fits.push({ key: 'fit', label: 'fit (phase 2)', g: phase2.fit.g, x: region.grid, colorVar: FIT_COLOR,
                primary: true });
        }
        chart.render({
            x: data.x,
            y: data.y,
            f: truth ? data.x.map(data.fAt) : null,
            fits,
            extremes: [],
            edges: data.edges,
            band: region ? { lb: region.lower, ub: region.upper, label: region.label, x: region.grid } : null,
            boxesLabel: null,
        });
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
        const key = bandKey(state, data);
        let result = bandCache.get(key);
        if (!result) {
            result = computeBand(state, data);
            remember(bandCache, key, result);
        }
        const region = result.region;
        let phase2 = null;
        if (region) {
            region.dataAt ??= dataPositions(region.grid, data.x);
            const sk = starKey(state, data);
            let star = starCache.get(sk);
            if (!star) {
                star = leavingAmbition(regionSpec(region, { ...phase2Options(state), ambition: 0 }));
                remember(starCache, sk, star);
            }
            phase2 = { star, fit: computeFit(state, region, star) };
        }
        current = { data, result, phase2 };
        const truthCheck = region && data.fAt ? checkBand(region, region.grid.map(data.fAt)) : null;
        controls.sync(state, {
            columns: data.columns,
            nsegAuto: defaultSegments(data.m),
            lambdaChosen: region && region.params.lambdaSource !== 'given' ? region.params.lambdaChosen : null,
            ambition: phase2 ? { a: phase2.fit.a, star: phase2.star.ambition, capped: phase2.star.capped,
                clamped: phase2.fit.clamped } : null,
        });
        draw();
        renderReadouts(readouts, data, result, state.showTruth ? truthCheck : null, phase2, state);
        const note = linkNote ? ` ${linkNote}` : '';
        if (result.error) {
            status.classed('error', true).text(`${data.label}: no band (${result.error})${note}`);
        }
        else {
            status.classed('error', Boolean(linkNote)).text(`${data.label}: n = ${fmtInt(data.n)}; ${region.label}; `
                + `fit ED ${fmt(phase2.fit.ed)}.${note}`);
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
