// App entry point (phase 1): state → data → band region → chart and readouts.
//
// The band takes milliseconds to tens of milliseconds (one λ search, one Monte Carlo of B = 1000 cheap draws), so it
// runs on the main thread and is cached by data and settings. URL state comes in phase 2.

import * as d3 from 'd3';
import { simulate, signalFunction, SIGNALS } from '../core/signals.js';
import { parseTable, prepareXY, rowIndex } from '../core/data.js';
import { readDataset } from '../core/datasets.js';
import { bandRegion, checkBand } from '../core/band.js';
import { defaultSegments } from '../core/pspline.js';
import { createControls } from './controls.js';
import { createMainChart } from './chart.js';
import { cellEdges } from './layout.js';
import { onThemeChange } from './theme.js';

const fmt = d3.format('.4~g');
const fmtInt = d3.format(',');
const CACHE_LIMIT = 24; // bands kept
const BAND_B = 1000; // Monte Carlo draws for the band's critical value
const BAND_SEED = 1;
const CENTER_COLOR = 'series-1';
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
};

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

function renderReadouts(container, data, result, truthCheck) {
    const sel = d3.select(container);
    sel.selectAll('*').remove();
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
        const dl = sel.append('dl');
        for (const r of rows) {
            dl.append('dt').text(r.label);
            dl.append('dd').text(r.value);
            if (r.note)
                dl.append('dd').attr('class', 'dd-note').text(r.note);
        }
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

async function startup() {
    const exampleList = await loadExamples();
    const examples = new Map(exampleList.map(e => [e.id, e]));
    const exampleTexts = new Map(); // id → file text, or {error}
    const status = d3.select('#status');
    const readouts = document.getElementById('readouts');
    const chart = createMainChart(document.getElementById('main-chart'));
    const bandCache = new Map();
    let current = null; // {data, result} drawn

    const controls = createControls(document.getElementById('controls'), state, patch => {
        Object.assign(state, patch);
        update();
    }, exampleList);

    function draw() {
        if (!current)
            return;
        const { data, result } = current;
        const region = result.region;
        const truth = data.fAt && state.showTruth;
        chart.render({
            x: data.x,
            y: data.y,
            f: truth ? data.x.map(data.fAt) : null,
            fits: region ? [{ key: 'center', label: 'P-spline (band center)', g: region.center, x: region.grid,
                colorVar: CENTER_COLOR, primary: true }] : [],
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
        current = { data, result };
        const region = result.region;
        const truthCheck = region && data.fAt ? checkBand(region, region.grid.map(data.fAt)) : null;
        controls.sync(state, {
            columns: data.columns,
            nsegAuto: defaultSegments(data.m),
            lambdaChosen: region && region.params.lambdaSource !== 'given' ? region.params.lambdaChosen : null,
        });
        draw();
        renderReadouts(readouts, data, result, state.showTruth ? truthCheck : null);
        if (result.error)
            status.classed('error', true).text(`${data.label}: no band (${result.error})`);
        else
            status.classed('error', false).text(`${data.label}: n = ${fmtInt(data.n)}; ${region.label}.`);
        document.body.dataset.ready = 'true';
    }

    window.addEventListener('resize', draw);
    onThemeChange(draw);
    update();
}

startup().catch(err => {
    d3.select('#status').classed('error', true).text(`Error: ${err.message}`);
    console.error(err);
});
