// Control panel: data source (examples, simulated signals, pasted or dropped CSV), the phase-2 fit's knobs and the
// phase-1 band settings. Patterned on the earlier project's controls.js (el, field, debounce are the same helpers). Emits
// partial state patches.

import { SIGNALS } from '../core/signals.js';
import { MAX_N, MAX_SEGMENTS, AMBITION_RANGE, LOG_KAPPA_RANGE, LOG_EPSILON_RANGE } from './urlstate.js';
import { fmtExp } from './format.js';

const DEBOUNCE = 300; // ms for typed input
const USER = 'user';
const AMBITION_STEP = 0.1; // decades
const FLAT_STEP = 0.05;
const UNDERSMOOTH_OPTIONS = [
    [1, 'none: λ as chosen'],
    [1 / 4, 'λ × 1/4'],
    [1 / 16, 'λ × 1/16 (default)'],
    [1 / 64, 'λ × 1/64'],
];
const LAMBDA_OPTIONS = [
    ['gcv', 'chosen by GCV'],
    ['aic', 'chosen by AIC'],
    ['given', 'given (log λ below)'],
];
const COVARIANCE_OPTIONS = [
    ['bayes', 'Bayesian (covers smooth f)'],
    ['sandwich', 'sandwich (covers E f̂)'],
];
const DIFF_OPTIONS = [[1, '1: toward flat'], [2, '2: toward straight'], [3, '3: toward quadratic']];
const PENALTY_OPTIONS = [
    ['quadratic', 'squared differences'],
    ['sparse', 'number of bends or steps'],
];

function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'text')
            node.textContent = v;
        else if (k in node && typeof v !== 'string')
            node[k] = v;
        else
            node.setAttribute(k, v);
    }
    node.append(...children);
    return node;
}

function field(labelText, input, hint) {
    const children = [el('span', { class: 'field-label', text: labelText }), input];
    if (hint)
        children.push(hint instanceof Node ? hint : el('span', { class: 'field-hint', text: hint }));
    return el('label', { class: 'field' }, children);
}

function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

/**
 * Build the controls.
 *
 * @param {HTMLElement} container
 * @param {object} state    initial state (see main.js)
 * @param {function(object): void} onChange  receives a partial state patch
 * @param {Array<{id: string, label: string}>} [examples]  bundled data sets (data/index.json)
 * @returns {{sync: function(object, {columns: string[], nsegAuto: number|null, lambdaChosen: number|null,
 *     ambition: {a: number, star: number, capped: boolean, clamped: boolean, sparse: boolean}|null}): void}}  sync
 *     refreshes the parts that depend on state or data
 */
export function createControls(container, state, onChange, examples = []) {
    const later = debounce(onChange, DEBOUNCE);
    const numberInput = (key, attrs, parse = Number) => {
        const input = el('input', { type: 'number', value: String(state[key] ?? ''), ...attrs });
        input.addEventListener('input', () => {
            const v = parse(input.value);
            if (input.value !== '' && Number.isFinite(v) && input.checkValidity())
                later({ [key]: v });
        });
        return input;
    };
    const select = (key, options, parse = v => v) => {
        const node = el('select', {}, options.map(([value, text]) => el('option', { value: String(value), text })));
        node.value = String(state[key]);
        node.addEventListener('change', () => onChange({ [key]: parse(node.value) }));
        return node;
    };

    // --- data
    const source = el('select', { id: 'source' });
    if (examples.length) {
        source.append(el('optgroup', { label: 'Examples' },
            examples.map(e => el('option', { value: `example:${e.id}`, text: e.label }))));
    }
    const sim = el('optgroup', { label: 'Simulated' });
    for (const [key, spec] of Object.entries(SIGNALS))
        sim.append(el('option', { value: key, text: spec.label }));
    source.append(sim, el('option', { value: USER, text: 'Your data (paste or drop CSV)' }));
    source.value = state.source;
    source.addEventListener('change', () => {
        const patch = { source: source.value };
        const spec = SIGNALS[source.value];
        if (spec) {
            patch.n = Math.min(spec.n, MAX_N);
            patch.sigma = spec.sigma;
            nInput.value = String(patch.n);
            sigmaInput.value = String(patch.sigma);
        }
        onChange(patch);
    });
    const nInput = numberInput('n', { min: 10, max: MAX_N, step: 1 }, v => Math.round(Number(v)));
    const sigmaInput = numberInput('sigma', { min: 0, step: 'any' });
    const seedInput = numberInput('seed', { step: 1 }, v => Math.round(Number(v)));
    const newSeed = el('button', { type: 'button', text: 'New seed' });
    newSeed.addEventListener('click', () => {
        const seed = Math.floor(Math.random() * 1e6);
        seedInput.value = String(seed);
        onChange({ seed });
    });
    const truth = el('input', { type: 'checkbox', checked: state.showTruth });
    truth.addEventListener('change', () => onChange({ showTruth: truth.checked }));
    const simBox = el('div', { class: 'group' }, [
        field('n', nInput),
        field('noise σ', sigmaInput),
        el('div', { class: 'row' }, [field('seed', seedInput), newSeed]),
        el('label', { class: 'check' }, [truth, ' show true f']),
    ]);

    const text = el('textarea', {
        rows: 6, spellcheck: false,
        placeholder: 'Paste CSV/TSV here or drop a file. A header row is optional.',
    });
    text.value = state.userText;
    text.addEventListener('input', () => later({ userText: text.value }));
    const loadText = t => {
        text.value = t;
        onChange({ userText: t, xCol: null, yCol: null });
    };
    text.addEventListener('dragover', e => e.preventDefault());
    text.addEventListener('drop', async e => {
        e.preventDefault();
        const file = e.dataTransfer?.files?.[0];
        if (file)
            loadText(await file.text());
    });
    const file = el('input', { type: 'file', accept: '.csv,.tsv,.txt,text/csv,text/plain' });
    file.addEventListener('change', async () => {
        if (file.files[0])
            loadText(await file.files[0].text());
    });
    const xCol = el('select');
    const yCol = el('select');
    xCol.addEventListener('change', () => onChange({ xCol: Number(xCol.value) }));
    yCol.addEventListener('change', () => onChange({ yCol: Number(yCol.value) }));
    const userBox = el('div', { class: 'group' }, [text, file, field('x column', xCol), field('y column', yCol)]);

    // --- band
    const alphaInput = numberInput('alpha', { min: 0.5, max: 0.999, step: 0.01 });
    const covariance = select('covariance', COVARIANCE_OPTIONS);
    const undersmooth = select('undersmooth', UNDERSMOOTH_OPTIONS, Number);
    const lambdaMode = select('lambdaMode', LAMBDA_OPTIONS);
    const lambdaHint = el('span', { class: 'field-hint' });
    const logLambdaInput = numberInput('logLambda', { step: 0.5 });
    const logLambdaField = field('log λ (natural)', logLambdaInput);
    const undersmoothField = field('undersmoothing', undersmooth,
        'a smaller λ narrows the smoothing, so the band also covers bends; it widens the band');
    const nsegInput = el('input', { type: 'number', min: 3, max: MAX_SEGMENTS, step: 1, placeholder: 'auto' });
    nsegInput.value = state.nseg === null ? '' : String(state.nseg);
    const nsegHint = el('span', { class: 'field-hint' });
    nsegInput.addEventListener('input', () => {
        if (nsegInput.value === '')
            later({ nseg: null });
        else if (nsegInput.checkValidity())
            later({ nseg: Math.round(Number(nsegInput.value)) });
    });
    const diff = select('diff', DIFF_OPTIONS, Number);

    // --- fit
    const slider = (key, min, max, step, show) => {
        const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step) });
        input.value = String(state[key]);
        const out = el('output', { text: show(state[key]) });
        input.addEventListener('input', () => {
            const v = Number(input.value);
            out.textContent = show(v);
            onChange({ [key]: v });
        });
        return { input, out, row: el('div', { class: 'slider' }, [input, out]) };
    };
    const signed = v => (v > 0 ? `+${v.toFixed(1)}` : v < 0 ? `−${(-v).toFixed(1)}` : '0');
    const ambition = slider('ambition', AMBITION_RANGE[0], AMBITION_RANGE[1], AMBITION_STEP, signed);
    const ambitionEnds = el('div', { class: 'slider-ends' }, [el('span', { text: '← detail' }),
        el('span', { text: 'simplicity →' })]);
    const ambitionHint = el('span', { class: 'field-hint' });
    const penalty = select('penalty', PENALTY_OPTIONS);
    const flat = slider('flat', 0, 1, FLAT_STEP, v => `${Math.round(100 * v)}%`);
    const logKappa = numberInput('logKappa', { min: LOG_KAPPA_RANGE[0], max: LOG_KAPPA_RANGE[1], step: 0.5 });
    const logEpsilon = numberInput('logEpsilon', { min: LOG_EPSILON_RANGE[0], max: LOG_EPSILON_RANGE[1], step: 0.5 });
    const advanced = el('details', {}, [
        el('summary', { text: 'Advanced' }),
        field('log₁₀ κ', logKappa, 'weight on the squared distance outside the region: larger is harder'),
        field('log₁₀ ε', logEpsilon, 'weight on the squared distance from the center: the pull inside the region'),
    ]);

    container.append(
        el('h2', { text: 'Data' }),
        field('source', source),
        simBox,
        userBox,
        el('h2', { text: 'Fit' }),
        el('p', { class: 'field-hint', text: 'A second P-spline, penalized severely for leaving the region, slightly '
            + 'for leaving its center, and for ambition: pressure toward straight (or flat).' }),
        field('ambition penalty', penalty, 'squared: straighter (or flatter) everywhere. Number: straight (or flat) '
            + 'pieces joined by short bends (or steps), by the adaptive ridge, an approximation'),
        field('ambition (decades from a*)', el('div', {}, [ambition.row, ambitionEnds]), ambitionHint),
        field('toward flat', flat.row, 'share of the ambition on first differences (flat) instead of second '
            + '(straight)'),
        advanced,
        el('h2', { text: 'Region' }),
        el('p', { class: 'field-hint', text: 'A P-spline fit and a simultaneous band around it.' }),
        field('coverage α', alphaInput, 'simultaneous over the check grid; Monte Carlo, B = 1000'),
        field('covariance', covariance),
        field('λ', lambdaMode, lambdaHint),
        logLambdaField,
        undersmoothField,
        field('segments', nsegInput, nsegHint),
        field('penalty order', diff, 'large λ pulls the P-spline toward a polynomial of degree order − 1'),
    );

    let lastColumns = null;
    function sync(st, { columns = [], nsegAuto = null, lambdaChosen = null, ambition: amb = null }) {
        simBox.hidden = !SIGNALS[st.source];
        userBox.hidden = st.source !== USER;
        const key = columns.join('\u0000');
        if (key !== lastColumns) {
            lastColumns = key;
            xCol.replaceChildren(el('option', { value: '-1', text: '(row number)' }),
                ...columns.map((name, j) => el('option', { value: String(j), text: name })));
            yCol.replaceChildren(...columns.map((name, j) => el('option', { value: String(j), text: name })));
        }
        xCol.value = String(st.xCol);
        yCol.value = String(st.yCol);
        const given = st.lambdaMode === 'given';
        logLambdaField.hidden = !given;
        undersmoothField.hidden = given;
        lambdaHint.textContent = !given && lambdaChosen !== null
            ? `${st.lambdaMode.toUpperCase()} chose λ = ${lambdaChosen.toPrecision(3)} (log λ = ${Math.log(lambdaChosen).toFixed(2)})`
            : '';
        nsegHint.textContent = nsegAuto !== null ? `empty = auto (${nsegAuto} for these data)` : 'empty = auto';
        if (document.activeElement !== alphaInput)
            alphaInput.value = String(st.alpha);
        if (amb === null) {
            ambitionHint.textContent = '0 is a*, the largest ambition before the fit leaves the region';
        }
        else if (amb.capped) {
            ambitionHint.textContent = amb.sparse
                ? `the fit never leaves the region: a* is the top of the range (${fmtExp(amb.star)}), where the fit `
                    + 'has no bends or steps, so above 0 nothing changes'
                : `the fit never leaves the region: a* is the largest ambition computed reliably `
                    + `(${fmtExp(amb.star)}), so above 0 nothing changes`;
        }
        else {
            ambitionHint.textContent = `0 is a* = ${fmtExp(amb.star)}, the largest ambition before the fit `
                + `leaves the region; now a = ${fmtExp(amb.a)}`
                + (amb.clamped ? (amb.sparse ? ' (the top of the range)' : ' (the largest computed reliably)') : '');
        }
    }

    return { sync };
}
