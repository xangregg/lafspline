// Control panel: data source (examples, simulated signals, pasted or dropped CSV), the phase-1 region's settings and
// the phase-2 fit's knobs, in the order they act. Explanations sit behind ⓘ icons (info.js) to keep the panel
// compact; text that depends on the data (the λ GCV chose, a*) is updated in them by sync. Patterned on the earlier project's
// controls.js (el, field, debounce are the same helpers). Emits partial state patches.

import { SIGNALS } from '../core/signals.js';
import { LOG_LAMBDA_RANGE } from '../core/pspline.js';
import {
    MAX_N, MAX_SEGMENTS, AMBITION_RANGE, LOG_KAPPA_RANGE, LOG_EPSILON_RANGE, AN_MAX_SEGMENTS, TAU_RANGE,
} from './urlstate.js';
import { fmtExp } from './format.js';
import { infoIcon, setInfo } from './info.js';

const DEBOUNCE = 300; // ms for typed input
const USER = 'user';
const AMBITION_STEP = 0.1; // decades
const FLAT_STEP = 0.05;
const LOG_LAMBDA_STEP = 0.1; // natural log units on the given-λ slider
const UNDERSMOOTH_OPTIONS = [
    [1, 'none: λ as chosen'],
    [1 / 4, 'λ × 1/4'],
    [1 / 16, 'λ × 1/16 (default)'],
    [1 / 64, 'λ × 1/64'],
];
const LAMBDA_OPTIONS = [
    ['gcv', 'chosen by GCV'],
    ['aic', 'chosen by AIC'],
    ['given', 'given (slider)'],
];
const COVARIANCE_OPTIONS = [
    ['bayes', 'Bayesian (covers smooth f)'],
    ['sandwich', 'sandwich (covers E f̂)'],
];
const DIFF_OPTIONS = [[1, '1: toward flat'], [2, '2: toward straight'], [3, '3: toward quadratic']];
const REGION_OPTIONS = [
    ['band', 'P-spline band'],
    ['boot', 'bootstrap band'],
    ['An', 'Aₙ (multiresolution)'],
];
const WEIGHT_OPTIONS = [
    ['matched', 'like resampled counts'],
    ['dirichlet', 'Dirichlet'],
];
const FAMILY_OPTIONS = [
    ['blocks', 'blocks (the paper\'s, lengths doubling)'],
    ['quarter', 'overlapping, ¼-width steps'],
    ['every', 'overlapping, every position'],
    ['all', 'all intervals (small n)'],
];
const BOX_OPTIONS = [
    ['tightest', 'the level of the tightest box'],
    ['all', 'all levels (outlines)'],
    ['off', 'none'],
];
const PENALTY_OPTIONS = [
    ['quadratic', 'squared differences'],
    ['sparse', 'number of bends or steps'],
];

const TIPS = {
    source: 'Bundled example data sets, signals simulated with Gaussian noise, or your own data pasted or dropped '
        + 'as CSV.',
    n: 'Number of observations, at x = i/n.',
    sigma: 'Standard deviation of the Gaussian noise added to the signal.',
    seed: 'Seed of the simulated noise: the same seed gives the same data.',
    truth: 'Draw the signal the data were simulated from, and check whether the region contains it.',
    xCol: 'The column for x; (row number) uses the order of the rows.',
    yCol: 'The column for y.',
    regionHead: 'Phase 1: a confidence region for f, from the data alone. The fit (phase 2) must stay inside it.',
    region: 'P-spline band: a simultaneous band around a P-spline fit, which covers f if f is smooth. Bootstrap '
        + 'band: around the same fit, from refits with random fractional weights on the observations, so it can be '
        + 'asymmetric and wider where the noise is; it covers the smoothed f. Aₙ (Davies, Kovac & Meise 2009): '
        + 'bounds on the sums of the residuals over intervals of the observations, assuming nothing about the '
        + 'smoothness of f.',
    alphaBand: 'The probability that the band contains f at every point of the check grid at once (the data x and '
        + '4 points per segment), by Monte Carlo with B = 1000.',
    alphaBoot: 'The probability that the band contains the smoothed f at every point of the check grid at once (the '
        + 'data x and 4 points per segment), by the bootstrap with B = 1000 replicates.',
    weights: 'Every observation gets a positive weight in each replicate, and the weights sum to n. Like resampled '
        + 'counts: the same mean, variance and skewness (about a fifth of the weights are below 0.1, as a count is 0 '
        + 'with probability 0.37); it held its coverage best in simulations. Dirichlet: the Bayesian bootstrap of '
        + 'Rubin (1981), or fractional-random-weight bootstrap; its weights are twice as skewed as counts. With '
        + 'skewed noise its band\'s asymmetry came nearer the truth, but its coverage was lower.',
    alphaAn: 'τ is calibrated by Monte Carlo so that Aₙ contains f with this probability.',
    family: 'The intervals of observations Aₙ checks. Blocks: the paper\'s, of lengths 1, 2, 4, … in fixed '
        + 'positions. Overlapping: windows of each length shifted by ¼ of the length or by one observation, which '
        + 'catch runs of residuals that straddle block edges. All: every interval (at most 300 distinct x). Denser '
        + 'families need a larger τ for the same coverage.',
    calibrate: 'Choose τ by simulating Aₙ\'s multiscale statistic on pure noise for this n, design and intervals, '
        + 'so that the coverage is α. Otherwise τ is given.',
    tau: 'Aₙ\'s threshold is c = σ̂·√(τ log n). The paper\'s default is 3; its coverage then depends on n and the '
        + 'intervals.',
    boxes: 'Each box spans an interval of observations, at the mean of y there ± c/√|I|; its tick is the fit\'s mean '
        + 'over the interval, which must lie inside. Tightest: the level of the interval with the largest ratio.',
    covariance: 'Bayesian: σ̂²(BᵀB + λDᵀD)⁻¹, which widens the band for the smoothing bias on average, so it covers '
        + 'a smooth f. Sandwich: the frequentist covariance, which covers only the smoother\'s mean E f̂.',
    lambda: 'The smoothing parameter of the band\'s P-spline: the weight on the squared differences of its '
        + 'coefficients. A larger λ gives a smoother fit and a narrower band. Chosen by GCV (generalized '
        + 'cross-validation) or AIC, or given.',
    logLambda: `Natural log of λ, over the range GCV searches (${LOG_LAMBDA_RANGE[0]} to ${LOG_LAMBDA_RANGE[1]}).`,
    undersmooth: 'Multiplies the chosen λ. A smaller λ follows bends more closely, so the band covers them too, at '
        + 'the cost of a wider band; with 1/16 the band contained smooth test signals in 94.5–98% of simulations.',
    diff: 'Order of the coefficient differences in the band\'s penalty: a large λ pulls the P-spline toward a '
        + 'polynomial of degree order − 1.',
    nseg: 'Equal segments of the cubic B-spline basis, for the band and the fit. Empty: automatic, about half the '
        + 'number of distinct x, at most 40.',
    nsegAn: ` For Aₙ at most ${AN_MAX_SEGMENTS}: its long intervals make each solve dense.`,
    fitHead: 'Phase 2: a second P-spline, penalized severely for leaving the region, slightly for distance from its '
        + 'center, and for ambition: pressure toward straight (or flat) lines.',
    penalty: 'Squared differences: the fit gets straighter (or flatter) everywhere, smoothly. Number of bends or '
        + 'steps: straight (or flat) pieces joined by short bends (or steps), by the adaptive ridge, which '
        + 'approximates the count.',
    ambition: 'Decades from a*, the largest ambition before the fit leaves the region by more than 1% of its '
        + 'half-width. 0 is the simplest fit inside; below it the fit follows more detail, above it the fit '
        + 'leaves the region.',
    flat: 'Share of the ambition on first differences (toward flat) rather than second (toward straight).',
    kappa: 'Weight on the squared distance outside the region, per squared half-width: larger makes the region '
        + 'harder.',
    epsilon: 'Weight on the squared distance from the region\'s center: a slight pull inside the region, which also '
        + 'makes the fit unique.',
};

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

/**
 * A labeled control. tip: the explanation behind an ⓘ after the label, a string or an icon from infoIcon (to
 * update it later).
 */
function field(labelText, input, tip) {
    const head = el('span', { class: 'field-label' }, [labelText]);
    if (tip)
        head.append(tip instanceof Node ? tip : infoIcon(tip, labelText));
    return el('label', { class: 'field' }, [head, input]);
}

function check(input, labelText, tip) {
    return el('label', { class: 'check' }, [input, ` ${labelText}`, infoIcon(tip, labelText)]);
}

function heading(text, tip) {
    return el('h2', {}, [text, infoIcon(tip, text)]);
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
 *     lambdaUsed: number|null, ambition: {a: number, star: number, capped: boolean, clamped: boolean,
 *     sparse: boolean}|null}): void}}  sync refreshes the parts that depend on state or data; lambdaChosen is the
 *     λ GCV or AIC chose, lambdaUsed the band's λ (after undersmoothing)
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
    const slider = (key, min, max, step, show) => {
        const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step) });
        input.value = String(state[key]);
        const out = el('output', { text: show(state[key]) });
        input.addEventListener('input', () => {
            const v = Number(input.value);
            out.textContent = show(v);
            onChange({ [key]: v });
        });
        const set = v => {
            input.value = String(v);
            out.textContent = show(v);
        };
        return { input, out, set, row: el('div', { class: 'slider' }, [input, out]) };
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
        el('div', { class: 'row' }, [field('n', nInput, TIPS.n), field('noise σ', sigmaInput, TIPS.sigma)]),
        el('div', { class: 'row' }, [field('seed', seedInput, TIPS.seed), newSeed]),
        check(truth, 'show true f', TIPS.truth),
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
    const userBox = el('div', { class: 'group' }, [text, file,
        el('div', { class: 'row' }, [field('x column', xCol, TIPS.xCol), field('y column', yCol, TIPS.yCol)])]);

    // --- region: Aₙ
    const regionKind = select('region', REGION_OPTIONS);
    const family = select('family', FAMILY_OPTIONS);
    const calibrate = el('input', { type: 'checkbox', checked: state.calibrate });
    calibrate.addEventListener('change', () => onChange({ calibrate: calibrate.checked }));
    const tauInput = numberInput('tau', { min: TAU_RANGE[0], max: TAU_RANGE[1], step: 0.1 });
    const tauField = field('τ', tauInput, TIPS.tau);
    const boxes = select('boxes', BOX_OPTIONS);
    const anBox = el('div', { class: 'group' }, [
        field('intervals', family, TIPS.family),
        check(calibrate, 'calibrate τ for coverage α', TIPS.calibrate),
        tauField,
        field('boxes on the chart', boxes, TIPS.boxes),
    ]);

    // --- region: band
    const alphaInput = numberInput('alpha', { min: 0.5, max: 0.999, step: 0.01 });
    const alphaInfo = infoIcon(TIPS.alphaBand, 'coverage α');
    const alphaField = field('coverage α', alphaInput, alphaInfo);
    const covariance = select('covariance', COVARIANCE_OPTIONS);
    const covarianceField = field('covariance', covariance, TIPS.covariance);
    const weightsField = field('weights', select('bootWeights', WEIGHT_OPTIONS), TIPS.weights);
    const undersmooth = select('undersmooth', UNDERSMOOTH_OPTIONS, Number);
    let lambdaUsed = null; // the band's λ, so that switching to a given λ starts from it
    const lambdaMode = el('select', {}, LAMBDA_OPTIONS.map(([value, t]) => el('option', { value, text: t })));
    lambdaMode.value = state.lambdaMode;
    lambdaMode.addEventListener('change', () => {
        const patch = { lambdaMode: lambdaMode.value };
        if (lambdaMode.value === 'given' && lambdaUsed !== null) {
            const v = Math.round(Math.log(lambdaUsed) / LOG_LAMBDA_STEP) * LOG_LAMBDA_STEP;
            patch.logLambda = Number(Math.min(LOG_LAMBDA_RANGE[1], Math.max(LOG_LAMBDA_RANGE[0], v)).toFixed(1));
            logLambda.set(patch.logLambda);
        }
        onChange(patch);
    });
    const lambdaInfo = infoIcon(TIPS.lambda, 'λ');
    const logLambda = slider('logLambda', LOG_LAMBDA_RANGE[0], LOG_LAMBDA_RANGE[1], LOG_LAMBDA_STEP,
        v => v.toFixed(1));
    const logLambdaInfo = infoIcon(TIPS.logLambda, 'log λ');
    const logLambdaField = field('log λ', logLambda.row, logLambdaInfo);
    const undersmoothField = field('undersmoothing', undersmooth, TIPS.undersmooth);
    const diff = select('diff', DIFF_OPTIONS, Number);
    const bandBox = el('div', { class: 'group' }, [
        covarianceField,
        weightsField,
        field('λ', lambdaMode, lambdaInfo),
        logLambdaField,
        undersmoothField,
        field('penalty order', diff, TIPS.diff),
    ]);
    const nsegInput = el('input', { type: 'number', min: 3, max: MAX_SEGMENTS, step: 1, placeholder: 'auto' });
    nsegInput.value = state.nseg === null ? '' : String(state.nseg);
    nsegInput.addEventListener('input', () => {
        if (nsegInput.value === '')
            later({ nseg: null });
        else if (nsegInput.checkValidity())
            later({ nseg: Math.round(Number(nsegInput.value)) });
    });
    const nsegInfo = infoIcon(TIPS.nseg, 'segments');

    // --- fit
    const signed = v => (v > 0 ? `+${v.toFixed(1)}` : v < 0 ? `−${(-v).toFixed(1)}` : '0');
    const ambition = slider('ambition', AMBITION_RANGE[0], AMBITION_RANGE[1], AMBITION_STEP, signed);
    const ambitionEnds = el('div', { class: 'slider-ends' }, [el('span', { text: '← detail' }),
        el('span', { text: 'simplicity →' })]);
    const ambitionInfo = infoIcon(TIPS.ambition, 'ambition');
    const penalty = select('penalty', PENALTY_OPTIONS);
    const flat = slider('flat', 0, 1, FLAT_STEP, v => `${Math.round(100 * v)}%`);
    const logKappa = numberInput('logKappa', { min: LOG_KAPPA_RANGE[0], max: LOG_KAPPA_RANGE[1], step: 0.5 });
    const logEpsilon = numberInput('logEpsilon', { min: LOG_EPSILON_RANGE[0], max: LOG_EPSILON_RANGE[1], step: 0.5 });
    const advanced = el('details', {}, [
        el('summary', { text: 'Advanced' }),
        el('div', { class: 'row' }, [field('log₁₀ κ', logKappa, TIPS.kappa),
            field('log₁₀ ε', logEpsilon, TIPS.epsilon)]),
    ]);

    container.append(
        el('h2', { text: 'Data' }),
        field('source', source, TIPS.source),
        simBox,
        userBox,
        heading('Region', TIPS.regionHead),
        field('region', regionKind, TIPS.region),
        alphaField,
        anBox,
        bandBox,
        field('segments', nsegInput, nsegInfo),
        heading('Fit', TIPS.fitHead),
        field('ambition penalty', penalty, TIPS.penalty),
        field('ambition (decades from a*)', el('div', {}, [ambition.row, ambitionEnds]), ambitionInfo),
        field('toward flat', flat.row, TIPS.flat),
        advanced,
    );

    let lastColumns = null;
    function sync(st, { columns = [], nsegAuto = null, lambdaChosen = null, lambdaUsed: used = null,
        ambition: amb = null }) {
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
        if (used !== null)
            lambdaUsed = used;
        const given = st.lambdaMode === 'given';
        logLambdaField.hidden = !given;
        undersmoothField.hidden = given;
        const chosen = lambdaChosen !== null
            ? ` ${st.lambdaMode.toUpperCase()} chose λ = ${fmtExp(lambdaChosen)} (log λ = `
                + `${Math.log(lambdaChosen).toFixed(2)}) for these data.`
            : '';
        setInfo(lambdaInfo, TIPS.lambda + chosen);
        setInfo(logLambdaInfo, `${TIPS.logLambda} Now λ = ${fmtExp(Math.exp(st.logLambda))}.`);
        const an = st.region === 'An';
        anBox.hidden = !an;
        bandBox.hidden = an;
        tauField.hidden = !an || st.calibrate;
        alphaField.hidden = an && !st.calibrate;
        covarianceField.hidden = st.region !== 'band';
        weightsField.hidden = st.region !== 'boot';
        setInfo(alphaInfo, an ? TIPS.alphaAn : st.region === 'boot' ? TIPS.alphaBoot : TIPS.alphaBand);
        nsegInput.placeholder = nsegAuto !== null ? `auto (${nsegAuto})` : 'auto';
        setInfo(nsegInfo, TIPS.nseg + (an ? TIPS.nsegAn : ''));
        if (document.activeElement !== alphaInput)
            alphaInput.value = String(st.alpha);
        let now = '';
        if (amb?.capped) {
            now = amb.sparse
                ? ` The fit never leaves the region: a* is the top of the range (${fmtExp(amb.star)}), where the fit `
                    + 'has no bends or steps, so above 0 nothing changes.'
                : ' The fit never leaves the region: a* is the largest ambition computed reliably '
                    + `(${fmtExp(amb.star)}), so above 0 nothing changes.`;
        }
        else if (amb) {
            now = ` Here a* = ${fmtExp(amb.star)}; now a = ${fmtExp(amb.a)}`
                + (amb.clamped ? (amb.sparse ? ' (the top of the range).' : ' (the largest computed reliably).') : '.');
        }
        setInfo(ambitionInfo, TIPS.ambition + now);
    }

    return { sync };
}
