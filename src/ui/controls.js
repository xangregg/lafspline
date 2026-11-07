// Control panel: data source (examples, simulated signals, pasted or dropped CSV), the phase-1 region's settings and
// the phase-2 fit's knobs, in the order they act. Explanations sit behind ⓘ icons (info.js) to keep the panel
// compact; text that depends on the data (the λ GCV chose, a*) is updated in them by sync. Patterned on the controls
// of the project the copied modules came from (VENDORED.md; el, field, debounce are the same helpers). Emits partial
// state patches.

import { SIGNALS } from '../core/signals.js';
import { LOG_LAMBDA_RANGE } from '../core/pspline.js';
import { lambdaSourceName, MIN_POINTS_PER_PERIOD } from '../core/thumb.js';
import {
    MAX_N, MAX_SEGMENTS, LOG_KAPPA_RANGE, LOG_EPSILON_RANGE, AN_MAX_SEGMENTS, TAU_RANGE,
    BEND_LENGTH_RANGE, SHAPE_COST_RANGE, POWER_RANGE, FEATURE_SCALE_START,
} from './urlstate.js';
import { fmtExp } from './format.js';
import { infoIcon, setInfo } from './info.js';

const DEBOUNCE = 300; // ms for typed input
const USER = 'user';
const AMBITION_STEP = 0.1; // decades
const FLAT_STEP = 0.05;
const BEND_LENGTH_STEP = 0.005; // fraction of the x range
const POWER_STEP = 0.05; // the power penalty's q
const LOG_LAMBDA_STEP = 0.1; // natural log units on the given-λ slider
const FEATURE_SCALE_STEP = 0.01; // share of the x range on the feature-scale slider
const AN_SHARE_STEP = 0.01; // share of the x range on Aₙ's shortest-interval slider
const FEATURE_SCALE_MIN = 0.02; // the slider's low end: finer than a few knot spacings the basis cannot follow
// The sliders span less than a link may hold (urlstate.js AMBITION_RANGE, FEATURE_SCALE_RANGE, AN_SHARE_RANGE), for
// finer steps where the useful values are; a value beyond the span shows its thumb at the nearer end, and its own
// value beside it.
const AMBITION_SLIDER = [-6, 1]; // decades of a/a*
const FEATURE_SCALE_MAX = 0.4; // share of the x range
const AN_SHARE_MAX = 0.3; // share of the x range
const UNDERSMOOTH_OPTIONS = [
    [1, 'none: λ as chosen'],
    [1 / 4, 'λ × 1/4 (default)'],
    [1 / 16, 'λ × 1/16'],
    [1 / 64, 'λ × 1/64'],
];
const LAMBDA_OPTIONS = [
    ['gcv', 'chosen by GCV'],
    ['aic', 'chosen by AIC'],
    ['silverman', 'Silverman\'s rule (x spread)'],
    ['rot', 'Fan & Gijbels\' rule of thumb'],
    ['scale', 'by feature scale'],
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
    ['shaped', 'shaped bends'],
    ['power', '|second differences|^q'],
];

const TIPS = {
    source: 'Where the data come from:\n'
        + '- **Examples**: bundled data sets.\n'
        + '- **Local data**: data sets kept on this machine (localdata/), listed only when served locally.\n'
        + '- **Simulated**: signals with Gaussian noise added.\n'
        + '- **Your data**: CSV pasted or dropped.',
    n: 'Number of observations, at x = i/n.',
    sigma: 'Standard deviation of the Gaussian noise added to the signal.',
    seed: 'Seed of the simulated noise: the same seed gives the same data.',
    xCol: 'The column for x; (row number) uses the order of the rows.',
    yCol: 'The column for y.',
    regionHead: 'Phase 1: a confidence region for f, from the data alone. The fit (phase 2) must stay inside it.',
    region: 'The region the fit must stay inside:\n'
        + '- **P-spline band**: a simultaneous band around a P-spline fit, which covers f if f is smooth.\n'
        + '- **Bootstrap band**: around the same fit, from refits with random fractional weights on the observations, '
        + 'so it can be asymmetric and wider where the noise is; it covers the smoothed f.\n'
        + '- **Aₙ** (Davies, Kovac & Meise 2009): bounds on the sums of the residuals over intervals of the '
        + 'observations, assuming nothing about the smoothness of f.',
    alphaBand: 'The probability that the band contains f at every point of the check grid at once (the data x and '
        + '4 points per segment), by Monte Carlo with B = 1000.',
    alphaBoot: 'The probability that the band contains the smoothed f at every point of the check grid at once (the '
        + 'data x and 4 points per segment), by the bootstrap with B = 2000 replicates (1000 above 1000 '
        + 'observations).',
    weights: 'Every observation gets a positive weight in each replicate, and the weights sum to n.\n'
        + '- **Like resampled counts**: the same mean, variance and skewness (about a fifth of the weights are below '
        + '0.1, as a count is 0 with probability 0.37); it held its coverage best in simulations.\n'
        + '- **Dirichlet**: the Bayesian bootstrap of Rubin (1981), or fractional-random-weight bootstrap; its weights '
        + 'are twice as skewed as counts. With skewed noise its band\'s asymmetry came nearer the truth, but its '
        + 'coverage was lower.',
    alphaAn: 'τ is calibrated by Monte Carlo so that Aₙ contains f with this probability.',
    family: 'The intervals of observations Aₙ checks:\n'
        + '- **Blocks**: the paper\'s, of lengths 1, 2, 4, … in fixed positions.\n'
        + '- **Overlapping**: windows of each length shifted by ¼ of the length or by one observation, which catch '
        + 'runs of residuals that straddle block edges.\n'
        + '- **All**: every interval (at most 300 distinct x).\n'
        + 'Denser families need a larger τ for the same coverage.',
    anShare: 'Every interval Aₙ checks spans at least this share of the x range (an interval spans from midway to '
        + 'the previous x to midway to the next, so where x is sparse a lone observation can be enough). Of the '
        + 'paper\'s blocks, those that span enough stay as they are; runs of shorter blocks are merged with each other '
        + 'into parts about equally wide, and a run too short even then is widened over its neighbors. So Aₙ answers '
        + 'only to features at least this wide, much as the band\'s feature scale does. 5% by default; "all" keeps '
        + 'every interval as it is, the paper\'s family.\n'
        + 'With fewer intervals a calibrated τ is smaller, so the wide intervals are tighter. In simulations, 5% held '
        + 'f about as often as every interval did under Gaussian noise (92.8–99.3%) and more often under skewed or '
        + 'uneven noise (16–56% at n = 500, against 0–2.5%), where single observations fall outside the Gaussian '
        + 'tail.',
    calibrate: 'Choose τ by simulating Aₙ\'s multiscale statistic on pure noise for this n, design and intervals, '
        + 'so that the coverage is α. Otherwise τ is given.',
    tau: 'Aₙ\'s threshold is c = σ̂·√(τ log n). The paper\'s default is 3; its coverage then depends on n and the '
        + 'intervals.',
    boxes: 'Each box spans an interval of observations, at the mean of y there ± c/√|I|; its tick is the fit\'s mean '
        + 'over the interval, which must lie inside.\n'
        + '- **Tightest**: the level of the interval with the largest ratio.\n'
        + '- **All levels**: every interval, as outlines.\n'
        + '- **None**: no boxes.',
    covariance: 'The covariance the band is built from:\n'
        + '- **Bayesian**: σ̂²(BᵀB + λDᵀD)⁻¹, which widens the band for the smoothing bias on average, so it covers '
        + 'a smooth f.\n'
        + '- **Sandwich**: the frequentist covariance, which covers only the smoother\'s mean E f̂.',
    lambda: 'The smoothing parameter of the band\'s P-spline: the weight on the squared differences of its '
        + 'coefficients. A larger λ gives a smoother fit and a narrower band.\n'
        + '- **GCV**: chosen by generalized cross-validation.\n'
        + '- **AIC**: chosen by Akaike\'s information criterion.\n'
        + '- **Silverman\'s rule**: the normal-reference kernel width 0.9·min(sd, IQR/1.34)·n^(−1/5) of x, from the '
        + 'spread of x and n alone; it knows nothing of y.\n'
        + '- **Fan & Gijbels\' rule of thumb**: the kernel width that would be best for local linear regression if '
        + 'f were the quartic fitted to the data, from that quartic\'s residual σ and curvature.\n'
        + '- **By feature scale**: the λ at which a sine whose period is the given share of the x range keeps half '
        + 'its amplitude: features longer than that are followed, shorter ones smoothed away. From x alone, so a '
        + 'scale means the same on every data set, except that a period must span at least '
        + `${MIN_POINTS_PER_PERIOD} observations; no undersmoothing.\n`
        + '- **Given**: set with the slider.\n'
        + 'The rules give a kernel width, and λ is the one whose P-spline has the same effective dimension as a '
        + 'Gaussian local linear smoother of that width. They are here for comparison: the bands\' coverage was '
        + 'simulated with GCV.',
    logLambda: `Natural log of λ, over the range GCV searches (${LOG_LAMBDA_RANGE[0]} to ${LOG_LAMBDA_RANGE[1]}).`,
    featureScale: 'The shortest features of interest, as a share of the x range: λ is set so that a sine of this '
        + 'period keeps half its amplitude in the band\'s P-spline. Smaller follows more detail, with a wider band. '
        + `It is raised to span at least ${MIN_POINTS_PER_PERIOD} observations (${MIN_POINTS_PER_PERIOD}/n of the `
        + 'range), since shorter waves are noise to the P-spline. Below about three knot spacings (7.5% with 40 '
        + 'segments) the basis cannot follow it.',
    undersmooth: 'Multiplies the chosen λ. A smaller λ follows bends more closely, so the band covers them too, at '
        + 'the cost of a wider band; with 1/4 or 1/16 the P-spline band contained smooth test signals in 94–98.5% of '
        + 'simulations.',
    diff: 'Order of the coefficient differences in the band\'s penalty: a large λ pulls the P-spline toward a '
        + 'polynomial of degree order − 1.\n'
        + '- **1**: toward a constant.\n'
        + '- **2**: toward a straight line.\n'
        + '- **3**: toward a quadratic.',
    nseg: 'Equal segments of the cubic B-spline basis, for the band and the fit. Empty: automatic, about half the '
        + 'number of distinct x, at most 40.',
    nsegAn: ` For Aₙ at most ${AN_MAX_SEGMENTS}: its long intervals make each solve dense.`,
    fitHead: 'Phase 2: a second P-spline, penalized severely for leaving the region, slightly for distance from its '
        + 'center, and for ambition: pressure toward straight (or flat) lines.',
    penalty: 'What ambition penalizes:\n'
        + '- **Squared differences**: the fit gets straighter (or flatter) everywhere, smoothly.\n'
        + '- **Number of bends or steps**: every bend (or step) costs the same, whatever its size or direction, so '
        + 'the cheapest turn is one sharp corner: straight (or flat) pieces joined by short bends (or steps). By the '
        + 'adaptive ridge, which approximates the count.\n'
        + '- **Shaped bends**: the same count, plus three costs on the shape of the fit, each with its knob below. '
        + 'A bend pays for its sharpness, so a large turn (the top of a wave) spreads into a rounded bend while a '
        + 'small one stays short (bend length); each local maximum or minimum costs extra (per extremum), and so does '
        + 'each change of bending direction (per inflection), so the fit rises and levels off where the region lets '
        + 'it. With all three at 0 it is the number of bends.\n'
        + '- **|Second differences|^q**: their sizes to a power q between 0 and 2, toward straight (and toward flat '
        + 'with the first differences\' share).',
    q: 'The power on the size of each second difference:\n'
        + '- **2**: the squared differences.\n'
        + '- **1**: ℓ₁ trend filtering, straight pieces joined by kinks.\n'
        + '- **Toward 0**: the number of bends.\n'
        + 'A smaller q concentrates the turning into fewer, sharper bends; a larger one spreads it out. Below 1 the '
        + 'fit is a local optimum. With a share toward flat, the first differences get the same power: flat pieces '
        + 'and steps.',
    bendLength: 'How round shaped bends are: a bend curves so that it leaves its tangent line by one half-width of '
        + 'the region (its median) over this fraction of the x range. 0 gives the sharp bends of the plain count. A '
        + 'bend\'s width grows with how far it turns, and the region limits it. Rounding\'s cost levels off for a '
        + 'turn the region forces to be sharp, such as a spike narrower than this length, so such a turn does not '
        + 'dominate the fit.',
    extremumCost: 'Extra cost of each local maximum or minimum, in plain bends. The fit tries stretches forced '
        + 'monotone and keeps one when the whole cost falls, so a fit that can rise and level off inside the region '
        + 'does so instead of peaking and falling back. 0: extremes cost nothing extra.',
    inflectionCost: 'Extra cost of each change of bending direction (an inflection), in plain bends: discourages '
        + 'wiggles even where the fit keeps rising. The fit tries stretches kept bending one way, as for extremes.',
    ambition: 'Decades from a*, the largest ambition before the fit leaves the region by more than 1% of its '
        + 'half-width. 0 is the simplest fit inside; below it the fit follows more detail, above it the fit '
        + 'leaves the region.',
    flat: 'Share of the ambition on first differences (toward flat) rather than second (toward straight). With '
        + 'shaped bends, the steps are counted plainly and their corners are not rounded; the bends of longer '
        + 'slopes are.',
    kappa: 'Weight on the squared distance outside the region, per squared half-width: larger makes the region '
        + 'harder.',
    epsilon: 'Weight on the squared distance from the region\'s center: a slight pull inside the region, which also '
        + 'makes the fit unique.',
    displayHead: 'What the chart draws. A hidden layer still sets the axes, keeps its values in the tooltip and its '
        + 'readouts; nothing is refitted.',
    showTruth: 'The signal the data were simulated from.',
    showData: 'The observations.',
    showCenter: 'The P-spline the band is built around (for the P-spline band, its center).',
    showBand: 'The phase-1 confidence band the fit must stay inside.',
    showFit: 'The least ambitious fit inside the region (phase 2).',
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
 * A select of the data sources: the example data sets, the local ones, the simulated signals and, unless
 * opts.user is false, the user's pasted data. Values are 'example:id', 'local:id', a SIGNALS key or 'user'.
 *
 * @param {{id: string, label: string}[]} examples
 * @param {{id: string, label: string}[]} local
 * @param {{user?: boolean}} [opts]
 * @returns {HTMLSelectElement}
 */
export function sourceSelect(examples, local, opts = {}) {
    const source = el('select');
    if (examples.length) {
        source.append(el('optgroup', { label: 'Examples' },
            examples.map(e => el('option', { value: `example:${e.id}`, text: e.label }))));
    }
    if (local.length) {
        source.append(el('optgroup', { label: 'Local data (not published)' },
            local.map(e => el('option', { value: `local:${e.id}`, text: e.label }))));
    }
    const sim = el('optgroup', { label: 'Simulated' });
    for (const [key, spec] of Object.entries(SIGNALS))
        sim.append(el('option', { value: key, text: spec.label }));
    source.append(sim);
    if (opts.user ?? true)
        source.append(el('option', { value: USER, text: 'Your data (paste or drop CSV)' }));
    return source;
}

/**
 * Build the controls.
 *
 * @param {HTMLElement} container
 * @param {object} state    initial state (see main.js)
 * @param {function(object): void} onChange  receives a partial state patch
 * @param {Array<{id: string, label: string}>} [examples]  bundled data sets (data/index.json)
 * @param {Array<{id: string, label: string}>} [local]  data sets in localdata/ (this machine only)
 * @returns {{sync: function(object, {columns: string[], nsegAuto: number|null, lambdaChosen: number|null,
 *     lambdaUsed: number|null, scaleRaised: {scale: number, minPoints: number, n: number}|null,
 *     ambition: {a: number, star: number, capped: boolean, clamped: boolean,
 *     sparse: boolean}|null}): void}}  sync refreshes the parts that depend on state or data; lambdaChosen is the
 *     λ GCV or AIC chose, lambdaUsed the band's λ (after undersmoothing), scaleRaised the feature scale used when
 *     the n-based floor raised it (scaleLambda)
 */
export function createControls(container, state, onChange, examples = [], local = []) {
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
    const source = sourceSelect(examples, local);
    source.id = 'source';
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
    const sizeRow = el('div', { class: 'row' }, [field('n', nInput, TIPS.n), field('noise σ', sigmaInput, TIPS.sigma)]);
    const simBox = el('div', { class: 'group' }, [
        sizeRow,
        el('div', { class: 'row' }, [field('seed', seedInput, TIPS.seed), newSeed]),
    ]);
    const sourceField = field('source', source, TIPS.source);
    const galleryNote = el('p', { class: 'field-hint gallery-note',
        text: 'Each pane chooses its data set above its chart. Simulated signals use their own n and σ and this '
            + 'seed; the other settings apply to every pane.' });

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
    const anShare = slider('anShare', 0, AN_SHARE_MAX, AN_SHARE_STEP,
        v => (v > 0 ? `${Math.round(100 * v)}%` : 'all'));
    const calibrateField = check(calibrate, 'calibrate τ for coverage α', TIPS.calibrate);
    const anBox = el('div', { class: 'group' }, [
        field('intervals', family, TIPS.family),
        field('shortest interval', anShare.row, TIPS.anShare),
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
        // a fixed start, not the scale the current λ amounts to, so the same setting means the same on every data set
        if (lambdaMode.value === 'scale') {
            patch.featureScale = FEATURE_SCALE_START;
            featureScale.set(patch.featureScale);
        }
        onChange(patch);
    });
    const lambdaInfo = infoIcon(TIPS.lambda, 'λ');
    const logLambda = slider('logLambda', LOG_LAMBDA_RANGE[0], LOG_LAMBDA_RANGE[1], LOG_LAMBDA_STEP,
        v => v.toFixed(1));
    const logLambdaInfo = infoIcon(TIPS.logLambda, 'log λ');
    const logLambdaField = field('log λ', logLambda.row, logLambdaInfo);
    const undersmoothField = field('undersmoothing', undersmooth, TIPS.undersmooth);
    const featureScale = slider('featureScale', FEATURE_SCALE_MIN, FEATURE_SCALE_MAX, FEATURE_SCALE_STEP,
        v => `${Math.round(100 * v)}%`);
    const featureScaleHint = el('span', { class: 'field-hint' });
    const featureScaleField = field('feature scale', featureScale.row, TIPS.featureScale);
    featureScaleField.append(featureScaleHint);
    const diff = select('diff', DIFF_OPTIONS, Number);
    const bandBox = el('div', { class: 'group' }, [
        covarianceField,
        weightsField,
        field('λ', lambdaMode, lambdaInfo),
        logLambdaField,
        featureScaleField,
        undersmoothField,
    ]);
    const diffField = field('penalty order', diff, TIPS.diff);
    const nsegInput = el('input', { type: 'number', min: 3, max: MAX_SEGMENTS, step: 1, placeholder: 'auto' });
    nsegInput.value = state.nseg === null ? '' : String(state.nseg);
    nsegInput.addEventListener('input', () => {
        if (nsegInput.value === '')
            later({ nseg: null });
        else if (nsegInput.checkValidity())
            later({ nseg: Math.round(Number(nsegInput.value)) });
    });
    const nsegInfo = infoIcon(TIPS.nseg, 'segments');
    // Settings rarely changed from their defaults, as in the Fit section's Advanced; each still shows only for the
    // regions it applies to.
    const regionAdvanced = el('details', {}, [
        el('summary', { text: 'Advanced' }),
        weightsField,
        diffField,
        field('segments', nsegInput, nsegInfo),
        calibrateField,
        tauField,
    ]);

    // --- fit
    const signed = v => (v > 0 ? `+${v.toFixed(1)}` : v < 0 ? `−${(-v).toFixed(1)}` : '0');
    const ambition = slider('ambition', AMBITION_SLIDER[0], AMBITION_SLIDER[1], AMBITION_STEP, signed);
    const ambitionEnds = el('div', { class: 'slider-ends' }, [el('span', { text: '← detail' }),
        el('span', { text: 'simplicity →' })]);
    const ambitionInfo = infoIcon(TIPS.ambition, 'ambition');
    const penalty = select('penalty', PENALTY_OPTIONS);
    const flat = slider('flat', 0, 1, FLAT_STEP, v => `${Math.round(100 * v)}%`);
    const flatField = field('toward flat', flat.row, TIPS.flat);
    const bendLength = slider('bendLength', BEND_LENGTH_RANGE[0], BEND_LENGTH_RANGE[1], BEND_LENGTH_STEP,
        v => v.toFixed(3));
    const bendLengthField = field('bend length', bendLength.row, TIPS.bendLength);
    const power = slider('q', POWER_RANGE[0], POWER_RANGE[1], POWER_STEP, v => v.toFixed(2));
    const powerField = field('power q', power.row, TIPS.q);
    const costAttrs = { min: SHAPE_COST_RANGE[0], max: SHAPE_COST_RANGE[1], step: 0.5 };
    const shapeCosts = el('div', { class: 'row' }, [
        field('per extremum', numberInput('extremumCost', costAttrs), TIPS.extremumCost),
        field('per inflection', numberInput('inflectionCost', costAttrs), TIPS.inflectionCost),
    ]);
    const logKappa = numberInput('logKappa', { min: LOG_KAPPA_RANGE[0], max: LOG_KAPPA_RANGE[1], step: 0.5 });
    const logEpsilon = numberInput('logEpsilon', { min: LOG_EPSILON_RANGE[0], max: LOG_EPSILON_RANGE[1], step: 0.5 });
    // Shaped bends' knobs are rarely changed from their defaults, so they sit here rather than beside the penalty.
    const advanced = el('details', {}, [
        el('summary', { text: 'Advanced' }),
        bendLengthField,
        shapeCosts,
        el('div', { class: 'row' }, [field('log₁₀ κ', logKappa, TIPS.kappa),
            field('log₁₀ ε', logEpsilon, TIPS.epsilon)]),
    ]);

    // --- display: chart layers only (main.js redraws without refitting)
    const toggle = (key, labelText) => {
        const input = el('input', { type: 'checkbox', checked: state[key] });
        input.addEventListener('change', () => onChange({ [key]: input.checked }));
        return check(input, labelText, TIPS[key]);
    };
    const truthToggle = toggle('showTruth', 'true f');
    const centerToggle = toggle('showCenter', 'phase-1 fit');
    const bandToggle = toggle('showBand', 'confidence band');
    const boxesField = field('Aₙ boxes', select('boxes', BOX_OPTIONS), TIPS.boxes);

    container.append(
        el('h2', { text: 'Data' }),
        sourceField,
        galleryNote,
        simBox,
        userBox,
        heading('Region', TIPS.regionHead),
        field('region', regionKind, TIPS.region),
        alphaField,
        anBox,
        bandBox,
        regionAdvanced,
        heading('Fit', TIPS.fitHead),
        field('ambition penalty', penalty, TIPS.penalty),
        field('ambition (decades from a*)', el('div', {}, [ambition.row, ambitionEnds]), ambitionInfo),
        flatField,
        powerField,
        advanced,
        heading('Display', TIPS.displayHead),
        toggle('showData', 'data points'),
        truthToggle,
        centerToggle,
        bandToggle,
        boxesField,
        toggle('showFit', 'phase-2 fit'),
    );

    let lastColumns = null;
    function sync(st, { columns = [], nsegAuto = null, lambdaChosen = null, lambdaUsed: used = null,
        scaleRaised = null, ambition: amb = null }) {
        // the gallery's panes choose their own data; the seed and true f apply to its simulated ones
        const gallery = st.view === 'gallery';
        sourceField.hidden = gallery;
        galleryNote.hidden = !gallery;
        simBox.hidden = !gallery && !SIGNALS[st.source];
        sizeRow.hidden = gallery;
        truthToggle.hidden = !gallery && !SIGNALS[st.source];
        userBox.hidden = gallery || st.source !== USER;
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
        const byScale = st.lambdaMode === 'scale';
        logLambdaField.hidden = !given;
        featureScaleField.hidden = !byScale;
        // the band uses a larger scale than the slider's when the data have too few points for it
        featureScaleHint.hidden = !scaleRaised;
        if (scaleRaised) {
            featureScaleHint.textContent = `Raised to ${+(100 * scaleRaised.scale).toPrecision(3)}% for these data: `
                + `a period spans at least ${scaleRaised.minPoints} of the ${scaleRaised.n} observations.`;
        }
        undersmoothField.hidden = given || byScale;
        const chosen = lambdaChosen !== null
            ? `\n${lambdaSourceName(st.lambdaMode)} chose λ = ${fmtExp(lambdaChosen)} (log λ = `
                + `${Math.log(lambdaChosen).toFixed(2)}) for these data.`
            : '';
        setInfo(lambdaInfo, TIPS.lambda + chosen);
        setInfo(logLambdaInfo, `${TIPS.logLambda} Now λ = ${fmtExp(Math.exp(st.logLambda))}.`);
        const an = st.region === 'An';
        anBox.hidden = !an;
        bandBox.hidden = an;
        diffField.hidden = an;
        calibrateField.hidden = !an;
        // Aₙ has no band or phase-1 fit to draw, only its boxes
        centerToggle.hidden = an;
        bandToggle.hidden = an;
        boxesField.hidden = !an;
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
        powerField.hidden = st.penalty !== 'power';
        bendLengthField.hidden = st.penalty !== 'shaped';
        shapeCosts.hidden = st.penalty !== 'shaped';
    }

    return { sync };
}
