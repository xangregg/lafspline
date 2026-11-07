// The animated export: the fit built up step by step (the data, the P-spline, its band, the fit inside the band,
// the fit alone), each step the chart with only its layers and a banner saying what the step adds. Written as an
// animated GIF (core/gif.js), which plays almost anywhere, or an animated SVG (CSS keyframes, no script), sharp at
// any size where SVG animates (browsers, GitHub). For Aₙ the band's steps become its boxes: the paper's blocks at a
// fine level, the region's own blocks (merged to the shortest interval), then the fit at its tightest level.
//
// The frames are drawn by a second chart off screen, so the page's chart is not touched; it has the page chart's
// width, so the frames look as the page does. The banner text follows the readouts' claims, shortened: the band
// says what it covers under which assumptions, and a fit that leaves the band says so.

import * as d3 from 'd3';
import { createMainChart } from './chart.js';
import { encodeGif, gifPalette, indexPixels } from '../core/gif.js';
import { lambdaSourceName } from '../core/thumb.js';
import { constraintBoxes } from '../core/region.js';
import { familyLevels } from '../core/intervals.js';
import { LEAVE_TOL } from '../core/fit.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const HEADLINE_SIZE = 1.5; // banner headline font size, in the legend's font sizes
const DETAIL_SIZE = 1.05; // banner detail font size, in the legend's font sizes
const BANNER_LINE = 1.3; // banner line height, in its font sizes
const BANNER_PAD = 10; // px around the banner text
const BANNER_GAP = 8; // px between the banner and the chart's legend
const MAX_HEADLINE_LINES = 2; // longer headlines are cut with an ellipsis
const MAX_DETAIL_LINES = 3;
const STEP_SECONDS = 3.5; // s each step is shown
const LAST_SECONDS = 7; // s the finished fit is shown before the loop starts over
const MAX_FINE_BOXES = 64; // Aₙ's first boxes: the first level of the paper's blocks with at most this many
const FIT_COLOR = 'series-1'; // the phase-2 fit's color (main.js), for Aₙ's ticks
const GIF_SCALE = 2; // GIF pixels per CSS pixel of the chart, as the PNG export
const GIF_COLORS = 256;
const ALL_LAYERS = ['data', 'truth', 'center', 'band', 'fit'];

const fmt = d3.format('.3~g');
const fmtInt = d3.format(',');

/**
 * @typedef {object} StoryStep  one step of the animation, one frame
 * @property {number} number  from 1
 * @property {number} seconds  how long the frame is shown
 * @property {string} headline
 * @property {string} detail
 * @property {string[]} show  the chart layers drawn ('data', 'center', 'band', 'fit')
 * @property {object|null} [boxes]  Aₙ boxes to draw (chart.js BoxesSpec, with its legend label)
 */

/**
 * The frames of the animation for a computed view.
 *
 * @param {{data: object, result: object, phase2: object}} view  as main.js computes it; for Aₙ with result.levels
 *     and phase2.atData set (main.js anLevels)
 * @param {object} st  the settings (for the ambition penalty)
 * @returns {{steps?: StoryStep[], error?: string}}
 */
export function storySteps(view, st) {
    const { data, result, phase2 } = view;
    if (!result.region || !phase2)
        return { error: 'Nothing to animate until there is a fit.' };
    const story = data.story ?? {};
    const xy = data.columns.length >= 2 ? ` of ${data.columns[1]} against ${data.columns[0]}` : '';
    const first = {
        headline: story.intro ?? data.label,
        detail: [story.intro ? data.label : '', `${fmtInt(data.n)} observations${xy}`,
            data.fAt ? `simulated: ${data.notes[0]}` : ''].filter(Boolean).join('; '),
        show: ['data'],
    };
    const middle = result.an ? anSteps(view, st) : bandSteps(view, st);
    const fit = phase2.fit;
    const pieces = piecesPhrase(fit);
    const last = {
        headline: 'The least ambitious fit',
        detail: story.outro ?? `${pieces ? `${capitalize(pieces)}, ` : ''}${fit.leaving > 0 ? 'mostly ' : ''}`
            + `inside ${result.an ? 'Aₙ' : `the ${Math.round(100 * result.region.params.alpha)}% band`}`,
        show: ['data', 'fit'],
    };
    const frames = [first, ...middle, last];
    const steps = frames.map((frame, k) => ({ number: k + 1,
        seconds: k === frames.length - 1 ? LAST_SECONDS : STEP_SECONDS, ...frame }));
    return { steps };
}

function bandSteps(view, st) {
    const { result, phase2 } = view;
    const p = result.region.params;
    const boot = p.covariance === 'bootstrap';
    const pct = `${Math.round(100 * p.alpha)}%`;
    const fit = phase2.fit;
    const G = result.region.grid.length;
    const leaves = fit.leaving > 0
        ? `; it leaves the band by more than 1% of its half-width at ${fmtInt(fit.leaving)} of ${fmtInt(G)} points`
        : '';
    const pieces = piecesPhrase(fit);
    return [
        {
            headline: 'Fit a P-spline',
            detail: `A smooth curve through the data (penalized B-splines, ${lambdaPhrase(p)}; effective dimension `
                + `${fmt(p.ed)})`,
            show: ['data', 'center'],
        },
        {
            headline: `Add a ${pct} confidence band${boot ? ' (bootstrap)' : ''}`,
            detail: bandPhrase(p),
            show: ['data', 'center', 'band'],
        },
        {
            headline: 'Fit the simplest curve inside the band',
            detail: `A second P-spline, penalized severely for leaving the band and for ${ambitionPhrase(st)}`
                + `${pieces ? `: ${pieces}` : ''}${leaves}`,
            show: ['data', 'center', 'band', 'fit'],
        },
    ];
}

/**
 * Aₙ's steps: the paper's blocks at a fine level (the first with at most MAX_FINE_BOXES boxes), before any shortest
 * interval merges them; then the region's own finest level, the blocks merged to span at least the shortest
 * interval (or, with none, every level at once); then the fit with the boxes of its tightest level and their ticks
 * (the fit's mean over each interval). Before the fit, the boxes have no ticks and no violations: nothing is checked
 * yet.
 */
function anSteps(view, st) {
    const { data, result, phase2 } = view;
    const p = result.an.params;
    const c = result.an.scale;
    const share = result.family?.minShare ?? 0;
    const levels = result.levels.filter(level => level.length > 0);
    const boxesOf = level => constraintBoxes(data.y, phase2.atData, level, c);
    const spec = (lvls, extra) => ({ limit: 1 + LEAVE_TOL, fitColorVar: FIT_COLOR, faint: false, ticks: false,
        levels: lvls.map(level => ({ boxes: boxesOf(level), emphasize: -1 })), ...extra });
    // the paper's blocks: the family without the shortest interval (the region's own levels when there is none)
    const raw = share > 0 ? familyLevels(data.x, { ...result.family, minShare: 0 }).filter(level => level.length > 0)
        : levels;
    let fine = raw.findIndex(level => level.length <= MAX_FINE_BOXES);
    if (fine < 0)
        fine = raw.length - 1;
    const skipped = fine > 0 ? `; ${fine === 1 ? 'the finer level' : `the ${fine} finer levels`} (up to `
        + `${fmtInt(raw[0].length)} blocks) ${fine === 1 ? 'is' : 'are'} not drawn` : '';
    const tau = p.calibration ? `τ = ${fmt(p.tau)}, calibrated` : `τ = ${fmt(p.tau)}, given`;
    const steps = [{
        headline: 'Average the data in small blocks',
        detail: `Each box: the mean of y over a block (here ${levelCounts(raw[fine], 'blocks')}) ± c/√(its count), c = `
            + `σ̂·√(τ log n) = ${fmt(c)} from the noise σ̂ = ${fmt(p.sigma)} and ${tau}`
            + `${share > 0 ? '; the paper\'s blocks, before the shortest interval merges them' : ''}${skipped}`,
        show: ['data'],
        boxes: spec([raw[fine]], { label: `Aₙ, ${levelCounts(raw[fine], 'blocks')}${share > 0 ? ', before merging'
            : ''}` }),
    }];
    const count = `${fmtInt(result.intervals.length)} intervals in all`;
    if (share > 0) {
        steps.push({
            headline: `Merge blocks narrower than ${fmt(100 * share)}% of the x range`,
            detail: `Aₙ checks these (${levelCounts(levels[0], 'blocks')}) and their unions at every coarser scale, `
                + `${count}. `
                + capitalize(anPhrase(p, share)),
            show: ['data'],
            boxes: spec([levels[0]], { label: `Aₙ, finest level (${levelCounts(levels[0], 'blocks')})` }),
        });
    }
    else {
        steps.push({
            headline: 'Check every scale at once',
            detail: `Aₙ checks the blocks and their unions at every coarser scale, ${count}. `
                + capitalize(anPhrase(p, share)),
            show: ['data'],
            boxes: spec(levels.slice(fine), { faint: true, label: `Aₙ boxes, ${levels.length - fine} levels drawn` }),
        });
    }
    const fit = phase2.fit;
    const pieces = piecesPhrase(fit);
    const tight = result.intervals[fit.argMax];
    const tightLevel = result.levels[tight.level];
    const emphasize = tightLevel.findIndex(iv => iv.lo === tight.lo && iv.hi === tight.hi);
    const misses = fit.leaving > 0
        ? `; it misses ${fmtInt(fit.leaving)} of ${fmtInt(result.intervals.length)} boxes by more than 1%` : '';
    steps.push({
        headline: 'Fit the simplest curve whose averages lie in every box',
        detail: `A P-spline, penalized severely for leaving Aₙ and for ${ambitionPhrase(st)}${pieces ? `: ${pieces}`
            : ''}${misses}. Drawn: the level where it comes closest to a box's edge (${levelCounts(tightLevel)}), each `
            + 'tick its average over the interval',
        show: ['data', 'fit'],
        boxes: { limit: 1 + LEAVE_TOL, fitColorVar: FIT_COLOR, faint: false,
            levels: [{ boxes: boxesOf(tightLevel), emphasize }],
            label: `Aₙ boxes at the tightest level (${levelCounts(tightLevel)}); ticks: the fit's means` },
    });
    return steps;
}

/** What Aₙ covers, shortened from main.js shortClaim. */
function anPhrase(p, share) {
    const level = p.calibration
        ? `It holds f with probability ≈ ${fmt(p.calibration.alpha)} (τ calibrated)`
        : `τ = ${fmt(p.tau)} is given, so its coverage is not calibrated`;
    const narrow = share > 0 ? `; features narrower than ${fmt(100 * share)}% of the x range are not checked` : '';
    return `${level}, if the noise is independent Gaussian; no smoothness is assumed${narrow}`;
}

/** 'intervals of 24–30 observations' or '25-observation intervals' (main.js levelCounts), or of blocks. */
function levelCounts(level, noun = 'intervals') {
    const counts = level.map(iv => iv.hi - iv.lo + 1);
    const lo = d3.min(counts);
    const hi = d3.max(counts);
    return lo === hi ? `${fmtInt(lo)}-observation ${noun}` : `${noun} of ${fmtInt(lo)}–${fmtInt(hi)} observations`;
}

function lambdaPhrase(p) {
    if (p.lambdaSource === 'scale')
        return `λ for features about ${fmt(100 * p.rule.scale)}% of the x range wide`;
    if (p.lambdaSource === 'given')
        return 'λ given';
    const under = p.undersmooth && p.undersmooth !== 1 ? ` × ${fractionText(p.undersmooth)}` : '';
    return `λ by ${lambdaSourceName(p.lambdaSource)}${under}`;
}

function fractionText(v) {
    return v < 1 ? `1/${fmt(1 / v)}` : fmt(v);
}

/** What the band covers, shortened from main.js shortClaim. */
function bandPhrase(p) {
    if (p.covariance === 'bootstrap') {
        return `It holds the P-spline's mean (the trend, smoothed) at every x with probability ≈ ${p.alpha}, by `
            + 'resampling; the noise need not be Gaussian';
    }
    if (p.covariance === 'sandwich') {
        return `It holds the P-spline's mean (the trend, smoothed) at every x with probability ≈ ${p.alpha}, for `
            + 'independent Gaussian noise';
    }
    return `It holds the trend at every x with probability ≈ ${p.alpha}, if it is smooth and the noise independent `
        + 'Gaussian';
}

/** What the ambition penalty charges, after 'penalized … for'. */
function ambitionPhrase(st) {
    if (st.penalty === 'quadratic')
        return st.flat === 1 ? 'its slope' : 'its curvature';
    if (st.penalty === 'power')
        return 'the sizes of its bends';
    const kinds = [...(st.flat < 1 ? ['bend'] : []), ...(st.flat > 0 ? ['step'] : []),
        ...(st.penalty === 'shaped' && st.flat < 1 ? ['peak', 'wiggle'] : [])];
    return `each ${kinds.length > 1 ? `${kinds.slice(0, -1).join(', ')} and ${kinds.at(-1)}` : kinds[0]}`;
}

/** '3 straight pieces and 2 flat pieces' for a fit with pieces; '' for the others. */
function piecesPhrase(fit) {
    if (!fit.pieces)
        return '';
    const parts = fit.pieces.orders.map(o => `${fmtInt(o.pieces)} ${o.order === 2 ? 'straight' : 'flat'} `
        + `piece${o.pieces === 1 ? '' : 's'}`);
    return parts.join(' and ');
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * The frames as standalone SVG documents of one size: each step's chart, drawn off screen with only its layers,
 * under a banner. The charts are aligned at the bottom, so the plot stays put while the legend above it changes.
 *
 * @param {HTMLElement} pageChart  the page's chart container, for the width and the fonts
 * @param {object} model  the page chart's model (main.js chartModel), whose hidden layers and boxes are replaced
 *     per step
 * @param {StoryStep[]} steps
 * @param {{details?: boolean}} [opts]  details: draw the detail line (default true)
 * @returns {{frames: {svg: SVGSVGElement, text: string}[], width: number, height: number}}
 */
export function storyFrames(pageChart, model, steps, opts = {}) {
    const details = opts.details ?? true;
    const width = pageChart.clientWidth;
    const host = document.createElement('div');
    host.style.cssText = `position: absolute; left: -100000px; top: 0; width: ${width}px;`;
    document.body.append(host);
    try {
        const chart = createMainChart(host);
        const charts = steps.map((step, k) => {
            chart.setBoxes(null);
            chart.render({ ...model, hidden: new Set(ALL_LAYERS.filter(l => !step.show.includes(l))),
                boxesLabel: step.boxes?.label ?? null });
            chart.setBoxes(step.boxes ?? null);
            const svg = chart.toSvg();
            // each frame's clip-path ids made unique, as the animated SVG holds them all
            const text = svg.text.replace(/plot-clip-[a-z0-9]+/g, m => `${m}-f${k}`);
            return { ...svg, node: new DOMParser().parseFromString(text, 'image/svg+xml').documentElement };
        });
        const legend = getComputedStyle(host.querySelector('.legend'));
        const font = legend.fontFamily;
        const base = parseFloat(legend.fontSize);
        const colors = themeColors();
        const ctx = document.createElement('canvas').getContext('2d');
        const measure = (size, weight) => text => {
            ctx.font = `${weight} ${size}px ${font}`;
            return ctx.measureText(text).width;
        };
        const headSize = base * HEADLINE_SIZE;
        const detailSize = base * DETAIL_SIZE;
        const total = steps[steps.length - 1].number;
        const counterText = k => `${steps[k].number}/${total}`;
        const counterWidth = measure(detailSize, 'normal')(`${total}/${total}`);
        const room = width - 2 * BANNER_PAD;
        const wrapped = steps.map(step => ({
            head: wrapLines(step.headline, room - counterWidth - BANNER_PAD, measure(headSize, '600'),
                MAX_HEADLINE_LINES),
            detail: details && step.detail ? wrapLines(step.detail, room, measure(detailSize, 'normal'),
                MAX_DETAIL_LINES) : [],
        }));
        const headLines = Math.max(...wrapped.map(w => w.head.length));
        const detailLines = Math.max(...wrapped.map(w => w.detail.length));
        const bannerHeight = 2 * BANNER_PAD + headLines * headSize * BANNER_LINE
            + detailLines * detailSize * BANNER_LINE;
        const chartHeight = Math.max(...charts.map(c => c.height));
        const height = Math.ceil(bannerHeight + BANNER_GAP + chartHeight);
        const frames = charts.map((c, k) => {
            const svg = svgEl('svg', { xmlns: SVG_NS, width, height, viewBox: `0 0 ${width} ${height}`,
                'font-family': font });
            svg.append(svgEl('rect', { width: '100%', height: '100%', fill: colors.surface }));
            svg.append(banner(wrapped[k], counterText(k), { width, bannerHeight, headSize, detailSize, colors }));
            c.node.setAttribute('x', 0);
            c.node.setAttribute('y', bannerHeight + BANNER_GAP + chartHeight - c.height);
            svg.append(document.importNode(c.node, true));
            return { svg, text: new XMLSerializer().serializeToString(svg) };
        });
        return { frames, width, height };
    }
    finally {
        host.remove();
    }
}

function banner(lines, counter, { width, bannerHeight, headSize, detailSize, colors }) {
    const g = svgEl('g', { class: 'banner' });
    g.append(svgEl('rect', { width, height: bannerHeight, fill: colors.page }));
    g.append(svgEl('line', { x1: 0, x2: width, y1: bannerHeight - 0.5, y2: bannerHeight - 0.5, stroke: colors.grid }));
    // each line centered in its line box
    const headLine = headSize * BANNER_LINE;
    const detailLine = detailSize * BANNER_LINE;
    const at = (text, x, y, attrs) => g.append(svgText(text, { x, y, 'dominant-baseline': 'central', ...attrs }));
    lines.head.forEach((line, i) => {
        at(line, BANNER_PAD, BANNER_PAD + (i + 0.5) * headLine, { 'font-size': headSize, 'font-weight': 600,
            fill: colors.ink });
    });
    at(counter, width - BANNER_PAD, BANNER_PAD + 0.5 * headLine, { 'font-size': detailSize, 'text-anchor': 'end',
        fill: colors.muted });
    const top = BANNER_PAD + lines.head.length * headLine;
    lines.detail.forEach((line, i) => {
        at(line, BANNER_PAD, top + (i + 0.5) * detailLine, { 'font-size': detailSize, fill: colors.ink2 });
    });
    return g;
}

/** Text broken at spaces into at most maxLines lines `room` wide; the last is cut with an ellipsis if need be. */
function wrapLines(text, room, measure, maxLines) {
    const lines = [];
    let line = '';
    for (const word of text.split(' ')) {
        const next = line ? `${line} ${word}` : word;
        if (line && measure(next) > room) {
            lines.push(line);
            line = word;
        }
        else {
            line = next;
        }
    }
    lines.push(line);
    if (lines.length <= maxLines)
        return lines;
    const kept = lines.slice(0, maxLines);
    let last = kept[maxLines - 1];
    while (last && measure(`${last}…`) > room)
        last = last.slice(0, last.lastIndexOf(' ') > 0 ? last.lastIndexOf(' ') : last.length - 1);
    kept[maxLines - 1] = `${last}…`;
    return kept;
}

/**
 * One SVG document that plays the frames in a loop, by CSS keyframes switching each frame's opacity (step-end, so
 * no fades), with no script. A viewer that does not animate SVG shows the last frame, the finished fit.
 *
 * @param {{frames: {svg: SVGSVGElement}[], width: number, height: number}} story  from storyFrames
 * @param {number[]} durations  s per frame
 * @returns {string}
 */
export function animatedSvg(story, durations) {
    const { frames, width, height } = story;
    const total = durations.reduce((a, b) => a + b, 0);
    const out = svgEl('svg', { xmlns: SVG_NS, width, height, viewBox: `0 0 ${width} ${height}` });
    const pct = t => `${(100 * t / total).toFixed(4)}%`;
    let css = `.frame { opacity: 0; animation: ${total}s step-end infinite; }\n.frame.last { opacity: 1; }\n`;
    let t = 0;
    frames.forEach((frame, k) => {
        const from = t;
        t += durations[k];
        const stops = [];
        stops.push(`0% { opacity: ${k === 0 ? 1 : 0}; }`);
        if (k > 0)
            stops.push(`${pct(from)} { opacity: 1; }`);
        if (k < frames.length - 1)
            stops.push(`${pct(t)} { opacity: 0; }`);
        stops.push(`100% { opacity: ${k === frames.length - 1 ? 1 : 0}; }`);
        css += `@keyframes frame${k} { ${stops.join(' ')} }\n.f${k} { animation-name: frame${k}; }\n`;
    });
    const style = svgEl('style');
    style.textContent = css;
    out.append(style);
    frames.forEach((frame, k) => {
        const g = svgEl('g', { class: `frame f${k}${k === frames.length - 1 ? ' last' : ''}` });
        for (const child of [...frame.svg.childNodes])
            g.append(child.cloneNode(true));
        for (const attr of ['font-family'])
            g.setAttribute(attr, frame.svg.getAttribute(attr));
        out.append(g);
    });
    return new XMLSerializer().serializeToString(out);
}

/**
 * The frames as an animated GIF, looping, at GIF_SCALE pixels per CSS pixel over one palette.
 *
 * @param {{frames: {text: string}[], width: number, height: number}} story  from storyFrames
 * @param {number[]} durations  s per frame
 * @returns {Promise<{blob: Blob, width: number, height: number, colors: number}>}
 */
export async function animatedGif(story, durations) {
    const w = Math.round(story.width * GIF_SCALE);
    const h = Math.round(story.height * GIF_SCALE);
    const pixels = [];
    for (const frame of story.frames)
        pixels.push(await rasterPixels(frame.text, w, h));
    const palette = gifPalette(pixels, GIF_COLORS);
    const cache = new Map();
    const frames = pixels.map((rgba, k) => ({ indices: indexPixels(rgba, palette, cache), delay: durations[k] }));
    const bytes = encodeGif({ width: w, height: h, palette, frames });
    return { blob: new Blob([bytes], { type: 'image/gif' }), width: w, height: h, colors: palette.length / 3 };
}

async function rasterPixels(svgText, w, h) {
    const url = URL.createObjectURL(new Blob([svgText], { type: 'image/svg+xml' }));
    try {
        const img = new Image();
        img.src = url;
        await img.decode();
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, w, h);
        return ctx.getImageData(0, 0, w, h).data;
    }
    finally {
        URL.revokeObjectURL(url);
    }
}

function themeColors() {
    const css = getComputedStyle(document.documentElement);
    const v = name => css.getPropertyValue(name).trim();
    return { surface: v('--surface'), page: v('--page'), ink: v('--ink'), ink2: v('--ink-2'), muted: v('--muted'),
        grid: v('--grid') };
}

function svgEl(tag, attrs = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs))
        node.setAttribute(k, v);
    return node;
}

function svgText(text, attrs) {
    const node = svgEl('text', attrs);
    node.textContent = text;
    return node;
}
