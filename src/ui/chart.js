// Main chart: data points (canvas, so n in the thousands stays fast), true signal and fit lines plus
// axes (SVG on top), a crosshair tooltip, and an optional highlighted interval driven by the residual
// view. Several fits can be drawn; the primary one is thicker and its extremes are marked.

import * as d3 from 'd3';
import { MARGIN, xScale, cellIndex, sizeCanvas } from './layout.js';
import { themeColors, cssColor } from './theme.js';
import { showTooltip, hideTooltip } from './tooltip.js';

const HEIGHT = 340; // px
const POINT_ALPHA = 0.55;
const MARK_ROW = 9; // px above the x axis for extreme-plateau marks
const MARK_SIZE = 42; // d3.symbol area, px²
const MARK_INSET = 2; // px trimmed from each end of a plateau mark so adjacent plateaus stay distinct
const MAX_MARKS = 100; // above this many extremes the marks are an unreadable band; the count is in the legend
const PRIMARY_WIDTH = 2.5; // px
const OVERLAY_WIDTH = 1.5; // px
// A spacing more than GAP_FACTOR × the median spacing is a gap: a step fit has no supported value there,
// so instead of a jump at the cell boundary (the midpoint) it gets a dotted connector between the data.
const GAP_FACTOR = 3;
const LONE_TICK = 3; // px half-width of the tick for a single point stranded between gaps
const KINK_RADIUS = 4; // px
const fmt = d3.format('.4~g');
// Below this magnitude tick labels get no thousands separator, so years read 1880, not 1,880 (style guides
// also leave four-digit numbers ungrouped). Larger values keep d3's default grouping.
const UNGROUPED_BELOW = 1e4;

// d3's default tick format is ',f' with a precision fitted to the tick step; 'f' is the same without grouping.
function tickFormat(scale, count) {
    const [a, b] = scale.domain();
    return Math.max(Math.abs(a), Math.abs(b)) < UNGROUPED_BELOW
        ? scale.tickFormat(count, 'f')
        : scale.tickFormat(count);
}

/**
 * @typedef {object} ChartModel
 * @property {Float64Array} x       sorted x
 * @property {Float64Array} y
 * @property {Float64Array|null} f  true signal at x (simulated data only)
 * @property {ChartFit[]} fits      fits to draw, primary last
 * @property {{type: 'max'|'min', start: number, end: number}[]} extremes  local extremes of the primary
 * @property {Float64Array} edges   cell edges (layout.cellEdges)
 * @property {{lb: Float64Array, ub: Float64Array, label: string, x?: Float64Array}|null} band  confidence band; at
 *     the data x, or at its own sorted abscissae x (which must include every data x; the band then also sets the
 *     y scale)
 * @property {{lb: Float64Array, ub: Float64Array, label: string}|null} [exact]  exact LP bounds, dashed
 * @property {string|null} [boxesLabel]  legend text for the Aₙ boxes layer (null: no legend item)
 * @property {{points: Array<{x: number, y: number}>, label: string, plural: string}} [markers]  features of the
 *     primary fit (kinks, inflections) marked with circles at (x, y), named in the legend and tooltip
 */

/**
 * @typedef {object} BoxesSpec  Aₙ constraint boxes to draw (see region.constraintBoxes)
 * @property {Array<{boxes: import('../core/region.js').ConstraintBoxes, emphasize: number}>} levels
 * @property {boolean} faint     outlines only, no fit ticks (for many levels at once)
 * @property {number} limit      ratio above which a box is violated
 * @property {string} fitColorVar  CSS custom property for the fit-mean ticks
 */

/**
 * @typedef {object} ChartFit
 * @property {string} key
 * @property {string} label
 * @property {Float64Array} g        fit at x (or at the fit's own x)
 * @property {Float64Array} [x]      the fit's own sorted abscissae (a smooth curve on a grid that includes every data
 *                                   x); drawn as a line, not as steps
 * @property {boolean} step          draw as a step function over the cells (piecewise-constant fits)
 * @property {{start: number, end: number}[]} runs  runs of equal g (for step drawing)
 * @property {string} colorVar       CSS custom property holding the color
 * @property {boolean} primary
 * @property {number[]} [breakAfter]  observation indices i after which the fit jumps (no line from i to i + 1;
 *                                    a dotted connector shows the gap instead)
 */

/**
 * @param {HTMLElement} container
 * @returns {{render: function(ChartModel): void, highlight: function({lo: number, hi: number}|null): void}}
 */
export function createMainChart(container) {
    container.classList.add('chart');
    const legend = d3.select(container).append('div').attr('class', 'legend');
    const plot = d3.select(container).append('div').attr('class', 'plot').style('height', `${HEIGHT}px`);
    const canvas = plot.append('canvas').node();
    const boxCanvas = plot.append('canvas').node(); // Aₙ boxes: above the points, below the lines
    const svg = plot.append('svg').attr('height', HEIGHT);
    const clipId = `plot-clip-${Math.random().toString(36).slice(2)}`;
    const clipRect = svg.append('clipPath').attr('id', clipId).append('rect');
    const gGrid = svg.append('g').attr('class', 'grid');
    const gBand = svg.append('g').attr('class', 'band').attr('clip-path', `url(#${clipId})`);
    const gHighlight = svg.append('rect').attr('class', 'interval-highlight').attr('visibility', 'hidden');
    const gGaps = svg.append('g').attr('class', 'gap-connectors').attr('clip-path', `url(#${clipId})`);
    const gLines = svg.append('g').attr('clip-path', `url(#${clipId})`);
    const gKinks = svg.append('g').attr('class', 'kink-marks').attr('clip-path', `url(#${clipId})`);
    const gMarks = svg.append('g').attr('class', 'extreme-marks');
    const gAxisX = svg.append('g').attr('class', 'axis');
    const gAxisY = svg.append('g').attr('class', 'axis');
    const cross = svg.append('g').attr('class', 'crosshair').attr('visibility', 'hidden');
    cross.append('line').attr('y1', MARGIN.top).attr('y2', HEIGHT - MARGIN.bottom);
    const crossDot = cross.append('circle').attr('r', 4);
    const overlay = svg.append('rect').attr('class', 'overlay')
        .attr('y', MARGIN.top).attr('height', HEIGHT - MARGIN.top - MARGIN.bottom);

    let model = null;
    let sx = null;
    let sy = null;
    let width = 0;
    let boxSpec = null;

    // Each box spans its interval's cells horizontally and mean_I(y) ± c/√|I| vertically; the tick is
    // the fit's mean over the interval. A tick outside its box is a violated constraint.
    function drawBoxes() {
        const ctx = sizeCanvas(boxCanvas, width, HEIGHT);
        ctx.clearRect(0, 0, width, HEIGHT);
        if (!boxSpec || !model)
            return;
        const colors = themeColors();
        const tickColor = cssColor(boxSpec.fitColorVar);
        const edges = model.edges;
        ctx.save();
        ctx.beginPath();
        ctx.rect(MARGIN.left, MARGIN.top, width - MARGIN.left - MARGIN.right, HEIGHT - MARGIN.top - MARGIN.bottom);
        ctx.clip();
        for (const { boxes, emphasize } of boxSpec.levels) {
            for (let k = 0; k < boxes.lo.length; k++) {
                const x0 = sx(edges[boxes.lo[k]]);
                const x1 = sx(edges[boxes.hi[k] + 1]);
                const top = sy(boxes.yMean[k] + boxes.half[k]);
                const bottom = sy(boxes.yMean[k] - boxes.half[k]);
                const violated = boxes.ratios[k] > boxSpec.limit;
                const w = Math.max(1, x1 - x0);
                ctx.globalAlpha = violated ? 0.18 : boxSpec.faint ? 0 : 0.07;
                ctx.fillStyle = violated ? colors.violation : colors.ink;
                ctx.fillRect(x0, top, w, bottom - top);
                ctx.globalAlpha = violated ? 0.9 : boxSpec.faint ? 0.12 : 0.55;
                ctx.strokeStyle = violated ? colors.violation : colors['ink-2'];
                ctx.lineWidth = k === emphasize ? 2 : 1;
                ctx.strokeRect(x0 + 0.5, top + 0.5, Math.max(0, w - 1), bottom - top);
                if (!boxSpec.faint) {
                    ctx.globalAlpha = 1;
                    ctx.strokeStyle = violated ? colors.violation : tickColor;
                    ctx.lineWidth = k === emphasize ? 3 : 2;
                    const yv = sy(boxes.gMean[k]);
                    ctx.beginPath();
                    ctx.moveTo(x0, yv);
                    ctx.lineTo(x0 + w, yv);
                    ctx.stroke();
                }
            }
        }
        ctx.restore();
    }

    /**
     * Show or clear Aₙ constraint boxes without redrawing the rest of the chart (cheap on hover).
     *
     * @param {BoxesSpec|null} spec
     */
    function setBoxes(spec) {
        boxSpec = spec;
        drawBoxes();
    }

    function render(m) {
        model = m;
        // Anything showing values from the previous model is now stale.
        cross.attr('visibility', 'hidden');
        gHighlight.attr('visibility', 'hidden');
        hideTooltip();
        width = container.clientWidth;
        const colors = themeColors();
        const { x, y, f, edges } = m;
        const n = x.length;
        sx = xScale(edges, width);
        let lo = d3.min(y);
        let hi = d3.max(y);
        if (f) {
            lo = Math.min(lo, d3.min(f));
            hi = Math.max(hi, d3.max(f));
        }
        // fits can leave the data's range (e.g. a smooth fit extrapolating below 0); never clip them away
        for (const fit of m.fits) {
            lo = Math.min(lo, d3.min(fit.g));
            hi = Math.max(hi, d3.max(fit.g));
        }
        if (m.band?.x) {
            for (const v of [m.band.lb, m.band.ub]) {
                lo = Math.min(lo, d3.min(v.filter(Number.isFinite)));
                hi = Math.max(hi, d3.max(v.filter(Number.isFinite)));
            }
        }
        sy = d3.scaleLinear().domain([lo, hi]).nice().range([HEIGHT - MARGIN.bottom, MARGIN.top]);

        svg.attr('width', width);
        clipRect.attr('x', MARGIN.left).attr('y', MARGIN.top)
            .attr('width', width - MARGIN.left - MARGIN.right).attr('height', HEIGHT - MARGIN.top - MARGIN.bottom);
        overlay.attr('x', MARGIN.left).attr('width', width - MARGIN.left - MARGIN.right);
        gAxisX.attr('transform', `translate(0,${HEIGHT - MARGIN.bottom})`)
            .call(d3.axisBottom(sx).ticks(width / 90).tickFormat(tickFormat(sx, width / 90)).tickSizeOuter(0));
        gAxisY.attr('transform', `translate(${MARGIN.left},0)`)
            .call(d3.axisLeft(sy).ticks(HEIGHT / 50).tickFormat(tickFormat(sy, HEIGHT / 50)).tickSizeOuter(0));
        gGrid.selectAll('line').data(sy.ticks(HEIGHT / 50)).join('line')
            .attr('x1', MARGIN.left).attr('x2', width - MARGIN.right)
            .attr('y1', d => sy(d)).attr('y2', d => sy(d));

        const ctx = sizeCanvas(canvas, width, HEIGHT);
        ctx.clearRect(0, 0, width, HEIGHT);
        ctx.fillStyle = colors.muted;
        ctx.globalAlpha = POINT_ALPHA;
        const r = n > 2000 ? 1.5 : n > 500 ? 2 : 2.5; // px; smaller points when dense
        for (let i = 0; i < n; i++) {
            ctx.beginPath();
            ctx.arc(sx(x[i]), sy(y[i]), r, 0, 2 * Math.PI);
            ctx.fill();
        }
        ctx.globalAlpha = 1;

        const line = v => d3.line().x((_, i) => sx(x[i])).y((_, i) => sy(v[i]))(v);

        // Band: infinite bounds are drawn to just past the plot edge (the tooltip shows ±∞). Where
        // lb > ub the band is empty: the assumption is inconsistent with the data there.
        gBand.selectAll('*').remove();
        if (m.band) {
            const [y0, y1] = sy.range(); // bottom, top in px
            const clampPx = v => Math.max(y1 - 10, Math.min(y0 + 10, sy(v)));
            const px = v => (v === Infinity ? y1 - 10 : v === -Infinity ? y0 + 10 : clampPx(v));
            const { lb, ub } = m.band;
            const bx = m.band.x ?? x;
            const ok = i => lb[i] <= ub[i];
            gBand.append('path').attr('class', 'band-fill')
                .attr('d', d3.area().defined((_, i) => ok(i)).x((_, i) => sx(bx[i]))
                    .y0((_, i) => px(lb[i])).y1((_, i) => px(ub[i]))(lb));
            gBand.append('path').attr('class', 'band-empty')
                .attr('d', d3.area().defined((_, i) => !ok(i)).x((_, i) => sx(bx[i]))
                    .y0((_, i) => px(ub[i])).y1((_, i) => px(lb[i]))(lb));
            for (const v of [lb, ub]) {
                gBand.append('path').attr('class', 'band-edge')
                    .attr('d', d3.line().defined(i => Number.isFinite(v[i])).x(i => sx(bx[i])).y(i => px(v[i]))(d3.range(bx.length)));
            }
            if (m.exact) {
                for (const v of [m.exact.lb, m.exact.ub]) {
                    gBand.append('path').attr('class', 'band-exact')
                        .attr('d', d3.line().defined(i => Number.isFinite(v[i])).x(i => sx(x[i])).y(i => px(v[i]))(d3.range(n)));
                }
            }
        }
        // gapAfter[i]: the spacing x[i] → x[i+1] is a gap (see GAP_FACTOR)
        const spacings = [];
        for (let i = 1; i < n; i++) {
            if (x[i] > x[i - 1])
                spacings.push(x[i] - x[i - 1]);
        }
        spacings.sort((a, b) => a - b);
        const typical = spacings.length ? spacings[Math.floor(spacings.length / 2)] : 0;
        const gapAfter = new Uint8Array(Math.max(0, n - 1));
        let anyGap = false;
        for (let i = 0; i + 1 < n; i++) {
            if (typical > 0 && x[i + 1] - x[i] > GAP_FACTOR * typical) {
                gapAfter[i] = 1;
                anyGap = true;
            }
        }

        // A piecewise-constant fit is drawn over each observation's cell, so plateaus show their extent,
        // except across gaps, where the solid line stops at the data and a dotted connector bridges it.
        const steps = fit => {
            let d = '';
            let dots = '';
            let prev = null;
            const segment = (a, b, yv) => {
                // a lone point between gaps would have zero width; give it a short tick
                if (b - a < 1e-9)
                    return `M${a - LONE_TICK},${yv}H${b + LONE_TICK}`;
                return `M${a},${yv}H${b}`;
            };
            for (const r of fit.runs) {
                const yv = sy(fit.g[r.start]);
                const gapBefore = r.start > 0 && gapAfter[r.start - 1];
                let a = gapBefore ? sx(x[r.start]) : sx(edges[r.start]);
                if (prev) {
                    if (gapBefore)
                        dots += `M${prev.x},${prev.y}L${a},${yv}`;
                    else
                        d += `M${prev.x},${prev.y}V${yv}`; // jump at the shared cell edge
                }
                for (let i = r.start; i < r.end; i++) {
                    if (gapAfter[i]) {
                        const left = sx(x[i]);
                        const right = sx(x[i + 1]);
                        d += segment(a, left, yv);
                        dots += `M${left},${yv}L${right},${yv}`;
                        a = right;
                    }
                }
                const gapAfterRun = r.end < n - 1 && gapAfter[r.end];
                const b = gapAfterRun ? sx(x[r.end]) : sx(edges[r.end + 1]);
                d += segment(a, b, yv);
                prev = { x: b, y: yv };
            }
            return { d, dots };
        };
        const series = [];
        if (f)
            series.push({ key: 'truth', label: 'true f', d: line(f), color: colors['series-2'], width: 1.5 });
        const gapPaths = [];
        for (const fit of m.fits) {
            const color = cssColor(fit.colorVar);
            let d;
            if (fit.x) {
                d = d3.line().x((_, j) => sx(fit.x[j])).y(v => sy(v))(fit.g);
            }
            else if (fit.step) {
                const st = steps(fit);
                d = st.d;
                if (st.dots)
                    gapPaths.push({ key: fit.key, d: st.dots, color });
            }
            else if (fit.breakAfter?.length) {
                // A break is a jump somewhere between two x: cut the line there and connect with dots.
                const cut = new Set(fit.breakAfter);
                const pts = [];
                for (let i = 0; i < n; i++) {
                    pts.push([x[i], fit.g[i]]);
                    if (cut.has(i))
                        pts.push(null);
                }
                d = d3.line().defined(p => p !== null).x(p => sx(p[0])).y(p => sy(p[1]))(pts);
                const dots = fit.breakAfter.map(i =>
                    `M${sx(x[i])},${sy(fit.g[i])}L${sx(x[i + 1])},${sy(fit.g[i + 1])}`).join('');
                gapPaths.push({ key: `${fit.key}-breaks`, d: dots, color, isBreak: true });
            }
            else {
                d = line(fit.g);
            }
            series.push({ key: fit.key, label: fit.label, d, color, width: fit.primary ? PRIMARY_WIDTH : OVERLAY_WIDTH });
        }
        gGaps.selectAll('path').data(gapPaths, g => g.key).join('path')
            .attr('class', 'gap-connector')
            .attr('stroke', g => g.color)
            .attr('d', g => g.d);
        gLines.selectAll('path').data(series, d => d.key).join('path')
            .attr('fill', 'none')
            .attr('stroke', d => d.color)
            .attr('stroke-width', d => d.width)
            .attr('stroke-linejoin', 'round')
            .attr('d', d => d.d);

        const primaryFit = m.fits.find(fit => fit.primary);
        const markers = primaryFit && m.markers ? m.markers.points : [];
        gKinks.selectAll('circle').data(markers).join('circle')
            .attr('r', KINK_RADIUS)
            .attr('cx', p => sx(p.x))
            .attr('cy', p => sy(p.y))
            .attr('fill', colors.surface)
            .attr('stroke', primaryFit ? cssColor(primaryFit.colorVar) : colors.ink)
            .attr('stroke-width', 1.5);

        const markY = HEIGHT - MARGIN.bottom - MARK_ROW;
        const shown = m.extremes.length <= MAX_MARKS ? m.extremes : [];
        const marks = gMarks.selectAll('g.mark').data(shown).join(enter => {
            const e = enter.append('g').attr('class', 'mark');
            e.append('line');
            e.append('path');
            return e;
        });
        // A plateau's extent stops at the data on the side of a gap, like the step line.
        const span = d => {
            const a = d.start > 0 && gapAfter[d.start - 1] ? x[d.start] : edges[d.start];
            const b = d.end < n - 1 && gapAfter[d.end] ? x[d.end] : edges[d.end + 1];
            return [sx(a), sx(b)];
        };
        marks.select('line')
            .attr('x1', d => {
                const [a, b] = span(d);
                return Math.min(a + MARK_INSET, (a + b) / 2);
            })
            .attr('x2', d => {
                const [a, b] = span(d);
                return Math.max(b - MARK_INSET, (a + b) / 2);
            })
            .attr('y1', markY).attr('y2', markY)
            .attr('stroke', colors.ink);
        marks.select('path')
            .attr('d', d3.symbol(d3.symbolTriangle, MARK_SIZE)())
            .attr('transform', d => {
                const [a, b] = span(d);
                return `translate(${(a + b) / 2},${markY})${d.type === 'min' ? ' rotate(180)' : ''}`;
            })
            .attr('fill', colors.ink);

        const items = [{ label: 'data', color: colors.muted, dot: true }, ...series];
        if (m.band)
            items.push({ label: m.band.label, color: colors['band'], box: true });
        if (m.band && m.exact)
            items.push({ label: m.exact.label, color: colors.ink, glyph: '┄' });
        if (gapPaths.some(g => !g.isBreak) && anyGap)
            items.push({ label: 'gap in x: step fit undefined between the data', color: colors['ink-2'], glyph: '┈' });
        if (gapPaths.some(g => g.isBreak))
            items.push({ label: 'break: the fit jumps somewhere in this gap', color: colors['ink-2'], glyph: '┈' });
        if (markers.length) {
            items.push({ label: `${markers.length === 1 ? m.markers.label : m.markers.plural} (${markers.length})`,
                color: primaryFit ? cssColor(primaryFit.colorVar) : colors.ink, glyph: '○' });
        }
        if (m.boxesLabel)
            items.push({ label: m.boxesLabel, color: colors['ink-2'], glyph: '▭' });
        if (shown.length)
            items.push({ label: 'local max / min plateau', color: colors.ink, glyph: '▲▼' });
        else if (m.extremes.length)
            items.push({ label: `${m.extremes.length} local extremes (marks hidden)`, color: colors.ink, glyph: '▲▼' });
        legend.selectAll('span.item').data(items, d => d.label).join(enter => {
            const s = enter.append('span').attr('class', 'item');
            s.append('span').attr('class', 'key');
            s.append('span').attr('class', 'label');
            return s;
        })
            .call(s => s.select('.key')
                .classed('box', d => Boolean(d.box))
                .classed('dot', d => Boolean(d.dot))
                .classed('glyph', d => Boolean(d.glyph))
                .text(d => d.glyph ?? '')
                .style('background', d => (d.glyph ? 'none' : d.color))
                .style('color', d => d.color))
            .call(s => s.select('.label').text(d => d.label));
        drawBoxes();
    }

    function highlight(interval) {
        if (!interval || !model) {
            gHighlight.attr('visibility', 'hidden');
            return;
        }
        const x0 = sx(model.edges[interval.lo]);
        const x1 = sx(model.edges[interval.hi + 1]);
        gHighlight.attr('visibility', 'visible')
            .attr('x', x0).attr('width', Math.max(1, x1 - x0))
            .attr('y', MARGIN.top).attr('height', HEIGHT - MARGIN.top - MARGIN.bottom);
    }

    // Index into a series given at its own abscissae xs (which include every data x) for observation i; with no
    // xs the series is at the data x.
    function atX(xs, i) {
        return xs ? d3.bisectLeft(xs, model.x[i]) : i;
    }

    overlay.on('pointermove', event => {
        if (!model)
            return;
        const [px] = d3.pointer(event);
        const i = cellIndex(model.edges, sx.invert(px));
        const cx = sx(model.x[i]);
        cross.attr('visibility', 'visible');
        cross.select('line').attr('x1', cx).attr('x2', cx);
        crossDot.attr('cx', cx).attr('cy', sy(model.y[i]));
        const colors = themeColors();
        const rows = [[fmt(model.y[i]), 'y', colors.muted]];
        if (model.f)
            rows.push([fmt(model.f[i]), 'true f', colors['series-2']]);
        for (const fit of [...model.fits].reverse())
            rows.push([fmt(fit.g[atX(fit.x, i)]), fit.label, cssColor(fit.colorVar)]);
        if (model.band) {
            const b = model.band;
            const k = atX(b.x, i);
            const show = v => (v === Infinity ? '+∞' : v === -Infinity ? '−∞' : fmt(v));
            rows.push([`[${show(b.lb[k])}, ${show(b.ub[k])}]`, b.lb[k] > b.ub[k] ? 'band (empty here)' : 'band']);
            if (model.exact)
                rows.push([`[${show(model.exact.lb[i])}, ${show(model.exact.ub[i])}]`, 'exact LP bounds']);
        }
        if (boxSpec && boxSpec.levels.length === 1) {
            const b = boxSpec.levels[0].boxes;
            const k = d3.bisectRight(b.lo, i) - 1;
            if (k >= 0 && i <= b.hi[k]) {
                const len = b.hi[k] - b.lo[k] + 1;
                rows.push([`${fmt(b.gMean[k])} in [${fmt(b.yMean[k] - b.half[k])}, ${fmt(b.yMean[k] + b.half[k])}]`,
                    `fit mean over its Aₙ box (#${b.lo[k] + 1}–${b.hi[k] + 1}, |I| = ${len})`]);
            }
        }
        const ext = model.extremes.find(e => e.start <= i && i <= e.end);
        const notes = [];
        if (ext) {
            notes.push(`In a local ${ext.type === 'max' ? 'maximum' : 'minimum'} plateau: #${ext.start + 1}–${ext.end + 1}, `
                + `x ∈ [${fmt(model.x[ext.start])}, ${fmt(model.x[ext.end])}]`);
        }
        for (const p of model.markers?.points ?? []) {
            if (cellIndex(model.edges, p.x) === i)
                notes.push(`A ${model.markers.label} of the fit at x = ${fmt(p.x)}.`);
        }
        const primaryBreaks = model.fits.find(fit => fit.primary)?.breakAfter ?? [];
        for (const b of primaryBreaks) {
            if (b === i || b + 1 === i)
                notes.push(`The fit jumps somewhere between x = ${fmt(model.x[b])} and ${fmt(model.x[b + 1])} (a break).`);
        }
        const note = notes.length ? notes.join(' ') : undefined;
        showTooltip(event.pageX, event.pageY, `x = ${fmt(model.x[i])}  (#${i + 1})`, rows, note);
    });
    overlay.on('pointerleave', () => {
        cross.attr('visibility', 'hidden');
        hideTooltip();
    });

    return { render, highlight, setBoxes };
}
