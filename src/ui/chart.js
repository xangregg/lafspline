// Main chart: data points (canvas, so n in the thousands stays fast), true signal and fit lines plus
// axes (SVG on top), a crosshair tooltip, and an optional highlighted interval driven by the residual
// view. Several fits can be drawn; the primary one is thicker and its extremes are marked.

import * as d3 from 'd3';
import { MARGIN, xScale, cellIndex, sizeCanvas } from './layout.js';
import { themeColors, cssColor } from './theme.js';
import { showTooltip, hideTooltip } from './tooltip.js';

const HEIGHT = 340; // px
const COMPACT_HEIGHT = 215; // px, a gallery pane's plot
const COMPACT_MARGIN = { top: 6, right: 6, bottom: 6, left: 6 }; // px: no axes
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
const SVG_NS = 'http://www.w3.org/2000/svg';
// Image export (toSvg): the legend's layout, which the page leaves to CSS flexbox
const LEGEND_KEY_GAP = 5; // px between a legend key and its label
const LEGEND_ITEM_GAP = 14; // px between legend items
const LEGEND_LINE = 1.45; // legend row height, in font sizes
const TITLE_GAP = 6; // px below the title
// computed styles copied onto the exported SVG's elements, so it needs no style sheet
const PAINT_PROPS = ['fill', 'stroke'];
const STYLE_PROPS = ['stroke-width', 'stroke-dasharray', 'opacity', 'font-size', 'font-family', 'font-weight'];
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
 * @property {Set<string>} [hidden]  layers not drawn: 'data', 'truth' (f), 'band', or a fit's key. They leave the
 *     legend but still set the scales (so toggling one does not move the others) and keep their rows in the tooltip.
 */

/**
 * @typedef {object} BoxesSpec  Aₙ constraint boxes to draw (see region.constraintBoxes)
 * @property {Array<{boxes: import('../core/region.js').ConstraintBoxes, emphasize: number}>} levels
 * @property {boolean} faint     outlines only, no fit ticks (for many levels at once)
 * @property {boolean} [ticks]   false: the boxes alone, as Aₙ is before any fit, with no ticks and no violations
 *                               marked (default: ticks unless faint)
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
 * @property {boolean} [afterBand]   its legend item follows the band's, so the band's item does not move when this
 *                                   fit is shown or hidden (the phase-2 fit, added after the band)
 * @property {number[]} [breakAfter]  observation indices i after which the fit jumps (no line from i to i + 1;
 *                                    a dotted connector shows the gap instead)
 */

/**
 * @param {HTMLElement} container
 * @param {{compact?: boolean, height?: number}} [opts]  compact: a gallery pane, shorter, with no axes, grid lines or
 *     legend of its own (legendItems gives the items for drawLegend); height: the plot's height in px, for a frame of
 *     a given size (the animated export), instead of the default
 * @returns {{render: function(ChartModel): void, highlight: function({lo: number, hi: number}|null): void,
 *     setBoxes: function(BoxesSpec|null): void, toSvg: function({title?: string}=): ({text: string, width: number,
 *     height: number}|null), legendItems: function(): object[]}}
 */
export function createMainChart(container, opts = {}) {
    const compact = opts.compact ?? false;
    const H = opts.height ?? (compact ? COMPACT_HEIGHT : HEIGHT);
    const M = compact ? COMPACT_MARGIN : MARGIN;
    container.classList.add('chart');
    if (compact)
        container.classList.add('compact');
    // a compact chart's legend is drawn once for the gallery (drawLegend), not above each chart
    const legend = d3.select(container).append('div').attr('class', 'legend').attr('hidden', compact ? '' : null);
    const plot = d3.select(container).append('div').attr('class', 'plot').style('height', `${H}px`);
    // grid lines in an SVG layer of their own under the canvases, so they are behind the points and boxes
    const gridSvg = plot.append('svg').attr('class', 'grid-layer').attr('height', H);
    const canvas = plot.append('canvas').node();
    const boxCanvas = plot.append('canvas').node(); // Aₙ boxes: above the points, below the lines
    const svg = plot.append('svg').attr('height', H);
    const clipId = `plot-clip-${Math.random().toString(36).slice(2)}`;
    const clipRect = svg.append('clipPath').attr('id', clipId).append('rect');
    const gGrid = gridSvg.append('g').attr('class', 'grid');
    const gBand = svg.append('g').attr('class', 'band').attr('clip-path', `url(#${clipId})`);
    const gHighlight = svg.append('rect').attr('class', 'interval-highlight').attr('visibility', 'hidden');
    const gGaps = svg.append('g').attr('class', 'gap-connectors').attr('clip-path', `url(#${clipId})`);
    const gLines = svg.append('g').attr('clip-path', `url(#${clipId})`);
    const gKinks = svg.append('g').attr('class', 'kink-marks').attr('clip-path', `url(#${clipId})`);
    const gMarks = svg.append('g').attr('class', 'extreme-marks');
    const gAxisX = svg.append('g').attr('class', 'axis');
    const gAxisY = svg.append('g').attr('class', 'axis');
    const cross = svg.append('g').attr('class', 'crosshair').attr('visibility', 'hidden');
    cross.append('line').attr('y1', M.top).attr('y2', H - M.bottom);
    const crossDot = cross.append('circle').attr('r', 4);
    const overlay = svg.append('rect').attr('class', 'overlay')
        .attr('y', M.top).attr('height', H - M.top - M.bottom);

    let model = null;
    let sx = null;
    let sy = null;
    let width = 0;
    let boxSpec = null;
    let legendItems = [];

    // Each box spans its interval's cells horizontally and mean_I(y) ± c/√|I| vertically; the tick is
    // the fit's mean over the interval. A tick outside its box is a violated constraint. The shapes are listed
    // once and drawn on the canvas or written into an exported SVG.
    function boxShapes() {
        if (!boxSpec || !model)
            return [];
        const colors = themeColors();
        const tickColor = cssColor(boxSpec.fitColorVar);
        const checked = boxSpec.ticks !== false;
        const edges = model.edges;
        const shapes = [];
        for (const { boxes, emphasize } of boxSpec.levels) {
            for (let k = 0; k < boxes.lo.length; k++) {
                const x0 = sx(edges[boxes.lo[k]]);
                const x1 = sx(edges[boxes.hi[k] + 1]);
                const top = sy(boxes.yMean[k] + boxes.half[k]);
                const bottom = sy(boxes.yMean[k] - boxes.half[k]);
                const violated = checked && boxes.ratios[k] > boxSpec.limit;
                const w = Math.max(1, x1 - x0);
                shapes.push({
                    x: x0, y: top, w, h: bottom - top,
                    fill: violated ? colors.violation : colors.ink,
                    fillAlpha: violated ? 0.18 : boxSpec.faint ? 0 : 0.07,
                    stroke: violated ? colors.violation : colors['ink-2'],
                    strokeAlpha: violated ? 0.9 : boxSpec.faint ? 0.12 : 0.55,
                    lineWidth: k === emphasize ? 2 : 1,
                    tick: boxSpec.faint || !checked ? null : {
                        y: sy(boxes.gMean[k]),
                        color: violated ? colors.violation : tickColor,
                        width: k === emphasize ? 3 : 2,
                    },
                });
            }
        }
        return shapes;
    }

    function drawBoxes() {
        const ctx = sizeCanvas(boxCanvas, width, H);
        ctx.clearRect(0, 0, width, H);
        const shapes = boxShapes();
        if (!shapes.length)
            return;
        ctx.save();
        ctx.beginPath();
        ctx.rect(M.left, M.top, width - M.left - M.right, H - M.top - M.bottom);
        ctx.clip();
        for (const b of shapes) {
            ctx.globalAlpha = b.fillAlpha;
            ctx.fillStyle = b.fill;
            ctx.fillRect(b.x, b.y, b.w, b.h);
            ctx.globalAlpha = b.strokeAlpha;
            ctx.strokeStyle = b.stroke;
            ctx.lineWidth = b.lineWidth;
            ctx.strokeRect(b.x + 0.5, b.y + 0.5, Math.max(0, b.w - 1), b.h);
            if (b.tick) {
                ctx.globalAlpha = 1;
                ctx.strokeStyle = b.tick.color;
                ctx.lineWidth = b.tick.width;
                ctx.beginPath();
                ctx.moveTo(b.x, b.tick.y);
                ctx.lineTo(b.x + b.w, b.tick.y);
                ctx.stroke();
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
        const hidden = m.hidden ?? new Set();
        sx = xScale(edges, width, M);
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
        sy = d3.scaleLinear().domain([lo, hi]).nice().range([H - M.bottom, M.top]);

        svg.attr('width', width);
        gridSvg.attr('width', width);
        // a chart narrower than its margins (for a moment while a pane opens) has no plot area, not a negative one
        const plotWidth = Math.max(0, width - M.left - M.right);
        clipRect.attr('x', M.left).attr('y', M.top)
            .attr('width', plotWidth).attr('height', H - M.top - M.bottom);
        overlay.attr('x', M.left).attr('width', plotWidth);
        // a compact chart has no axes or grid lines: its values are in the tooltip
        if (!compact) {
            gAxisX.attr('transform', `translate(0,${H - M.bottom})`)
                .call(d3.axisBottom(sx).ticks(width / 90).tickFormat(tickFormat(sx, width / 90)).tickSizeOuter(0));
            gAxisY.attr('transform', `translate(${M.left},0)`)
                .call(d3.axisLeft(sy).ticks(H / 50).tickFormat(tickFormat(sy, H / 50)).tickSizeOuter(0));
            gGrid.selectAll('line').data(sy.ticks(H / 50)).join('line')
                .attr('x1', M.left).attr('x2', width - M.right)
                .attr('y1', d => sy(d)).attr('y2', d => sy(d));
        }

        const ctx = sizeCanvas(canvas, width, H);
        ctx.clearRect(0, 0, width, H);
        ctx.fillStyle = colors.muted;
        ctx.globalAlpha = POINT_ALPHA;
        const r = pointRadius(n);
        for (let i = 0; i < (hidden.has('data') ? 0 : n); i++) {
            ctx.beginPath();
            ctx.arc(sx(x[i]), sy(y[i]), r, 0, 2 * Math.PI);
            ctx.fill();
        }
        ctx.globalAlpha = 1;

        const line = v => d3.line().x((_, i) => sx(x[i])).y((_, i) => sy(v[i]))(v);

        // Band: infinite bounds are drawn to just past the plot edge (the tooltip shows ±∞). Where
        // lb > ub the band is empty: the assumption is inconsistent with the data there.
        gBand.selectAll('*').remove();
        if (m.band && !hidden.has('band')) {
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
        if (f && !hidden.has('truth'))
            series.push({ key: 'truth', label: 'true f', d: line(f), color: colors['series-2'], width: 1.5 });
        const gapPaths = [];
        for (const fit of m.fits) {
            if (hidden.has(fit.key))
                continue;
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
            series.push({ key: fit.key, label: fit.label, d, color, width: fit.primary ? PRIMARY_WIDTH : OVERLAY_WIDTH,
                afterBand: fit.afterBand ?? false });
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

        const markY = H - M.bottom - MARK_ROW;
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

        const items = [...(hidden.has('data') ? [] : [{ label: 'data', color: colors.muted, dot: true }]),
            ...series.filter(s => !s.afterBand)];
        const bandShown = m.band && !hidden.has('band');
        if (bandShown)
            items.push({ label: m.band.label, color: colors['band'], box: true });
        if (bandShown && m.exact)
            items.push({ label: m.exact.label, color: colors.ink, glyph: '┄' });
        items.push(...series.filter(s => s.afterBand));
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
        drawLegend(legend.node(), items);
        legendItems = items;
        drawBoxes();
    }

    /**
     * The chart as a standalone SVG document, as drawn now: a title, the legend, the points and Aₙ boxes (canvas
     * layers on the page) as vector shapes, and the SVG layers with their computed styles written in, so it needs
     * neither the page's style sheet nor its fonts' CSS. The crosshair and hover highlight are left out.
     *
     * @param {{title?: string}} [opts]
     * @returns {{text: string, width: number, height: number}|null}  null before the first render
     */
    function toSvg(opts = {}) {
        if (!model)
            return null;
        const colors = themeColors();
        const legendStyle = getComputedStyle(legend.node());
        const fontSize = parseFloat(legendStyle.fontSize);
        const font = legendStyle.fontFamily;
        const ctx = document.createElement('canvas').getContext('2d');
        const textWidth = (text, weight = 'normal') => {
            ctx.font = `${weight} ${fontSize}px ${font}`;
            return ctx.measureText(text).width;
        };
        const out = svgEl('svg', { xmlns: SVG_NS, width, 'font-family': font, 'font-size': fontSize });
        out.append(svgEl('rect', { width: '100%', height: '100%', fill: colors.surface }));
        const lineHeight = fontSize * LEGEND_LINE;
        let top = 0;
        if (opts.title) {
            out.append(svgText(opts.title, { x: 0, y: fontSize * 1.1, 'font-size': fontSize * 1.1,
                'font-weight': 600, ...paint('fill', colors.ink) }));
            top = fontSize * 1.1 * LEGEND_LINE + TITLE_GAP;
        }
        // legend rows, wrapped at the chart's width as the page's flexbox does; a label too long for a row of its
        // own wraps at spaces onto further rows
        let lx = 0;
        let row = 0;
        for (const item of legendItems) {
            const keyWidth = item.glyph ? textWidth(item.glyph) : item.box ? 14 : item.dot ? 8 : 16;
            const room = width - keyWidth - LEGEND_KEY_GAP;
            const lines = wrapWords(item.label, room, textWidth);
            const itemWidth = keyWidth + LEGEND_KEY_GAP + Math.max(...lines.map(l => textWidth(l)));
            if (lx > 0 && (lx + itemWidth > width || lines.length > 1)) {
                lx = 0;
                row++;
            }
            const mid = top + row * lineHeight + lineHeight / 2;
            const ink = paint('fill', item.color);
            if (item.glyph)
                out.append(svgText(item.glyph, { x: lx, y: mid, 'dominant-baseline': 'central', ...ink }));
            else if (item.box)
                out.append(svgEl('rect', { x: lx, y: mid - 5, width: 14, height: 10, rx: 2, ...ink }));
            else if (item.dot)
                out.append(svgEl('circle', { cx: lx + 4, cy: mid, r: 4, ...ink }));
            else
                out.append(svgEl('rect', { x: lx, y: mid - 1, width: 16, height: 2, rx: 1, ...ink }));
            lines.forEach((line, k) => {
                out.append(svgText(line, { x: lx + keyWidth + LEGEND_KEY_GAP, y: mid + k * lineHeight,
                    'dominant-baseline': 'central', ...paint('fill', legendStyle.color) }));
            });
            row += lines.length - 1;
            lx = lines.length > 1 ? width : lx + itemWidth + LEGEND_ITEM_GAP;
        }
        top += legendItems.length ? (row + 1) * lineHeight + parseFloat(legendStyle.marginBottom) : 0;
        const height = Math.ceil(top + H);
        out.setAttribute('height', height);
        out.setAttribute('viewBox', `0 0 ${width} ${height}`);

        const plotG = svgEl('g', { transform: `translate(0,${top})` });
        out.append(plotG);
        const grid = gGrid.node().cloneNode(true);
        inlineStyles(gGrid.node(), grid);
        plotG.append(grid);
        const hidden = model.hidden ?? new Set();
        if (!hidden.has('data')) {
            const points = svgEl('g', paint('fill', colors.muted, POINT_ALPHA));
            const r = pointRadius(model.x.length);
            for (let i = 0; i < model.x.length; i++)
                points.append(svgEl('circle', { cx: round2(sx(model.x[i])), cy: round2(sy(model.y[i])), r }));
            plotG.append(points);
        }
        const shapes = boxShapes();
        if (shapes.length) {
            const boxG = svgEl('g', { 'clip-path': `url(#${clipId})` });
            for (const b of shapes) {
                boxG.append(svgEl('rect', { x: b.x + 0.5, y: b.y + 0.5, width: Math.max(0, b.w - 1), height: b.h,
                    ...paint('fill', b.fill, b.fillAlpha), ...paint('stroke', b.stroke, b.strokeAlpha),
                    'stroke-width': b.lineWidth }));
                if (b.tick) {
                    boxG.append(svgEl('line', { x1: b.x, x2: b.x + b.w, y1: b.tick.y, y2: b.tick.y,
                        ...paint('stroke', b.tick.color), 'stroke-width': b.tick.width }));
                }
            }
            plotG.append(boxG);
        }
        const skip = new Set([gHighlight.node(), cross.node(), overlay.node()]);
        for (const node of svg.node().children) {
            if (skip.has(node))
                continue;
            const copy = node.cloneNode(true);
            inlineStyles(node, copy);
            // clip-path definitions go first, as on the page
            if (node.tagName === 'clipPath')
                plotG.prepend(copy);
            else
                plotG.append(copy);
        }
        return { text: new XMLSerializer().serializeToString(out), width, height };
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
            .attr('y', M.top).attr('height', H - M.top - M.bottom);
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

    return { render, highlight, setBoxes, toSvg, legendItems: () => legendItems };
}

/**
 * Legend items (as a chart's legendItems) drawn into a container: a key and a label each.
 *
 * @param {HTMLElement} container
 * @param {{label: string, color: string, dot?: boolean, box?: boolean, glyph?: string}[]} items
 */
export function drawLegend(container, items) {
    d3.select(container).selectAll('span.item').data(items, d => d.label).join(enter => {
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
}

/** Point radius for n points, px: smaller when dense. */
function pointRadius(n) {
    return n > 2000 ? 1.5 : n > 500 ? 2 : 2.5;
}

function round2(v) {
    return Math.round(v * 100) / 100;
}

function svgEl(tag, attrs = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs))
        node.setAttribute(k, v);
    return node;
}

/** Text broken at spaces into lines at most `room` wide by measure (a single longer word keeps its own line). */
function wrapWords(text, room, measure) {
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
    return lines;
}

function svgText(text, attrs) {
    const node = svgEl('text', attrs);
    node.textContent = text;
    return node;
}

/**
 * A fill or stroke as SVG 1.1 attributes: rgba() split into its color and an opacity (times alpha), which
 * editors that read only SVG 1.1 paint (Illustrator, Inkscape's older releases) understand.
 */
function paint(prop, color, alpha = 1) {
    const m = /^rgba\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)[,\s/]+([\d.]+)\s*\)$/.exec(color);
    if (m)
        return { [prop]: `rgb(${m[1]}, ${m[2]}, ${m[3]})`, [`${prop}-opacity`]: Number(m[4]) * alpha };
    return alpha === 1 ? { [prop]: color } : { [prop]: color, [`${prop}-opacity`]: alpha };
}

// Copy each element's computed paint and stroke styles onto its clone (same tree shape) as attributes.
function inlineStyles(orig, copy) {
    const cs = getComputedStyle(orig);
    for (const prop of PAINT_PROPS) {
        const value = cs.getPropertyValue(prop);
        if (!value)
            continue;
        const attrs = paint(prop, value, Number(cs.getPropertyValue(`${prop}-opacity`) || 1));
        for (const [k, v] of Object.entries(attrs))
            copy.setAttribute(k, v);
    }
    for (const prop of STYLE_PROPS) {
        const value = cs.getPropertyValue(prop);
        if (value && !(prop === 'opacity' && value === '1') && !(prop === 'stroke-dasharray' && value === 'none'))
            copy.setAttribute(prop, value);
    }
    copy.removeAttribute('class');
    for (let i = 0; i < orig.children.length; i++)
        inlineStyles(orig.children[i], copy.children[i]);
}
