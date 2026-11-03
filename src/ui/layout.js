// Shared geometry so the main chart and the residual view line up on the same x scale.

import * as d3 from 'd3';

export const MARGIN = { top: 10, right: 16, bottom: 28, left: 60 }; // px

/**
 * Cell edges between design points: edges[i] is the left edge of observation i, edges[n] the right
 * edge of the last. Interior edges are midpoints; the outer edges extend half a neighbor spacing.
 *
 * @param {Float64Array} x  sorted
 * @returns {Float64Array}  length n + 1
 */
export function cellEdges(x) {
    const n = x.length;
    const e = new Float64Array(n + 1);
    if (n === 1) {
        e[0] = x[0] - 0.5;
        e[1] = x[0] + 0.5;
        return e;
    }
    for (let i = 1; i < n; i++)
        e[i] = (x[i - 1] + x[i]) / 2;
    e[0] = x[0] - (x[1] - x[0]) / 2;
    e[n] = x[n - 1] + (x[n - 1] - x[n - 2]) / 2;
    return e;
}

/**
 * x scale spanning all cells, in pixels within a plot of the given outer width.
 *
 * @param {Float64Array} edges  from cellEdges
 * @param {number} width        outer width, px
 * @returns {d3.ScaleLinear<number, number>}
 */
export function xScale(edges, width) {
    return d3.scaleLinear()
        .domain([edges[0], edges[edges.length - 1]])
        .range([MARGIN.left, width - MARGIN.right]);
}

/**
 * Index of the observation whose cell contains data coordinate xv (clamped to 0..n−1).
 *
 * @param {Float64Array} edges
 * @param {number} xv
 * @returns {number}
 */
export function cellIndex(edges, xv) {
    const n = edges.length - 1;
    const i = d3.bisectRight(edges, xv) - 1;
    return Math.max(0, Math.min(n - 1, i));
}

/**
 * Canvas sized for the device pixel ratio, with its context scaled to CSS pixels.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {number} width   CSS px
 * @param {number} height  CSS px
 * @returns {CanvasRenderingContext2D}
 */
export function sizeCanvas(canvas, width, height) {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return ctx;
}
