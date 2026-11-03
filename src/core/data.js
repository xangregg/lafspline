// User data: parse delimited text, pick x/y columns, sort by x, and describe spacing and ties
// (PLAN.md §1.10).

const SPACING_REL_TOL = 1e-6; // max |Δx − mean Δx| relative to the x range for "equally spaced"
const SNIFF_LINES = 5; // lines inspected to choose the delimiter

/**
 * Split delimited text into rows of string fields. Handles double-quoted fields (with "" escapes and
 * embedded delimiters/newlines). If no comma, tab or semicolon appears in the first lines, splits on
 * runs of whitespace.
 *
 * @param {string} text
 * @returns {string[][]}  non-empty rows
 */
export function parseDelimited(text) {
    const head = text.split(/\r?\n/).filter(line => line.trim() !== '').slice(0, SNIFF_LINES).join('\n');
    const counts = [',', '\t', ';'].map(d => [d, head.split(d).length - 1]);
    counts.sort((a, b) => b[1] - a[1]);
    if (counts[0][1] === 0) {
        return text.split(/\r?\n/)
            .map(line => line.trim())
            .filter(line => line !== '')
            .map(line => line.split(/\s+/));
    }
    const delim = counts[0][0];

    const rows = [];
    let row = [];
    let field = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (inQuotes) {
            if (ch === '"' && text[i + 1] === '"') {
                field += '"';
                i++;
            }
            else if (ch === '"') {
                inQuotes = false;
            }
            else {
                field += ch;
            }
        }
        else if (ch === '"') {
            inQuotes = true;
        }
        else if (ch === delim) {
            row.push(field);
            field = '';
        }
        else if (ch === '\n' || ch === '\r') {
            if (ch === '\r' && text[i + 1] === '\n')
                i++;
            row.push(field);
            rows.push(row);
            row = [];
            field = '';
        }
        else {
            field += ch;
        }
    }
    row.push(field);
    rows.push(row);
    return rows.filter(r => r.some(f => f.trim() !== ''));
}

/**
 * Parse a field as a number; empty or non-numeric fields give NaN (unlike Number(''), which is 0).
 *
 * @param {string} s
 * @returns {number}
 */
export function parseNumber(s) {
    const t = s.trim();
    if (t === '')
        return NaN;
    return Number(t);
}

/**
 * Parse delimited text into named numeric columns. The first row is a header if any of its fields is
 * non-numeric; otherwise columns are named col1, col2, ….
 *
 * @param {string} text
 * @returns {{names: string[], columns: Float64Array[], rows: number}}  NaN marks missing/non-numeric
 */
export function parseTable(text) {
    const rows = parseDelimited(text);
    if (rows.length === 0)
        return { names: [], columns: [], rows: 0 };
    const width = Math.max(...rows.map(r => r.length));
    const hasHeader = rows[0].some(f => f.trim() !== '' && Number.isNaN(parseNumber(f)));
    const names = [];
    for (let j = 0; j < width; j++) {
        const fromHeader = hasHeader ? (rows[0][j] ?? '').trim() : '';
        names.push(fromHeader || `col${j + 1}`);
    }
    const body = hasHeader ? rows.slice(1) : rows;
    const columns = names.map((_, j) => Float64Array.from(body, r => parseNumber(r[j] ?? '')));
    return { names, columns, rows: body.length };
}

/**
 * @typedef {object} PreparedData
 * @property {Float64Array} x        sorted x, user units
 * @property {Float64Array} y        y reordered to match x
 * @property {Int32Array} order      order[i] = original row of sorted observation i
 * @property {number} dropped        rows dropped because x or y was missing/non-finite
 * @property {number} ties           number of i with x[i] = x[i−1]
 * @property {boolean} equallySpaced true if n ≥ 2, no ties and all Δx equal within tolerance
 * @property {number} xMin
 * @property {number} xMax
 * @property {Float64Array} t        x rescaled to [0, 1] (all 0 if the x range is 0)
 */

/**
 * Drop incomplete rows, stable-sort by x (ties keep their original order) and describe the design.
 *
 * @param {ArrayLike<number>} xRaw
 * @param {ArrayLike<number>} yRaw  same length as xRaw
 * @returns {PreparedData}
 */
export function prepareXY(xRaw, yRaw) {
    const keep = [];
    for (let i = 0; i < xRaw.length; i++) {
        if (Number.isFinite(xRaw[i]) && Number.isFinite(yRaw[i]))
            keep.push(i);
    }
    // Array.prototype.sort is stable, so tied x keep their original row order.
    keep.sort((a, b) => xRaw[a] - xRaw[b]);
    const n = keep.length;
    const order = Int32Array.from(keep);
    const x = Float64Array.from(keep, i => xRaw[i]);
    const y = Float64Array.from(keep, i => yRaw[i]);

    let ties = 0;
    for (let i = 1; i < n; i++) {
        if (x[i] === x[i - 1])
            ties++;
    }
    const xMin = n > 0 ? x[0] : NaN;
    const xMax = n > 0 ? x[n - 1] : NaN;
    const range = xMax - xMin;
    const t = new Float64Array(n);
    if (range > 0) {
        for (let i = 0; i < n; i++)
            t[i] = (x[i] - xMin) / range;
    }

    let equallySpaced = n >= 2 && ties === 0 && range > 0;
    if (equallySpaced) {
        const step = range / (n - 1);
        for (let i = 1; i < n; i++) {
            if (Math.abs(x[i] - x[i - 1] - step) > SPACING_REL_TOL * range) {
                equallySpaced = false;
                break;
            }
        }
    }
    return { x, y, order, dropped: xRaw.length - n, ties, equallySpaced, xMin, xMax, t };
}

/**
 * x = 1, 2, …, n, for data given as a single column.
 *
 * @param {number} n
 * @returns {Float64Array}
 */
export function rowIndex(n) {
    return Float64Array.from({ length: n }, (_, i) => i + 1);
}
