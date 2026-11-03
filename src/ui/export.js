// CSV export of the data, the fits and the band, one row per observation (sorted by x, ties kept).
// Values are written at full precision (shortest round-trip form); infinite band edges as Inf/-Inf,
// which R, pandas and JMP all read. No DOM use, so it is testable in Node.

/**
 * @typedef {object} ExportColumn
 * @property {string} name
 * @property {ArrayLike<number>} values  one per observation
 */

function cell(v) {
    if (v === Infinity)
        return 'Inf';
    if (v === -Infinity)
        return '-Inf';
    if (Number.isNaN(v))
        return '';
    return String(v);
}

function quote(name) {
    return /[",\n\r]/.test(name) ? `"${name.replace(/"/g, '""')}"` : name;
}

/**
 * Build CSV text from equal-length columns.
 *
 * @param {ExportColumn[]} columns
 * @returns {string}  header plus one line per row, '\n'-terminated
 */
export function toCsv(columns) {
    const n = columns.length ? columns[0].values.length : 0;
    for (const col of columns) {
        if (col.values.length !== n)
            throw new RangeError(`column ${col.name} has ${col.values.length} values, expected ${n}`);
    }
    const lines = [columns.map(c => quote(c.name)).join(',')];
    for (let i = 0; i < n; i++)
        lines.push(columns.map(c => cell(c.values[i])).join(','));
    return lines.join('\n') + '\n';
}

/**
 * The export columns for what is on screen: x, y, the true f (simulated data), every drawn fit (primary
 * first), and the band and exact bounds when shown.
 *
 * @param {object} p
 * @param {ArrayLike<number>} p.x
 * @param {ArrayLike<number>} p.y
 * @param {ArrayLike<number>|null} p.f            true regression function, or null
 * @param {{key: string, g: ArrayLike<number>}[]} p.fits  primary first
 * @param {{key: string, lb: ArrayLike<number>, ub: ArrayLike<number>}|null} p.band
 * @param {{lb: ArrayLike<number>, ub: ArrayLike<number>}|null} p.exact
 * @param {{x?: string, y?: string}} [p.names]   column names from the user's data
 * @returns {ExportColumn[]}
 */
export function exportColumns({ x, y, f, fits, band, exact, names = {} }) {
    const cols = [
        { name: names.x ?? 'x', values: x },
        { name: names.y ?? 'y', values: y },
    ];
    if (f)
        cols.push({ name: 'f_true', values: f });
    for (const fit of fits)
        cols.push({ name: `fit_${fit.key}`, values: fit.g });
    if (band) {
        cols.push({ name: `band_${band.key}_lower`, values: band.lb });
        cols.push({ name: `band_${band.key}_upper`, values: band.ub });
    }
    if (exact) {
        cols.push({ name: 'exact_lower', values: exact.lb });
        cols.push({ name: 'exact_upper', values: exact.ub });
    }
    return cols;
}
