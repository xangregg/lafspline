// CSV export of the data, the fits and the band, one row per observation (sorted by x, ties kept), and (lafspline)
// the same layers on the check grid.
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

// A name a spreadsheet would read as a formula (=, +, -, @, or a tab or carriage return first) gets a leading
// apostrophe, so a column named in pasted data, which a shared link can carry, cannot run as one in an export.
function quote(name) {
    const safe = /^[=+\-@\t\r]/.test(name) ? `'${name}` : name;
    return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
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

/**
 * Values on a grid that contains every data x, at the observations: at[k] is the grid index of the k-th distinct
 * x, and observations that share an x share its value.
 *
 * @param {ArrayLike<number>} values  on the grid
 * @param {ArrayLike<number>} at      grid index per distinct x, ascending
 * @param {ArrayLike<number>} x       observations' x, sorted (ties allowed)
 * @returns {Float64Array}
 */
export function atObservations(values, at, x) {
    const out = new Float64Array(x.length);
    let k = -1;
    for (let i = 0; i < x.length; i++) {
        if (i === 0 || x[i] !== x[i - 1])
            k++;
        out[i] = values[at[k]];
    }
    return out;
}

/**
 * lafspline's two export tables: one row per observation (x, y, the true f for simulated data, and each layer
 * where it is at that x) and one row per check-grid point (where the region and the fits are computed and
 * checked, so it is the finer and more faithful curve).
 *
 * @param {object} p
 * @param {ArrayLike<number>} p.x        sorted observations' x
 * @param {ArrayLike<number>} p.y
 * @param {((x: number) => number)|null} p.fAt  true regression function, or null
 * @param {ArrayLike<number>} p.grid     sorted, contains every data x
 * @param {ArrayLike<number>} p.at       grid index per distinct x
 * @param {{name: string, values: ArrayLike<number>}[]} p.layers  on the grid
 * @param {{x?: string, y?: string}} [p.names]  column names from the data
 * @returns {{observations: ExportColumn[], grid: ExportColumn[]}}
 */
export function fitTables({ x, y, fAt, grid, at, layers, names = {} }) {
    const xName = names.x ?? 'x';
    const observations = [{ name: xName, values: x }, { name: names.y ?? 'y', values: y }];
    const onGrid = [{ name: xName, values: grid }];
    if (fAt) {
        observations.push({ name: 'f_true', values: Float64Array.from(x, fAt) });
        onGrid.push({ name: 'f_true', values: Float64Array.from(grid, fAt) });
    }
    for (const layer of layers) {
        observations.push({ name: layer.name, values: atObservations(layer.values, at, x) });
        onGrid.push(layer);
    }
    return { observations, grid: onGrid };
}
