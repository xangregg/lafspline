// Example data sets bundled as CSV files in data/ (see data/README.md). A file may start with metadata lines
//     # key: value
// (keys: label and source required; x and y name the columns, default the first two; note optional), followed
// by an ordinary CSV/TSV table that data.js parses. tools/data-index.mjs turns the files into data/index.json,
// which the app reads because a static site cannot list a directory.

import { parseTable, prepareXY } from './data.js';

export const META_KEYS = ['label', 'source', 'x', 'y', 'note'];
const REQUIRED = ['label', 'source'];
const ID_PATTERN = /^[a-z0-9][a-z0-9-]*$/;
const MIN_DISTINCT_X = 3;

/**
 * Split leading '# key: value' lines from the table.
 *
 * @param {string} text
 * @returns {{meta: Record<string, string>, body: string, errors: string[]}}
 */
export function splitMetadata(text) {
    const lines = text.split(/\r?\n/);
    const meta = {};
    const errors = [];
    let k = 0;
    for (; k < lines.length; k++) {
        const line = lines[k].trim();
        if (line === '')
            continue;
        if (!line.startsWith('#'))
            break;
        const match = /^#\s*([A-Za-z]+)\s*:\s*(.*)$/.exec(line);
        if (!match) {
            errors.push(`line ${k + 1}: expected '# key: value'`);
            continue;
        }
        const key = match[1].toLowerCase();
        if (!META_KEYS.includes(key))
            errors.push(`line ${k + 1}: unknown key '${key}' (known: ${META_KEYS.join(', ')})`);
        else
            meta[key] = match[2].trim();
    }
    for (const key of REQUIRED) {
        if (!meta[key])
            errors.push(`missing '# ${key}: …'`);
    }
    return { meta, body: lines.slice(k).join('\n'), errors };
}

/**
 * Read a data set file: metadata, the chosen columns, sorted by x.
 *
 * @param {string} text
 * @returns {{meta: Record<string, string>, names: string[], xName: string, yName: string,
 *     x: Float64Array, y: Float64Array, n: number, m: number, dropped: number, ties: number, errors: string[]}}
 */
export function readDataset(text) {
    const { meta, body, errors } = splitMetadata(text);
    const table = parseTable(body);
    const names = table.names;
    const pick = (wanted, fallback, role) => {
        if (wanted === undefined)
            return fallback;
        const j = names.indexOf(wanted);
        if (j < 0)
            errors.push(`${role} column '${wanted}' not found (columns: ${names.join(', ')})`);
        return j;
    };
    const xj = pick(meta.x, 0, 'x');
    const yj = pick(meta.y, 1, 'y');
    if (names.length < 2)
        errors.push('needs at least two columns');
    const empty = new Float64Array(0);
    if (errors.length || xj < 0 || yj < 0 || names.length < 2)
        return { meta, names, xName: '', yName: '', x: empty, y: empty, n: 0, m: 0, dropped: 0, ties: 0, errors };
    const d = prepareXY(table.columns[xj], table.columns[yj]);
    let m = d.x.length > 0 ? 1 : 0;
    for (let i = 1; i < d.x.length; i++) {
        if (d.x[i] !== d.x[i - 1])
            m++;
    }
    if (m < MIN_DISTINCT_X)
        errors.push(`needs at least ${MIN_DISTINCT_X} distinct x with a y (has ${m})`);
    return { meta, names, xName: names[xj], yName: names[yj], x: d.x, y: d.y, n: d.x.length, m, dropped: d.dropped,
        ties: d.ties, errors };
}

/**
 * Data set id from a file name: 'nile-flow.csv' → 'nile-flow'.
 *
 * @param {string} fileName
 * @returns {string|null}  null if the name is not lower-case letters, digits and hyphens ending in .csv
 */
export function datasetId(fileName) {
    const m = /^(.*)\.csv$/.exec(fileName);
    return m && ID_PATTERN.test(m[1]) ? m[1] : null;
}

/**
 * The index entry for a file (what the app's dataset list needs).
 *
 * @param {string} fileName
 * @param {string} text
 * @returns {{entry: object|null, errors: string[]}}
 */
export function indexEntry(fileName, text) {
    const id = datasetId(fileName);
    const d = readDataset(text);
    const errors = [...d.errors];
    if (!id)
        errors.unshift(`file name must be lower-case letters, digits and hyphens, ending in .csv`);
    if (errors.length)
        return { entry: null, errors };
    const entry = { id, file: fileName, label: d.meta.label, source: d.meta.source, x: d.xName, y: d.yName, n: d.n, m: d.m };
    if (d.meta.note)
        entry.note = d.meta.note;
    return { entry, errors };
}
