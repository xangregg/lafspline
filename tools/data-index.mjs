// Build data/index.json from the CSV files in data/ (the app's list of example data sets; a static site
// cannot list a directory). Exits 1, writing nothing, if any file is invalid. See data/README.md.
// Also writes localdata/index.json, the names of the local data sets, for servers that do not list directories
// (WebStorm's); git ignores it with the rest of localdata/. A local file that is invalid is left out with a warning.
//
// Usage: node tools/data-index.mjs [--check]   (--check: exit 1 if data/index.json is out of date, write nothing)

import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { indexEntry } from '../src/core/datasets.js';

const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'data');
const INDEX = join(DATA_DIR, 'index.json');
const LOCAL_DIR = join(DATA_DIR, '..', 'localdata');
const LOCAL_INDEX = join(LOCAL_DIR, 'index.json');

/**
 * The index for the CSV files in a directory, sorted by label.
 *
 * @param {string} dir
 * @returns {{index: object[], errors: string[]}}
 */
export function buildIndex(dir) {
    const index = [];
    const errors = [];
    for (const file of readdirSync(dir).filter(f => f.endsWith('.csv')).sort()) {
        const r = indexEntry(file, readFileSync(join(dir, file), 'utf8'));
        if (r.entry)
            index.push(r.entry);
        for (const e of r.errors)
            errors.push(`${file}: ${e}`);
    }
    index.sort((a, b) => a.label.localeCompare(b.label));
    return { index, errors };
}

export function indexText(index) {
    return JSON.stringify(index, null, 2) + '\n';
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const check = process.argv.includes('--check');
    const { index, errors } = buildIndex(DATA_DIR);
    if (errors.length) {
        for (const e of errors)
            console.error(e);
        process.exit(1);
    }
    const text = indexText(index);
    const current = existsSync(INDEX) ? readFileSync(INDEX, 'utf8') : '';
    if (check) {
        if (text !== current) {
            console.error('data/index.json is out of date: run npm run data:index');
            process.exit(1);
        }
        console.log(`data/index.json is up to date (${index.length} data sets)`);
    }
    else {
        writeFileSync(INDEX, text);
        console.log(`wrote data/index.json: ${index.map(e => `${e.id} (n = ${e.n})`).join(', ') || 'no data sets'}`);
        writeLocalIndex();
    }
}

/**
 * localdata/index.json: a JSON array of the CSV file names there that the app can read, sorted. As in the app
 * (src/ui/main.js loadLocalExamples), a file without a label or source is listed by its name.
 */
function writeLocalIndex() {
    if (!existsSync(LOCAL_DIR))
        return;
    const names = [];
    for (const file of readdirSync(LOCAL_DIR).filter(f => f.endsWith('.csv')).sort()) {
        const defaults = `# label: ${file}\n# source: localdata/${file} (local)\n`;
        const r = indexEntry(file, defaults + readFileSync(join(LOCAL_DIR, file), 'utf8'));
        if (r.entry)
            names.push(file);
        else
            console.warn(`localdata/${file} left out: ${r.errors.join('; ')}`);
    }
    writeFileSync(LOCAL_INDEX, JSON.stringify(names, null, 2) + '\n');
    console.log(`wrote localdata/index.json: ${names.join(', ') || 'no data sets'}`);
}
