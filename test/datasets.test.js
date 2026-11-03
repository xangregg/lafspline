import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { splitMetadata, readDataset, datasetId, indexEntry } from '../src/core/datasets.js';
import { buildIndex, indexText } from '../tools/data-index.mjs';

const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'data');

const SAMPLE = `# label: A test set
# source: made up for this test
# x: when
# y: level
# note: tied x at 2
id,when,level
a,1,10
b,2,11
c,2,12
d,,13
e,3,14
f,4,x
`;

test('metadata lines are read, and unknown or missing keys are reported', () => {
    const { meta, body, errors } = splitMetadata(SAMPLE);
    assert.deepEqual(meta, { label: 'A test set', source: 'made up for this test', x: 'when', y: 'level', note: 'tied x at 2' });
    assert.ok(body.startsWith('id,when,level'));
    assert.deepEqual(errors, []);
    const bad = splitMetadata('# label: x\n# colour: red\n#oops\nx,y\n1,2\n');
    assert.equal(bad.errors.length, 3, bad.errors.join('; '));
    assert.match(bad.errors.join(' '), /unknown key 'colour'/);
    assert.match(bad.errors.join(' '), /missing '# source/);
});

test('the named columns are used, incomplete rows dropped, ties kept, and x sorted', () => {
    const d = readDataset(SAMPLE);
    assert.deepEqual(d.errors, []);
    assert.equal(d.xName, 'when');
    assert.equal(d.yName, 'level');
    assert.deepEqual(Array.from(d.x), [1, 2, 2, 3]);
    assert.deepEqual(Array.from(d.y), [10, 11, 12, 14]);
    assert.equal(d.dropped, 2);
    assert.equal(d.ties, 1);
    assert.equal(d.m, 3);

    const defaults = readDataset('# label: l\n# source: s\nt,v\n3,1\n1,2\n2,3\n');
    assert.deepEqual(Array.from(defaults.x), [1, 2, 3]);
    assert.match(readDataset('# label: l\n# source: s\n# x: nope\nt,v\n1,1\n2,2\n3,3\n').errors[0], /x column 'nope' not found/);
    assert.match(readDataset('# label: l\n# source: s\nt,v\n1,1\n1,2\n').errors[0], /at least 3 distinct x/);
});

test('ids come from safe file names only', () => {
    assert.equal(datasetId('nile-flow.csv'), 'nile-flow');
    assert.equal(datasetId('Nile.csv'), null);
    assert.equal(datasetId('a b.csv'), null);
    assert.equal(datasetId('x.tsv'), null);
    const { entry } = indexEntry('test-set.csv', SAMPLE);
    assert.deepEqual(entry, { id: 'test-set', file: 'test-set.csv', label: 'A test set', source: 'made up for this test',
        x: 'when', y: 'level', n: 4, m: 3, note: 'tied x at 2' });
});

test('data/index.json is up to date with the CSV files and every file is valid', () => {
    const { index, errors } = buildIndex(DATA_DIR);
    assert.deepEqual(errors, []);
    assert.equal(readFileSync(join(DATA_DIR, 'index.json'), 'utf8'), indexText(index), 'run npm run data:index');
});
