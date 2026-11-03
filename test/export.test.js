import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toCsv, exportColumns } from '../src/ui/export.js';
import { parseTable } from '../src/core/data.js';

test('toCsv writes full precision, Inf for infinite bounds, and quotes awkward names', () => {
    const csv = toCsv([
        { name: 'x', values: [0.1, 0.2] },
        { name: 'a,b "c"', values: [1 / 3, -Infinity] },
        { name: 'u', values: [Infinity, NaN] },
    ]);
    assert.equal(csv, 'x,"a,b ""c""",u\n0.1,0.3333333333333333,Inf\n0.2,-Inf,\n');
});

test('toCsv rejects ragged columns', () => {
    assert.throws(() => toCsv([{ name: 'a', values: [1, 2] }, { name: 'b', values: [1] }]), RangeError);
});

test('exportColumns orders data, truth, fits, band, exact; the app can read its own export back', () => {
    const x = Float64Array.from([1, 2, 2, 3]);
    const y = Float64Array.from([0.5, 1.5, 2.5, 3]);
    const cols = exportColumns({
        x, y, f: null, names: { x: 'age', y: 'correct' },
        fits: [{ key: 'minTV2', g: [1, 2, 2, 3] }, { key: 'taut', g: [0.5, 2, 2, 3] }],
        band: { key: 'monoUp', lb: [-Infinity, 1, 1, 2], ub: [1, 3, 3, Infinity] },
        exact: { lb: [0, 1, 1, 2], ub: [1, 2.5, 2.5, 4] },
    });
    assert.deepEqual(cols.map(c => c.name), ['age', 'correct', 'fit_minTV2', 'fit_taut',
        'band_monoUp_lower', 'band_monoUp_upper', 'exact_lower', 'exact_upper']);
    const table = parseTable(toCsv(cols));
    assert.deepEqual(table.names.slice(0, 3), ['age', 'correct', 'fit_minTV2']);
    assert.deepEqual(Array.from(table.columns[1]), [0.5, 1.5, 2.5, 3]);
});
