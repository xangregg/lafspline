import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseDelimited, parseTable, parseNumber, prepareXY, rowIndex } from '../src/core/data.js';

test('comma CSV with header and quoted fields', () => {
    const t = parseTable('time,"value, mV",note\n1,2.5,"a ""b"""\n2,3,\n');
    assert.deepEqual(t.names, ['time', 'value, mV', 'note']);
    assert.equal(t.rows, 2);
    assert.deepEqual(Array.from(t.columns[0]), [1, 2]);
    assert.deepEqual(Array.from(t.columns[1]), [2.5, 3]);
    assert.ok(t.columns[2].every(Number.isNaN));
});

test('headerless tab, semicolon and whitespace tables; CRLF line endings', () => {
    for (const text of ['1\t10\r\n2\t20\r\n', '1;10\n2;20', '  1   10\n 2 20\n\n']) {
        const t = parseTable(text);
        assert.deepEqual(t.names, ['col1', 'col2']);
        assert.deepEqual(Array.from(t.columns[1]), [10, 20]);
    }
});

test('parseNumber treats empty as missing, not zero', () => {
    assert.ok(Number.isNaN(parseNumber('  ')));
    assert.equal(parseNumber(' -1.5e2 '), -150);
    assert.ok(Number.isNaN(parseNumber('abc')));
});

test('quoted newline stays inside the field', () => {
    assert.deepEqual(parseDelimited('a,"x\ny"\n1,2'), [['a', 'x\ny'], ['1', '2']]);
});

test('prepareXY drops incomplete rows, sorts stably, counts ties', () => {
    const x = [3, 1, NaN, 2, 1, 5];
    const y = [30, 10, 99, 20, 11, NaN];
    const d = prepareXY(x, y);
    assert.deepEqual(Array.from(d.x), [1, 1, 2, 3]);
    assert.deepEqual(Array.from(d.y), [10, 11, 20, 30]); // tie keeps original row order
    assert.deepEqual(Array.from(d.order), [1, 4, 3, 0]);
    assert.equal(d.dropped, 2);
    assert.equal(d.ties, 1);
    assert.equal(d.equallySpaced, false);
    assert.deepEqual(Array.from(d.t), [0, 0, 0.5, 1]);
});

test('equal spacing detection', () => {
    assert.ok(prepareXY([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]).equallySpaced); // float steps within tol
    assert.ok(!prepareXY([0, 1, 3], [1, 2, 3]).equallySpaced);
    assert.ok(prepareXY(rowIndex(5), [5, 4, 3, 2, 1]).equallySpaced);
});
