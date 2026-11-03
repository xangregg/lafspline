import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runs, localExtremes, stepSigns } from '../src/core/extremes.js';

test('runs merge values equal within the relative tolerance', () => {
    const f = [1, 1, 1 + 1e-13, 2, 2, 1];
    assert.deepEqual(runs(f).map(r => [r.start, r.end]), [[0, 2], [3, 4], [5, 5]]);
});

test('local extremes with plateaus; endpoint runs not counted', () => {
    const f = [0, 0, 3, 3, 3, 1, 2, 2, 5, 4];
    const { extremes, nMax, nMin } = localExtremes(f);
    assert.deepEqual(extremes.map(e => [e.type, e.start, e.end]), [['max', 2, 4], ['min', 5, 5], ['max', 8, 8]]);
    assert.equal(nMax, 2);
    assert.equal(nMin, 1);
    assert.equal(localExtremes([5, 5, 5]).extremes.length, 0);
    assert.equal(localExtremes([1, 2, 3, 4]).extremes.length, 0); // monotone
});

test('step signs: monotone segment direction, 0 inside extreme plateaus', () => {
    const f = [0, 0, 3, 3, 3, 1, 1, 2];
    // steps: 0→0 (+, rising segment), 0→3 (+), 3→3 (0, plateau of max), 3→3 (0), 3→1 (−),
    //        1→1 (0, plateau of min), 1→2 (+)
    assert.deepEqual(Array.from(stepSigns(f)), [1, 1, 0, 0, -1, 0, 1]);
    assert.deepEqual(Array.from(stepSigns([2, 2, 2])), [0, 0]);
    assert.deepEqual(Array.from(stepSigns([3, 3, 1, 1])), [-1, -1, -1]);
});

test('step signs with split plateaus: up to the midpoint of a max, down after (reverse for a min)', () => {
    const f = [0, 0, 3, 3, 3, 3, 1, 1, 1, 2];
    // max plateau 2..5 (mid 3): steps 2→3 up, 3→4 and 4→5 down; min plateau 6..8 (mid 7): 6→7 down, 7→8 up
    assert.deepEqual(Array.from(stepSigns(f, { plateaus: 'split' })), [1, 1, 1, -1, -1, -1, -1, 1, 1]);
    assert.deepEqual(Array.from(stepSigns(f)), [1, 1, 0, 0, 0, -1, 0, 0, 1]);
});

