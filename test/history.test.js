import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHistory, COALESCE_MS } from '../src/ui/history.js';

const START = { ambition: -3.4, flat: 0.15, region: 'boot' };

test('undo and redo step through recorded changes; a new change drops the redo branch', () => {
    const h = createHistory(START);
    assert.equal(h.canUndo(), false);
    assert.equal(h.undo(), null);
    assert.equal(h.record({ ...START, region: 'An' }, 0), true);
    assert.equal(h.record({ ...START, region: 'An', flat: 0.5 }, 5000), true);
    assert.deepEqual(h.undo(), { ...START, region: 'An' });
    assert.deepEqual(h.undo(), START);
    assert.equal(h.canUndo(), false);
    assert.deepEqual(h.redo(), { ...START, region: 'An' });
    assert.equal(h.canRedo(), true);
    h.record({ ...START, region: 'An', ambition: 0 }, 10000);
    assert.equal(h.canRedo(), false);
    assert.equal(h.redo(), null);
    assert.deepEqual(h.current(), { ...START, region: 'An', ambition: 0 });
});

test('a gesture on one setting coalesces; another setting or a pause starts a new entry', () => {
    const h = createHistory(START);
    // a slider dragged: many changes to the same setting, each soon after the last
    for (const [k, v] of [-3, -2.5, -2, -1.5].entries())
        h.record({ ...START, ambition: v }, 100 * k);
    assert.deepEqual(h.undo(), START, 'one undo takes back the whole drag');
    h.redo();
    h.record({ ...START, ambition: -1.5, flat: 0.3 }, 450);
    h.record({ ...START, ambition: -1, flat: 0.3 }, 500 + COALESCE_MS);
    assert.deepEqual(h.undo(), { ...START, ambition: -1.5, flat: 0.3 });
    assert.deepEqual(h.undo(), { ...START, ambition: -1.5 });
    // coalescing can be refused (a reset, a link)
    const g = createHistory(START);
    g.record({ ...START, flat: 0 }, 0);
    g.record({ ...START, flat: 1 }, 10, { coalesce: false });
    assert.deepEqual(g.undo(), { ...START, flat: 0 });
});

test('no entry for no change; a gesture back to its start leaves nothing to undo', () => {
    const h = createHistory(START);
    assert.equal(h.record({ ...START }, 0), false);
    assert.equal(h.canUndo(), false);
    h.record({ ...START, ambition: -2 }, 0);
    h.record({ ...START }, 100);
    assert.equal(h.canUndo(), false);
    assert.deepEqual(h.current(), START);
});

test('the oldest entries go past the limit, and snapshots are copies', () => {
    const h = createHistory(START, { limit: 3 });
    for (let k = 1; k <= 5; k++)
        h.record({ ...START, ambition: k }, 10000 * k);
    assert.deepEqual(h.undo(), { ...START, ambition: 4 });
    assert.deepEqual(h.undo(), { ...START, ambition: 3 });
    assert.equal(h.undo(), null);
    const snap = h.current();
    snap.ambition = 99;
    assert.equal(h.current().ambition, 3);
});
