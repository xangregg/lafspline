import { test } from 'node:test';
import assert from 'node:assert/strict';
import { multiresolutionLevels, multiresolution, allIntervals, overlappingLevels, familyLevels, familyIntervals,
    familyText, xIntervalFamily, intervalSpan, ALL_INTERVALS_MAX } from '../src/core/intervals.js';

const SIZES = [1, 2, 3, 7, 8, 9, 100, 500, 1000, 1023, 1024, 1025];

test('each level of multiresolution(n, λ) partitions 0..n−1', () => {
    for (const lambda of [2, 1.5, 3]) {
        for (const n of SIZES) {
            const levels = multiresolutionLevels(n, lambda);
            for (const level of levels) {
                let next = 0;
                for (const { lo, hi } of level) {
                    assert.equal(lo, next, `n=${n} λ=${lambda}: gap or overlap at ${lo}`);
                    assert.ok(hi >= lo);
                    next = hi + 1;
                }
                assert.equal(next, n, `n=${n} λ=${lambda}: level does not reach n`);
            }
            // top level is the whole range
            const top = levels[levels.length - 1];
            assert.deepEqual([top[0].lo, top[0].hi, top.length], [0, n - 1, 1]);
        }
    }
});

test('multiresolution contains every singleton, has no duplicates, and is O(n)', () => {
    for (const n of SIZES) {
        const fam = multiresolution(n, 2);
        const keys = new Set(fam.map(iv => `${iv.lo},${iv.hi}`));
        assert.equal(keys.size, fam.length, 'duplicates');
        for (let i = 0; i < n; i++)
            assert.ok(keys.has(`${i},${i}`), `missing singleton ${i}`);
        assert.ok(fam.length <= 2 * n + Math.log2(n) + 1, `n=${n}: ${fam.length} intervals`);
    }
});

test('dyadic blocks for n = 8 are exactly the expected intervals', () => {
    const fam = multiresolution(8, 2).map(iv => [iv.lo, iv.hi, iv.level]);
    assert.deepEqual(fam, [
        [0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0], [5, 5, 0], [6, 6, 0], [7, 7, 0],
        [0, 1, 1], [2, 3, 1], [4, 5, 1], [6, 7, 1],
        [0, 3, 2], [4, 7, 2],
        [0, 7, 3],
    ]);
});

test('allIntervals(n) has n(n+1)/2 distinct intervals', () => {
    const fam = allIntervals(20);
    assert.equal(fam.length, 210);
    assert.equal(new Set(fam.map(iv => `${iv.lo},${iv.hi}`)).size, 210);
});

test('λ ≤ 1 is rejected', () => {
    assert.throws(() => multiresolution(10, 1), RangeError);
});

test('overlapping windows have one length per level, step ≈ length/offsets, and reach both ends', () => {
    for (const n of [1, 2, 7, 10, 100, 1025]) {
        for (const offsets of [1, 4, Infinity]) {
            const levels = overlappingLevels(n, 2, offsets);
            const top = levels[levels.length - 1];
            assert.deepEqual(top.map(iv => [iv.lo, iv.hi]), [[0, n - 1]]);
            for (const level of levels) {
                const W = level[0].hi - level[0].lo + 1;
                const step = Math.max(1, Math.round(W / offsets));
                assert.equal(level[0].lo, 0);
                assert.equal(level[level.length - 1].hi, n - 1, `n=${n} W=${W}: last window does not end at n − 1`);
                level.forEach((iv, i) => {
                    assert.equal(iv.hi - iv.lo + 1, W, `n=${n} W=${W}: window of another length`);
                    if (i > 0 && i < level.length - 1)
                        assert.equal(iv.lo - level[i - 1].lo, step);
                });
            }
            assert.equal(levels[0].length, n, 'every singleton');
        }
    }
    // every position: n − W + 1 windows of each length
    for (const level of overlappingLevels(100, 2, Infinity))
        assert.equal(level.length, 100 - (level[0].hi - level[0].lo) );
});

test('denser families contain the sparser ones (λ = 2): quarter steps ⊂ every position ⊂ all intervals', () => {
    const x = Float64Array.from({ length: 77 }, (_, i) => i);
    const keys = family => new Set(familyIntervals(x, family).map(iv => `${iv.lo},${iv.hi}`));
    const quarter = keys({ kind: 'overlapping', lambda: 2, offsets: 4 });
    const every = keys({ kind: 'overlapping', lambda: 2, offsets: Infinity });
    const all = keys({ kind: 'all' });
    const blocks = keys({ kind: 'blocks', lambda: 2 });
    const subset = (a, b) => [...a].every(k => b.has(k));
    assert.ok(subset(quarter, every) && subset(every, all) && subset(blocks, all));
    assert.ok(quarter.size > blocks.size && every.size > quarter.size);
    assert.equal(all.size, 77 * 78 / 2);
});

test('families over tied x never split ties, and blocks equal the λ family', () => {
    const x = Float64Array.of(1, 1, 2, 3, 3, 3, 4, 5, 5, 6);
    const groupStarts = new Set([0, 2, 3, 6, 7, 9]);
    for (const family of [{ kind: 'blocks', lambda: 2 }, { kind: 'overlapping', lambda: 2, offsets: 4 }, { kind: 'all' }]) {
        for (const level of familyLevels(x, family)) {
            for (const { lo, hi } of level) {
                assert.ok(groupStarts.has(lo), `${family.kind}: interval starts inside a tie group`);
                assert.ok(hi === x.length - 1 || groupStarts.has(hi + 1), `${family.kind}: interval ends inside a tie group`);
            }
        }
    }
    assert.deepEqual(familyIntervals(x, { kind: 'blocks', lambda: 2 }), xIntervalFamily(x, 2));
    assert.equal(familyText({ kind: 'overlapping', lambda: 2, offsets: 4 }), 'λ = 2, 1/4-width steps');
    const big = Float64Array.from({ length: ALL_INTERVALS_MAX + 1 }, (_, i) => i);
    assert.throws(() => familyLevels(big, { kind: 'all' }), RangeError);
});

// uneven x with ties: dense near 0, sparse after, two tied pairs
const UNEVEN = Float64Array.from([0, 0.01, 0.02, 0.02, 0.05, 0.3, 0.31, 1, 2, 2, 4, 7, 7.5, 10]);

test('an interval spans midway to its neighbors (clipped to the range)', () => {
    const tol = 1e-12; // units of x
    assert.ok(Math.abs(intervalSpan(UNEVEN, { lo: 0, hi: 0 }) - 0.005) < tol);
    assert.ok(Math.abs(intervalSpan(UNEVEN, { lo: 10, hi: 10 }) - (5.5 - 3)) < tol); // x = 4, between 2 and 7
    assert.ok(Math.abs(intervalSpan(UNEVEN, { lo: 8, hi: 9 }) - (3 - 1.5)) < tol); // the tied pair at 2
    assert.ok(Math.abs(intervalSpan(UNEVEN, { lo: 13, hi: 13 }) - (10 - 8.75)) < tol);
    assert.equal(intervalSpan(UNEVEN, { lo: 0, hi: 13 }), 10);
});

test('with a shortest span blocks long enough stay, runs of short blocks merge, and every part spans enough', () => {
    const full = familyLevels(UNEVEN, { kind: 'blocks', lambda: 2 });
    for (const share of [0.05, 0.2, 0.5, 1]) {
        const shortest = share * 10;
        const levels = familyLevels(UNEVEN, { kind: 'blocks', lambda: 2, minShare: share });
        assert.equal(levels.length, full.length, 'level numbers stay array indices');
        for (const [k, level] of levels.entries()) {
            const starts = new Set(full[k].map(iv => iv.lo));
            const ends = new Set(full[k].map(iv => iv.hi));
            for (const iv of level) {
                assert.equal(iv.level, k);
                assert.ok(intervalSpan(UNEVEN, iv) >= shortest * (1 - 1e-9) || level.length === 1);
            }
            // every observation is in some part, and a block long enough is a part as it was
            for (let i = 0; i < UNEVEN.length; i++)
                assert.ok(level.some(iv => iv.lo <= i && i <= iv.hi));
            for (const iv of full[k]) {
                if (intervalSpan(UNEVEN, iv) >= shortest)
                    assert.ok(level.some(w => w.lo === iv.lo && w.hi === iv.hi));
            }
            // a part not widened is a run of whole blocks
            for (const iv of level.filter(w => starts.has(w.lo) && ends.has(w.hi)))
                assert.ok(full[k].some(b => b.lo === iv.lo) && full[k].some(b => b.hi === iv.hi));
        }
    }
    // an even design: every singleton is short, so the finest level becomes about 1/share parts, a partition
    const even = Float64Array.from({ length: 100 }, (_, i) => i);
    const finest = familyLevels(even, { kind: 'blocks', lambda: 2, minShare: 0.05 })[0];
    assert.ok(finest.length >= 15 && finest.length <= 20, `${finest.length} parts`);
    let next = 0;
    for (const iv of finest) {
        assert.equal(iv.lo, next);
        next = iv.hi + 1;
    }
    assert.equal(next, 100);
    // 60 distinct intervals, against 199 with every block and 105 when each short block was widened on its own
    assert.equal(familyIntervals(even, { kind: 'blocks', lambda: 2, minShare: 0.05 }).length, 60);
});

test('with a shortest span short intervals are widened to it, ties kept together, duplicates removed', () => {
    for (const family of [{ kind: 'overlapping', lambda: 2, offsets: 4 }, { kind: 'all' }]) {
        const full = familyLevels(UNEVEN, family);
        for (const share of [0.05, 0.2, 0.5, 1]) {
            const levels = familyLevels(UNEVEN, { ...family, minShare: share });
            assert.equal(levels.length, full.length);
            for (const [k, level] of levels.entries()) {
                assert.equal(new Set(level.map(iv => `${iv.lo},${iv.hi}`)).size, level.length);
                for (const iv of level) {
                    assert.ok(intervalSpan(UNEVEN, iv) >= share * 10 * (1 - 1e-9) || (iv.lo === 0 && iv.hi === 13));
                    assert.ok(iv.lo === 0 || UNEVEN[iv.lo] !== UNEVEN[iv.lo - 1], 'ties are never split');
                    assert.ok(iv.hi === 13 || UNEVEN[iv.hi] !== UNEVEN[iv.hi + 1], 'ties are never split');
                }
                // every original interval lies inside a widened one of its level, and one spanning enough is kept
                for (const iv of full[k]) {
                    assert.ok(level.some(w => w.lo <= iv.lo && w.hi >= iv.hi));
                    if (intervalSpan(UNEVEN, iv) >= share * 10)
                        assert.ok(level.some(w => w.lo === iv.lo && w.hi === iv.hi));
                }
            }
        }
        assert.equal(familyIntervals(UNEVEN, { ...family, minShare: 1 }).length, 1);
    }
    // where x is sparse a widened block holds fewer observations than where it is dense
    const finest = familyLevels(UNEVEN, { kind: 'blocks', lambda: 2, minShare: 0.2 })[0];
    const count = iv => iv.hi - iv.lo + 1;
    assert.ok(count(finest[0]) > count(finest.at(-1)), finest.map(count).join(','));
    assert.equal(familyText({ kind: 'blocks', lambda: 2, minShare: 0.05 }), 'λ = 2, spanning at least 5% of x');
});
