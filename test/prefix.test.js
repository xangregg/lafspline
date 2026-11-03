import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prefixSums, windowSum, windowMean } from '../src/core/prefix.js';
import { createRng } from '../src/core/rng.js';

test('window sums and means match direct summation (tol 1e-12)', () => {
    const rng = createRng(3);
    const v = Float64Array.from({ length: 50 }, () => rng.normal());
    const s = prefixSums(v);
    assert.equal(s.length, 51);
    assert.equal(s[0], 0);
    for (let lo = 0; lo < v.length; lo++) {
        for (let hi = lo; hi < v.length; hi++) {
            let direct = 0;
            for (let i = lo; i <= hi; i++)
                direct += v[i];
            assert.ok(Math.abs(windowSum(s, lo, hi) - direct) < 1e-12);
            assert.ok(Math.abs(windowMean(s, lo, hi) - direct / (hi - lo + 1)) < 1e-12);
        }
    }
});
