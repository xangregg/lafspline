import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fmtExp } from '../src/ui/format.js';

test('fmtExp: two significant digits, plain from 1e-3 to 1e4, a power of ten outside', () => {
    // Regression: 530 came out as 5.3e+2 (toPrecision's own exponent form).
    assert.equal(fmtExp(530), '530');
    assert.equal(fmtExp(5300), '5300');
    assert.equal(fmtExp(0.0012), '0.0012');
    assert.equal(fmtExp(0.01), '0.01');
    assert.equal(fmtExp(1.25), '1.3');
    assert.equal(fmtExp(-47.2), '-47');
    assert.equal(fmtExp(2.8e-8), '2.8 × 10⁻⁸');
    assert.equal(fmtExp(42000), '4.2 × 10⁴');
    assert.equal(fmtExp(0), '0');
});
