import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fmtExp, sentence } from '../src/ui/format.js';

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

test('sentence: capitalizes a leading Latin letter and adds a period, but leaves a leading symbol as it is', () => {
    assert.equal(sentence('the fit stays inside'), 'The fit stays inside.');
    assert.equal(sentence('Already a sentence.'), 'Already a sentence.');
    assert.equal(sentence('is it?'), 'Is it?');
    // Regression: σ̂ came out as Σ̂ (a sum) and λ as Λ in the readouts' tips.
    assert.equal(sentence('σ̂ ÷ 0.12 from neighboring observations'), 'σ̂ ÷ 0.12 from neighboring observations.');
    assert.equal(sentence('λ = 2, singletons included'), 'λ = 2, singletons included.');
    assert.equal(sentence('τ = 3'), 'τ = 3.');
    assert.equal(sentence('2n LPs'), '2n LPs.');
    // Regression: a leading Latin symbol was capitalized too (a* read A*, c/√|I| read C/√|I|).
    assert.equal(sentence('a*, the largest ambition before the fit leaves the region'),
        'a*, the largest ambition before the fit leaves the region.');
    assert.equal(sentence('a·∫|g\'\'|^q: q = 2 is the squared differences'), 'a·∫|g\'\'|^q: q = 2 is the squared differences.');
    assert.equal(sentence('c/√|I|, units of y'), 'c/√|I|, units of y.');
    assert.equal(sentence('n = 50 observations'), 'n = 50 observations.');
    // ... while the article "a" still starts a sentence
    assert.equal(sentence('a sine of this period keeps half its amplitude'), 'A sine of this period keeps half its amplitude.');
});
