// Prefix sums and window means. Windows use 0-based inclusive indices [lo, hi].

/**
 * Prefix sums S with S[0] = 0 and S[k] = v[0] + … + v[k−1] (length n + 1).
 *
 * @param {ArrayLike<number>} v
 * @returns {Float64Array}
 */
export function prefixSums(v) {
    const n = v.length;
    const s = new Float64Array(n + 1);
    for (let i = 0; i < n; i++)
        s[i + 1] = s[i] + v[i];
    return s;
}

/**
 * Sum of v[lo..hi] (inclusive) from its prefix sums.
 *
 * @param {Float64Array} s  prefix sums from `prefixSums`
 * @param {number} lo       0-based, inclusive
 * @param {number} hi       0-based, inclusive, hi ≥ lo
 * @returns {number}
 */
export function windowSum(s, lo, hi) {
    return s[hi + 1] - s[lo];
}

/**
 * Mean of v[lo..hi] (inclusive) from its prefix sums.
 *
 * @param {Float64Array} s  prefix sums from `prefixSums`
 * @param {number} lo       0-based, inclusive
 * @param {number} hi       0-based, inclusive, hi ≥ lo
 * @returns {number}
 */
export function windowMean(s, lo, hi) {
    return (s[hi + 1] - s[lo]) / (hi - lo + 1);
}
