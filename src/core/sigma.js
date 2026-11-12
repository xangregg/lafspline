// Noise-scale estimates: first differences (paper eq. 10) and local linear residuals (GSJS), both
// tie-aware when x is given.
//
// Tied x. With tied observations the within-x spread is pure noise whatever the trend, while differences
// between neighboring rows depend on the arbitrary order of tied rows. Given x, both estimators therefore
// use order-free pseudo-residuals, each ~ N(0, σ²) when f is smooth enough:
//   pure error   (yᵢ − ȳ_j)·√(k_j/(k_j − 1)) for each observation of an x with k_j ≥ 2 replicates,
//                weighted (k_j − 1)/k_j so a group counts its k_j − 1 degrees of freedom
//   between x    eq. 10: (ȳ_{j+1} − ȳ_j)/√(1/k_j + 1/k_{j+1});  GSJS: the local linear residual of the
//                x-means scaled by its standard deviation (weight 1 each)
// and σ̂ = weighted median |pseudo-residual| / Φ⁻¹(0.75). When the replicates carry at least as many
// degrees of freedom as there are between-x terms, the between-x terms are dropped: they only add trend
// bias (measured: steep trend with 1–4 replicates, eq. 10 gives 1.63σ, pure error 0.99σ). Without ties
// both estimators reduce exactly to their usual form.
//
// Few distinct y. When y takes only a few values (0/1 outcomes, small counts), neighbors are often equal, and more
// than half the pseudo-residuals can be exactly 0: the median is then 0 whatever the noise. In that case only, σ̂ is
// the root mean square of the same pseudo-residuals instead (as in Rice 1984, Ann. Statist. 12:1215–1230), which
// estimates σ under the same model; for Bernoulli noise it gives about √(mean p(1 − p)).

// For white noise with sd σ, yᵢ₊₁ − yᵢ ~ N(0, 2σ²): the median of |N(0, 2σ²)| is σ·√2·Φ⁻¹(0.75), so eq. 10 divides
// the median |difference| by Φ⁻¹(0.75)·√2 ≈ 0.954 (scaleOf with sd √2).
const PHI_INV_075 = 0.6744897501960817; // Φ⁻¹(0.75): median of |N(0, 1)|

/**
 * Median of an array (copies; does not modify the input). Mean of the middle pair for even length.
 *
 * @param {ArrayLike<number>} v  non-empty
 * @returns {number}
 */
export function median(v) {
    const a = Float64Array.from(v).sort();
    const m = a.length >> 1;
    return a.length % 2 === 1 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/**
 * Weighted median: the smallest value at which the cumulative weight reaches half the total.
 *
 * @param {ArrayLike<number>} v
 * @param {ArrayLike<number>} w  positive weights, same length
 * @returns {number}
 */
export function weightedMedian(v, w) {
    const idx = Array.from(v, (_, i) => i).sort((a, b) => v[a] - v[b]);
    let total = 0;
    for (const wi of w)
        total += wi;
    let acc = 0;
    for (const i of idx) {
        acc += w[i];
        if (acc >= total / 2)
            return v[i];
    }
    return v[idx[idx.length - 1]];
}

// Group sorted x into runs of ties: start[j] = first observation of group j, start[m] = n.
function groupsOf(x) {
    const n = x.length;
    const start = [0];
    for (let i = 1; i < n; i++) {
        if (x[i] !== x[i - 1])
            start.push(i);
    }
    start.push(n);
    return Int32Array.from(start);
}

/**
 * Pure-error pseudo-residuals and x-means of tied groups (see the header comment).
 *
 * @returns {{values: number[], weights: number[], mean: Float64Array, count: Int32Array, df: number}}
 */
function pureError(y, start) {
    const m = start.length - 1;
    const mean = new Float64Array(m);
    const count = new Int32Array(m);
    const values = [];
    const weights = [];
    let df = 0;
    for (let j = 0; j < m; j++) {
        const k = start[j + 1] - start[j];
        let sum = 0;
        for (let i = start[j]; i < start[j + 1]; i++)
            sum += y[i];
        mean[j] = sum / k;
        count[j] = k;
        df += k - 1;
        if (k < 2)
            continue;
        const scale = Math.sqrt(k / (k - 1));
        for (let i = start[j]; i < start[j + 1]; i++) {
            values.push(Math.abs(y[i] - mean[j]) * scale);
            weights.push((k - 1) / k);
        }
    }
    return { values, weights, mean, count, df };
}

// The scale of pseudo-residuals |r| whose sd is `sd` per unit σ: their weighted median over its expected value, or,
// when the median is 0 (few distinct y, header comment), their weighted root mean square over sd.
function scaleOf(values, weights, sd) {
    const w = weights ?? Array.from(values, () => 1);
    const med = weights ? weightedMedian(values, weights) : median(values);
    if (med > 0)
        return med / (PHI_INV_075 * sd);
    let sum = 0;
    let total = 0;
    for (let i = 0; i < values.length; i++) {
        sum += w[i] * values[i] * values[i];
        total += w[i];
    }
    return total > 0 ? Math.sqrt(sum / total) / sd : 0;
}

// Combine pure-error and between-x pseudo-residuals by the rule in the header comment.
function pooled(pe, between) {
    if (pe.df > 0 && pe.df >= between.length)
        return scaleOf(pe.values, pe.weights, 1);
    return scaleOf([...pe.values, ...between], [...pe.weights, ...between.map(() => 1)], 1);
}

/**
 * How the tie-aware estimators use replicates for this design.
 *
 * @param {ArrayLike<number>} x  sorted
 * @returns {{df: number, between: number, pureErrorOnly: boolean}}  df = replicate degrees of freedom
 *     (n − number of distinct x); between = number of between-x terms (distinct x − 1)
 */
export function replicateInfo(x) {
    const start = groupsOf(x);
    const m = start.length - 1;
    const df = x.length - m;
    return { df, between: m - 1, pureErrorOnly: df > 0 && df >= m - 1 };
}

/**
 * σ̂ = median(|y₂ − y₁|, …, |yₙ − yₙ₋₁|) / (Φ⁻¹(0.75)·√2)   (eq. 10).
 *
 * Positively biased when there is signal, which only makes Aₙ more conservative. Assumes y is
 * ordered by x. With x given and tied, uses the order-free pure-error / between-x-means form described
 * in the header comment; without ties (or without x) it is exactly eq. 10.
 *
 * @param {ArrayLike<number>} y  responses ordered by x, length ≥ 2
 * @param {ArrayLike<number>} [x]  sorted x, to handle ties
 * @returns {number}  σ̂, in the units of y
 */
export function sigmaMAD(y, x) {
    if (x) {
        const start = groupsOf(x);
        if (start.length - 1 < y.length) {
            const pe = pureError(y, start);
            const between = [];
            for (let j = 0; j + 1 < pe.mean.length; j++)
                between.push(Math.abs(pe.mean[j + 1] - pe.mean[j]) / Math.sqrt(1 / pe.count[j] + 1 / pe.count[j + 1]));
            return pooled(pe, between);
        }
    }
    const n = y.length;
    if (n < 2)
        throw new RangeError('sigmaMAD needs at least 2 observations');
    const d = new Float64Array(n - 1);
    for (let i = 1; i < n; i++)
        d[i - 1] = Math.abs(y[i] - y[i - 1]);
    return scaleOf(d, null, Math.SQRT2);
}

/**
 * Trend-robust noise scale from local linear pseudo-residuals (Gasser, Sroka & Jennen-Steinmetz 1986,
 * Biometrika 73:625–633), in a median form for robustness like eq. (10).
 *
 * For interior points with x_{i−1} < x_{i+1}, the pseudo-residual is the gap between yᵢ and the line
 * through its neighbors,
 *     eᵢ = aᵢ y_{i−1} + bᵢ y_{i+1} − yᵢ,   aᵢ = (x_{i+1} − xᵢ)/(x_{i+1} − x_{i−1}),   bᵢ = 1 − aᵢ,
 * scaled by cᵢ = 1/√(aᵢ² + bᵢ² + 1) so that cᵢeᵢ ~ N(0, σ²) when f is linear on [x_{i−1}, x_{i+1}].
 * Then σ̂ = median|cᵢeᵢ| / Φ⁻¹(0.75). A locally linear trend cancels exactly at any spacing, so unlike
 * eq. (10) a steep smooth signal does not inflate it; curvature and jumps still do, less so.
 *
 * With tied x, uses the order-free pure-error / between-x-means form in the header comment (the
 * between-x terms are these residuals computed on the x-means, scaled by their standard deviations).
 *
 * @param {ArrayLike<number>} x  sorted (ties allowed)
 * @param {ArrayLike<number>} y  responses ordered by x
 * @returns {number}  σ̂, units of y
 */
export function sigmaGSJS(x, y) {
    const start = groupsOf(x);
    const m = start.length - 1;
    if (m < y.length) {
        // tied x: pure error, plus the GSJS residuals of the x-means, each scaled by its sd
        const pe = pureError(y, start);
        const xs = Float64Array.from({ length: m }, (_, j) => x[start[j]]);
        const between = [];
        for (let j = 1; j + 1 < m; j++) {
            const span = xs[j + 1] - xs[j - 1];
            const a = (xs[j + 1] - xs[j]) / span;
            const b = 1 - a;
            const e = a * pe.mean[j - 1] + b * pe.mean[j + 1] - pe.mean[j];
            between.push(Math.abs(e) / Math.sqrt(a * a / pe.count[j - 1] + b * b / pe.count[j + 1] + 1 / pe.count[j]));
        }
        if (pe.values.length === 0 && between.length === 0)
            throw new RangeError('sigmaGSJS needs at least 3 distinct x or replicated observations');
        return pooled(pe, between);
    }
    const n = y.length;
    const r = [];
    for (let i = 1; i + 1 < n; i++) {
        const span = x[i + 1] - x[i - 1];
        if (!(span > 0))
            continue;
        const a = (x[i + 1] - x[i]) / span;
        const b = 1 - a;
        const e = a * y[i - 1] + b * y[i + 1] - y[i];
        r.push(Math.abs(e) / Math.sqrt(a * a + b * b + 1));
    }
    if (r.length === 0)
        throw new RangeError('sigmaGSJS needs at least 3 observations with distinct outer x');
    return scaleOf(r, null, 1);
}
