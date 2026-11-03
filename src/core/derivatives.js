// Discrete derivatives by divided differences (PLAN.md §1.6, paper eqs. 44–46), and grouping of tied x.
//
// Dᵏ_j = k!·g[t_j, …, t_{j+k}] (k-th divided difference scaled by k!), computed by the recursion
//     Dᵏ_j = k·(Dᵏ⁻¹_{j+1} − Dᵏ⁻¹_j) / (t_{j+k} − t_j),   D⁰_j = g_j.
// For equal spacing h this is Δᵏg/hᵏ, i.e. the paper's Δ⁽ᵏ⁾ = n(·) iterated when h = 1/n, and it is
// exact (= g⁽ᵏ⁾) for polynomials of degree ≤ k at any spacing.

/**
 * Group sorted x into runs of equal values (ties).
 *
 * @param {ArrayLike<number>} x  sorted ascending
 * @returns {{group: Int32Array, xs: Float64Array, count: Int32Array}}  group[i] = index of x[i]'s
 *     distinct value; xs = distinct values; count[j] = observations at xs[j]
 */
export function tieGroups(x) {
    const n = x.length;
    const group = new Int32Array(n);
    const xs = [];
    const count = [];
    for (let i = 0; i < n; i++) {
        if (i === 0 || x[i] !== x[i - 1]) {
            xs.push(x[i]);
            count.push(0);
        }
        group[i] = xs.length - 1;
        count[xs.length - 1]++;
    }
    return { group, xs: Float64Array.from(xs), count: Int32Array.from(count) };
}

/**
 * Coefficients of Dᵏ_j as a linear combination of g_j, …, g_{j+k}, for every j = 0..m−k−1.
 *
 * @param {ArrayLike<number>} t  distinct abscissae, strictly increasing, length m
 * @param {number} k  derivative order, 0 ≤ k < m
 * @returns {Float64Array[]}  rows[j] has length k + 1; Dᵏ_j = Σ_r rows[j][r]·g_{j+r}
 */
export function differenceRows(t, k) {
    const m = t.length;
    let rows = Array.from({ length: m }, () => Float64Array.of(1));
    for (let order = 1; order <= k; order++) {
        const next = [];
        for (let j = 0; j + order < m; j++) {
            const scale = order / (t[j + order] - t[j]);
            const row = new Float64Array(order + 1);
            const a = rows[j];
            const b = rows[j + 1];
            for (let r = 0; r < order; r++) {
                row[r] -= scale * a[r];
                row[r + 1] += scale * b[r];
            }
            next.push(row);
        }
        rows = next;
    }
    return rows;
}

/**
 * Numeric k-th derivative values Dᵏ_j of g at distinct abscissae t.
 *
 * @param {ArrayLike<number>} t  strictly increasing, length m
 * @param {ArrayLike<number>} g  values at t
 * @param {number} k
 * @returns {Float64Array}  length m − k
 */
export function derivative(t, g, k) {
    let d = Float64Array.from(g);
    for (let order = 1; order <= k; order++) {
        const next = new Float64Array(d.length - 1);
        for (let j = 0; j < next.length; j++)
            next[j] = order * (d[j + 1] - d[j]) / (t[j + order] - t[j]);
        d = next;
    }
    return d;
}

/**
 * Total variation of the k-th derivative, TV(g⁽ᵏ⁾) = Σ_j |Dᵏ_{j+1} − Dᵏ_j| (≈ ∫|g⁽ᵏ⁺¹⁾|). This is the
 * true total variation; the paper's eq. (44) is n times larger (PLAN.md Known issue 10).
 *
 * @param {ArrayLike<number>} t
 * @param {ArrayLike<number>} g
 * @param {number} k
 * @returns {number}
 */
export function totalVariation(t, g, k) {
    const d = derivative(t, g, k);
    let tv = 0;
    for (let j = 1; j < d.length; j++)
        tv += Math.abs(d[j] - d[j - 1]);
    return tv;
}

/**
 * Supremum norm of the k-th derivative, max_j |Dᵏ_j| (eq. 46).
 *
 * @param {ArrayLike<number>} t
 * @param {ArrayLike<number>} g
 * @param {number} k
 * @returns {number}
 */
export function supNorm(t, g, k) {
    let s = 0;
    for (const v of derivative(t, g, k))
        s = Math.max(s, Math.abs(v));
    return s;
}
