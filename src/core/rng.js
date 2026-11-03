// Seeded pseudo-random numbers. All randomness in the app and tests goes through here so that results
// are reproducible from a seed. Generator: xoshiro128** (period 2^128 − 1), state seeded by splitmix32
// so that nearby integer seeds give unrelated streams.

function splitmix32(seed) {
    let s = seed >>> 0;
    return function next() {
        s = (s + 0x9e3779b9) >>> 0;
        let z = s;
        z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
        z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
        return (z ^ (z >>> 16)) >>> 0;
    };
}

function rotl(x, k) {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/**
 * @typedef {object} Rng
 * @property {function(): number} uniform  uniform on [0, 1), 32-bit resolution
 * @property {function(): number} normal   standard normal N(0, 1)
 */

/**
 * Create a seeded generator.
 *
 * @param {number} seed  any number; its low 32 bits are used
 * @returns {Rng}
 */
export function createRng(seed) {
    const init = splitmix32(seed);
    let s0 = init();
    let s1 = init();
    let s2 = init();
    let s3 = init();
    if ((s0 | s1 | s2 | s3) === 0)
        s0 = 1; // all-zero state is a fixed point

    function nextUint32() {
        const result = Math.imul(rotl(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
        const t = (s1 << 9) >>> 0;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = rotl(s3, 11);
        return result;
    }

    function uniform() {
        return nextUint32() / 4294967296;
    }

    // Box–Muller, caching the second variate of each pair.
    let spare = NaN;
    function normal() {
        if (!Number.isNaN(spare)) {
            const z = spare;
            spare = NaN;
            return z;
        }
        const u1 = 1 - uniform(); // (0, 1], avoids log(0)
        const u2 = uniform();
        const r = Math.sqrt(-2 * Math.log(u1));
        spare = r * Math.sin(2 * Math.PI * u2);
        return r * Math.cos(2 * Math.PI * u2);
    }

    return { uniform, normal };
}

/**
 * Draw n independent standard normals.
 *
 * @param {number} n
 * @param {Rng} rng
 * @returns {Float64Array}
 */
export function normals(n, rng) {
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++)
        z[i] = rng.normal();
    return z;
}
