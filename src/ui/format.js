// Number and text formats shared by the controls and the readouts.

const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const PLAIN_MIN = 1e-3; // below this magnitude (and from PLAIN_MAX up) a power of ten is clearer than digits
const PLAIN_MAX = 1e4;

/**
 * Two significant digits; very small or large values with the power of ten written out: 2.8 × 10⁻⁸, 0.0012, 530,
 * 4.2 × 10⁵.
 *
 * @param {number} v
 * @returns {string}
 */
export function fmtExp(v) {
    const m = Math.abs(v);
    if (v === 0 || !Number.isFinite(v))
        return String(v);
    if (m >= PLAIN_MIN && m < PLAIN_MAX)
        return String(Number(v.toPrecision(2))); // toPrecision alone writes 530 as 5.3e+2
    return v.toExponential(1).replace(/e\+?(-?)(\d+)/,
        (_, sign, e) => ` × 10${sign ? '⁻' : ''}${[...e].map(d => SUPERSCRIPT[d]).join('')}`);
}

/**
 * A readout note as a sentence: a leading word capitalized and a period added. A note that starts with a symbol
 * keeps its case: capitalized, σ̂ would read Σ̂ (a sum), λ Λ, a* A* and c/√|I| C/√|I|. A word is two Latin letters,
 * or the article "a" before a space; a lone letter otherwise is taken for a symbol.
 *
 * @param {string} text
 * @returns {string}
 */
export function sentence(text) {
    const t = /^(?:[a-z]{2}|a )/.test(text) ? text.charAt(0).toUpperCase() + text.slice(1) : text;
    return /[.!?]$/.test(t) ? t : `${t}.`;
}
