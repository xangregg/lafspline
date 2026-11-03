// Chart colors come from CSS custom properties (styles.css) so light/dark values live in one place.
// Canvas drawing can't use var(), so read the resolved values at render time.

const NAMES = [
    'surface', 'ink', 'ink-2', 'muted', 'grid', 'axis',
    'series-1', 'series-2', 'highlight', 'violation',
    'seq-0', 'seq-1', 'seq-2', 'seq-3', 'seq-4', 'band',
];

/**
 * Resolved chart colors for the current color scheme.
 *
 * @returns {Record<string, string>}  keys as in NAMES, e.g. colors['series-1']
 */
export function themeColors() {
    const style = getComputedStyle(document.documentElement);
    const out = {};
    for (const name of NAMES)
        out[name] = style.getPropertyValue(`--${name}`).trim();
    return out;
}

/**
 * Resolved value of one CSS custom property, e.g. cssColor('fit-taut').
 *
 * @param {string} name  without the leading --
 * @returns {string}
 */
export function cssColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}

/**
 * Call `fn` when the OS color scheme changes.
 *
 * @param {function(): void} fn
 */
export function onThemeChange(fn) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', fn);
}
