// Write data/cleveland-curve.csv: the artificial example from Cleveland's Visualizing Data (1993), three smooth
// pieces stitched together plus Gaussian noise. Seeded, so rerunning reproduces the file exactly.
//
//   x = 5 + 5 i / 149, i = 0 … 149
//   piece k = 0, 1, 2 covers i = 50k … 50k + 49, with z = (i − 50k) / 49 running 0 … 1:
//   f = 5 − √(1 − z), then 5 + sin(2πz), then 5 + z²/2
//   y = f + N(0, NOISE_SD²)
//
// Each piece starts or ends at f = 5, so at each seam two neighboring points both have f = 5; the first piece
// ends in a vertical tangent (the slope of −√(1 − z) is unbounded at z = 1).
//
// Usage: node tools/make-cleveland-curve.mjs

import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRng } from '../src/core/rng.js';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'data', 'cleveland-curve.csv');
const SEED = 1993;
const PIECE = 50;           // points per piece
const PIECES = 3;
const X_START = 5;
const X_SPAN = 5;
const LEVEL = 5;
const NOISE_SD = 0.2;
const DIGITS = 6;           // decimals written

const pieces = [
    z => -Math.sqrt(1 - z),
    z => Math.sin(2 * Math.PI * z),
    z => z * z / 2,
];

const n = PIECE * PIECES;
const rng = createRng(SEED);
const lines = [
    '# label: Cleveland curve, three pieces',
    '# source: artificial example from W. S. Cleveland, Visualizing Data (Hobart Press, 1993); this draw by '
        + 'tools/make-cleveland-curve.mjs, seed ' + SEED,
    '# x: x',
    '# y: y',
    `# note: three smooth pieces of 50 points each, f = 5 − √(1 − z), 5 + sin(2πz), 5 + z²/2 with z from 0 to 1 `
        + `in each; y = f + N(0, ${NOISE_SD}²); column f is the noise-free curve; the first piece ends in a `
        + `vertical tangent`,
    'x,y,f',
];
for (let i = 0; i < n; i++) {
    const k = Math.floor(i / PIECE);
    const z = (i - k * PIECE) / (PIECE - 1);
    const x = X_START + X_SPAN * i / (n - 1);
    const f = LEVEL + pieces[k](z);
    const y = f + NOISE_SD * rng.normal();
    lines.push([x, y, f].map(v => v.toFixed(DIGITS)).join(','));
}
writeFileSync(OUT, lines.join('\n') + '\n');
console.log(`wrote ${OUT} (n = ${n})`);
