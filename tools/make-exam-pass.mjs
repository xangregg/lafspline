// Write data/exam-pass-simulated.csv: binary outcomes from a logistic curve, the textbook setting of logistic
// regression (pass or fail against hours of study). Seeded, so rerunning reproduces the file exactly.
//
//   hours ~ uniform on [HOURS_MIN, HOURS_MAX], sorted (each student studies a different amount)
//   p = 1 / (1 + exp(−SLOPE·(hours − MIDPOINT)))
//   pass = 1 with probability p, else 0
//
// Column p is the true probability, as cleveland-curve.csv keeps its noise-free f.
//
// Usage: node tools/make-exam-pass.mjs

import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRng } from '../src/core/rng.js';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'data', 'exam-pass-simulated.csv');
const SEED = 2026;
const N = 200; // students
const HOURS_MIN = 0.5; // hours of study
const HOURS_MAX = 6.5; // hours of study
const MIDPOINT = 3; // hours at which p = ½
const SLOPE = 1.5; // per hour: the log odds of passing rise by this much for each hour
const DIGITS = 2; // decimals written for hours
const P_DIGITS = 4; // decimals written for p

const rng = createRng(SEED);
const hours = Array.from({ length: N }, () => HOURS_MIN + (HOURS_MAX - HOURS_MIN) * rng.uniform())
    .map(h => Number(h.toFixed(DIGITS)))
    .sort((a, b) => a - b);
const lines = [
    '# label: Exam pass or fail by hours of study (simulated)',
    `# source: simulated by tools/make-exam-pass.mjs, seed ${SEED}`,
    '# x: hours',
    '# y: pass',
    `# note: ${N} students, hours uniform on ${HOURS_MIN}–${HOURS_MAX}; pass is 1 or 0, drawn with probability `
        + `p = 1/(1 + e^(−${SLOPE}(hours − ${MIDPOINT}))), a logistic curve through ½ at ${MIDPOINT} hours; column p `
        + 'is the true probability. The noise is Bernoulli: its spread p(1 − p) is largest at the midpoint and '
        + 'shrinks toward the ends',
    '# intro: Pass or fail by hours of study: each dot is one student (simulated)',
    `# outro: The truth: P(pass) = 1/(1 + e^(−${SLOPE}(hours − ${MIDPOINT}))), a logistic curve`,
    'hours,pass,p',
];
let passed = 0;
for (const h of hours) {
    const p = 1 / (1 + Math.exp(-SLOPE * (h - MIDPOINT)));
    const pass = rng.uniform() < p ? 1 : 0;
    passed += pass;
    lines.push(`${h.toFixed(DIGITS)},${pass},${p.toFixed(P_DIGITS)}`);
}
writeFileSync(OUT, lines.join('\n') + '\n');
console.log(`wrote ${OUT} (n = ${N}, ${passed} passed)`);
