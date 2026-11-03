// Scaffold entry point (phase 0): loads an example data set and draws it with the chart copied from the earlier project,
// and runs a smoke LP in the solver worker so the vendored HiGHS is exercised in the browser. Replaced in
// phase 1 by the real app.

import { readDataset } from '../core/datasets.js';
import { createMainChart } from './chart.js';
import { cellEdges } from './layout.js';
import { onThemeChange } from './theme.js';
import { solveLP } from './lp.js';

const EXAMPLE = 'data/nile-flow.csv';
const SMOKE_LP = 'Maximize\n obj: x + 2 y\nSubject To\n c1: x + y <= 4\n c2: x + 3 y <= 6\nEnd';

const statusEl = document.getElementById('status');

async function start() {
    const chart = createMainChart(document.getElementById('main-chart'));
    const response = await fetch(EXAMPLE);
    if (!response.ok)
        throw new Error(`${EXAMPLE}: HTTP ${response.status}`);
    const d = readDataset(await response.text());
    if (d.errors.length)
        throw new Error(d.errors.join('; '));
    const model = { x: d.x, y: d.y, f: null, fits: [], extremes: [], edges: cellEdges(d.x), band: null,
        boxesLabel: null };
    chart.render(model);
    onThemeChange(() => chart.render(model));

    const sol = await solveLP(SMOKE_LP);
    statusEl.textContent = `${d.meta.label}: n = ${d.n}. Solver check: ${sol.status}, objective ${sol.objective}.`;
    document.body.dataset.ready = 'true';
}

start().catch(err => {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
});
