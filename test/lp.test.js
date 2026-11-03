import { test } from 'node:test';
import assert from 'node:assert/strict';
import loadHighs from 'highs';
import { solveLP, solveModel, primalByName } from '../src/core/lp-solve.js';

const TOL = 1e-9; // absolute tolerance on objective and primal values

// The same small LP in both forms: maximize x + 2y subject to x + y ≤ 4, x + 3y ≤ 6, x, y ≥ 0.
// Optimum at the vertex (3, 1), objective 5.
const SMOKE_LP = 'Maximize\n obj: x + 2 y\nSubject To\n c1: x + y <= 4\n c2: x + 3 y <= 6\nEnd';

test('HiGHS solves the smoke LP from text', async () => {
    const highs = await loadHighs();
    const sol = solveLP(highs, SMOKE_LP);
    assert.equal(sol.status, 'Optimal');
    assert.ok(Math.abs(sol.objective - 5) < TOL, `objective ${sol.objective}`);
    assert.deepEqual(sol.names, ['x', 'y']);
    assert.ok(sol.primal instanceof Float64Array);
    const byName = primalByName(sol);
    assert.ok(Math.abs(byName.get('x') - 3) < TOL);
    assert.ok(Math.abs(byName.get('y') - 1) < TOL);
});

test('HiGHS reports infeasibility rather than a bogus solution', async () => {
    const highs = await loadHighs();
    const sol = solveLP(highs, 'Minimize\n obj: x\nSubject To\n c1: x >= 2\n c2: x <= 1\nEnd');
    assert.equal(sol.status, 'Infeasible');
});

test('solveModel solves the smoke LP given as ModelData', async () => {
    const highs = await loadHighs();
    const model = {
        sense: highs.constants.objectiveSense.maximize,
        numCols: 2,
        numRows: 2,
        colCost: [1, 2],
        colLower: [0, 0],
        colUpper: [highs.infinity, highs.infinity],
        rowLower: [-highs.infinity, -highs.infinity],
        rowUpper: [4, 6],
        matrix: {
            format: 'csc',
            numRows: 2, numCols: 2,
            starts: [0, 2, 4], indices: [0, 1, 0, 1], values: [1, 1, 1, 3],
        },
    };
    const sol = solveModel(highs, model);
    assert.equal(sol.status, 'optimal');
    assert.ok(Math.abs(sol.objective - 5) < TOL);
    assert.ok(Math.abs(sol.x[0] - 3) < TOL && Math.abs(sol.x[1] - 1) < TOL);
});
