// Module Web Worker that owns the HiGHS (WebAssembly) solver, keeping solves off the UI thread.
// Started from src/ui/lp.js with `new Worker(url, {type: 'module'})`.
//
// Protocol:
//   {id, lp, options}                  → {id, status, objective, names, primal: Float64Array, ms}
//   {id, task: 'model', model, options} → {id, status, objective, x: Float64Array|null, ms}
//                                        (HiGHS ModelData, src/core/lp-solve.js solveModel)
//   failures                           → {id, error}
//
// highs.mjs locates highs.wasm next to itself via import.meta.url, so no locateFile is needed.

import loadHighs from '../vendor/highs/highs.mjs';
import { solveLP, solveModel } from './core/lp-solve.js';

let highsPromise = null;

function getHighs() {
    if (!highsPromise)
        highsPromise = loadHighs();
    return highsPromise;
}

self.addEventListener('message', async event => {
    const { id, task, lp, model, options } = event.data;
    try {
        const highs = await getHighs();
        if (task === 'model') {
            const result = solveModel(highs, model, options);
            self.postMessage({ id, ...result }, result.x ? [result.x.buffer] : []);
        }
        else {
            const result = solveLP(highs, lp, options);
            self.postMessage({ id, ...result }, [result.primal.buffer]);
        }
    }
    catch (err) {
        // A failed solve can leave the Wasm heap in a bad state (e.g. an LP parse error aborts it);
        // start fresh next time.
        highsPromise = null;
        self.postMessage({ id, error: String(err?.message ?? err) });
    }
});
