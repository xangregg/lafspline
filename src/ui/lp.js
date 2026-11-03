// Promise API over the solver worker (src/lp-worker.js). One worker, requests matched by id. A solve cannot be
// interrupted from outside the worker, so cancelAll() terminates it; the next request starts a fresh one.

const WORKER_URL = new URL('../lp-worker.js', import.meta.url);

let worker = null;
let nextId = 1;
const pending = new Map();

function getWorker() {
    if (worker)
        return worker;
    worker = new Worker(WORKER_URL, { type: 'module' });
    worker.addEventListener('message', event => {
        const { id, error, progress, ...result } = event.data;
        const entry = pending.get(id);
        if (!entry)
            return;
        if (progress !== undefined) {
            entry.onProgress?.(progress);
            return;
        }
        pending.delete(id);
        if (error)
            entry.reject(new Error(error));
        else
            entry.resolve(result);
    });
    worker.addEventListener('error', event => {
        // Worker failed to load or threw outside a message handler: fail everything in flight.
        const err = new Error(`LP worker error: ${event.message ?? 'failed to load'}`);
        for (const entry of pending.values())
            entry.reject(err);
        pending.clear();
        worker = null;
    });
    return worker;
}

function request(message, onProgress) {
    const w = getWorker();
    const id = nextId++;
    return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject, onProgress });
        w.postMessage({ id, ...message });
    });
}

/**
 * Stop everything in flight: terminate the worker and reject pending requests with an error whose
 * `cancelled` property is true. Cheap when idle.
 */
export function cancelAll() {
    if (!worker)
        return;
    worker.terminate();
    worker = null;
    const err = Object.assign(new Error('cancelled'), { cancelled: true });
    for (const entry of pending.values())
        entry.reject(err);
    pending.clear();
}

/**
 * Solve a model given as HiGHS ModelData (src/core/lp-solve.js solveModel) in the worker.
 *
 * @param {object} model       ModelData (with a Hessian for a QP)
 * @param {object} [options]   HiGHS options
 * @returns {Promise<{status: string, objective: number, x: Float64Array|null, ms: number}>}
 */
export function solveModelInWorker(model, options = {}) {
    return request({ task: 'model', model, options });
}

/**
 * Solve an LP in the worker.
 *
 * @param {string} lp          problem in CPLEX LP format
 * @param {object} [options]   HiGHS options
 * @returns {Promise<{status: string, objective: number, names: string[], primal: Float64Array, ms: number}>}
 */
export function solveLP(lp, options = {}) {
    return request({ lp, options });
}
