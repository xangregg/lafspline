// Thin, pure wrapper around a loaded HiGHS instance. The caller supplies the instance (Node tests load
// it from the `highs` dev dependency, the browser worker from vendor/highs/highs.mjs), so this module
// has no imports and runs unchanged in both places.

/**
 * @typedef {object} LPSolution
 * @property {string} status       HiGHS model status, e.g. 'Optimal', 'Infeasible'
 * @property {number} objective    objective value (NaN if HiGHS reported none)
 * @property {string[]} names      column names in HiGHS column order
 * @property {Float64Array} primal primal values aligned with `names`
 * @property {number} ms           wall-clock solve time, milliseconds
 */

/**
 * Convert the result of `highs.solve()` (columns keyed by name) into column-ordered arrays.
 *
 * Column order is HiGHS's internal index, which for LP text is order of first appearance; callers
 * that need a particular variable should look it up by name rather than assume a position.
 *
 * @param {object} raw  object returned by `highs.solve(lpText, options)`
 * @returns {Omit<LPSolution, 'ms'>}
 */
export function normalizeSolution(raw) {
    const columns = Object.values(raw.Columns ?? {});
    columns.sort((a, b) => a.Index - b.Index);
    const names = columns.map(col => col.Name);
    const primal = new Float64Array(columns.length);
    for (let i = 0; i < columns.length; i++)
        primal[i] = columns[i].Primal;
    const objective = typeof raw.ObjectiveValue === 'number' ? raw.ObjectiveValue : NaN;
    return { status: raw.Status, objective, names, primal };
}

/**
 * Solve a CPLEX-LP-format problem with a loaded HiGHS instance.
 *
 * @param {{solve: function(string, object): object}} highs  instance from the HiGHS loader
 * @param {string} lp          problem in CPLEX LP format
 * @param {object} [options]   HiGHS options; solver log output is off unless overridden
 * @returns {LPSolution}
 */
export function solveLP(highs, lp, options = {}) {
    const start = performance.now();
    const raw = highs.solve(lp, { output_flag: false, ...options });
    const ms = performance.now() - start;
    return { ...normalizeSolution(raw), ms };
}

/**
 * Map a solution's primal values by column name.
 *
 * @param {{names: string[], primal: Float64Array}} solution
 * @returns {Map<string, number>}
 */
export function primalByName(solution) {
    const map = new Map();
    for (let i = 0; i < solution.names.length; i++)
        map.set(solution.names[i], solution.primal[i]);
    return map;
}

/**
 * Name of a HiGHS model status code, e.g. 7 → 'optimal'.
 *
 * @param {{constants: {modelStatus: Record<string, number>}}} highs
 * @param {number} code
 * @returns {string}
 */
export function modelStatusName(highs, code) {
    for (const [name, value] of Object.entries(highs.constants.modelStatus)) {
        if (value === code)
            return name;
    }
    return `status ${code}`;
}

/**
 * Solve a model given as HiGHS ModelData (sparse arrays) with a loaded HiGHS instance. The native
 * model is always disposed.
 *
 * @param {object} highs       instance from the HiGHS loader
 * @param {object} model       ModelData: numCols, numRows, colCost, colLower, colUpper, rowLower,
 *                             rowUpper, matrix {format, numRows, numCols, starts, indices, values}
 * @param {object} [options]   HiGHS options; solver log output is off unless overridden
 * @param {string[]} [infoNames]  HiGHS info items to report after the run, e.g. 'qp_iteration_count' (an item the
 *     run did not produce is reported as NaN)
 * @returns {{status: string, objective: number, x: Float64Array|null, ms: number, info: Object<string, number>}}
 *     x = column values (null unless optimal); ms = wall-clock time including model setup
 */
export function solveModel(highs, model, options = {}, infoNames = []) {
    const start = performance.now();
    return highs.withModel(model, mdl => {
        mdl.options.set({ output_flag: false, ...options });
        try {
            mdl.run();
        }
        catch (err) {
            // highs-js throws when HiGHS's run returns an error, e.g. model status 'solveError' when the simplex
            // diverges on a badly conditioned LP; the model status is still readable, so report it like any other
            // status and let the caller fall back instead of losing the whole computation
            if (err?.operation !== 'run' || typeof err.status !== 'number')
                throw err;
        }
        const code = mdl.getModelStatus();
        const optimal = code === highs.constants.modelStatus.optimal;
        const x = optimal ? Float64Array.from(mdl.getSolution().colValue) : null;
        const objective = optimal ? mdl.getObjectiveValue() : NaN;
        const info = {};
        for (const name of infoNames) {
            try {
                info[name] = Number(mdl.info.get(name));
            }
            catch {
                info[name] = NaN;
            }
        }
        return { status: modelStatusName(highs, code), objective, x, ms: performance.now() - start, info };
    });
}
