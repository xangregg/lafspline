// Undo and redo for the settings: a list of snapshots and a position in it. No DOM, so it runs in Node for tests.
//
// A change records a snapshot. Changes to the same settings in quick succession (a slider dragged, a number typed)
// coalesce into one entry, so one undo takes back the whole gesture. Recording after an undo drops the entries
// ahead (the redo branch), as editors do.

export const COALESCE_MS = 1000; // ms: a change to the same settings within this of the last one joins its entry
const DEFAULT_LIMIT = 200; // entries kept; the oldest go first

/** The keys whose values differ between two snapshots (values compared as JSON). */
function changedKeys(a, b) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    return [...keys].filter(k => JSON.stringify(a[k]) !== JSON.stringify(b[k])).sort();
}

/**
 * A settings history starting at a snapshot.
 *
 * @param {object} initial  a snapshot: setting name → value
 * @param {{limit?: number}} [opts]  limit: entries kept (DEFAULT_LIMIT)
 * @returns {{current: function(): object, canUndo: function(): boolean, canRedo: function(): boolean,
 *     record: function(object, number, {coalesce?: boolean}=): boolean, undo: function(): object|null,
 *     redo: function(): object|null}}  record(snapshot, at ms, {coalesce}) returns false when nothing changed;
 *     coalesce (default true) lets it join the last entry; undo and redo return the snapshot to restore, or null
 */
export function createHistory(initial, opts = {}) {
    const limit = opts.limit ?? DEFAULT_LIMIT;
    // keys: the settings this entry changed from the one before; at: when it was last recorded
    const entries = [{ snapshot: { ...initial }, keys: '', at: -Infinity }];
    let index = 0;
    return {
        current: () => ({ ...entries[index].snapshot }),
        canUndo: () => index > 0,
        canRedo: () => index < entries.length - 1,
        record(snapshot, at, { coalesce = true } = {}) {
            const top = entries[index];
            const changed = changedKeys(top.snapshot, snapshot);
            if (changed.length === 0)
                return false;
            const atEnd = index === entries.length - 1;
            // a continued gesture: the same settings again, soon, on the newest entry (never the initial one)
            if (coalesce && atEnd && index > 0 && at - top.at < COALESCE_MS) {
                const fromPrevious = changedKeys(entries[index - 1].snapshot, snapshot);
                if (fromPrevious.join() === top.keys) {
                    top.snapshot = { ...snapshot };
                    top.at = at;
                    return true;
                }
                if (fromPrevious.length === 0) {
                    // the gesture went back to where it started: no change left to undo
                    entries.pop();
                    index--;
                    return true;
                }
            }
            entries.splice(index + 1);
            entries.push({ snapshot: { ...snapshot }, keys: changed.join(), at });
            if (entries.length > limit)
                entries.shift();
            index = entries.length - 1;
            return true;
        },
        undo() {
            if (index === 0)
                return null;
            index--;
            return { ...entries[index].snapshot };
        },
        redo() {
            if (index === entries.length - 1)
                return null;
            index++;
            return { ...entries[index].snapshot };
        },
    };
}
