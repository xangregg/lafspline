// One shared tooltip element. Content is built from [value, label] rows with textContent, since labels
// may come from user CSV headers.

let el = null;

function getEl() {
    if (!el) {
        el = document.createElement('div');
        el.className = 'tooltip';
        el.hidden = true;
        document.body.appendChild(el);
    }
    return el;
}

/**
 * Show the tooltip near a page position.
 *
 * @param {number} pageX
 * @param {number} pageY
 * @param {string} title  first line
 * @param {Array<[string, string, string?]>} rows  [value, label, optional swatch color]
 * @param {string} [note]  optional last line
 * @param {boolean} [alert=false]  style the note as a warning
 */
export function showTooltip(pageX, pageY, title, rows, note, alert = false) {
    const tip = getEl();
    tip.replaceChildren();
    const head = document.createElement('div');
    head.className = 'tooltip-title';
    head.textContent = title;
    tip.appendChild(head);
    for (const [value, label, color] of rows) {
        const row = document.createElement('div');
        row.className = 'tooltip-row';
        if (color) {
            const key = document.createElement('span');
            key.className = 'tooltip-key';
            key.style.background = color;
            row.appendChild(key);
        }
        const v = document.createElement('strong');
        v.textContent = value;
        const l = document.createElement('span');
        l.textContent = ` ${label}`;
        row.append(v, l);
        tip.appendChild(row);
    }
    if (note) {
        const n = document.createElement('div');
        n.className = alert ? 'tooltip-note alert' : 'tooltip-note';
        n.textContent = note;
        tip.appendChild(n);
    }
    tip.hidden = false;
    const pad = 14; // px from the pointer
    const { width, height } = tip.getBoundingClientRect();
    let left = pageX + pad;
    if (left + width > window.scrollX + document.documentElement.clientWidth)
        left = pageX - pad - width;
    let top = pageY + pad;
    if (top + height > window.scrollY + document.documentElement.clientHeight)
        top = pageY - pad - height;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
}

export function hideTooltip() {
    getEl().hidden = true;
}
