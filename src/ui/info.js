// Explanations on demand: a small ⓘ after a label shows its text in a popup on hover, keyboard focus or tap, so the
// controls and readouts stay compact. One shared popup, listeners delegated on the document (installed with the
// first icon). Text goes in with textContent only, through a small markup (render): each line is a paragraph,
// lines that start with '- ' make a bulleted list, and **…** is bold.

const GAP = 6; // px between the icon and the popup
const MARGIN = 8; // px kept clear of the viewport edges

let tip = null;
let shownFor = null; // the icon whose text is showing
let pinned = false; // opened by a click or tap: stays until another click, Escape or a click elsewhere

function popup() {
    if (!tip) {
        tip = document.createElement('div');
        tip.className = 'info-tip';
        tip.id = 'info-tip';
        tip.setAttribute('role', 'tooltip');
        tip.hidden = true;
        document.body.appendChild(tip);
    }
    return tip;
}

/** A run of text with **…** in bold, as text nodes and <strong>s. */
function inline(tag, text) {
    const node = document.createElement(tag);
    text.split('**').forEach((part, i) => {
        if (i % 2 === 1)
            node.appendChild(document.createElement('strong')).textContent = part;
        else if (part)
            node.appendChild(document.createTextNode(part));
    });
    return node;
}

/** The tip's markup into node: a paragraph per line, consecutive '- ' lines as one list. */
function render(node, text) {
    node.replaceChildren();
    let list = null;
    for (const line of text.split('\n')) {
        const item = /^- (.*)$/.exec(line);
        if (item) {
            list ??= node.appendChild(document.createElement('ul'));
            list.appendChild(inline('li', item[1]));
        }
        else {
            list = null;
            if (line.trim())
                node.appendChild(inline('p', line));
        }
    }
}

function show(icon) {
    const t = popup();
    render(t, icon.dataset.tip);
    t.hidden = false;
    if (shownFor && shownFor !== icon)
        shownFor.removeAttribute('aria-describedby');
    shownFor = icon;
    icon.setAttribute('aria-describedby', t.id);
    const r = icon.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const { width, height } = t.getBoundingClientRect();
    const left = Math.max(MARGIN, Math.min(r.left - MARGIN, vw - width - MARGIN));
    const below = r.bottom + GAP + height <= vh - MARGIN || r.top - GAP - height < MARGIN;
    t.style.left = `${left + window.scrollX}px`;
    t.style.top = `${(below ? r.bottom + GAP : r.top - GAP - height) + window.scrollY}px`;
}

function hide() {
    if (tip)
        tip.hidden = true;
    shownFor?.removeAttribute('aria-describedby');
    shownFor?.setAttribute('aria-expanded', 'false');
    shownFor = null;
    pinned = false;
}

let installed = false;

function install() {
    installed = true;
    const iconOf = e => e.target.closest?.('.info');
    document.addEventListener('pointerover', e => {
        const icon = iconOf(e);
        if (icon && !pinned)
            show(icon);
    });
    document.addEventListener('pointerout', e => {
        const icon = iconOf(e);
        if (icon && icon === shownFor && !pinned)
            hide();
    });
    document.addEventListener('focusin', e => {
        const icon = iconOf(e);
        if (icon)
            show(icon);
    });
    document.addEventListener('focusout', e => {
        if (iconOf(e) && iconOf(e) === shownFor && !pinned)
            hide();
    });
    const toggle = icon => {
        if (pinned && shownFor === icon) {
            hide();
            return;
        }
        show(icon);
        pinned = true;
        icon.setAttribute('aria-expanded', 'true');
    };
    document.addEventListener('click', e => {
        const icon = iconOf(e);
        if (icon) {
            // inside a <label>, a click would otherwise focus or toggle the labeled control
            e.preventDefault();
            toggle(icon);
        }
        else if (pinned && !e.target.closest?.('.info-tip')) {
            hide();
        }
    });
    document.addEventListener('keydown', e => {
        const icon = iconOf(e);
        if (icon && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            toggle(icon);
        }
        else if (e.key === 'Escape' && shownFor) {
            hide();
        }
    });
    window.addEventListener('resize', hide);
}

/**
 * An ⓘ icon that explains something.
 *
 * @param {string} text  the explanation: a paragraph per line, '- ' lines as a bulleted list, **…** bold
 * @param {string} [about]  what it explains, for screen readers ("About λ")
 * @returns {HTMLSpanElement}
 */
export function infoIcon(text, about = '') {
    if (!installed)
        install();
    const icon = document.createElement('span');
    icon.className = 'info';
    icon.textContent = 'i';
    icon.tabIndex = 0;
    icon.setAttribute('role', 'button');
    icon.setAttribute('aria-label', about ? `About ${about}` : 'Explain');
    icon.setAttribute('aria-expanded', 'false');
    icon.dataset.tip = text;
    return icon;
}

/**
 * Change an icon's explanation (for text that depends on the data), refreshing the popup if it shows.
 *
 * @param {HTMLElement} icon  from infoIcon
 * @param {string} text
 */
export function setInfo(icon, text) {
    icon.dataset.tip = text;
    if (shownFor === icon)
        show(icon);
}
