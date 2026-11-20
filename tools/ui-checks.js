// In-page UI regression checks, run by `npm run check:ui` (tools/check-page.mjs --eval-file). Each failed check calls
// console.error, which check-page reports as a problem (exit status 1). Returns one {key, ok} per check.
//
// 1. A chart narrower than its margins draws no rect of negative width (an SVG error in the console). (Regression:
//    the plot's clip and hover rects got width −38 when the charts were squeezed below the margins, as they are for
//    a moment while the Claude desktop app's browser pane opens.)
// 2. A zoomed view of the main chart survives a change of a fit setting. (Regression: simulated data are rebuilt on
//    every change, and the chart took the new arrays for a new data set and reset the view.)
// 3. An upper bound a newly computed band lies above says "No fit" in the readouts, and clearing it brings the fit
//    back. (Regression: the readouts read the region's ladder, which that path never computed, and threw.)
// 4. Anchors typed in the Fit section reach the chart as diamonds and the readouts as a row.
(async () => {
    const wait = ms => new Promise(r => setTimeout(r, ms));
    const results = [];

    const charts = document.getElementById('charts');
    charts.style.width = '40px';
    window.dispatchEvent(new Event('resize')); // the charts redraw on resize
    await wait(400);
    const negative = [...document.querySelectorAll('#charts rect')]
        .filter(r => parseFloat(r.getAttribute('width')) < 0).length;
    charts.style.width = '';
    window.dispatchEvent(new Event('resize'));
    await wait(400);
    if (negative)
        console.error(`${negative} rect(s) with a negative width in a chart narrower than its margins`);
    results.push({ key: 'narrow charts', ok: negative === 0 });

    const svg = () => document.querySelector('#main-chart .plot > svg:not(.grid-layer)');
    const yTicks = () => [...svg().querySelectorAll('.axis')][1].textContent;
    const full = yTicks();
    const hit = svg().querySelector('.axis-hit.y');
    const box = hit.getBoundingClientRect();
    hit.dispatchEvent(new WheelEvent('wheel', { deltaY: -300, clientX: box.x + 5, clientY: box.y + box.height / 2,
        bubbles: true, cancelable: true }));
    await wait(200);
    const zoomed = yTicks();
    const ranges = [...document.querySelectorAll('#controls input[type=range]')];
    const slider = ranges[ranges.length - 1];
    const before = slider.value;
    slider.value = String(Number(slider.value) + 3 * Number(slider.step || 0.01));
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.dispatchEvent(new Event('change', { bubbles: true }));
    await wait(3000);
    const kept = zoomed !== full && yTicks() === zoomed;
    if (!kept)
        console.error(`the zoomed view was not kept across a settings change (${full} → ${zoomed} → ${yTicks()})`);
    results.push({ key: 'zoom kept', ok: kept });
    slider.value = before;
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.dispatchEvent(new Event('change', { bubbles: true }));
    await wait(3000);

    const SETTLE = 3000; // ms: inputs are debounced, then the fit is recomputed
    const readoutText = () => document.getElementById('readouts').innerText;
    const type = async (input, value) => {
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        await wait(SETTLE);
    };
    const field = label => [...document.querySelectorAll('#controls .field')]
        .find(f => f.querySelector('.field-label')?.textContent.trim().startsWith(label));
    const upper = field('bounds')?.querySelectorAll('input')[1];
    if (!upper) {
        console.error('no bounds inputs in the Fit section');
        results.push({ key: 'bounds', ok: false });
    }
    else {
        // a new region (another coverage) computed while the bounds fail, so nothing about it is cached yet
        const alpha = field('coverage')?.querySelector('input');
        const alphaBefore = alpha.value;
        await type(upper, '-1e9');
        await type(alpha, '0.93');
        const refused = /No fit: the region lies above the upper bound/.test(readoutText());
        await type(upper, '');
        await type(alpha, alphaBefore);
        const back = !/No fit/.test(readoutText()) && /fit [∈∉]|fit within/.test(readoutText());
        if (!refused || !back)
            console.error(`bounds: refused ${refused}, fit back after clearing ${back}`);
        results.push({ key: 'bounds', ok: refused && back });
    }
    const anchorsInput = field('anchors')?.querySelector('input');
    if (!anchorsInput) {
        console.error('no anchors input in the Fit section');
        results.push({ key: 'anchors', ok: false });
    }
    else {
        const box = svg().querySelector('.anchor-marks');
        await type(anchorsInput, '0.5, 0; 0.25, 0.1');
        const drawn = box.querySelectorAll('path').length;
        const row = readoutText().includes('(0.25, 0.1), (0.5, 0)'); // sorted by x
        await type(anchorsInput, '');
        const cleared = box.querySelectorAll('path').length === 0;
        if (drawn !== 2 || !row || !cleared)
            console.error(`anchors: ${drawn} drawn, readout row ${row}, cleared ${cleared}`);
        results.push({ key: 'anchors', ok: drawn === 2 && row && cleared });
    }

    return results;
})()
