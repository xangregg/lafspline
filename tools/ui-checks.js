// In-page UI regression checks, run by `npm run check:ui` (tools/check-page.mjs --eval-file). Each failed check calls
// console.error, which check-page reports as a problem (exit status 1). Returns one {key, ok} per check.
//
// 1. A chart narrower than its margins draws no rect of negative width (an SVG error in the console). (Regression:
//    the plot's clip and hover rects got width −38 when the charts were squeezed below the margins, as they are for
//    a moment while the Claude desktop app's browser pane opens.)
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

    return results;
})()
