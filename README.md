# lafspline: least ambitious fit, P-spline version

**Live:** https://xangregg.github.io/lafspline/

lafspline finds the simplest curve the data allow. It works in two phases, both with P-splines (B-splines on
equally spaced knots with a penalty on the differences of their coefficients):

1. **A confidence region** for the regression function, from the data alone: a bootstrap band (the default) or a
   P-spline band around a first P-spline fit, or Aₙ, the multiresolution region of Davies, Kovac & Meise, which
   bounds the residuals' sums over intervals at every scale.
2. **The least ambitious fit**: a second P-spline, penalized severely for leaving the region, slightly for its
   distance from the region's center, and for *ambition*, its departure from a straight line or a constant. The
   ambition can be the squared differences, the number of bends or steps, shaped bends (that number with costs for
   rounding, extremes and inflections), or the differences' sizes to a power.

Every fit reports which region it was checked against, whether it stays inside, and what the region covers under
which assumptions; approximate methods say so. The page's ⓘ tips and its About panel explain each setting.

It runs entirely in the browser: no server, no build step, no data leaves the page. Choose an example data set or
a simulated signal, or paste or drop your own CSV. Settings live in the address bar, so a link shares a view; the
Gallery shows several data sets with the same settings; Export saves the data and fits as CSV, the chart as SVG,
PNG or JPEG, and the fit built up step by step as an animated GIF or SVG.

## Running it locally

Plain ES modules, served as static files (Node 20 or later for the tests):

```
npm run serve      # python3 -m http.server 8002; then open http://localhost:8002/
npm test           # node --test
```

d3 comes from jsDelivr through the import map in `index.html`, so the page needs a network connection and a browser
with import maps (Chrome or Edge 89, Firefox 108, Safari 16.4 or later). `npm install` is needed only for the
tests and tools that use the HiGHS solver.

## Layout

- `src/core/`: the algorithms, pure JavaScript that runs unchanged in Node and the browser (P-splines, the bands,
  Aₙ, the phase-2 fits, the GIF encoder).
- `src/ui/`: the page (charts with d3, controls, readouts, link state, exports).
- `data/`: the example data sets, CSV with `# label:` and `# source:` lines; see `data/README.md` to add one.
- `test/`: `node:test` tests, some against fixtures computed in R.
- `tools/`: simulations and checks behind the coverage figures the page cites, and a headless-Chrome page check.
- `PLAN.md`: the design, decisions and studies; `VENDORED.md`: code adapted from the earlier project it follows.

## Data

The example data sets are third-party measurements, included for demonstration with their sources cited in each
file and shown with every fit. They are not covered by lafspline's license; their own terms apply.

## Credits

lafspline follows [tautfit](https://github.com/xangregg/tautfit), an implementation of P. L. Davies, A. Kovac and
M. Meise (2009), *Ann. Statist.* 37(5B):2597–2625. P-splines after P. H. C. Eilers and B. D. Marx (1996), *Statist.
Sci.* 11(2):89–121, and their book *Practical Smoothing: The Joys of P-splines* (2021). Further references are in
the About panel and PLAN.md.

## License

[0BSD](LICENSE). The vendored HiGHS build in `vendor/highs/` is under its own MIT license.
