# Example data sets

Each `.csv` file here appears in the app's dataset list under **Examples** and in `npm run diagnose`.

## Adding one

1. Save the data as `data/<id>.csv`. The id is lower-case letters, digits and hyphens (e.g. `nile-flow.csv`).
2. Start the file with metadata lines, then an ordinary CSV (or TSV) table with a header row:

   ```
   # label: Nile annual flow at Aswan, 1871–1970
   # source: R datasets::Nile (Durbin & Koopman 2001); units 10⁸ m³
   # x: year
   # y: flow
   # note: level shift near 1898
   year,flow
   1871,1120
   1872,1160
   ```

   - `label` (required): the name in the dataset list.
   - `source` (required): where the data come from, shown with every fit, so the provenance travels with
     the data.
   - `x`, `y`: column names to use; default the first two columns.
   - `note`: anything a reader should know (known features, units, caveats such as serial correlation).
   - `intro`, `outro`: optional text for the animated export (Export → Steps as animated GIF or SVG). `intro` is the
     first step's headline, in place of the label, for a reader who does not know the data; `outro` is the last
     step's detail line, for context such as a date that bears on the fit (the Montreal Protocol, a speed limit).

   Rows with a missing or non-numeric x or y are dropped (and counted in the readouts). Tied x are fine.
3. Run `npm run data:index` to regenerate `data/index.json`. It reports any problem with a file and writes
   nothing until all files are valid. `npm test` fails if the index is out of date.

## Keeping one local

A data set that should not be published (its license does not allow redistribution) goes in `localdata/` instead:
git ignores it there, and the app lists it when run from this machine. See `localdata/README.md`.
