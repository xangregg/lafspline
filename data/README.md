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

   Rows with a missing or non-numeric x or y are dropped (and counted in the readouts). Tied x are fine.
3. Run `npm run data:index` to regenerate `data/index.json`. It reports any problem with a file and writes
   nothing until all files are valid. `npm test` fails if the index is out of date.

The fits assume independent noise with constant variance. Real data often have serial correlation or
noise that changes with x; note it, since the region's coverage claim does not hold then.

## License

The files in `data/` are third-party measurements, included for demonstration with their sources cited. They
are not covered by this repository's license; see each file's `# source:` line. Measurements themselves are
facts, but cite the original publication (not only the package or site the file was taken from), and check a
data set's terms of use before adding it: some are restricted (e.g. non-commercial only) or ask for a specific
citation. Copy numbers, not documentation text, beyond a short attributed quotation.
