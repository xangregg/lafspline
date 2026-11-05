# Local data sets

CSV files here appear in the app's data list under **Local data (not published)** when the app runs from this
machine. Git ignores everything in this directory except this README, so the files are never committed or deployed.

Use the same format as `data/` (see `data/README.md`): optional `# label:`, `# source:`, `# x:`, `# y:` and
`# note:` lines, then the table. A file without a label or source is listed by its name. File names are lower-case
letters, digits and hyphens, ending in `.csv`.

- With `npm run serve` (Python's server, which lists directories), nothing needs indexing: reload the page after
  adding or removing a file.
- With other servers (WebStorm's built-in one), the app reads `index.json` here instead: run `npm run data:index`
  after adding or removing a file. It writes the index (ignored by git too) and warns about files the app can't
  read.

A link to a local data set (`#source=local:…`) works only where the same file is in `localdata/`.
