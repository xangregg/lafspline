# Local data sets

CSV files here appear in the app's data list under **Local data (not published)** when the app runs from this
machine (`npm run serve`, which lists directories). Git ignores everything in this directory except this README,
so the files are never committed or deployed.

Use the same format as `data/` (see `data/README.md`): optional `# label:`, `# source:`, `# x:`, `# y:` and
`# note:` lines, then the table. A file without a label or source is listed by its name. File names are lower-case
letters, digits and hyphens, ending in `.csv`. Nothing needs indexing: reload the page after adding a file.

A link to a local data set (`#source=local:…`) works only where the same file is in `localdata/`.
