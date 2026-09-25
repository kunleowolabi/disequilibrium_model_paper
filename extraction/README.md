# Extraction pipeline

This folder builds `data/intermediate/nerc_disco_quarterly_clean.csv` from
raw NERC quarterly report PDFs (`data/raw/nerc_reports/`). It runs in two
stages plus a cleaning pass, and it is **not required to reproduce the
paper's results** — its output is already provided directly under
`data/intermediate/`. It is included for transparency, not as a step you
need to re-run.

## Why two stages

NERC's report formats are inconsistent across years: table locations move,
some quarters present Plant Availability Factor as a formal table and
others as a bar chart, and a handful of early reports are scanned images
with no extractable text layer. A single extraction pass could not
reliably handle this range of formats, so the pipeline runs in two stages.

### Stage 1 — `batch_extract.py`

Runs a first, full pass across every report in `data/raw/nerc_reports/`,
for four tables per report: plant availability (PAF), energy
offtake/contracted capacity (PCC), ATC&C loss, and metering. For each
table it locates the relevant pages by content (DISCO names, unit markers
such as `MWh/h` or `PCC`, and a scoring heuristic that favours pages
containing a formal `Table N` label), then sends the located page text to
Claude for structured JSON extraction.

Its output records, for every report and table, whether extraction
succeeded and — if it failed — why (no matching pages found, an empty
result from Claude, a connection error, or an apparent image PDF).

### Stage 2 — `retry_extract.py`

Reads Stage 1's failure record and retries only the specific report/table
combinations that failed, using broadened page-finding logic for known
problem cases — in particular, a looser `find_paf_pages` that also
recognises PAF presented as a bar chart rather than a formal table, since
this was the single largest source of Stage 1 failures. Each retried
report is checked first for whether it is a scanned image PDF (under 200
characters of extractable text across its first five pages); if so, it is
recorded as unrecoverable by this method rather than retried blindly.

Both stages write incrementally after every extraction and can be
resumed from where they left off if interrupted.

### Cleaning — `clean_panel.py`

Merges Stage 1 and Stage 2's raw JSON output into a single DISCO-quarter
panel, normalises DISCO and plant names across reports (which are not
spelled consistently — e.g. "Port Harcourt" vs "PHED", plant-name
variants), and resolves duplicate report periods where more than one file
covers the same quarter. This is the script that produces
`nerc_disco_quarterly_clean.csv`.

## What this pipeline does not resolve

Some raw-panel issues were identified and corrected downstream, not here:

- **The 2023Q3 Plant Availability Factor was extracted incorrectly**
  (36.07% instead of the report's own stated 33.31%) and was corrected
  manually against the source report before use — see
  `data/estimation/README.md` and the paper's Section 4.2.
- **A small number of quarters' PAF figures (2022Q4–2023Q2) were not
  present in any extractable table format** and were instead reconstructed
  from narrative text in the corresponding reports, outside this
  pipeline.
- **The NERC "registered" to "active registered" customer redefinition**
  (2024→2025) is a reporting-definition change in the source data, not
  an extraction error, and is not corrected here; it is why customer
  counts do not appear in the final specification.

## Requirements
pdfplumber
anthropic


Both stages require `ANTHROPIC_API_KEY` set in the environment and the
source PDFs present in `data/raw/nerc_reports/`. Neither is distributed
by default given the API dependency and the size of the PDF corpus; see
`data/raw/nerc_reports/MANIFEST.md` for the list of source reports and
where to obtain them (NERC's public report archive).

## Running

```bash
export ANTHROPIC_API_KEY=your_key_here
python batch_extract.py
python retry_extract.py
python clean_panel.py
```

Stage 1 must complete (or at least produce its output file) before Stage 2
runs, since Stage 2 reads Stage 1's failure record directly.