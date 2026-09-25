
## Reproducing the results

**The reliable entry point is `data/estimation/estimation_sample_v4.csv`.**
Everything from that file onward — model fitting, robustness checks,
leave-one-out tests, figures — is deterministic, pure Python (plus one R
script for independent validation), and will reproduce exactly.

```bash
pip install -r requirements.txt

# Headline estimate
python estimation/fit_headline.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Robustness suite (clustered SEs, out-of-sample test, IGR sensitivity, tariff shock)
python estimation/robustness.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Independence assumption test (fixed-rho sensitivity)
python estimation/model2_correlated.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Leave-one-DISCO-out / leave-one-quarter-out
python estimation/leave_one_out.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Figures
python figures/make_figures.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv
```

**What is *not* independently reproducible without additional access:**
the `extraction/` pipeline that builds `nerc_disco_quarterly_clean.csv`
from the raw NERC PDFs uses an LLM-assisted extraction step and requires
both the source PDFs (`data/raw/nerc_reports/`, not all redistributed here
— see `data/raw/nerc_reports/MANIFEST.md`) and an Anthropic API key. This
step is documented for transparency but is not required to reproduce the
paper's results, since its output (`nerc_disco_quarterly_clean.csv`) is
provided directly under `data/intermediate/`.

The R validation script (`validation/validate_markets_package.R`) requires
R and the `markets` package (Karapanagiotis, 2024), installable via
`install.packages("markets")`.

## Data provenance

| File | Source | Notes |
|---|---|---|
| `nerc_disco_quarterly_clean.csv` | NERC quarterly reports, 2022Q4–2026Q1 | Extracted and cleaned; see `paper/` Section 4 for construction decisions |
| `Nigeria_States_IGR_2016_2024.csv` | National Bureau of Statistics | State-level Internally Generated Revenue, single primary source |
| `estimation_sample_v4.csv` | Derived | Final panel: 11 DISCOs × 14 quarters = 154 observations |

The estimation window (2022Q4–2026Q1) and several variable choices are
constrained by what NERC's public reporting archive makes available and
comparable, not by a theoretically preferred design. See "Data-constrained
research design" in `paper/Paper_2_revised.docx`, Section 4.2, for the
full account of these constraints.

## Citation

If you use this code or data, please cite the accompanying paper (full
citation in `paper/`).

## License

Code is released under the MIT License (see `LICENSE`). NERC and NBS data
are public regulatory/statistical releases; no additional rights are
claimed over them.