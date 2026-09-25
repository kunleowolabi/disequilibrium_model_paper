# Estimating Suppressed Electricity Demand in a Rationed Grid: A Disequilibrium Approach for Nigeria

This repository contains the data, estimation code, and validation scripts
for a paper estimating suppressed electricity demand on Nigeria's connected
distribution network, using a maximum-likelihood disequilibrium model
(Fair and Jaffee, 1972; Maddala and Nelson, 1974) applied to a purpose-built
panel of quarterly regulatory data.

## Headline result

Averaged across 154 DISCO-quarters (11 distribution companies, 2022Q4–2026Q1),
Nigeria's existing connected network is estimated to draw **5.34 GW** on
average against **3.43 GW** actually delivered — a suppressed demand gap of
**1.91 GW**, or 56 per cent above current delivery. The estimate is
stable across specifications, holds out of sample, and is corroborated by
an independent estimator. Full detail, caveats, and robustness checks are
in `paper/Paper_2_revised.docx`.

## What this model does, in one line

Observed grid offtake is the smaller of two unobserved quantities — demand
and supply — because Nigeria's tariffs do not clear the market and supply is
administratively rationed. Rather than assuming how much consumption
understates demand, as prior estimates for Nigeria have done, this model
recovers demand and supply as two separately specified equations, jointly
fit by maximum likelihood, with the demand- or supply-constrained status
of every observation determined by the data rather than assumed in
advance.


## Reproducing the results

**The reliable entry point is `data/estimation/estimation_sample_v4.csv`.**
Everything from that file onward — model fitting, robustness checks,
leave-one-out tests, figures — is deterministic, pure Python (plus one R
script for independent validation), and will reproduce exactly.

```bash
pip install -r requirements.txt

# Headline estimate and IGR sensitivity table
python estimation/rerun_full_pipeline_v4.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Cluster-robust standard errors and out-of-sample test
python estimation/rerun_robustness_v4.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Regime-probability breakdown and fixed-rho sensitivity (independence assumption)
python estimation/phase10_regime_and_rho_sensitivity.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Leave-one-DISCO-out / leave-one-quarter-out (use the corrected version — see estimation/README.md)
python estimation/phase11b_leave_one_out_corrected.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv

# Structured JSON snapshot of the fitted model (outputs/results_v4.json)
python estimation/generate_results_json.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv outputs/results_v4.json

# Figures
python figures/make_figures.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv
```

See `estimation/README.md` for what each script produces and which paper
section it supports.

**What is *not* independently reproducible without additional access:**
the `extraction/` pipeline that builds `nerc_disco_quarterly_clean.csv`
from the raw NERC PDFs runs in two stages (`batch_extract.py`, then
`retry_extract.py`) using an LLM-assisted extraction step, and requires
both the source PDFs (`data/raw/nerc_reports/`, not all redistributed here
— see `data/raw/nerc_reports/MANIFEST.md`) and an Anthropic API key. This
pipeline is documented in `extraction/README.md` for transparency but is
not required to reproduce the paper's results, since its output
(`nerc_disco_quarterly_clean.csv`) is provided directly under
`data/intermediate/`.

The R validation script (`validation/validate_markets_package.R`) requires
R and the `markets` package (Karapanagiotis, 2024), installable via
`install.packages("markets")`.

## Data provenance

| File | Location | Source | Notes |
|---|---|---|---|
| `Nigeria_States_IGR_2016_2024.csv` | `data/raw/` | National Bureau of Statistics | State-level Internally Generated Revenue, single primary source, all 37 states |
| `nerc_disco_quarterly_clean.csv` | `data/intermediate/` | NERC quarterly reports | Extracted and cleaned by `extraction/`; see `paper/` Section 4 for construction decisions |
| `disco_igr_by_year_v2.csv` | `data/intermediate/` | Derived from the NBS IGR file | State IGR aggregated to DISCO franchise areas |
| `estimation_sample_v4.csv` | `data/estimation/` | Derived | Final estimation panel: 11 DISCOs × 14 quarters = 154 observations |

The estimation window (2022Q4–2026Q1) and several variable choices are
constrained by what NERC's public reporting archive makes available and
comparable, not by a theoretically preferred design. See "Data-constrained
research design" in `paper/Paper_2_revised.docx`, Section 4.2, for the
full account of these constraints.

## Citation

Owolabi, A. (2026) *Estimating Suppressed Electricity Demand in a Rationed
Grid: A Disequilibrium Approach for Nigeria*. Working paper. Available at:
<https://github.com/kunleowolabi/disequilibrium_model_paper> (Accessed: September 2026).

```bibtex
@techreport{owolabi2026suppresseddemand,
  title  = {Estimating Suppressed Electricity Demand in a Rationed Grid: A Disequilibrium Approach for Nigeria},
  author = {Owolabi, Ayokunle},
  year   = {2026},
  month  = {September},
  type   = {Working paper},
  url    = {https://github.com/kunleowolabi/disequilibrium_model_paper}
}
```

## License

Code is released under the MIT License (see `LICENSE`). NERC and NBS data
are public regulatory/statistical releases; no additional rights are
claimed over them.