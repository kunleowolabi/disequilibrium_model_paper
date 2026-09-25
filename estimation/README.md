# Estimation

Scripts that fit the disequilibrium model and run every robustness check
reported in the paper. All read from `data/estimation/estimation_sample_v4.csv`
and `data/intermediate/disco_igr_by_year_v2.csv`.

## Files

| Script | What it produces | Paper section |
|---|---|---|
| `rerun_full_pipeline_v4.py` | Headline fit (5.34 GW / 1.91 GW) and the static-year IGR sensitivity table | 6.1, 6.4 |
| `rerun_robustness_v4.py` | Cluster-robust standard errors (Table 4, 5); out-of-sample test; writes `estimation_sample_v4_for_R.csv` for the R validation step | 6.2, 6.4 |
| `phase10_regime_and_rho_sensitivity.py` | Full-sample regime-probability breakdown (which DISCO-quarters are demand- vs supply-constrained); fixed-ρ sensitivity table (Table 7) | 6.4, 6.5 |
| `phase11b_leave_one_out_corrected.py`* | Leave-one-DISCO-out and leave-one-quarter-out sensitivity (Table A1) | 6.4, 7.3 |

\* **Use the `phase11b_...corrected` version, not `phase11_leave_one_out.py`.**
An earlier version of this script (`phase11_leave_one_out.py`, retained here
for the record) had a genuine bug: it fit the model on a reduced sample
(one DISCO excluded) and then aggregated the fitted demand over that same
reduced sample, which conflates two different things — whether the
*estimated coefficients* are sensitive to a DISCO's presence, against the
purely arithmetic effect of removing that DISCO's own demand from the
national total. The corrected version fits on the reduced sample but
always evaluates the fitted equation on the full, unchanged 154-observation
panel, isolating genuine coefficient sensitivity. The two versions give
materially different numbers for some DISCOs (e.g. excluding Abuja: the
buggy version and the corrected version happen to agree closely by
coincidence for this particular case, but this should not be assumed to
hold in general — always use the corrected script's output).

## Running order

None of these scripts depend on each other's *output files* — each reads
directly from the estimation sample — but `rerun_robustness_v4.py` should
run before the R validation step in `validation/`, since it writes the
CSV that script consumes.

```bash
python rerun_full_pipeline_v4.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv
python rerun_robustness_v4.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv
python phase10_regime_and_rho_sensitivity.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv
python phase11b_leave_one_out_corrected.py data/estimation/estimation_sample_v4.csv data/intermediate/disco_igr_by_year_v2.csv
```

## A note on code structure

These scripts each contain their own copy of the core likelihood,
gradient, and multi-start fitting routines rather than importing from a
shared module. This is a real characteristic of how the project was
built incrementally, not a deliberate design choice, and it means the
same ~150 lines of model machinery are duplicated four times across this
folder. A refactor into a shared `model.py` would reduce that duplication
and is a natural piece of housekeeping, but was not done here in order to
keep each script's output exactly reproducible against what is reported
in the paper.