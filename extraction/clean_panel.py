"""
Clean NERC Panel CSVs — Apply normalization and quality fixes
==============================================================
Reads the raw CSVs from build_panel.py, applies 10 cleanup tasks,
outputs cleaned versions ready for modeling.

Input:  nerc_plant_quarterly.csv, nerc_disco_quarterly.csv
Output: nerc_plant_quarterly_clean.csv, nerc_disco_quarterly_clean.csv

Usage:  python3 clean_panel.py
"""

import pandas as pd
import numpy as np
import re
import sys

PLANT_IN = "nerc_plant_quarterly.csv"
DISCO_IN = "nerc_disco_quarterly.csv"
PLANT_OUT = "nerc_plant_quarterly_clean.csv"
DISCO_OUT = "nerc_disco_quarterly_clean.csv"

# ── Task 4: Canonical plant name mapping ───────────────────────────────────────
# Maps every observed variant to a canonical plant name.
# Built from examining the full plant name list, period ranges, and PAF continuity.

PLANT_ALIASES = {
    # Afam IV-V (old gas turbines)
    "AFAM (I-V)":       "Afam IV-V",
    "Afam I-V":         "Afam IV-V",
    "Afam IV - V":      "Afam IV-V",
    "Afam IV_V":        "Afam IV-V",
    "Afam V-VI":        "Afam IV-V",  # likely mislabel, same unit block
    "Afam":             "Afam IV-V",  # standalone "Afam" in 2018Q3
    "Afam_1":           "Afam IV-V",  # 2024+ naming for the old block

    # Afam VI (NIPP, separate from IV-V)
    "AFAM VI":          "Afam VI",
    "Afam_2":           "Afam VI",    # 2024+ naming
    "Afam_2I":          "Afam VI",    # typo in 2024Q3

    # AES (decommissioned barge)
    "AES BARGE":        "AES",
    "AES":              "AES",

    # Alaoji
    "Alaoji NIPP":      "Alaoji",
    "Alaoji_1":         "Alaoji",

    # ASCO (decommissioned barge)
    "ASCO":             "ASCO",

    # Azura
    "Azura IPP":        "Azura",
    "Azura-Edo IPP":    "Azura",
    "Azura-Ede IPP":    "Azura",  # typo
    "AZURA-EDO IPP":    "Azura",
    "Azura":            "Azura",

    # Calabar NIPP → becomes Odukpani in later reports
    "CALABAR NIPP":     "Odukpani",
    "Calabar NIPP":     "Odukpani",
    "Odukpani (Calabar) NIPP": "Odukpani",
    "Odukpani NIPP":    "Odukpani",
    "Odukpani":         "Odukpani",
    "Odukpani_1":       "Odukpani",

    # Dadin Kowa
    "Dadin Kowa":       "Dadin Kowa",
    "Dadin Kowa Hydro": "Dadin Kowa",
    "Dadin Kowa hydro": "Dadin Kowa",
    "Dadin-Kowa_1":     "Dadin Kowa",
    "Dadinkowa Hydro":  "Dadin Kowa",

    # Delta
    "Delta":            "Delta",
    "Delta GS":         "Delta",
    "Delta_1":          "Delta",
    "Transcorp Ughelli": "Delta",  # Transcorp acquired Delta/Ughelli

    # Egbin
    "EGBIN":            "Egbin",
    "Egbin":            "Egbin",
    "Egbin_1":          "Egbin",
    "Egbin ST":         "Egbin",
    "Egbin ST(GAS)":    "Egbin",
    "Egbin ST(Gas)":    "Egbin",
    "Egbin ST-6":       "Egbin",
    "Egbin St-6":       "Egbin",
    "Egbin ST6":        "Egbin",

    # Gbarain
    "GBARAIN":          "Gbarain",
    "Gbarain":          "Gbarain",
    "Gbarain NIPP":     "Gbarain",

    # Geregu Gas (original plant)
    "GEREGU":           "Geregu",
    "Geregu":           "Geregu",
    "Geregu Gas":       "Geregu",
    "Geregu_1":         "Geregu",

    # Geregu NIPP (second plant at same site)
    "GEREGU NIPP":      "Geregu NIPP",
    "GEREGU IPP":       "Geregu NIPP",  # likely same — IPP/NIPP confusion
    "Geregu NIPP":      "Geregu NIPP",
    "Geregu_2":         "Geregu NIPP",

    # Ibom Power
    "Ibom":             "Ibom Power",
    "Ibom Power":       "Ibom Power",
    "Ibom Power_1":     "Ibom Power",
    "Ibom power_1":     "Ibom Power",

    # Igbafo (small plant, appears 2024+)
    "Igbafo_1":         "Igbafo",

    # Ihovbor (original NIPP block)
    "IHOVBOR":          "Ihovbor",
    "Ihovbor NIPP":     "Ihovbor",
    "Ihobvor NIPP":     "Ihovbor",  # typo
    "Ihovbor_1":        "Ihovbor",

    # Ihovbor 2 (expansion, appears 2024+)
    "Ihovbor_2":        "Ihovbor 2",

    # Ikeja (embedded generation, appears 2024+)
    "Ikeja_1":          "Ikeja",

    # Jebba
    "JEBBA":            "Jebba",
    "Jebba":            "Jebba",
    "Jebba_1":          "Jebba",

    # Kainji
    "KAINJI":           "Kainji",
    "Kainji":           "Kainji",
    "Kainji_1":         "Kainji",

    # Okpai
    "Okpai":            "Okpai",
    "Okpai_1":          "Okpai",
    "Agip":             "Okpai",  # Agip operates Okpai

    # Olorunsogo Gas (original plant)
    "OLORUNSOGO":       "Olorunsogo",
    "Olorunsogo":       "Olorunsogo",
    "Olorunsogo Gas":   "Olorunsogo",
    "Olorunsogo_1":     "Olorunsogo",

    # Olorunsogo NIPP (second plant)
    "OLORUNSOGO NIPP":  "Olorunsogo NIPP",
    "Olorunsogo NIPP":  "Olorunsogo NIPP",
    "Olorunsogo NIPP (second entry)": "Olorunsogo NIPP",
    "Olorunsogo_2":     "Olorunsogo NIPP",

    # Omoku
    "OMOKU":            "Omoku",
    "Omoku":            "Omoku",
    "Omoku_1":          "Omoku",

    # Omotosho Gas (original plant)
    "OMOTOSHO":         "Omotosho",
    "Omotosho":         "Omotosho",
    "Omotosho Gas":     "Omotosho",
    "Omotosho_1":       "Omotosho",

    # Omotosho NIPP (second plant)
    "OMOTOSHO NIPP":    "Omotosho NIPP",
    "OMOTOSHO NAIPP":   "Omotosho NIPP",  # garbled but identifiable
    "Omotosho NIPP":    "Omotosho NIPP",
    "Omotosho_2":       "Omotosho NIPP",

    # Paras Energy
    "PARAS":            "Paras",
    "Paras":            "Paras",
    "Paras Energy":     "Paras",

    # Rivers IPP
    "Rivers IPP":       "Rivers IPP",
    "Rivers_1":         "Rivers IPP",

    # Sapele ST (old steam plant)
    "SAPELE":           "Sapele ST",
    "Sapele":           "Sapele ST",
    "Sapele ST":        "Sapele ST",
    "Sapele Steam_1":   "Sapele ST",

    # Sapele NIPP / GT NIPP (gas turbine addition)
    "SAPELE IPP":       "Sapele NIPP",
    "Sapele NIPP":      "Sapele NIPP",
    "Sapele GT NIPP":   "Sapele NIPP",
    "Sapele IPP":       "Sapele NIPP",
    "Sapele_2":         "Sapele NIPP",

    # Shiroro
    "SHIRORO":          "Shiroro",
    "Shiroro":          "Shiroro",
    "Shiroro_1":        "Shiroro",

    # Taopex
    "Taopex":           "Taopex",
    "Taopex Energy":    "Taopex",

    # Trans-Amadi
    "Trans Amadi":      "Trans-Amadi",
    "Trans Amadi_1":    "Trans-Amadi",
    "Trans-Amadi":      "Trans-Amadi",
    "Transami":         "Trans-Amadi",

    # Zungeru
    "Zungeru_1":        "Zungeru",
    "Zungeru Hydro":    "Zungeru",

    # Misc real but rare
    "Bonga":            "Bonga",       # offshore, 1 appearance
    "Egbema":           "Egbema",      # 1 appearance
    "Mabon":            "Mabon",       # 1 appearance, possibly Calabar-related
    "Notore":           "Notore",      # fertilizer plant, embedded gen
    "Dudkon NIPP":      "Duduku NIPP", # likely Duduku, typo
    "Dudoku NIPP":      "Duduku NIPP",
}

# ── Task 5: Garbled names to DROP (from chart misreading) ──────────────────────
# These are clearly corrupted strings from Claude misreading bar chart text.
# PAF values are real but can't be reliably assigned to plants.

GARBLED_DROP = {
    # 2017Q4 garbled
    "ABDOIMPOW", "ARIUNS OGO", "BORNIPP", "EGBINOOKPO",
    "LAOJI NSIPAP", "NESCO RIVERS TIAPP/UGHELLI", "NNSJIOGOTN",
    "PELE INHIOPV", "RIANNPS-AMI",
    # 2018Q1 garbled
    "APRIVARIVE", "BAFALAMO", "EGIT IPP", "ERMO", "HELLOI",
    "IMMADOI", "KOLA OPI", "NASIN-BAJO", "NI IPP", "NSIPEP",
    "OUMU", "PGERLO", "PNIPP", "ROBO IPP", "UGBIN",
    "UNKETSONI IPP", "UPON SHIP IPP", "VIJINGIB", "WSHAIRE",
    # 2019Q3 numbered placeholders
    "Plant3", "Plant4", "Plant5", "Plant6", "Plant8", "Plant9",
    "Plant10", "Plant11", "Plant12", "Plant14", "Plant15", "Plant16",
    "Plant17", "Plant18", "Plant20", "Plant21", "Plant22", "Plant23",
    "Plant24", "Plant25", "Plant26",
    # Explicit unknown
    "Unknown Plant",
}

# ── Task 2: DISCO aliases ──────────────────────────────────────────────────────

DISCO_ALIASES = {
    "APL": "Aba",  # Aba Power Limited
}

# ── Task 1: DISCO rows to drop ─────────────────────────────────────────────────

DISCO_DROP = {"All DisCos (Aggregate)", "disco"}

# ── Periods with known partial PAF extraction (Task 9) ─────────────────────────

PARTIAL_PAF_PERIODS = {
    "2017Q3": 8,   # chart format, only 8 plants read
    "2019Q2": 4,   # chart format, only 4 plants read
    "2019Q4": 5,   # chart format, only 5 plants read
    "2020Q2": 5,   # chart format, only 5 plants read
    "2020Q3": 7,   # chart format, only 7 plants read
    "2023Q1": 5,   # chart format, only 5 plants read
}


# ── Execute cleanup ────────────────────────────────────────────────────────────

def clean_plant_panel(df):
    print(f"Plant panel input: {len(df)} rows, {df['plant'].nunique()} unique plants\n")
    log = []

    # Task 5 + 6 + 7: Drop garbled, PlantN, and Unknown
    mask_garbled = df["plant"].isin(GARBLED_DROP)
    n_garbled = mask_garbled.sum()
    df = df[~mask_garbled].copy()
    log.append(f"Task 5/6/7: Dropped {n_garbled} garbled/placeholder/unknown rows")

    # Task 4: Normalize plant names
    def normalize_plant(name):
        if name in PLANT_ALIASES:
            return PLANT_ALIASES[name]
        # Fallback: check case-insensitive
        for alias, canonical in PLANT_ALIASES.items():
            if alias.lower() == name.lower():
                return canonical
        return name

    original_names = df["plant"].nunique()
    df["plant"] = df["plant"].apply(normalize_plant)
    new_names = df["plant"].nunique()
    log.append(f"Task 4: Normalized plant names: {original_names} → {new_names} unique")

    # Handle duplicates after normalization (e.g., both "Egbin" and "Egbin_1" in same period)
    # Keep the one with more data (more non-null fields)
    def row_completeness(row):
        return row[["installed_mw", "available_mw", "paf_pct"]].notna().sum()

    before_dedup = len(df)
    df["_completeness"] = df.apply(row_completeness, axis=1)
    df = df.sort_values("_completeness", ascending=False)
    df = df.drop_duplicates(subset=["period", "plant"], keep="first")
    df = df.drop(columns=["_completeness"])
    n_dedup = before_dedup - len(df)
    if n_dedup > 0:
        log.append(f"Task 4: Removed {n_dedup} duplicate plant-period rows after normalization")

    # Task 9: Add coverage flag
    df["paf_coverage"] = "full"
    for period, count in PARTIAL_PAF_PERIODS.items():
        mask = df["period"] == period
        if mask.any():
            df.loc[mask, "paf_coverage"] = "partial"
    n_partial = (df["paf_coverage"] == "partial").sum()
    log.append(f"Task 9: Flagged {n_partial} rows as partial PAF coverage")

    # Task 8: Fix installed_mw = 0 → null for periods where it wasn't reported
    # If ALL plants in a period have installed_mw == null or 0, set to null
    for period in df["period"].unique():
        mask = df["period"] == period
        installed = df.loc[mask, "installed_mw"]
        if installed.isna().all() or (installed.fillna(0) == 0).all():
            df.loc[mask, "installed_mw"] = np.nan
            df.loc[mask, "available_mw"] = np.nan

    # Sort
    df = df.sort_values(["year", "quarter", "plant"]).reset_index(drop=True)

    print("  Plant panel cleanup log:")
    for l in log:
        print(f"    {l}")
    print(f"  Output: {len(df)} rows, {df['plant'].nunique()} unique plants")
    print(f"  Periods: {df['period'].nunique()}")
    print(f"  PAF coverage: {(df['paf_pct'].notna()).sum()}/{len(df)} have PAF %")
    print(f"  Installed MW: {(df['installed_mw'].notna()).sum()}/{len(df)} have installed MW")
    print()
    return df


def clean_disco_panel(df, clean_plant_df):
    print(f"DISCO panel input: {len(df)} rows, {df['disco'].nunique()} unique DISCOs\n")
    log = []

    # Task 1: Drop aggregate/header rows
    mask_drop = df["disco"].isin(DISCO_DROP)
    n_drop = mask_drop.sum()
    df = df[~mask_drop].copy()
    log.append(f"Task 1: Dropped {n_drop} aggregate/header DISCO rows")

    # Task 2: Merge APL into Aba
    n_apl = (df["disco"] == "APL").sum()
    df["disco"] = df["disco"].replace(DISCO_ALIASES)
    # Deduplicate if APL and Aba now overlap
    before = len(df)
    df = df.drop_duplicates(subset=["period", "disco"], keep="first")
    n_merged = n_apl - (before - len(df) - n_apl) if before != len(df) else 0
    log.append(f"Task 2: Renamed {n_apl} APL rows to Aba")

    # Task 3: Flag Aba as non-model entity
    df["in_model"] = True
    aba_mask = df["disco"] == "Aba"
    df.loc[aba_mask, "in_model"] = False
    log.append(f"Task 3: Flagged {aba_mask.sum()} Aba rows as in_model=False")

    # Task 8 + 10: Recompute system-level metrics from cleaned plant data
    # Drop old system columns
    sys_cols = ["system_installed_mw", "system_available_mw", "plant_count", "avg_paf_pct"]
    for col in sys_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Recompute from cleaned plant panel
    if not clean_plant_df.empty:
        # Only use full-coverage periods for avg_paf to avoid bias
        full_paf = clean_plant_df[clean_plant_df["paf_coverage"] == "full"]

        sys_metrics = full_paf.groupby("period").agg(
            system_installed_mw=("installed_mw", lambda x: x.sum() if x.notna().any() else np.nan),
            system_available_mw=("available_mw", lambda x: x.sum() if x.notna().any() else np.nan),
            plant_count=("plant", "count"),
            avg_paf_pct=("paf_pct", "mean"),
        ).reset_index()

        # Fix: if sum was 0 because all values were NaN, set to NaN
        sys_metrics.loc[sys_metrics["system_installed_mw"] == 0, "system_installed_mw"] = np.nan
        sys_metrics.loc[sys_metrics["system_available_mw"] == 0, "system_available_mw"] = np.nan

        df = df.merge(sys_metrics, on="period", how="left")
        n_sys = sys_metrics["period"].nunique()
        log.append(f"Task 10: Recomputed system metrics for {n_sys} full-coverage periods")
    else:
        log.append("Task 10: Skipped — no plant data available")

    # Sort
    df = df.sort_values(["year", "quarter", "disco"]).reset_index(drop=True)

    print("  DISCO panel cleanup log:")
    for l in log:
        print(f"    {l}")
    print(f"  Output: {len(df)} rows, {df['disco'].nunique()} unique DISCOs")
    print(f"  Model DISCOs: {df[df['in_model']]['disco'].nunique()}")
    print(f"  Periods: {df['period'].nunique()}")
    print()

    # Coverage report
    print("  Variable coverage (model DISCOs only):")
    model_df = df[df["in_model"]]
    for col in model_df.columns:
        if col in ("period", "year", "quarter", "disco", "in_model"):
            continue
        non_null = model_df[col].notna().sum()
        total = len(model_df)
        pct = 100 * non_null / total if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {col:30s}  {non_null:4d}/{total}  {bar} {pct:.0f}%")
    print()

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load
    try:
        plant_df = pd.read_csv(PLANT_IN)
    except FileNotFoundError:
        print(f"ERROR: {PLANT_IN} not found. Run build_panel.py first.")
        sys.exit(1)

    try:
        disco_df = pd.read_csv(DISCO_IN)
    except FileNotFoundError:
        print(f"ERROR: {DISCO_IN} not found. Run build_panel.py first.")
        sys.exit(1)

    print(f"{'='*60}")
    print("NERC Panel Cleanup")
    print(f"{'='*60}\n")

    # Clean plant panel first (DISCO panel depends on it)
    clean_plants = clean_plant_panel(plant_df)

    # Clean DISCO panel with recomputed system metrics
    clean_discos = clean_disco_panel(disco_df, clean_plants)

    # Save
    clean_plants.to_csv(PLANT_OUT, index=False)
    print(f"Saved: {PLANT_OUT}")

    clean_discos.to_csv(DISCO_OUT, index=False)
    print(f"Saved: {DISCO_OUT}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY\n")

    periods = sorted(clean_plants["period"].unique())
    print(f"Plant panel: {len(clean_plants)} rows, {clean_plants['plant'].nunique()} plants, "
          f"{len(periods)} periods ({periods[0]}–{periods[-1]})")

    model_discos = clean_discos[clean_discos["in_model"]]
    d_periods = sorted(model_discos["period"].unique())
    print(f"DISCO panel: {len(model_discos)} model rows, {model_discos['disco'].nunique()} DISCOs, "
          f"{len(d_periods)} periods ({d_periods[0]}–{d_periods[-1]})")

    # Show canonical plant list
    print(f"\nCanonical plants ({clean_plants['plant'].nunique()}):")
    for p in sorted(clean_plants["plant"].unique()):
        count = len(clean_plants[clean_plants["plant"] == p])
        print(f"  {p:25s}  {count:2d} quarters")

    print(f"\nCanonical DISCOs ({model_discos['disco'].nunique()}):")
    for d in sorted(model_discos["disco"].unique()):
        count = len(model_discos[model_discos["disco"] == d])
        print(f"  {d:25s}  {count:2d} quarters")