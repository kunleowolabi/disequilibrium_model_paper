"""
Phase 11b: CORRECTED leave-one-out -- fit on reduced sample, evaluate on
the FULL original 154-obs panel
============================================================================
Fixes a real flaw in the first version: it fitted AND aggregated over the
same reduced sample, so "excluding a DISCO" mechanically removed that
DISCO's own demand from the national total, conflating a trivial
arithmetic effect with genuine coefficient sensitivity. This version
fits on the reduced sample but always evaluates D_grid on the full,
unchanged 154-observation panel, isolating whether the ESTIMATED
COEFFICIENTS are sensitive to a DISCO's or quarter's presence.

Usage:
  python3 phase11b_leave_one_out_corrected.py estimation_sample_v4.csv disco_igr_by_year_v2.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import sys

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "estimation_sample_v4.csv"
IGR_FILE = sys.argv[2] if len(sys.argv) > 2 else "disco_igr_by_year_v2.csv"
IGR_YEAR_COL = "2023"

FE_COLS = ["fe_abuja", "fe_benin", "fe_eko", "fe_enugu", "fe_ibadan",
           "fe_ikeja", "fe_jos", "fe_kaduna", "fe_kano", "fe_port_harcourt"]
DISCO_ORDER = ["Abuja", "Benin", "Eko", "Enugu", "Ibadan", "Ikeja",
               "Jos", "Kaduna", "Kano", "Port Harcourt", "Yola"]
LAGOS_DISCOS = {"Eko", "Ikeja"}
SUPPLY_CONTINUOUS = ["pcc", "avg_paf"]
SEASON_COLS = ["cos_q", "sin_q"]
DEMAND_VARS = ["log_igr", "log_igr_x_lagos", "cos_q", "sin_q"]
DEMAND_CONTINUOUS = ["log_igr", "log_igr_x_lagos"]

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -3.0, 6.0
SIGMA1_BOUNDS = (5, 40)
SIGMA2_BOUNDS = (3, 40)
HEADLINE_GAP = 1.91
N_DEMAND = 1 + len(DEMAND_VARS)


def add_harmonic_seasonal(df):
    df = df.copy()
    a = 2 * np.pi * df["quarter"] / 4
    df["cos_q"], df["sin_q"] = np.cos(a), np.sin(a)
    return df


def merge_igr(df, f, col):
    igr = pd.read_csv(f, index_col="disco"); igr.columns = [str(c) for c in igr.columns]
    df = df.copy()
    df["igr"] = df["disco"].map(igr[col]); df["log_igr"] = np.log(df["igr"])
    df["lagos_group"] = df["disco"].isin(LAGOS_DISCOS).astype(int)
    df["log_igr_x_lagos"] = df["log_igr"] * df["lagos_group"]
    return df


def standardize(df, cols):
    return {c: df[c].mean() for c in cols}, {c: df[c].std() for c in cols}


def build_demand_design(df, m, s):
    cols = [np.ones(len(df))]
    for v in DEMAND_VARS:
        cols.append((df[v].values - m[v]) / s[v] if v in DEMAND_CONTINUOUS else df[v].values)
    return np.column_stack(cols)


def build_demand_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in DEMAND_VARS])


def build_supply_design(df, m, s, active_fe):
    cols = [np.ones(len(df))]
    for v in SUPPLY_CONTINUOUS:
        cols.append((df[v].values - m[v]) / s[v])
    cols.append(df["cos_q"].values); cols.append(df["sin_q"].values)
    for v in active_fe:
        cols.append(df[v].values)
    return np.column_stack(cols)


def unpack_theta(theta, nd, ns):
    b1, b2 = theta[:nd], theta[nd:nd + ns]
    s1 = np.exp(np.clip(theta[nd + ns], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    s2 = np.exp(np.clip(theta[nd + ns + 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    return b1, b2, s1, s2


def compute_components(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2 = unpack_theta(theta, nd, ns)
    mu1, mu2 = X1 @ b1, X2 @ b2
    z1 = np.clip((Q - mu1) / s1, -30, 30); z2 = np.clip((Q - mu2) / s2, -30, 30)
    f1, f2 = stats.norm.pdf(z1) / s1, stats.norm.pdf(z2) / s2
    F1, F2 = stats.norm.cdf(z1), stats.norm.cdf(z2)
    g = np.clip(f1 * (1 - F2) + f2 * (1 - F1), 1e-300, None)
    return b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g


def nll(theta, X1, X2, Q, nd, ns):
    *_, g = compute_components(theta, X1, X2, Q, nd, ns)
    return -np.sum(np.log(g))


def grad(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g = compute_components(theta, X1, X2, Q, nd, ns)
    w1 = (1 - F2) * z1 / s1 + f2; w2 = (1 - F1) * z2 / s2 + f1
    d1 = (f1 * w1)[:, None] * X1; d2 = (f2 * w2)[:, None] * X2
    l1 = f1 * ((1 - F2) * (z1 ** 2 - 1) + f2 * z1 * s1)
    l2 = f2 * ((1 - F1) * (z2 ** 2 - 1) + f1 * z2 * s2)
    return np.concatenate([-np.sum(d1 / g[:, None], axis=0), -np.sum(d2 / g[:, None], axis=0),
                            [-np.sum(l1 / g)], [-np.sum(l2 / g)]])


def get_starting_values(df, X1, X2, Q):
    r = df["offtake_ratio"].values
    dm, sm = r < 0.95, r >= 0.95
    if dm.sum() < 3 or sm.sum() < 3:
        b1, *_ = np.linalg.lstsq(X1, Q, rcond=None); s1 = np.std(Q - X1 @ b1)
        b2, *_ = np.linalg.lstsq(X2, Q, rcond=None); s2 = np.std(Q - X2 @ b2)
        return np.concatenate([b1, b2, [np.log(max(s1, 1e-3))], [np.log(max(s2, 1e-3))]])
    b1, *_ = np.linalg.lstsq(X1[dm], Q[dm], rcond=None); s1 = np.std(Q[dm] - X1[dm] @ b1)
    b2, *_ = np.linalg.lstsq(X2[sm], Q[sm], rcond=None); s2 = np.std(Q[sm] - X2[sm] @ b2)
    return np.concatenate([b1, b2, [np.log(max(s1, 1e-3))], [np.log(max(s2, 1e-3))]])


def unstandardize_coefs(b, var_names, continuous_cols, means, stds):
    b_orig = b.copy(); adj = 0.0
    for i, name in enumerate(var_names):
        if name in continuous_cols:
            adj += b[i + 1] * means[name] / stds[name]
            b_orig[i + 1] = b[i + 1] / stds[name]
    b_orig[0] = b[0] - adj
    return b_orig


def fit_get_demand_coefs(fit_df, active_fe, n_starts=12, seed=11, label="fit"):
    """Fit on fit_df (the reduced sample). Return ONLY the unstandardized
    demand coefficients b1 -- that's all we need, since the demand
    equation has no DISCO fixed effects and can be evaluated on any
    DISCO regardless of whether it was in the fitting sample."""
    n_supply = 1 + len(SUPPLY_CONTINUOUS) + len(SEASON_COLS) + len(active_fe)
    dm, ds = standardize(fit_df, DEMAND_CONTINUOUS)
    sm, ss = standardize(fit_df, SUPPLY_CONTINUOUS)
    X1 = build_demand_design(fit_df, dm, ds)
    X2 = build_supply_design(fit_df, sm, ss, active_fe)
    Q = fit_df["Q"].values

    n_total = N_DEMAND + n_supply + 2
    bounds = [(None, None)] * n_total
    bounds[N_DEMAND + n_supply] = (np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
    bounds[N_DEMAND + n_supply + 1] = (np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))

    rng = np.random.default_rng(seed); valid = []
    for i in range(n_starts):
        j = 0.0 if i == 0 else rng.uniform(0.1, 0.8)
        t0 = get_starting_values(fit_df, X1, X2, Q)
        if j > 0: t0 = t0 + rng.normal(0, j, size=t0.shape)
        t0[N_DEMAND + n_supply] = np.clip(t0[N_DEMAND + n_supply], np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
        t0[N_DEMAND + n_supply + 1] = np.clip(t0[N_DEMAND + n_supply + 1], np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))
        res = minimize(nll, t0, args=(X1, X2, Q, N_DEMAND, n_supply), jac=grad,
                        method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})
        _, _, s1, s2 = unpack_theta(res.x, N_DEMAND, n_supply)
        if -650 <= -res.fun <= -300: valid.append(res)

    if not valid:
        print(f"  [{label}] DEGENERATE ({len(valid)}/{n_starts})")
        return None
    best = min(valid, key=lambda r: r.fun)
    b1s = best.x[:N_DEMAND]
    b1 = unstandardize_coefs(b1s, DEMAND_VARS, DEMAND_CONTINUOUS, dm, ds)
    print(f"  [{label}] {len(valid)}/{n_starts} valid  -LL={best.fun:.3f}  "
          f"log_igr={b1[1]:.2f}  log_igr_x_lagos={b1[2]:.2f}")
    return b1


def evaluate_on_full_panel(full_df, b1):
    """Predict demand for ALL 154 observations using these coefficients,
    regardless of which subsample fitted them."""
    X1_full = build_demand_design_raw(full_df)
    mu1 = X1_full @ b1
    Q_full = full_df["Q"].values
    suppressed = np.maximum(mu1 - Q_full, 0)
    df_t = full_df.copy(); df_t["suppressed"] = suppressed
    by_p = df_t.groupby("period")["suppressed"].sum() / 1000
    obs_gw = df_t.groupby("period")["Q"].sum().mean() / 1000
    gap_gw = by_p.mean()
    return obs_gw + gap_gw, gap_gw


def main():
    print("=" * 70, "\nPHASE 11b: CORRECTED leave-one-out (fit reduced, evaluate on full panel)\n" + "=" * 70)

    df_full = pd.read_csv(DATA_FILE)
    df_full = add_harmonic_seasonal(df_full)
    df_full = merge_igr(df_full, IGR_FILE, IGR_YEAR_COL)
    print(f"\nLoaded {len(df_full)} observations (this fixed panel is used for ALL evaluation)")

    print(f"\n{'='*70}\nPART 1: Leave-one-DISCO-out (fit reduced, evaluate on all 11)\n{'='*70}")
    disco_results = []
    for excluded in DISCO_ORDER:
        fit_sample = df_full[df_full["disco"] != excluded].copy()
        active_fe = [c for c in FE_COLS if c != f"fe_{excluded.lower().replace(' ', '_')}"]
        b1 = fit_get_demand_coefs(fit_sample, active_fe, label=f"excl. {excluded}")
        if b1 is not None:
            dgrid, gap = evaluate_on_full_panel(df_full, b1)
            disco_results.append((excluded, dgrid, gap))
            print(f"      -> evaluated on FULL panel: D_grid={dgrid:.2f} GW  gap={gap:.2f} GW")
        else:
            disco_results.append((excluded, None, None))

    print(f"\n{'-'*70}\nPART 1 SUMMARY (corrected)\n{'-'*70}")
    print(f"{'Excluded DISCO':20s}{'D_grid (GW)':>14s}{'Gap (GW)':>12s}{'Delta gap':>12s}")
    for name, dgrid, gap in disco_results:
        if dgrid is None:
            print(f"{name:20s}{'DEGENERATE':>14s}")
        else:
            delta = gap - HEADLINE_GAP
            flag = "  <-- demand-revealing DISCO" if name in {"Abuja", "Eko", "Ikeja"} else ""
            print(f"{name:20s}{dgrid:14.2f}{gap:12.2f}{delta:+12.2f}{flag}")

    print(f"\n{'='*70}\nPART 2: Leave-one-quarter-out (fit reduced, evaluate on all 14)\n{'='*70}")
    periods = sorted(df_full["period"].unique(), key=lambda x: (int(x[:4]), int(x[-1])))
    quarter_results = []
    for excluded_q in periods:
        fit_sample = df_full[df_full["period"] != excluded_q].copy()
        b1 = fit_get_demand_coefs(fit_sample, FE_COLS, label=f"excl. {excluded_q}")
        if b1 is not None:
            dgrid, gap = evaluate_on_full_panel(df_full, b1)
            quarter_results.append((excluded_q, dgrid, gap))
            print(f"      -> evaluated on FULL panel: D_grid={dgrid:.2f} GW  gap={gap:.2f} GW")
        else:
            quarter_results.append((excluded_q, None, None))

    print(f"\n{'-'*70}\nPART 2 SUMMARY (corrected)\n{'-'*70}")
    print(f"{'Excluded quarter':20s}{'D_grid (GW)':>14s}{'Gap (GW)':>12s}{'Delta gap':>12s}")
    for name, dgrid, gap in quarter_results:
        if dgrid is None:
            print(f"{name:20s}{'DEGENERATE':>14s}")
        else:
            delta = gap - HEADLINE_GAP
            print(f"{name:20s}{dgrid:14.2f}{gap:12.2f}{delta:+12.2f}")

    all_gaps = [g for _, _, g in disco_results + quarter_results if g is not None]
    if all_gaps:
        print(f"\n{'='*70}\nOVERALL RANGE\n{'='*70}")
        print(f"  Gap: min={min(all_gaps):.2f} GW, max={max(all_gaps):.2f} GW, headline={HEADLINE_GAP:.2f} GW")

    print("\nDone.")


if __name__ == "__main__":
    main()