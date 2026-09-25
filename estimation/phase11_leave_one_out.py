"""
Phase 11: Leave-one-DISCO-out and leave-one-quarter-out robustness
============================================================================
Directly answers the question Phase 10 raised: the demand-constrained
regime is concentrated in three DISCOs (Abuja, Eko, Ikeja). This tests
whether the headline D_grid/gap depends on any single DISCO or quarter
being in the sample.

Part 1: Refit the model 11 times, each time excluding one DISCO
        entirely. Report D_grid and gap each time.
Part 2: Refit the model 14 times, each time excluding one quarter
        entirely. Report D_grid and gap each time.

Uses the same rho=0 headline specification and fitting machinery
verified throughout this project (standardization, analytic gradient,
multi-start, degenerate-fit filtering).

Usage:
  python3 phase11_leave_one_out.py estimation_sample_v4.csv disco_igr_by_year_v2.csv
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
HEADLINE_DGRID = 5.34
HEADLINE_GAP = 1.91


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


def build_supply_design(df, m, s):
    cols = [np.ones(len(df))]
    for v in SUPPLY_CONTINUOUS:
        cols.append((df[v].values - m[v]) / s[v])
    cols.append(df["cos_q"].values); cols.append(df["sin_q"].values)
    for v in FE_COLS:
        cols.append(df[v].values)
    return np.column_stack(cols)


def build_demand_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in DEMAND_VARS])


def build_supply_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in SUPPLY_CONTINUOUS] +
                            [df["cos_q"].values, df["sin_q"].values] + [df[v].values for v in FE_COLS])


N_DEMAND = 1 + len(DEMAND_VARS)


def get_n_supply(active_fe_cols):
    return 1 + len(SUPPLY_CONTINUOUS) + len(SEASON_COLS) + len(active_fe_cols)


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
        # fallback if a leave-out split leaves too few points for a regime
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


def fit_and_get_gw(df, active_fe_cols, n_starts=12, seed=11, label="fit"):
    """Fit the model on `df` (already restricted to the desired subsample),
    using only the FE columns for DISCOs actually present. Returns
    D_grid (GW), gap (GW), -LL, or None if degenerate."""
    global FE_COLS
    saved_fe = FE_COLS
    FE_COLS = active_fe_cols  # swap active FE set for this fit only

    n_supply = get_n_supply(active_fe_cols)
    dm, ds = standardize(df, DEMAND_CONTINUOUS)
    sm, ss = standardize(df, SUPPLY_CONTINUOUS)
    X1, X2 = build_demand_design(df, dm, ds), build_supply_design(df, sm, ss)
    Q = df["Q"].values

    n_total = N_DEMAND + n_supply + 2
    bounds = [(None, None)] * n_total
    bounds[N_DEMAND + n_supply] = (np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
    bounds[N_DEMAND + n_supply + 1] = (np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))

    rng = np.random.default_rng(seed); valid = []
    for i in range(n_starts):
        j = 0.0 if i == 0 else rng.uniform(0.1, 0.8)
        t0 = get_starting_values(df, X1, X2, Q)
        if j > 0: t0 = t0 + rng.normal(0, j, size=t0.shape)
        t0[N_DEMAND + n_supply] = np.clip(t0[N_DEMAND + n_supply], np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
        t0[N_DEMAND + n_supply + 1] = np.clip(t0[N_DEMAND + n_supply + 1], np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))
        res = minimize(nll, t0, args=(X1, X2, Q, N_DEMAND, n_supply), jac=grad,
                        method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})
        _, _, s1, s2 = unpack_theta(res.x, N_DEMAND, n_supply)
        if -650 <= -res.fun <= -300: valid.append(res)

    FE_COLS = saved_fe  # restore

    if not valid:
        print(f"  [{label}] DEGENERATE ({len(valid)}/{n_starts})")
        return None

    best = min(valid, key=lambda r: r.fun)
    b1s, b2s, s1, s2 = unpack_theta(best.x, N_DEMAND, n_supply)
    b1 = unstandardize_coefs(b1s, DEMAND_VARS, DEMAND_CONTINUOUS, dm, ds)

    X1r = build_demand_design_raw(df)
    mu1 = X1r @ b1
    Q_arr = df["Q"].values
    suppressed = np.maximum(mu1 - Q_arr, 0)
    df_t = df.copy(); df_t["suppressed"] = suppressed
    by_p = df_t.groupby("period")["suppressed"].sum() / 1000
    obs_gw = df_t.groupby("period")["Q"].sum().mean() / 1000
    gap_gw = by_p.mean()
    dgrid_gw = obs_gw + gap_gw

    print(f"  [{label}] {len(valid)}/{n_starts} valid  -LL={best.fun:.3f}  "
          f"D_grid={dgrid_gw:.2f} GW  gap={gap_gw:.2f} GW")
    return dgrid_gw, gap_gw, -best.fun


def main():
    print("=" * 70, "\nPHASE 11: Leave-one-out robustness\n" + "=" * 70)

    df_full = pd.read_csv(DATA_FILE)
    df_full = add_harmonic_seasonal(df_full)
    df_full = merge_igr(df_full, IGR_FILE, IGR_YEAR_COL)
    print(f"\nLoaded {len(df_full)} observations")
    print(f"Headline (full sample): D_grid={HEADLINE_DGRID:.2f} GW, gap={HEADLINE_GAP:.2f} GW\n")

    # ══════════════════════════════════════════════
    # PART 1: Leave-one-DISCO-out
    # ══════════════════════════════════════════════
    print(f"{'='*70}\nPART 1: Leave-one-DISCO-out (11 refits)\n{'='*70}")
    disco_results = []
    for excluded in DISCO_ORDER:
        sub = df_full[df_full["disco"] != excluded].copy()
        active_fe = [c for c in FE_COLS if c != f"fe_{excluded.lower().replace(' ', '_')}"]
        r = fit_and_get_gw(sub, active_fe, label=f"excl. {excluded}")
        if r:
            dgrid, gap, ll = r
            disco_results.append((excluded, dgrid, gap, ll))
        else:
            disco_results.append((excluded, None, None, None))

    print(f"\n{'-'*70}\nPART 1 SUMMARY\n{'-'*70}")
    print(f"{'Excluded DISCO':20s}{'D_grid (GW)':>14s}{'Gap (GW)':>12s}{'Delta gap':>12s}")
    for name, dgrid, gap, ll in disco_results:
        if dgrid is None:
            print(f"{name:20s}{'DEGENERATE':>14s}")
        else:
            delta = gap - HEADLINE_GAP
            flag = "  <-- demand-revealing DISCO" if name in {"Abuja", "Eko", "Ikeja"} else ""
            print(f"{name:20s}{dgrid:14.2f}{gap:12.2f}{delta:+12.2f}{flag}")

    # ══════════════════════════════════════════════
    # PART 2: Leave-one-quarter-out
    # ══════════════════════════════════════════════
    print(f"\n{'='*70}\nPART 2: Leave-one-quarter-out (14 refits)\n{'='*70}")
    periods = sorted(df_full["period"].unique(), key=lambda x: (int(x[:4]), int(x[-1])))
    quarter_results = []
    for excluded_q in periods:
        sub = df_full[df_full["period"] != excluded_q].copy()
        r = fit_and_get_gw(sub, FE_COLS, label=f"excl. {excluded_q}")
        if r:
            dgrid, gap, ll = r
            quarter_results.append((excluded_q, dgrid, gap, ll))
        else:
            quarter_results.append((excluded_q, None, None, None))

    print(f"\n{'-'*70}\nPART 2 SUMMARY\n{'-'*70}")
    print(f"{'Excluded quarter':20s}{'D_grid (GW)':>14s}{'Gap (GW)':>12s}{'Delta gap':>12s}")
    for name, dgrid, gap, ll in quarter_results:
        if dgrid is None:
            print(f"{name:20s}{'DEGENERATE':>14s}")
        else:
            delta = gap - HEADLINE_GAP
            print(f"{name:20s}{dgrid:14.2f}{gap:12.2f}{delta:+12.2f}")

    # ══════════════════════════════════════════════
    # Overall range
    # ══════════════════════════════════════════════
    all_gaps = [g for _, _, g, _ in disco_results + quarter_results if g is not None]
    if all_gaps:
        print(f"\n{'='*70}\nOVERALL RANGE ACROSS ALL 25 LEAVE-ONE-OUT REFITS\n{'='*70}")
        print(f"  Gap: min={min(all_gaps):.2f} GW, max={max(all_gaps):.2f} GW, "
              f"headline={HEADLINE_GAP:.2f} GW")

    print("\nDone.")


if __name__ == "__main__":
    main()