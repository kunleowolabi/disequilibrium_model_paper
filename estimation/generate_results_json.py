"""
Generate outputs/results_v4.json -- a structured snapshot of the fitted
headline model, for downstream use (tables, figures, or programmatic
access) without re-running the full estimation each time.
============================================================================
Reuses the exact fitting logic verified in rerun_robustness_v4.py (same
functions, same defaults, same cluster-robust SE computation) so the
JSON cannot drift from the paper's reported numbers.

Usage:
  python3 generate_results_json.py estimation_sample_v4.csv disco_igr_by_year_v2.csv ../outputs/results_v4.json
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import json
import sys
import os
from datetime import datetime, timezone

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "estimation_sample_v4.csv"
IGR_FILE = sys.argv[2] if len(sys.argv) > 2 else "disco_igr_by_year_v2.csv"
OUT_FILE = sys.argv[3] if len(sys.argv) > 3 else "../outputs/results_v4.json"
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
N_DEMAND = 1 + len(DEMAND_VARS)
N_SUPPLY = 1 + len(SUPPLY_CONTINUOUS) + len(SEASON_COLS) + len(FE_COLS)


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


def neg_log_likelihood(theta, X1, X2, Q, nd, ns):
    *_, g = compute_components(theta, X1, X2, Q, nd, ns)
    return -np.sum(np.log(g))


def per_obs_neg_score(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g = compute_components(theta, X1, X2, Q, nd, ns)
    w1 = (1 - F2) * z1 / s1 + f2; w2 = (1 - F1) * z2 / s2 + f1
    d1 = (f1 * w1)[:, None] * X1; d2 = (f2 * w2)[:, None] * X2
    l1 = f1 * ((1 - F2) * (z1 ** 2 - 1) + f2 * z1 * s1)
    l2 = f2 * ((1 - F1) * (z2 ** 2 - 1) + f1 * z2 * s2)
    n = len(Q); n_params = nd + ns + 2
    scores = np.zeros((n, n_params))
    scores[:, :nd] = -d1 / g[:, None]
    scores[:, nd:nd + ns] = -d2 / g[:, None]
    scores[:, nd + ns] = -l1 / g
    scores[:, nd + ns + 1] = -l2 / g
    return scores


def neg_log_likelihood_grad(theta, X1, X2, Q, nd, ns):
    return per_obs_neg_score(theta, X1, X2, Q, nd, ns).sum(axis=0)


def get_starting_values(df, X1, X2, Q):
    r = df["offtake_ratio"].values
    dm, sm = r < 0.95, r >= 0.95
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


def main():
    df = pd.read_csv(DATA_FILE)
    df = add_harmonic_seasonal(df)
    df = merge_igr(df, IGR_FILE, IGR_YEAR_COL)
    n_obs = len(df)
    print(f"Loaded {n_obs} observations")

    # ── Fit (multi-start, identical protocol) ──
    dm, ds = standardize(df, DEMAND_CONTINUOUS)
    sm, ss = standardize(df, SUPPLY_CONTINUOUS)
    X1, X2 = build_demand_design(df, dm, ds), build_supply_design(df, sm, ss)
    Q = df["Q"].values
    n_total = N_DEMAND + N_SUPPLY + 2
    bounds = [(None, None)] * n_total
    bounds[N_DEMAND + N_SUPPLY] = (np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
    bounds[N_DEMAND + N_SUPPLY + 1] = (np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))

    rng = np.random.default_rng(11)
    valid = []
    for i in range(12):
        jitter = 0.0 if i == 0 else rng.uniform(0.1, 0.8)
        t0 = get_starting_values(df, X1, X2, Q)
        if jitter > 0: t0 = t0 + rng.normal(0, jitter, size=t0.shape)
        t0[N_DEMAND + N_SUPPLY] = np.clip(t0[N_DEMAND + N_SUPPLY], np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
        t0[N_DEMAND + N_SUPPLY + 1] = np.clip(t0[N_DEMAND + N_SUPPLY + 1], np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))
        res = minimize(neg_log_likelihood, t0, args=(X1, X2, Q, N_DEMAND, N_SUPPLY),
                        jac=neg_log_likelihood_grad, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})
        _, _, s1, s2 = unpack_theta(res.x, N_DEMAND, N_SUPPLY)
        if -650 <= -res.fun <= -300: valid.append(res)

    if not valid:
        print("ERROR: degenerate. Aborting JSON generation."); sys.exit(1)
    best = min(valid, key=lambda r: r.fun)
    theta_hat = best.x
    n_converged_to_best = sum(1 for r in valid if abs(r.fun - best.fun) < 0.01)
    print(f"{len(valid)}/12 valid runs, {n_converged_to_best} converged to the reported optimum")
    print(f"-LL = {best.fun:.4f}")

    # ── Cluster-robust SEs ──
    eps = 1e-4; n_params = len(theta_hat)
    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        tp, tm = theta_hat.copy(), theta_hat.copy()
        tp[i] += eps; tm[i] -= eps
        H[:, i] = (neg_log_likelihood_grad(tp, X1, X2, Q, N_DEMAND, N_SUPPLY) -
                   neg_log_likelihood_grad(tm, X1, X2, Q, N_DEMAND, N_SUPPLY)) / (2 * eps)
    H = (H + H.T) / 2
    H_inv = np.linalg.inv(H)
    scores = per_obs_neg_score(theta_hat, X1, X2, Q, N_DEMAND, N_SUPPLY)
    df["_idx"] = np.arange(len(df))
    clusters = df["disco"].unique(); G = len(clusters); N = len(df); K = n_params
    B = np.zeros((K, K))
    for c in clusters:
        idx = df.loc[df["disco"] == c, "_idx"].values
        u_g = scores[idx].sum(axis=0)
        B += np.outer(u_g, u_g)
    corr = (G / (G - 1)) * ((N - 1) / (N - K))
    V_cluster = corr * (H_inv @ B @ H_inv)
    se_cluster = np.sqrt(np.diag(V_cluster))

    b1_std, b2_std, sigma1, sigma2 = unpack_theta(theta_hat, N_DEMAND, N_SUPPLY)
    b1_orig = unstandardize_coefs(b1_std, DEMAND_VARS, DEMAND_CONTINUOUS, dm, ds)
    b2_orig = unstandardize_coefs(b2_std, SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS, SUPPLY_CONTINUOUS, sm, ss)

    demand_names = ["intercept"] + DEMAND_VARS
    supply_names = ["intercept"] + SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS

    se_cluster_d = se_cluster[:N_DEMAND].copy()
    for i, name in enumerate(DEMAND_VARS):
        if name in DEMAND_CONTINUOUS:
            se_cluster_d[i + 1] /= ds[name]
    se_cluster_s = se_cluster[N_DEMAND:N_DEMAND + N_SUPPLY].copy()
    for i, name in enumerate(SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS):
        if name in SUPPLY_CONTINUOUS:
            se_cluster_s[i + 1] /= ss[name]

    # ── Headline GW figures ──
    X1_raw = np.column_stack([np.ones(len(df))] + [df[v].values for v in DEMAND_VARS])
    mu1 = X1_raw @ b1_orig
    suppressed = np.maximum(mu1 - df["Q"].values, 0)
    df["suppressed"] = suppressed
    by_period = df.groupby("period")["suppressed"].sum() / 1000
    observed_gw = df.groupby("period")["Q"].sum().mean() / 1000
    gap_gw = by_period.mean()
    dgrid_gw = observed_gw + gap_gw

    # ── Assemble JSON ──
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specification": "Lagos-group IGR elasticity, static 2023 IGR (headline)",
        "data_file": os.path.basename(DATA_FILE),
        "igr_file": os.path.basename(IGR_FILE),
        "n_observations": int(n_obs),
        "n_discos": int(df["disco"].nunique()),
        "n_periods": int(df["period"].nunique()),
        "fit": {
            "neg_log_likelihood": float(best.fun),
            "log_likelihood": float(-best.fun),
            "n_multistart_runs": 12,
            "n_valid_runs": len(valid),
            "n_converged_to_reported_optimum": int(n_converged_to_best),
            "sigma1": float(sigma1),
            "sigma2": float(sigma2),
        },
        "demand_equation": {
            name: {"coefficient": float(coef), "cluster_robust_se": float(se),
                   "t_stat": float(coef / se) if se else None}
            for name, coef, se in zip(demand_names, b1_orig, se_cluster_d)
        },
        "supply_equation": {
            name: {"coefficient": float(coef), "cluster_robust_se": float(se),
                   "t_stat": float(coef / se) if se else None}
            for name, coef, se in zip(supply_names, b2_orig, se_cluster_s)
        },
        "headline_results_gw": {
            "observed_offtake": round(float(observed_gw), 2),
            "estimated_demand_ceiling": round(float(dgrid_gw), 2),
            "suppressed_demand_gap": round(float(gap_gw), 2),
            "gap_pct_above_observed": round(float(100 * gap_gw / observed_gw), 1),
        },
    }

    os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {OUT_FILE}")
    print(f"Headline: D_grid={dgrid_gw:.2f} GW, gap={gap_gw:.2f} GW")
    print("(should match the paper: D_grid=5.34 GW, gap=1.91 GW -- if not, do not use this file)")


if __name__ == "__main__":
    main()