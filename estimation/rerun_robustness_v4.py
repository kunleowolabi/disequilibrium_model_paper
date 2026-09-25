"""
Re-run cluster-robust SEs, OOS test, and prep for R cross-validation --
all on the CORRECTED 154-obs sample (estimation_sample_v4.csv)
============================================================================
Keeps the paper internally consistent after the 2023Q3 recovery/correction.
Headline model unchanged: Lagos-group IGR elasticity, static 2023 IGR.

Part 1: Cluster-robust SEs by DISCO (full 154-obs sample) -- FULL demand
        AND supply tables printed, no truncation.
Part 2: Out-of-sample test (75/25 period split, 14 periods)
Part 3: Writes a fresh CSV for the R markets-package cross-validation

Usage:
  python3 rerun_robustness_v4.py estimation_sample_v4.csv disco_igr_by_year_v2.csv
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
N_DEMAND = 1 + len(DEMAND_VARS)
N_SUPPLY = 1 + len(SUPPLY_CONTINUOUS) + len(SEASON_COLS) + len(FE_COLS)


def add_harmonic_seasonal(df):
    df = df.copy()
    angle = 2 * np.pi * df["quarter"] / 4
    df["cos_q"] = np.cos(angle)
    df["sin_q"] = np.sin(angle)
    return df


def merge_igr_static(df, igr_file, year_col):
    igr = pd.read_csv(igr_file, index_col="disco")
    igr.columns = [str(c) for c in igr.columns]
    df = df.copy()
    df["igr"] = df["disco"].map(igr[year_col])
    df["log_igr"] = np.log(df["igr"])
    df["lagos_group"] = df["disco"].isin(LAGOS_DISCOS).astype(int)
    df["log_igr_x_lagos"] = df["log_igr"] * df["lagos_group"]
    return df


def standardize(df, cols):
    return {c: df[c].mean() for c in cols}, {c: df[c].std() for c in cols}


def build_demand_design(df, means, stds):
    cols = [np.ones(len(df))]
    for v in DEMAND_VARS:
        cols.append((df[v].values - means[v]) / stds[v] if v in DEMAND_CONTINUOUS else df[v].values)
    return np.column_stack(cols)


def build_supply_design(df, means, stds):
    cols = [np.ones(len(df))]
    for v in SUPPLY_CONTINUOUS:
        cols.append((df[v].values - means[v]) / stds[v])
    cols.append(df["cos_q"].values)
    cols.append(df["sin_q"].values)
    for v in FE_COLS:
        cols.append(df[v].values)
    return np.column_stack(cols)


def build_demand_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in DEMAND_VARS])


def build_supply_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in SUPPLY_CONTINUOUS] +
                            [df["cos_q"].values, df["sin_q"].values] + [df[v].values for v in FE_COLS])


def unpack_theta(theta, nd, ns):
    b1, b2 = theta[:nd], theta[nd:nd + ns]
    s1 = np.exp(np.clip(theta[nd + ns], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    s2 = np.exp(np.clip(theta[nd + ns + 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    return b1, b2, s1, s2


def compute_components(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2 = unpack_theta(theta, nd, ns)
    mu1, mu2 = X1 @ b1, X2 @ b2
    z1 = np.clip((Q - mu1) / s1, -30, 30)
    z2 = np.clip((Q - mu2) / s2, -30, 30)
    f1, f2 = stats.norm.pdf(z1) / s1, stats.norm.pdf(z2) / s2
    F1, F2 = stats.norm.cdf(z1), stats.norm.cdf(z2)
    g = np.clip(f1 * (1 - F2) + f2 * (1 - F1), 1e-300, None)
    return b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g


def neg_log_likelihood(theta, X1, X2, Q, nd, ns):
    *_, g = compute_components(theta, X1, X2, Q, nd, ns)
    return -np.sum(np.log(g))


def per_obs_neg_score(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g = compute_components(theta, X1, X2, Q, nd, ns)
    w1 = (1 - F2) * z1 / s1 + f2
    w2 = (1 - F1) * z2 / s2 + f1
    dg_db1 = (f1 * w1)[:, None] * X1
    dg_db2 = (f2 * w2)[:, None] * X2
    dg_dl1 = f1 * ((1 - F2) * (z1 ** 2 - 1) + f2 * z1 * s1)
    dg_dl2 = f2 * ((1 - F1) * (z2 ** 2 - 1) + f1 * z2 * s2)
    n = len(Q)
    n_params = nd + ns + 2
    scores = np.zeros((n, n_params))
    scores[:, :nd] = -dg_db1 / g[:, None]
    scores[:, nd:nd + ns] = -dg_db2 / g[:, None]
    scores[:, nd + ns] = -dg_dl1 / g
    scores[:, nd + ns + 1] = -dg_dl2 / g
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


def fit_with_multistart(df, n_starts=12, seed=11, label="fit"):
    dm, ds = standardize(df, DEMAND_CONTINUOUS)
    sm, ss = standardize(df, SUPPLY_CONTINUOUS)
    X1, X2 = build_demand_design(df, dm, ds), build_supply_design(df, sm, ss)
    Q = df["Q"].values
    n_total = N_DEMAND + N_SUPPLY + 2
    bounds = [(None, None)] * n_total
    bounds[N_DEMAND + N_SUPPLY] = (np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
    bounds[N_DEMAND + N_SUPPLY + 1] = (np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))
    rng = np.random.default_rng(seed); valid = []
    for i in range(n_starts):
        jitter = 0.0 if i == 0 else rng.uniform(0.1, 0.8)
        t0 = get_starting_values(df, X1, X2, Q)
        if jitter > 0:
            t0 = t0 + rng.normal(0, jitter, size=t0.shape)
        t0[N_DEMAND + N_SUPPLY] = np.clip(t0[N_DEMAND + N_SUPPLY], np.log(SIGMA1_BOUNDS[0]), np.log(SIGMA1_BOUNDS[1]))
        t0[N_DEMAND + N_SUPPLY + 1] = np.clip(t0[N_DEMAND + N_SUPPLY + 1], np.log(SIGMA2_BOUNDS[0]), np.log(SIGMA2_BOUNDS[1]))
        res = minimize(neg_log_likelihood, t0, args=(X1, X2, Q, N_DEMAND, N_SUPPLY),
                        jac=neg_log_likelihood_grad, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})
        _, _, s1, s2 = unpack_theta(res.x, N_DEMAND, N_SUPPLY)
        if -650 <= -res.fun <= -300:
            valid.append(res)
    print(f"  [{label}] {len(valid)}/{n_starts} valid")
    if not valid:
        return None
    best = min(valid, key=lambda r: r.fun)
    lls = sorted(set(round(r.fun, 2) for r in valid))
    print(f"  [{label}] -LL={best.fun:.4f}  distinct: {lls}")
    return best, X1, X2, Q, dm, ds, sm, ss


def main():
    print("=" * 70, "\nRE-RUN ROBUSTNESS ON CORRECTED 154-OBS SAMPLE\n" + "=" * 70)

    df = pd.read_csv(DATA_FILE)
    df = add_harmonic_seasonal(df)
    df = merge_igr_static(df, IGR_FILE, IGR_YEAR_COL)
    print(f"\nLoaded {len(df)} observations, {df['disco'].nunique()} clusters")

    # ══════════════════════════════════════════════
    # PART 1: Cluster-robust SEs, full sample -- FULL demand AND supply tables
    # ══════════════════════════════════════════════
    print(f"\n{'=' * 70}\nPART 1: Cluster-robust SEs (full 154-obs sample)\n{'=' * 70}")
    fit_result = fit_with_multistart(df, label="full sample")
    if fit_result is None:
        print("DEGENERATE. Stop."); return
    best, X1, X2, Q, dm, ds, sm, ss = fit_result
    theta_hat = best.x

    eps = 1e-4; n_params = len(theta_hat)
    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        tp, tm = theta_hat.copy(), theta_hat.copy()
        tp[i] += eps; tm[i] -= eps
        H[:, i] = (neg_log_likelihood_grad(tp, X1, X2, Q, N_DEMAND, N_SUPPLY) -
                   neg_log_likelihood_grad(tm, X1, X2, Q, N_DEMAND, N_SUPPLY)) / (2 * eps)
    H = (H + H.T) / 2
    H_inv = np.linalg.inv(H)
    se_naive = np.sqrt(np.diag(H_inv))

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

    b1_std, b2_std, s1, s2 = unpack_theta(theta_hat, N_DEMAND, N_SUPPLY)
    b1_orig = unstandardize_coefs(b1_std, DEMAND_VARS, DEMAND_CONTINUOUS, dm, ds)
    b2_orig = unstandardize_coefs(b2_std, SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS,
                                   SUPPLY_CONTINUOUS, sm, ss)

    demand_names = ["intercept"] + DEMAND_VARS
    supply_names = ["intercept"] + SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS

    se_naive_d = se_naive[:N_DEMAND].copy(); se_cluster_d = se_cluster[:N_DEMAND].copy()
    for i, name in enumerate(DEMAND_VARS):
        if name in DEMAND_CONTINUOUS:
            se_naive_d[i + 1] /= ds[name]; se_cluster_d[i + 1] /= ds[name]

    se_naive_s = se_naive[N_DEMAND:N_DEMAND + N_SUPPLY].copy()
    se_cluster_s = se_cluster[N_DEMAND:N_DEMAND + N_SUPPLY].copy()
    for i, name in enumerate(SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS):
        if name in SUPPLY_CONTINUOUS:
            se_naive_s[i + 1] /= ss[name]; se_cluster_s[i + 1] /= ss[name]

    print(f"\n  {'-'*70}\n  DEMAND EQUATION\n  {'-'*70}")
    print(f"  {'Variable':22s}{'Coef':>11s}{'SE(naive)':>12s}{'t(naive)':>10s}{'SE(cluster)':>13s}{'t(cluster)':>11s}")
    for name, coef, sen, sec in zip(demand_names, b1_orig, se_naive_d, se_cluster_d):
        tn = coef / sen if sen else float("nan"); tc = coef / sec if sec else float("nan")
        flag = "  **" if abs(tc) > 1.96 else ""
        print(f"  {name:22s}{coef:11.3f}{sen:12.3f}{tn:10.2f}{sec:13.3f}{tc:11.2f}{flag}")

    print(f"\n  {'-'*70}\n  SUPPLY EQUATION (full, all {len(supply_names)} terms)\n  {'-'*70}")
    print(f"  {'Variable':22s}{'Coef':>11s}{'SE(naive)':>12s}{'t(naive)':>10s}{'SE(cluster)':>13s}{'t(cluster)':>11s}")
    for name, coef, sen, sec in zip(supply_names, b2_orig, se_naive_s, se_cluster_s):
        tn = coef / sen if sen else float("nan"); tc = coef / sec if sec else float("nan")
        flag = "  **" if abs(tc) > 1.96 else ""
        print(f"  {name:22s}{coef:11.3f}{sen:12.3f}{tn:10.2f}{sec:13.3f}{tc:11.2f}{flag}")

    print(f"\n  sigma1={s1:.3f}  sigma2={s2:.3f}  -LL={best.fun:.4f}")

    # ══════════════════════════════════════════════
    # PART 2: OOS test
    # ══════════════════════════════════════════════
    print(f"\n{'=' * 70}\nPART 2: Out-of-sample test (154-obs, 14 periods)\n{'=' * 70}")
    df["period_sort"] = df["year"] * 4 + df["quarter"]
    cutoff = df["period_sort"].quantile(0.75)
    train = df[df["period_sort"] <= cutoff].copy()
    test = df[df["period_sort"] > cutoff].copy()
    print(f"Train: {len(train)} obs ({sorted(train['period'].unique())})")
    print(f"Test:  {len(test)} obs ({sorted(test['period'].unique())})")

    fit_train = fit_with_multistart(train, label="training subset")
    if fit_train is None:
        print("Training refit degenerate -- cannot complete OOS.")
    else:
        best_tr, X1_tr, X2_tr, Q_tr, dm_tr, ds_tr, sm_tr, ss_tr = fit_train
        b1s_tr, b2s_tr, s1_tr, s2_tr = unpack_theta(best_tr.x, N_DEMAND, N_SUPPLY)
        b1_tr = unstandardize_coefs(b1s_tr, DEMAND_VARS, DEMAND_CONTINUOUS, dm_tr, ds_tr)
        b2_tr = unstandardize_coefs(b2s_tr, SUPPLY_CONTINUOUS + SEASON_COLS + FE_COLS,
                                     SUPPLY_CONTINUOUS, sm_tr, ss_tr)

        X1_test = build_demand_design_raw(test)
        X2_test = build_supply_design_raw(test)
        mu1_test = X1_test @ b1_tr
        mu2_test = X2_test @ b2_tr
        pred_Q = np.minimum(mu1_test, mu2_test)
        actual_Q = test["Q"].values

        mae = np.mean(np.abs(pred_Q - actual_Q))
        rmse = np.sqrt(np.mean((pred_Q - actual_Q) ** 2))
        mape = np.mean(np.abs((pred_Q - actual_Q) / actual_Q)) * 100
        print(f"\nOverall: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%  (mean Q={actual_Q.mean():.1f})")

        z1 = np.clip((actual_Q - mu1_test) / s1_tr, -30, 30)
        z2 = np.clip((actual_Q - mu2_test) / s2_tr, -30, 30)
        f1, f2 = stats.norm.pdf(z1) / s1_tr, stats.norm.pdf(z2) / s2_tr
        F1, F2 = stats.norm.cdf(z1), stats.norm.cdf(z2)
        g = np.clip(f1 * (1 - F2) + f2 * (1 - F1), 1e-300, None)
        pi_test = (f1 * (1 - F2)) / g
        demand_side = pi_test > 0.5
        for label, mask in [("supply-constrained", ~demand_side), ("demand-constrained", demand_side)]:
            if mask.sum() > 0:
                m = np.mean(np.abs((pred_Q[mask] - actual_Q[mask]) / actual_Q[mask])) * 100
                print(f"  MAPE on {label}: {m:.1f}% (n={mask.sum()})")

    # ══════════════════════════════════════════════
    # PART 3: Write CSV for R cross-validation re-run
    # ══════════════════════════════════════════════
    print(f"\n{'=' * 70}\nPART 3: R cross-validation input\n{'=' * 70}")
    df.drop(columns=["_idx", "period_sort"], errors="ignore").to_csv(
        "estimation_sample_v4_for_R.csv", index=False)
    print("Saved estimation_sample_v4_for_R.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()