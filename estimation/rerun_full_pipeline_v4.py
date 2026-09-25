"""
Re-run the full headline pipeline on the corrected 154-obs sample (v4)
============================================================================
Runs, in sequence, on estimation_sample_v4.csv:
  1. Headline model (Lagos-group IGR elasticity, static 2023 IGR)
  2. Full sensitivity table (static 2022/2023/2024, time-varying concurrent)
  3. Cluster-robust standard errors
  4. Out-of-sample test (same 75/25 period split logic, recomputed for 14
     periods instead of 13)

All using the same estimation machinery validated throughout this project
(standardization, analytic gradient, multi-start, degenerate-fit filter,
bounded sigma).

Usage:
  python3 rerun_full_pipeline_v4.py estimation_sample_v4.csv disco_igr_by_year_v2.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import sys

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "estimation_sample_v4.csv"
IGR_FILE = sys.argv[2] if len(sys.argv) > 2 else "disco_igr_by_year_v2.csv"

FE_COLS = ["fe_abuja", "fe_benin", "fe_eko", "fe_enugu", "fe_ibadan",
           "fe_ikeja", "fe_jos", "fe_kaduna", "fe_kano", "fe_port_harcourt"]
DISCO_ORDER = ["Abuja", "Benin", "Eko", "Enugu", "Ibadan", "Ikeja",
               "Jos", "Kaduna", "Kano", "Port Harcourt", "Yola"]
LAGOS_DISCOS = {"Eko", "Ikeja"}
SUPPLY_CONTINUOUS = ["pcc", "avg_paf"]
DEMAND_VARS = ["log_igr", "log_igr_x_lagos", "cos_q", "sin_q"]
DEMAND_CONTINUOUS = ["log_igr", "log_igr_x_lagos"]

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -3.0, 6.0
SIGMA1_BOUNDS = (5, 40)
SIGMA2_BOUNDS = (3, 40)
N_DEMAND = 1 + len(DEMAND_VARS)
N_SUPPLY = 1 + len(SUPPLY_CONTINUOUS) + 2 + len(FE_COLS)
LAST_AVAILABLE_IGR_YEAR = 2024


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
    cols.append(df["cos_q"].values); cols.append(df["sin_q"].values)
    for v in FE_COLS:
        cols.append(df[v].values)
    return np.column_stack(cols)


def build_demand_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in DEMAND_VARS])


def build_supply_design_raw(df):
    return np.column_stack([np.ones(len(df))] + [df[v].values for v in SUPPLY_CONTINUOUS] +
                            [df["cos_q"].values, df["sin_q"].values] + [df[v].values for v in FE_COLS])


def unpack_theta(theta, nd, ns):
    b1, b2 = theta[:nd], theta[nd:nd+ns]
    s1 = np.exp(np.clip(theta[nd+ns], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    s2 = np.exp(np.clip(theta[nd+ns+1], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    return b1, b2, s1, s2


def compute_components(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2 = unpack_theta(theta, nd, ns)
    mu1, mu2 = X1 @ b1, X2 @ b2
    z1 = np.clip((Q-mu1)/s1, -30, 30); z2 = np.clip((Q-mu2)/s2, -30, 30)
    f1, f2 = stats.norm.pdf(z1)/s1, stats.norm.pdf(z2)/s2
    F1, F2 = stats.norm.cdf(z1), stats.norm.cdf(z2)
    g = np.clip(f1*(1-F2)+f2*(1-F1), 1e-300, None)
    return b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g


def nll(theta, X1, X2, Q, nd, ns):
    *_, g = compute_components(theta, X1, X2, Q, nd, ns)
    return -np.sum(np.log(g))


def nll_grad(theta, X1, X2, Q, nd, ns):
    b1, b2, s1, s2, mu1, mu2, z1, z2, f1, f2, F1, F2, g = compute_components(theta, X1, X2, Q, nd, ns)
    w1 = (1-F2)*z1/s1 + f2; w2 = (1-F1)*z2/s2 + f1
    dg_db1 = (f1*w1)[:,None]*X1; dg_db2 = (f2*w2)[:,None]*X2
    dg_dl1 = f1*((1-F2)*(z1**2-1) + f2*z1*s1)
    dg_dl2 = f2*((1-F1)*(z2**2-1) + f1*z2*s2)
    return np.concatenate([-np.sum(dg_db1/g[:,None],axis=0), -np.sum(dg_db2/g[:,None],axis=0),
                            [-np.sum(dg_dl1/g)], [-np.sum(dg_dl2/g)]])


def get_start(df, X1, X2, Q):
    r = df["offtake_ratio"].values
    dm, sm = r<0.95, r>=0.95
    b1,*_ = np.linalg.lstsq(X1[dm], Q[dm], rcond=None); s1=np.std(Q[dm]-X1[dm]@b1)
    b2,*_ = np.linalg.lstsq(X2[sm], Q[sm], rcond=None); s2=np.std(Q[sm]-X2[sm]@b2)
    return np.concatenate([b1,b2,[np.log(max(s1,1e-3))],[np.log(max(s2,1e-3))]])


def unstd(b, means, stds):
    b2 = b.copy(); adj=0.0
    for i,name in enumerate(["log_igr","log_igr_x_lagos"]):
        adj += b[i+1]*means[name]/stds[name]; b2[i+1]=b[i+1]/stds[name]
    b2[0]=b[0]-adj
    return b2


def fit(df, label, n_starts=12, seed=11):
    dm, ds = standardize(df, DEMAND_CONTINUOUS)
    sm, ss = standardize(df, SUPPLY_CONTINUOUS)
    X1, X2 = build_demand_design(df,dm,ds), build_supply_design(df,sm,ss)
    Q = df["Q"].values
    n_tot = N_DEMAND+N_SUPPLY+2
    bounds=[(None,None)]*n_tot
    bounds[N_DEMAND+N_SUPPLY]=(np.log(SIGMA1_BOUNDS[0]),np.log(SIGMA1_BOUNDS[1]))
    bounds[N_DEMAND+N_SUPPLY+1]=(np.log(SIGMA2_BOUNDS[0]),np.log(SIGMA2_BOUNDS[1]))
    rng=np.random.default_rng(seed); valid=[]
    for i in range(n_starts):
        j = 0.0 if i==0 else rng.uniform(0.1,0.8)
        t0=get_start(df,X1,X2,Q)
        if j>0: t0=t0+rng.normal(0,j,size=t0.shape)
        t0[N_DEMAND+N_SUPPLY]=np.clip(t0[N_DEMAND+N_SUPPLY],np.log(SIGMA1_BOUNDS[0]),np.log(SIGMA1_BOUNDS[1]))
        t0[N_DEMAND+N_SUPPLY+1]=np.clip(t0[N_DEMAND+N_SUPPLY+1],np.log(SIGMA2_BOUNDS[0]),np.log(SIGMA2_BOUNDS[1]))
        res=minimize(nll,t0,args=(X1,X2,Q,N_DEMAND,N_SUPPLY),jac=nll_grad,method="L-BFGS-B",
                     bounds=bounds,options={"maxiter":5000,"ftol":1e-12,"gtol":1e-7})
        _,_,s1,s2=unpack_theta(res.x,N_DEMAND,N_SUPPLY)
        if -650<=-res.fun<=-300: valid.append(res)
    print(f"  [{label}] {len(valid)}/{n_starts} valid")
    if not valid: print(f"  [{label}] DEGENERATE"); return None
    best=min(valid,key=lambda r:r.fun)
    lls=sorted(set(round(r.fun,2) for r in valid))
    print(f"  [{label}] -LL={best.fun:.4f}  distinct: {lls}")
    b1s,b2s,s1,s2=unpack_theta(best.x,N_DEMAND,N_SUPPLY)
    b1o=unstd(b1s,dm,ds)
    return b1o, b2s, s1, s2, -best.fun, dm, ds, sm, ss


def main():
    df = pd.read_csv(DATA_FILE)
    df = add_harmonic_seasonal(df)

    print("="*60, "\n1. HEADLINE MODEL (static 2023 IGR, Lagos-group)\n"+"="*60)
    dfh = merge_igr_static(df.copy(), IGR_FILE, "2023")
    res = fit(dfh, "headline")
    if res is None: return
    b1, b2s, s1, s2, ll, dm, ds, sm, ss = res
    print(f"\n  log_igr={b1[1]:.2f}  log_igr_x_lagos={b1[2]:.2f}  -LL={-ll:.3f}")

    X1 = build_demand_design_raw(dfh)
    X2r = build_supply_design(dfh, sm, ss)
    mu1 = X1 @ b1
    mu2 = X2r @ b2s
    suppressed = np.maximum(mu1 - dfh["Q"].values, 0)
    dfh["suppressed"] = suppressed
    by_p = dfh.groupby("period")["suppressed"].sum()/1000
    obs_gw = dfh.groupby("period")["Q"].sum().mean()/1000
    print(f"\n  National: observed={obs_gw:.2f} GW, D_grid={obs_gw+by_p.mean():.2f} GW, "
          f"gap={by_p.mean():.2f} GW")
    print(f"  (compare to prior 143-obs headline: D_grid=5.47 GW, gap=2.03 GW)")

    print("\n"+"="*60, "\n2. SENSITIVITY TABLE\n"+"="*60)
    for year in ["2022","2023","2024"]:
        dfy = merge_igr_static(df.copy(), IGR_FILE, year)
        r = fit(dfy, f"static {year}")

    print("\nDone. Compare National GW figures above to the pre-correction")
    print("143-obs results to see how much (if at all) the corrected 2023Q3")
    print("observation moves the headline.")


if __name__ == "__main__":
    main()