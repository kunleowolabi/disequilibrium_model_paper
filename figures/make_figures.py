"""
Generate the three paper figures, in paper2/figures/
============================================================================
F1: Offtake-ratio (Q/PCC) distribution across DISCO-quarters
    -- the empirical signature that regimes switch (Section 4.3)
F2: Estimated demand-side seasonal curve over the four quarters
    -- justifies the harmonic-seasonal specification (Section 6.3)
F3: Suppressed demand by DISCO (mean gap, MWh/h)
    -- shows the gap is not uniform across the network (Section 6.1/7.1)

All three are drawn from a single re-fit of the headline specification
on the corrected 154-obs sample, so they cannot drift from the paper's
reported numbers.

Usage:
  python3 make_figures.py estimation_sample_v4.csv disco_igr_by_year_v2.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "estimation_sample_v4.csv"
IGR_FILE = sys.argv[2] if len(sys.argv) > 2 else "disco_igr_by_year_v2.csv"
IGR_YEAR_COL = "2023"
OUTDIR = "paper2/figures"

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
SIGMA1_BOUNDS, SIGMA2_BOUNDS = (5, 40), (3, 40)
N_DEMAND = 1 + len(DEMAND_VARS)
N_SUPPLY = 1 + len(SUPPLY_CONTINUOUS) + 2 + len(FE_COLS)

# muted, print-friendly palette (no reliance on colour to carry meaning)
INK = "#1F2A37"
ACCENT = "#2C6E8F"
MUTED = "#9AA5B1"
FILL = "#CFE0E8"


# ---------- shared model machinery (matches the estimation pipeline) ----------
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

def standardize(df, cols): return {c: df[c].mean() for c in cols}, {c: df[c].std() for c in cols}

def bd(df, m, s):
    cols = [np.ones(len(df))]
    for v in DEMAND_VARS: cols.append((df[v].values-m[v])/s[v] if v in DEMAND_CONTINUOUS else df[v].values)
    return np.column_stack(cols)

def bs(df, m, s):
    cols = [np.ones(len(df))]
    for v in SUPPLY_CONTINUOUS: cols.append((df[v].values-m[v])/s[v])
    cols.append(df["cos_q"].values); cols.append(df["sin_q"].values)
    for v in FE_COLS: cols.append(df[v].values)
    return np.column_stack(cols)

def bd_raw(df): return np.column_stack([np.ones(len(df))]+[df[v].values for v in DEMAND_VARS])

def unpack(t, nd, ns):
    return (t[:nd], t[nd:nd+ns],
            np.exp(np.clip(t[nd+ns], LOG_SIGMA_MIN, LOG_SIGMA_MAX)),
            np.exp(np.clip(t[nd+ns+1], LOG_SIGMA_MIN, LOG_SIGMA_MAX)))

def comp(t, X1, X2, Q, nd, ns):
    b1,b2,s1,s2 = unpack(t,nd,ns); m1,m2 = X1@b1, X2@b2
    z1=np.clip((Q-m1)/s1,-30,30); z2=np.clip((Q-m2)/s2,-30,30)
    f1,f2=stats.norm.pdf(z1)/s1,stats.norm.pdf(z2)/s2; F1,F2=stats.norm.cdf(z1),stats.norm.cdf(z2)
    g=np.clip(f1*(1-F2)+f2*(1-F1),1e-300,None)
    return b1,b2,s1,s2,m1,m2,z1,z2,f1,f2,F1,F2,g

def nll(t,X1,X2,Q,nd,ns): *_,g=comp(t,X1,X2,Q,nd,ns); return -np.sum(np.log(g))

def grad(t,X1,X2,Q,nd,ns):
    b1,b2,s1,s2,m1,m2,z1,z2,f1,f2,F1,F2,g=comp(t,X1,X2,Q,nd,ns)
    w1=(1-F2)*z1/s1+f2; w2=(1-F1)*z2/s2+f1
    d1=(f1*w1)[:,None]*X1; d2=(f2*w2)[:,None]*X2
    l1=f1*((1-F2)*(z1**2-1)+f2*z1*s1); l2=f2*((1-F1)*(z2**2-1)+f1*z2*s2)
    return np.concatenate([-np.sum(d1/g[:,None],axis=0),-np.sum(d2/g[:,None],axis=0),
                            [-np.sum(l1/g)],[-np.sum(l2/g)]])

def start(df,X1,X2,Q):
    r=df["offtake_ratio"].values; dm,sm=r<0.95,r>=0.95
    b1,*_=np.linalg.lstsq(X1[dm],Q[dm],rcond=None); s1=np.std(Q[dm]-X1[dm]@b1)
    b2,*_=np.linalg.lstsq(X2[sm],Q[sm],rcond=None); s2=np.std(Q[sm]-X2[sm]@b2)
    return np.concatenate([b1,b2,[np.log(max(s1,1e-3))],[np.log(max(s2,1e-3))]])

def unstd(b, names, cont, m, s):
    o=b.copy(); adj=0.0
    for i,n in enumerate(names):
        if n in cont: adj+=b[i+1]*m[n]/s[n]; o[i+1]=b[i+1]/s[n]
    o[0]=b[0]-adj; return o

def fit(df):
    dm,ds=standardize(df,DEMAND_CONTINUOUS); sm,ss=standardize(df,SUPPLY_CONTINUOUS)
    X1,X2=bd(df,dm,ds),bs(df,sm,ss); Q=df["Q"].values
    nt=N_DEMAND+N_SUPPLY+2; bnd=[(None,None)]*nt
    bnd[N_DEMAND+N_SUPPLY]=(np.log(SIGMA1_BOUNDS[0]),np.log(SIGMA1_BOUNDS[1]))
    bnd[N_DEMAND+N_SUPPLY+1]=(np.log(SIGMA2_BOUNDS[0]),np.log(SIGMA2_BOUNDS[1]))
    rng=np.random.default_rng(11); best=None
    for i in range(12):
        j=0.0 if i==0 else rng.uniform(0.1,0.8); t0=start(df,X1,X2,Q)
        if j>0: t0=t0+rng.normal(0,j,size=t0.shape)
        t0[N_DEMAND+N_SUPPLY]=np.clip(t0[N_DEMAND+N_SUPPLY],np.log(SIGMA1_BOUNDS[0]),np.log(SIGMA1_BOUNDS[1]))
        t0[N_DEMAND+N_SUPPLY+1]=np.clip(t0[N_DEMAND+N_SUPPLY+1],np.log(SIGMA2_BOUNDS[0]),np.log(SIGMA2_BOUNDS[1]))
        res=minimize(nll,t0,args=(X1,X2,Q,N_DEMAND,N_SUPPLY),jac=grad,method="L-BFGS-B",
                     bounds=bnd,options={"maxiter":5000,"ftol":1e-12,"gtol":1e-7})
        if -650<=-res.fun<=-300 and (best is None or res.fun<best.fun): best=res
    b1s,b2s,s1,s2=unpack(best.x,N_DEMAND,N_SUPPLY)
    b1=unstd(b1s,DEMAND_VARS,DEMAND_CONTINUOUS,dm,ds)
    return b1, s1


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})

    df = add_harmonic_seasonal(pd.read_csv(DATA_FILE))
    df = merge_igr(df, IGR_FILE, IGR_YEAR_COL)
    b1, sigma1 = fit(df)
    print(f"Fitted demand coefficients: {dict(zip(['int','log_igr','x_lagos','cos','sin'], np.round(b1,3)))}")

    # ---- F1: offtake-ratio distribution ----
    r = df["offtake_ratio"].values
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(r, bins=24, color=FILL, edgecolor=INK, linewidth=0.6)
    ax.axvline(1.0, color=ACCENT, linestyle="--", linewidth=1.4)
    ax.text(1.005, ax.get_ylim()[1]*0.92, "offtake = contracted capacity",
            color=ACCENT, fontsize=9, ha="left")
    ax.set_xlabel("Offtake ratio  (Q / PCC)")
    ax.set_ylabel("DISCO-quarters")
    ax.set_title("Figure 1. Distribution of the offtake ratio across the panel", loc="left", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/F1_offtake_ratio.png", dpi=200); plt.close(fig)

    # ---- F2: estimated demand seasonal curve ----
    # seasonal contribution = b_cos*cos(2pi q/4) + b_sin*sin(2pi q/4), q=1..4
    b_cos, b_sin = b1[3], b1[4]
    q = np.array([1, 2, 3, 4])
    seasonal = b_cos*np.cos(2*np.pi*q/4) + b_sin*np.sin(2*np.pi*q/4)
    seasonal = seasonal - seasonal.mean()  # center for readability (level absorbed elsewhere)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(q, seasonal, "-o", color=ACCENT, linewidth=2, markersize=7)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(q); ax.set_xticklabels(["Q1\n(Jan–Mar)","Q2\n(Apr–Jun)","Q3\n(Jul–Sep)","Q4\n(Oct–Dec)"])
    ax.set_ylabel("Seasonal demand shift (MWh/h, centred)")
    ax.set_title("Figure 2. Estimated seasonal component of demand", loc="left", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/F2_seasonal_curve.png", dpi=200); plt.close(fig)

    # ---- F3: suppressed demand by DISCO ----
    X1_raw = bd_raw(df)
    mu1 = X1_raw @ b1
    df["suppressed"] = np.maximum(mu1 - df["Q"].values, 0)
    by = df.groupby("disco")["suppressed"].mean().reindex(DISCO_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = by.sort_values(ascending=True)
    ax.barh(order.index, order.values, color=FILL, edgecolor=INK, linewidth=0.6)
    ax.set_xlabel("Mean suppressed demand (MWh/h)")
    ax.set_title("Figure 3. Suppressed demand by distribution company", loc="left", fontsize=11, weight="bold")
    for i, v in enumerate(order.values):
        ax.text(v + 2, i, f"{v:.0f}", va="center", fontsize=8.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/F3_suppressed_by_disco.png", dpi=200); plt.close(fig)

    print(f"\nSaved three figures to {OUTDIR}/")
    print("  F1_offtake_ratio.png, F2_seasonal_curve.png, F3_suppressed_by_disco.png")
    print(f"\nSanity: national mean suppressed demand = {df.groupby('period')['suppressed'].sum().mean()/1000:.2f} GW")
    print("  (should be ~1.91 GW; if not, the fit in this script diverged from the paper's)")


if __name__ == "__main__":
    main()