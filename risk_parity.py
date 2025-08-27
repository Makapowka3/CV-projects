import warnings
warnings.filterwarnings("ignore")

import argparse, sys
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DAYS_PER_YEAR = 252

#assets
ASSETS = [
    #US equity / style
    ("SPY",    "S&P 500 (US Equity)"),
    ("QQQ",    "NASDAQ-100 (Growth)"),
    ("EFA",    "MSCI EAFE (Dev ex-US Equity)"),
    ("EEM",    "MSCI Emerging Markets"),
    #US bonds
    ("TLT",    "US Treasuries 20+Y"),
    ("IEF",    "US Treasuries 7-10Y"),
    ("LQD",    "US IG Corporate Bonds"),
    ("HYG",    "US High Yield Bonds"),
    #US real assets
    ("VNQ",    "US REITs"),
    ("GLD",    "Gold"),
    #UK/EU equity
    ("VUSA.L", "S&P 500 UCITS (GBP, LSE)"),
    ("CSPX.L", "S&P 500 UCITS Acc (GBP, LSE)"),
    ("VUKE.L", "FTSE 100 UCITS (GBP, LSE)"),
    ("IWDA.AS","MSCI World UCITS (EUR, AMS)"),
    ("EIMI.AS","MSCI EM UCITS (EUR, AMS)"),
    #UK/EU bonds
    ("AGGG.L", "Global Aggregate Bond UCITS (GBP, LSE)"),
    ("IGLT.L", "UK Gilts UCITS (GBP, LSE)"),
    #UK/EU real assets
    ("SGLN.L", "Physical Gold UCITS (GBP, LSE)"),
    ("IWDP.L", "Dev Mkts Property Yield UCITS (GBP, LSE)"),
]

PRESETS = {
    "US Core 4":        ["SPY","TLT","GLD","VNQ"],
    "US Balanced 6":    ["SPY","TLT","LQD","HYG","GLD","VNQ"],
    "UK/EU Core 4":     ["VUSA.L","AGGG.L","SGLN.L","IWDP.L"],
    "Diversified 7":    ["SPY","TLT","GLD","VNQ","EFA","EEM","LQD"],
}

def _print_presets():
    print("\nAvailable presets:")
    for i, name in enumerate(PRESETS.keys(), 1):
        print(f"{i:2d}. {name:15s} -> {PRESETS[name]}")

def _pick_preset_cli():
    names = list(PRESETS.keys())
    _print_presets()
    while True:
        sel = input("\nChoose a preset by NUMBER (or type a name): ").strip()
        if sel.isdigit():
            k = int(sel)
            if 1 <= k <= len(names):
                chosen = names[k-1]
                return chosen, PRESETS[chosen]
        if sel in PRESETS:
            return sel, PRESETS[sel]
        print("Invalid selection. Try again.")

def parse_args():
    p = argparse.ArgumentParser(description="Portfolio allocation with preset selection")
    p.add_argument("--preset", type=str, help="Preset name (e.g. 'US Core 4')")
    p.add_argument("--list-presets", action="store_true", help="List presets and exit")
    p.add_argument("--tickers", type=str, help="Comma-separated tickers to override preset (e.g. SPY,TLT,GLD,VNQ)")
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--rf", type=float, default=0.00, help="Annual risk-free rate (e.g. 0.02)")
    p.add_argument("--allow-short", action="store_true", help="Allow shorting in Markowitz optimisation")
    p.add_argument("--points", type=int, default=80, help="Number of points on efficient frontier")
    return p.parse_args()

#data and math
def get_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = data["Close"].dropna(how="all")
    if isinstance(px, pd.Series):  # single ticker
        px = px.to_frame()
    px = px.ffill().dropna(how="any", axis=0)
    available = [c for c in px.columns if c in tickers]
    if len(available) < 2:
        raise ValueError("Not enough valid tickers downloaded. Try different symbols or dates.")
    return px[available]

def simple_returns(px):
    return px.pct_change().dropna()

def annualize_ret_cov(rets):
    mu = rets.mean() * DAYS_PER_YEAR
    cov = rets.cov() * DAYS_PER_YEAR
    return mu, cov

def portfolio_performance(w, mu, cov, rf=0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def compute_cagr(price_series):
    if len(price_series) < 2:
        return np.nan
    total_return = price_series[-1] / price_series[0] - 1
    years = (price_series.index[-1] - price_series.index[0]).days / 365.25
    return (1 + total_return) ** (1/years) - 1 if years > 0 else np.nan

def max_drawdown(daily_returns):
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()

#markowitz
def markowitz_min_var(mu, cov, allow_short=False):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = None if allow_short else [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    obj = lambda w: w @ cov @ w
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def markowitz_max_sharpe(mu, cov, rf=0.0, allow_short=False):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = None if allow_short else [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -((ret - rf) / vol) if vol > 0 else 1e6
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def efficient_frontier(mu, cov, allow_short=False, points=80):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = None if allow_short else [(0.0, 1.0)] * n

    mu_min, mu_max = float(mu.min()), float(mu.max())
    targets = np.linspace(mu_min, mu_max, points)

    ws, risks, rets = [], [], []
    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        obj = lambda w: w @ cov @ w
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            r = float(w @ mu)
            v = float(np.sqrt(w @ cov @ w))
            ws.append(w); risks.append(v); rets.append(r)
    return np.array(ws), np.array(risks), np.array(rets)

#risk parity
def risk_contributions(w, cov):
    port_var = w @ cov @ w
    mrc = cov @ w
    rc = w * mrc
    return rc, port_var

def risk_parity_weights(cov, allow_short=False):
    n = cov.shape[0]
    w0 = np.ones(n) / n
    bounds = None if allow_short else [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    def obj(w):
        rc, port_var = risk_contributions(w, cov)
        if port_var <= 0:
            return 1e6
        target = port_var / n
        return ((rc - target)**2).sum()
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 10000, "ftol": 1e-12})
    return res.x

#plots and rep
def allocation_table(names, weights):
    df = pd.DataFrame({"Asset": names, "Weight": weights})
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df

def performance_table(name, w, rets, mu, cov, rf=0.0):
    ann_ret, ann_vol, sharpe = portfolio_performance(w, mu, cov, rf)
    port_daily = (rets @ w).fillna(0.0)
    cum = (1 + port_daily).cumprod()
    cagr = compute_cagr(cum)
    mdd = max_drawdown(port_daily)
    out = pd.DataFrame(
        {"Portfolio":[name],
         "Ann.Return":[ann_ret],
         "Ann.Vol":[ann_vol],
         "Sharpe":[sharpe],
         "CAGR":[cagr],
         "MaxDD":[mdd]}
    )
    return out, cum

def plot_frontier_and_points(risks, rets, pts, labels):
    plt.figure(figsize=(8,6))
    plt.scatter(risks, rets, s=12, alpha=0.7, label="Efficient Frontier")
    for (r, e), lbl in zip(pts, labels):
        plt.scatter([r], [e], s=60, marker="X", label=lbl)
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier & Key Portfolios")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cum_curves(curves_dict):
    plt.figure(figsize=(9,6))
    for name, series in curves_dict.items():
        plt.plot(series.index, series.values, label=name)
    plt.title("Cumulative Growth of 1")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_risk_contributions(name, w, cov, tickers):
    rc, _ = risk_contributions(w, cov)
    rc_share = rc / rc.sum()
    plt.figure(figsize=(7,5))
    plt.bar(tickers, rc_share)
    plt.title(f"Risk Contributions — {name}")
    plt.ylabel("Share of Portfolio Risk")
    plt.tight_layout()
    plt.show()

#m
def main():
    args = parse_args()

    if args.list_presets:
        _print_presets()
        sys.exit(0)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        preset_used = "Custom (CLI)"
    elif args.preset:
        if args.preset not in PRESETS:
            print(f"Unknown preset: {args.preset!r}. Use --list-presets to see options.")
            sys.exit(1)
        preset_used = args.preset
        tickers = PRESETS[preset_used]
    else:
        preset_used, tickers = _pick_preset_cli()

    print(f"\n[Universe] Using: {preset_used} -> {tickers}")
    print(f"[Config] start={args.start} end={args.end or 'today'} rf={args.rf} allow_short={args.allow_short} points={args.points}")

    #data
    px = get_prices(tickers, args.start, args.end)
    rets = simple_returns(px)
    mu, cov = annualize_ret_cov(rets)
    names = list(px.columns)
    n = len(names)

    #portfolios
    w_eq  = np.ones(n) / n
    w_min = markowitz_min_var(mu.values, cov.values, allow_short=args.allow_short)
    w_tan = markowitz_max_sharpe(mu.values, cov.values, rf=args.rf, allow_short=args.allow_short)
    w_rp  = risk_parity_weights(cov.values, allow_short=args.allow_short)

    #frontier
    _, risks, rets_front = efficient_frontier(mu.values, cov.values, allow_short=args.allow_short, points=args.points)

    #tables
    alloc_eq  = allocation_table(names, w_eq)
    alloc_min = allocation_table(names, w_min)
    alloc_tan = allocation_table(names, w_tan)
    alloc_rp  = allocation_table(names, w_rp)

    perf_eq,  cum_eq  = performance_table("Equal-Weight", w_eq,  rets, mu.values, cov.values, args.rf)
    perf_min, cum_min = performance_table("Min-Variance", w_min, rets, mu.values, cov.values, args.rf)
    perf_tan, cum_tan = performance_table("Max-Sharpe",   w_tan, rets, mu.values, cov.values, args.rf)
    perf_rp,  cum_rp  = performance_table("Risk Parity",  w_rp,  rets, mu.values, cov.values, args.rf)

    perf_all = pd.concat([perf_eq, perf_min, perf_tan, perf_rp], ignore_index=True)
    perf_all[["Ann.Return","Ann.Vol","Sharpe","CAGR","MaxDD"]] = perf_all[["Ann.Return","Ann.Vol","Sharpe","CAGR","MaxDD"]].round(4)

    #print summaries
    print("\n=== Selected Tickers ===")
    print(names)
    print("\n=== Allocation Tables ===")
    print("\nEqual-Weight\n", alloc_eq.to_string(index=False))
    print("\nMin-Variance\n", alloc_min.to_string(index=False))
    print("\nMax-Sharpe\n",   alloc_tan.to_string(index=False))
    print("\nRisk Parity\n",  alloc_rp.to_string(index=False))

    print("\n=== Performance (Annualized unless noted) ===")
    print(perf_all.to_string(index=False))

    #plots
    pts, labels = [], []
    for name, w in [("Equal-Weight", w_eq), ("Min-Var", w_min), ("Max-Sharpe", w_tan), ("Risk Parity", w_rp)]:
        r, v, _ = portfolio_performance(w, mu.values, cov.values, args.rf)
        pts.append((v, r)); labels.append(name)
    plot_frontier_and_points(risks, rets_front, pts, labels)

    curves = {
        "Equal-Weight": cum_eq,
        "Min-Variance": cum_min,
        "Max-Sharpe":   cum_tan,
        "Risk Parity":  cum_rp,
    }
    plot_cum_curves(curves)
    plot_risk_contributions("Risk Parity", w_rp, cov.values, names)
    plot_risk_contributions("Max-Sharpe", w_tan, cov.values, names)

if __name__ == "__main__":
    main()
