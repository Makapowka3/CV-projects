import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

#inputs with defaults
def get_float(prompt, default):
    s = input(f"{prompt} [default {default}]: ").strip()
    if s == "":
        return float(default)
    try:
        return float(s)
    except ValueError:
        print("Invalid number, using default.")
        return float(default)

def get_int(prompt, default):
    s = input(f"{prompt} [default {default}]: ").strip()
    if s == "":
        return int(default)
    try:
        return int(s)
    except ValueError:
        print("Invalid integer, using default.")
        return int(default)

#parameters
START_DATE   = "2010-01-01"
USE_BONDS    = True       #if False: SPY-only; if True: 60/40 SPY/TLT base
BASE_W_SPY   = 0.60
BASE_W_TLT   = 0.40
COST_BPS     = 0.0001     #per $ traded (1 bp = 0.01%)
REBAL_FREQ   = "D"        #'D' or 'W-FRI'

TARGET_VOL   = get_float("Enter target volatility (annualized, e.g., 0.10 for 10%)", 0.10)
LOOKBACK     = get_int("Enter lookback window in trading days (e.g., 20)", 20)
LEVERAGE_CAP = get_float("Enter leverage cap (e.g., 1.75)", 1.75)

#help funcs
def download_prices(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def annualized_vol(returns: pd.Series, lookback: int) -> pd.Series:
    return returns.rolling(lookback).std() * np.sqrt(252)

def max_drawdown(cum_curve: pd.Series) -> float:
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1.0
    return drawdown.min()

def sortino_ratio(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if downside.std() == 0 or downside.empty:
        return np.nan
    ann_excess = excess.mean() * 252
    ann_downside = downside.std() * np.sqrt(252)
    return ann_excess / ann_downside

def summarize_performance(daily_returns: pd.Series, label: str):
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        print(f"\n{label}: no data")
        return

    cum = (1 + daily_returns).cumprod()
    n_days = daily_returns.shape[0]
    ann_ret = cum.iloc[-1] ** (252 / n_days) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe  = np.nan if ann_vol == 0 else ann_ret / ann_vol
    mdd     = max_drawdown(cum)
    calmar  = np.nan if mdd == 0 else ann_ret / abs(mdd)
    sortino = sortino_ratio(daily_returns)

    print(f"\n── {label} ──")
    print(f"CAGR     : {ann_ret:6.2%}")
    print(f"Vol      : {ann_vol:6.2%}")
    print(f"Sharpe   : {sharpe:6.2f}")
    print(f"Max DD   : {mdd:6.2%}")
    print(f"Calmar   : {calmar:6.2f}")
    print(f"Sortino  : {sortino:6.2f}")

def align_rebalance(signal: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return signal.ffill()
    sampled = signal.resample(freq).last()
    return sampled.reindex(signal.index).ffill()

#data load
tickers = ["SPY", "TLT", "BIL"]  # BIL ~ T-Bill proxy (cash)
prices  = download_prices(tickers, START_DATE)
rets    = compute_returns(prices)

r_spy  = rets["SPY"].fillna(0.0)
r_tlt  = rets["TLT"].fillna(0.0) if "TLT" in rets.columns else pd.Series(0.0, index=rets.index)
r_cash = (rets["BIL"] if "BIL" in rets.columns else pd.Series(0.0, index=rets.index)).fillna(0.0)

#core backtest
def backtest_vol_target(target_vol: float,
                        lookback: int,
                        leverage_cap: float,
                        use_bonds: bool,
                        base_w_spy: float,
                        base_w_tlt: float,
                        cost_bps: float,
                        rebal_freq: str) -> dict:
    """
    Returns dict with:
      'strat_ret', 'weights', 'scale', 'turnover', 'bench_spy', 'bench_6040',
      and quick metrics ('sharpe','cagr','mdd') for grids.
    """
    #base portfolio
    if use_bonds and "TLT" in rets.columns:
        base_ret = base_w_spy * r_spy + base_w_tlt * r_tlt
    else:
        base_ret = r_spy
        use_bonds = False

    #realized vol
    realized_vol = annualized_vol(base_ret, lookback).shift(1)

    # scaling to target vol
    scale_raw = (target_vol / realized_vol).clip(lower=0, upper=leverage_cap)
    scale = align_rebalance(scale_raw, rebal_freq)

    #weights
    if use_bonds:
        w_spy  = base_w_spy * scale
        w_tlt  = base_w_tlt * scale
        w_cash = 1.0 - scale
    else:
        w_spy  = scale
        w_tlt  = pd.Series(0.0, index=rets.index)
        w_cash = 1.0 - scale

    weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt, "CASH": w_cash}).reindex(rets.index)

    #turnover & transaction costs
    weights_lag    = weights.shift(1).fillna(0.0)
    daily_turnover = (weights - weights_lag).abs().sum(axis=1)
    daily_costs    = daily_turnover * cost_bps

    #strategy returns
    strat_ret_gross = (w_spy * r_spy) + (w_tlt * r_tlt) + (w_cash * r_cash)
    strat_ret = strat_ret_gross - daily_costs

    #benchmarks
    bench_spy  = r_spy
    bench_6040 = 0.6 * r_spy + 0.4 * r_tlt

    #quick metrics
    strat = strat_ret.dropna()
    ann_vol = strat.std() * np.sqrt(252)
    if ann_vol == 0 or strat.empty:
        sharpe = np.nan
        cagr   = np.nan
        mdd    = np.nan
    else:
        cum = (1 + strat).cumprod()
        n_days = strat.shape[0]
        cagr   = cum.iloc[-1] ** (252 / n_days) - 1
        sharpe = (strat.mean() * 252) / ann_vol
        mdd    = max_drawdown(cum)

    return {
        "strat_ret": strat_ret,
        "weights": weights,
        "scale": scale,
        "turnover": daily_turnover,
        "bench_spy": bench_spy,
        "bench_6040": bench_6040,
        "sharpe": sharpe,
        "cagr": cagr,
        "mdd": mdd
    }

#base run for display
base = backtest_vol_target(
    target_vol=TARGET_VOL,
    lookback=LOOKBACK,
    leverage_cap=LEVERAGE_CAP,
    use_bonds=USE_BONDS,
    base_w_spy=BASE_W_SPY,
    base_w_tlt=BASE_W_TLT,
    cost_bps=COST_BPS,
    rebal_freq=REBAL_FREQ
)

summarize_performance(base["strat_ret"], "Vol-Target Strategy")
summarize_performance(base["bench_spy"], "Buy & Hold SPY")
if USE_BONDS:
    summarize_performance(base["bench_6040"], "Static 60/40")

ann_turnover = base["turnover"].mean() * 252
print(f"\nAvg annualized turnover (approx): {ann_turnover:6.2f}x")

#plot
cum_strat = (1 + base["strat_ret"]).cumprod()
cum_spy   = (1 + base["bench_spy"]).cumprod()
plot_df   = pd.DataFrame({"Strat": cum_strat, "SPY": cum_spy})
if USE_BONDS:
    plot_df["60/40"] = (1 + base["bench_6040"]).cumprod()

#equity curve
plt.figure(figsize=(10, 5))
plot_df.dropna().plot(ax=plt.gca())
plt.title("Equity Curves")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.show()

#scaling
plt.figure(figsize=(10, 3))
base["scale"].plot()
plt.title("Scaling (Effective Exposure to Base Portfolio)")
plt.ylabel("Scale (x)")
plt.grid(True)
plt.show()

#sharpe by LOOKBACK × LEVERAGE_CAP
lookback_grid     = [15, 20, 30, 60]
leverage_cap_grid = [1.00, 1.25, 1.50, 1.75, 2.00]

records = []
for lb in lookback_grid:
    for lev in leverage_cap_grid:
        res = backtest_vol_target(
            target_vol=TARGET_VOL,
            lookback=lb,
            leverage_cap=lev,
            use_bonds=USE_BONDS,
            base_w_spy=BASE_W_SPY,
            base_w_tlt=BASE_W_TLT,
            cost_bps=COST_BPS,
            rebal_freq=REBAL_FREQ
        )
        records.append({"LOOKBACK": lb, "LEVERAGE_CAP": lev, "Sharpe": res["sharpe"]})

grid_df = pd.DataFrame(records)
heat1 = grid_df.pivot(index="LOOKBACK", columns="LEVERAGE_CAP", values="Sharpe").reindex(index=lookback_grid, columns=leverage_cap_grid)

plt.figure(figsize=(7.5, 4.8))
im = plt.imshow(heat1.values, aspect='auto')
plt.title("Robustness: Sharpe by LOOKBACK × LEVERAGE_CAP")
plt.xticks(ticks=np.arange(len(leverage_cap_grid)), labels=[str(x) for x in leverage_cap_grid])
plt.yticks(ticks=np.arange(len(lookback_grid)), labels=[str(x) for x in lookback_grid])
plt.xlabel("LEVERAGE_CAP")
plt.ylabel("LOOKBACK (days)")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Sharpe")
for i, lb in enumerate(lookback_grid):
    for j, lev in enumerate(leverage_cap_grid):
        val = heat1.iloc[i, j]
        txt = f"{val:.2f}" if pd.notna(val) else "NA"
        plt.text(j, i, txt, ha='center', va='center', fontsize=9,
                 color='white' if pd.notna(val) and val < heat1.values.mean() else 'black')
plt.tight_layout()
plt.show()

# Sharpe by TARGET_VOL × LEVERAGE_CAP
target_vol_grid   = [0.10, 0.15, 0.20, 0.25]
leverage_cap_grid = [1.00, 1.25, 1.50, 1.75, 2.00]

records_tv = []
for tv in target_vol_grid:
    for lev in leverage_cap_grid:
        res = backtest_vol_target(
            target_vol=tv,
            lookback=LOOKBACK,  
            leverage_cap=lev,
            use_bonds=USE_BONDS,
            base_w_spy=BASE_W_SPY,
            base_w_tlt=BASE_W_TLT,
            cost_bps=COST_BPS,
            rebal_freq=REBAL_FREQ
        )
        records_tv.append({"TARGET_VOL": tv, "LEVERAGE_CAP": lev, "Sharpe": res["sharpe"]})

grid_tv = pd.DataFrame(records_tv)
heat2 = grid_tv.pivot(index="TARGET_VOL", columns="LEVERAGE_CAP", values="Sharpe").reindex(index=target_vol_grid, columns=leverage_cap_grid)

plt.figure(figsize=(7.5, 4.8))
im2 = plt.imshow(heat2.values, aspect='auto')
plt.title("Robustness: Sharpe by TARGET_VOL × LEVERAGE_CAP")
plt.xticks(ticks=np.arange(len(leverage_cap_grid)), labels=[str(x) for x in leverage_cap_grid])
plt.yticks(ticks=np.arange(len(target_vol_grid)), labels=[str(x) for x in target_vol_grid])
plt.xlabel("LEVERAGE_CAP")
plt.ylabel("TARGET_VOL (annualized)")
plt.colorbar(im2, fraction=0.046, pad=0.04, label="Sharpe")
for i, tv in enumerate(target_vol_grid):
    for j, lev in enumerate(leverage_cap_grid):
        val = heat2.iloc[i, j]
        txt = f"{val:.2f}" if pd.notna(val) else "NA"
        plt.text(j, i, txt, ha='center', va='center', fontsize=9,
                 color='white' if pd.notna(val) and val < heat2.values.mean() else 'black')
plt.tight_layout()
plt.show()
