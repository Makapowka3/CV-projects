import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

#settings
TX_COST       = 0.0005         # 0.05 % commission per trade
DAYS_PER_YEAR = 252

SPAN_LIMIT_YR = 5              # keep arrows only for recent N yrs if span > 5 yrs
LONG_SPAN_YR  = 8              # if span > 8 yrs, also thin arrows
ARROW_YRS     = 20              # “recent” window length (yrs)
ARROW_SKIP_N  = 3              # keep every N‑th trade when thinning
ARROW_SIZE    = 60             # marker size

GRID_FAST = range(5, 55, 5)    # 5–50   step 5
GRID_SLOW = range(20, 210, 10) # 20–200 step 10

#funcs
def get_data(tkr, start, end=None):
    return yf.download(tkr, start=start, end=end)

def add_signals(df, fast, slow):
    df[f"SMA{fast}"] = df["Close"].rolling(fast).mean()
    df[f"SMA{slow}"] = df["Close"].rolling(slow).mean()
    df.dropna(inplace=True)
    df["Signal"]   = (df[f"SMA{fast}"] > df[f"SMA{slow}"]).astype(int)
    df["Position"] = df["Signal"].diff()
    return df

def backtest(df):
    df["Market_Ret"] = df["Close"].pct_change()
    df["Strat_Ret"]  = df["Market_Ret"] * df["Signal"]
    df["Strat_Ret"] -= TX_COST * df["Position"].abs().fillna(0)
    df.dropna(inplace=True)
    df["Market_Cum"] = (1 + df["Market_Ret"]).cumprod()
    df["Strat_Cum"]  = (1 + df["Strat_Ret"]).cumprod()
    return df

#metric
def sharpe(series):
    mu, sig = series.mean(), series.std(ddof=0)
    return (mu / sig) * np.sqrt(DAYS_PER_YEAR) if sig else np.nan

def cagr(series, periods=DAYS_PER_YEAR):
    t = len(series) / periods
    return series.iloc[-1] ** (1 / t) - 1

def max_dd(series):
    roll_max = series.cummax()
    return 1 - (series / roll_max).min()

#grid
def grid_search(df_raw, metric="sharpe"):
    results = pd.DataFrame(index=GRID_FAST, columns=GRID_SLOW, dtype=float)
    for fast in GRID_FAST:
        for slow in GRID_SLOW:
            if fast >= slow:
                continue
            df = df_raw.copy()
            add_signals(df, fast, slow)
            backtest(df)
            if metric == "sharpe":
                score = sharpe(df["Strat_Ret"])
            elif metric == "cagr":
                score = cagr(df["Strat_Cum"])
            else:
                score = df["Strat_Cum"].iloc[-1]
            results.loc[fast, slow] = score
    return results

def plot_heatmap(df_metric, metric_name, tkr):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_metric, cmap="viridis", cbar_kws={"label": metric_name})
    plt.title(f"{tkr}: {metric_name} across SMA Grid (fast rows, slow cols)")
    plt.xlabel("Slow SMA length")
    plt.ylabel("Fast SMA length")
    plt.tight_layout()
    plt.show()

#chart
def single_run_chart(df, tkr, fast, slow):
    span_years = (df.index[-1] - df.index[0]).days / 365.25
    cutoff = df.index[-1] - pd.DateOffset(years=ARROW_YRS) if span_years > SPAN_LIMIT_YR else df.index[0]
    arrow_df = df[df.index >= cutoff].copy()

    if span_years > LONG_SPAN_YR:
        buys  = arrow_df[arrow_df.Position == 1].iloc[::ARROW_SKIP_N]
        sells = arrow_df[arrow_df.Position == -1].iloc[::ARROW_SKIP_N]
    else:
        buys  = arrow_df[arrow_df.Position == 1]
        sells = arrow_df[arrow_df.Position == -1]

    fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax_p.plot(df["Close"], label="Price", color="tab:blue", linewidth=1.2)
    ax_p.scatter(buys.index,  buys["Close"],  marker="^", color="green", s=ARROW_SIZE, label="Buy")
    ax_p.scatter(sells.index, sells["Close"], marker="v", color="red",   s=ARROW_SIZE, label="Sell")
    ax_p.set_title(f"{tkr} | SMA{fast}/{slow}")
    ax_p.legend(); ax_p.grid(True)

    ax_e.plot(df["Market_Cum"], label="Buy & Hold", linewidth=1.2)
    ax_e.plot(df["Strat_Cum"],  label="Strategy",   linewidth=1.2)
    ax_e.set_title("Cumulative Return (net of fees)")
    ax_e.legend(); ax_e.grid(True)

    plt.tight_layout(); plt.show()
#main
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    tkr   = input("Ticker (e.g. AAPL, MSFT, SPY): ").upper()
    fast  = int(input("Fast SMA (default 10): ")  or 10)
    slow  = int(input("Slow SMA (default 50): ")  or 50)
    start = input("Start YYYY‑MM‑DD (default 2015‑01‑01): ") or "2015-01-01"
    end   = input("End   YYYY‑MM‑DD (blank = today): ") or None

    raw = get_data(tkr, start, end)
    add_signals(raw, fast, slow)
    backtest(raw)

    #metric
    print("\n--- Single‑Run Metrics ---")
    print(f"Sharpe  : {sharpe(raw['Strat_Ret']):6.2f}")
    print(f"CAGR    : {cagr(raw['Strat_Cum']):6.2%}")
    print(f"Max DD  : {max_dd(raw['Strat_Cum']):6.2%}")
    print("---------------------------\n")

    single_run_chart(raw, tkr, fast, slow)

    #heatmap
    if input("Run SMA grid‑search heat‑map (y/n)? ").lower() == "y":
        metric_type = "sharpe"
        df_scores = grid_search(get_data(tkr, start, end), metric_type)
        plot_heatmap(df_scores, metric_type.capitalize(), tkr)
