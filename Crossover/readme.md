# Moving Average Crossover — Backtest & Heatmap

A compact Python tool to **generate trading signals** from a simple moving average (SMA) crossover, **backtest** the strategy with transaction costs, and **visualize** results (equity curves, buy/sell markers, and a parameter heatmap).

---

## Features
- **Signals:** SMA(fast) vs SMA(slow) crossover
- **Backtest:** daily returns, transaction costs, equity curves
- **Metrics:** Sharpe, CAGR, Max Drawdown
- **Plots:** 
  - Equity curves (Strategy vs Buy & Hold)
  - Buy/Sell markers on price
  - SMA grid-search **heatmap** for parameter tuning (Sharpe or CAGR)

---

## Methods (quick math)
 
- **Metrics:**
  - **Sharpe** (annualized): $\frac{\bar r}{\sigma_r}\sqrt{252}$
  - **CAGR:** $\left(\frac{\text{ending}}{\text{starting}}\right)^{1/\text{years}} - 1$
  - **MaxDD:** $1 - \min_t \frac{V_t}{\max_{s \le t} V_s}$
