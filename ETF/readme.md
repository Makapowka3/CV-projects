# Volatility Targeting with Leverage

This project implements a **volatility targeting framework** used in quantitative trading and asset management.  
The idea is to scale portfolio exposure up or down so that realized volatility matches a chosen target.

---

## Features
- **Base portfolios:** SPY-only or 60/40 (SPY/TLT)
- **Volatility targeting:** Scales exposure to reach a chosen annualized volatility
- **Leverage cap:** Prevents exposure from exceeding a set multiple
- **Rebalancing:** Daily or weekly
- **Transaction costs:** Included via turnover-based cost model
- **Performance metrics:** CAGR, Volatility, Sharpe, Max Drawdown, Sortino, Calmar
- **Plots:**  
  - Equity curves vs benchmarks  
  - Scaling (exposure over time)  
  - Heatmaps of Sharpe ratios across parameter grids

---

## Benchmarks
- **SPY Buy & Hold** (equities only)  
- **60/40 Portfolio** (60% SPY, 40% TLT, fixed weights)  
