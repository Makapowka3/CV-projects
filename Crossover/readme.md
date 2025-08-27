# Moving Average Crossover Strategy

This project implements a **simple technical trading strategy** based on moving averages.  
The idea is to generate buy/sell signals when a short-term moving average crosses a long-term moving average.

---

## Features
- **Signals:** Buy when short-term SMA > long-term SMA, sell otherwise  
- **Backtest engine:** Calculates daily returns, applies transaction costs, builds equity curves  
- **Performance metrics:** Sharpe ratio, CAGR (compound annual growth rate), Maximum Drawdown  
- **Plots:**  
  - Equity curves (Strategy vs Buy & Hold)  
  - Buy/Sell markers on the price series  
  - Heatmap grid-search for tuning SMA parameters

---

## Benchmarks
- **Buy & Hold** on the selected asset  
- **Crossover Strategy** with user-chosen SMA windows  
