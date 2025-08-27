# Portfolio Allocation & Optimisation

This project implements several **portfolio construction techniques** commonly used in quantitative finance.  
It compares Equal-Weight, Minimum Variance, Maximum Sharpe, and Risk Parity allocations, and plots the efficient frontier.

---

## Features
- **Data:** Historical asset prices via `yfinance`  
- **Portfolio methods:**  
  - Equal-Weight  
  - Markowitz Minimum Variance  
  - Markowitz Maximum Sharpe  
  - Risk Parity  
- **Efficient frontier:** Visualizes trade-offs between return and risk  
- **Performance metrics:** Annualized Return, Volatility, Sharpe, CAGR, Max Drawdown  
- **Plots:**  
  - Efficient frontier with key portfolios highlighted  
  - Cumulative equity growth curves  
  - Risk contribution charts (for Risk Parity and Max-Sharpe)  
- **Presets:** Includes US and UK/EU portfolios (e.g. SPY/TLT/GLD/VNQ, UCITS tickers)  

---

## Benchmarks
- **Equal-Weight Portfolio** (naïve diversification)  
- **Min-Variance Portfolio** (lowest risk allocation)  
- **Max-Sharpe Portfolio** (highest risk-adjusted return)  
- **Risk Parity Portfolio** (equal risk contributions from each asset)  
