# Limit Order Book Market Making Project

This project is a research-style simulation of a limit order book and market-making strategies.  
I built it to understand how real market makers think about inventory risk, volatility, and adverse selection, not just how to write trading code.

The focus is on behaviour and clarity, not speed or heavy optimisation.

---

## What this project does

- Implements a **FIFO limit order book** with price–time priority  
- Simulates market order flow with regime changes  
- Implements two market-making strategies:
  - **Fixed Spread** market maker (baseline)
  - **Avellaneda–Stoikov** style market maker with:
    - inventory-aware pricing  
    - volatility-dependent spreads  
    - dynamic quote placement  
- Measures performance using:
  - PnL  
  - Sharpe ratio (per step)  
  - drawdowns  
  - inventory risk  
  - fill rate  
  - adverse selection  

The aim is to show how better quoting logic improves risk-adjusted performance, not just raw profit.

---

## Project structure

```text
LOB-project/
│
├── src/
│   ├── orderbook.py
│   ├── simulator.py
│   ├── market_maker.py
│   └── metrics.py
│
├── tests/
│   └── test_orderbook.py
│
├── notebooks/
│   └── experiments.ipynb
│
└── README.md
```


---

## Strategies

### Fixed Spread

- Quotes a constant spread around mid price  
- No inventory control  
- No volatility awareness  

Trades frequently but can accumulate risky inventory and large drawdowns.

---

### Avellaneda–Stoikov

- Shifts quotes based on inventory  
- Widens spreads in volatile markets  
- Reduces adverse selection and stabilises PnL  

Trades less often, but with much better risk control.

---

## Main results (multi-seed experiment)

Each strategy was tested across **20 random seeds**.

| Strategy            | Mean PnL | Std PnL | Mean Sharpe | Worst Max DD | Inv Std | Adv Hit Rate |
|--------------------|---------:|--------:|------------:|-------------:|--------:|-------------:|
| Fixed Spread       | ~1.41    | ~6.61   | ~0.005      | ~20.31       | ~6.27   | ~0.40        |
| Avellaneda–Stoikov | **~4.33**| **~2.54**| **~0.033** | **~2.69**    | **~2.00**| **~0.37**   |

Summary:

- ~3× higher average PnL  
- Much smaller drawdowns  
- Strong inventory control  
- Lower adverse selection  

These results match what market-making theory predicts.

---

## Experiments

All experiments are in:

notebooks/experiments.ipynb


The notebook:
- runs both strategies  
- plots PnL and inventory  
- compares metrics  
- tests robustness across multiple seeds  

This is the core research output of the project.

---

## How to run

From the project root:
```python -m src.simulator```

Run tests:
```pytest```

Run experiments:
```jupyter notebook```

Then open:
```notebooks/experiments.ipynb```

## Why I built this

Most student projects focus on price prediction.
I wanted to explore how trading systems manage risk in real time.

This project helped me understand:

- why spreads widen in volatile markets
- why inventory control matters more than raw profit
- how adverse selection affects profitability
- how small modelling choices shape risk

It is simple by design, but the behaviour is realistic.
