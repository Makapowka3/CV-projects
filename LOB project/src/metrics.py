# src/metrics.py
"""
Metrics for LOB market-making simulations.

Inputs:
- history: List[dict] produced by Simulator (see src/simulator.py)
  Must contain at least: t, pnl, inventory, mm_fill_count, mm_fill_qty
  (mid/spread/regime optional but used if present)

What we compute (minimum CV-quality set):
- Total PnL, final inventory, cash
- Step returns (ΔPnL), mean/std
- Sharpe (annualization optional; by default raw step Sharpe)
- Max drawdown + drawdown duration (in steps)
- Inventory distribution stats (mean/std/min/max/percentiles)
- Fill rate + fill qty stats
- Adverse selection proxy (simple):
    For each MM fill event at time t, compare mid(t+h) - mid(t).
    If MM sold (inventory decreases), adverse when mid rises after fill.
    If MM bought (inventory increases), adverse when mid falls after fill.
  We infer trade side via inventory change between steps.

Usage:
    from src.metrics import compute_metrics, print_report
    m = compute_metrics(history)
    print_report(m)

You can also pass mid_horizon for adverse selection horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _quantile(sorted_xs: List[float], q: float) -> float:
    """
    q in [0,1]. Linear interpolation.
    """
    if not sorted_xs:
        return 0.0
    if q <= 0:
        return float(sorted_xs[0])
    if q >= 1:
        return float(sorted_xs[-1])

    n = len(sorted_xs)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_xs[lo])
    w = pos - lo
    return float(sorted_xs[lo] * (1 - w) + sorted_xs[hi] * w)


def _max_drawdown(pnl: List[float]) -> Tuple[float, int]:
    """
    Returns (max_drawdown, max_drawdown_duration_steps)
    Drawdown is computed on equity curve pnl (mark-to-market).
    """
    if not pnl:
        return 0.0, 0

    peak = pnl[0]
    max_dd = 0.0
    max_dur = 0
    dur = 0

    for x in pnl:
        if x >= peak:
            peak = x
            dur = 0
        else:
            dd = peak - x
            dur += 1
            if dd > max_dd:
                max_dd = dd
            if dur > max_dur:
                max_dur = dur

    return max_dd, max_dur


def _infer_mm_fill_sign(history: List[Dict[str, Any]], i: int) -> int:
    """
    Infer whether MM net bought or sold at step i via inventory change.
    Returns:
      +1 if net buy (inventory increased)
      -1 if net sell (inventory decreased)
       0 if no change / cannot infer
    """
    if i <= 0:
        return 0
    inv_now = history[i].get("inventory")
    inv_prev = history[i - 1].get("inventory")
    if inv_now is None or inv_prev is None:
        return 0
    if inv_now > inv_prev:
        return +1
    if inv_now < inv_prev:
        return -1
    return 0


@dataclass
class Metrics:
    steps: int
    total_pnl: float
    final_inventory: int
    final_cash: float

    mean_step_return: float
    std_step_return: float
    sharpe_step: float

    max_drawdown: float
    max_drawdown_duration: int

    inv_mean: float
    inv_std: float
    inv_min: int
    inv_max: int
    inv_p05: float
    inv_p50: float
    inv_p95: float

    fill_events: int
    fill_event_rate: float
    fill_qty_total: int
    fill_qty_mean_per_step: float

    adv_score: float
    adv_hit_rate: float         
    adv_count: int
    adv_horizon: int

    avg_spread_ticks: Optional[float] = None
    regime_active_frac: Optional[float] = None


def compute_metrics(
    history: List[Dict[str, Any]],
    *,
    mid_horizon: int = 10,
    sharpe_annualize: bool = False,
    steps_per_day: int = 24 * 60 * 60,
) -> Metrics:
    if not history:
        raise ValueError("history is empty")

    pnl = [float(h.get("pnl", 0.0)) for h in history]
    cash = float(history[-1].get("cash", 0.0))
    inv = int(history[-1].get("inventory", 0))

    rets = [pnl[i] - pnl[i - 1] for i in range(1, len(pnl))]
    mu = _mean(rets) if rets else 0.0
    sd = _std(rets) if rets else 0.0

    sharpe_step = 0.0
    if sd > 0:
        sharpe_step = mu / sd
        if sharpe_annualize:
            sharpe_step *= math.sqrt(max(1, steps_per_day))

    max_dd, max_dd_dur = _max_drawdown(pnl)

    inv_series = [int(h.get("inventory", 0)) for h in history]
    inv_sorted = sorted(inv_series)
    inv_mean = _mean([float(x) for x in inv_series])
    inv_std = _std([float(x) for x in inv_series])

    fill_events = sum(int(h.get("mm_fill_count", 0)) for h in history)
    fill_qty_total = sum(int(h.get("mm_fill_qty", 0)) for h in history)
    steps = len(history)
    fill_event_rate = fill_events / steps if steps else 0.0
    fill_qty_mean_per_step = fill_qty_total / steps if steps else 0.0

    spreads = [h.get("spread_ticks") for h in history if h.get("spread_ticks") is not None]
    avg_spread_ticks = _mean([float(s) for s in spreads]) if spreads else None

    regimes = [h.get("regime") for h in history if "regime" in h]
    regime_active_frac = None
    if regimes:
        regime_active_frac = sum(1 for r in regimes if r == "active") / len(regimes)

    mids = [h.get("mid") for h in history]
    has_mid = all(m is not None for m in mids)

    adv_moves: List[float] = []
    adv_hits = 0
    adv_count = 0

    if has_mid and mid_horizon > 0:
        for i in range(steps):
            if int(history[i].get("mm_fill_count", 0)) <= 0:
                continue
            j = i + mid_horizon
            if j >= steps:
                break

            sign = _infer_mm_fill_sign(history, i)
            if sign == 0:
                continue

            mid_now = float(mids[i])
            mid_fwd = float(mids[j])
            d = mid_fwd - mid_now

            adverse = (sign == -1 and d > 0) or (sign == +1 and d < 0)

            score = abs(d) if adverse else -abs(d)

            adv_moves.append(score)
            adv_count += 1
            if adverse:
                adv_hits += 1

    adv_score = _mean(adv_moves) if adv_moves else 0.0
    adv_hit_rate = (adv_hits / adv_count) if adv_count > 0 else 0.0

    return Metrics(
        steps=steps,
        total_pnl=pnl[-1],
        final_inventory=inv,
        final_cash=cash,
        mean_step_return=mu,
        std_step_return=sd,
        sharpe_step=sharpe_step,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        inv_mean=inv_mean,
        inv_std=inv_std,
        inv_min=int(min(inv_series)) if inv_series else 0,
        inv_max=int(max(inv_series)) if inv_series else 0,
        inv_p05=_quantile([float(x) for x in inv_sorted], 0.05),
        inv_p50=_quantile([float(x) for x in inv_sorted], 0.50),
        inv_p95=_quantile([float(x) for x in inv_sorted], 0.95),
        fill_events=fill_events,
        fill_event_rate=fill_event_rate,
        fill_qty_total=fill_qty_total,
        fill_qty_mean_per_step=fill_qty_mean_per_step,
        adv_score=adv_score,
        adv_hit_rate=adv_hit_rate,
        adv_count=adv_count,
        adv_horizon=mid_horizon,
        avg_spread_ticks=avg_spread_ticks,
        regime_active_frac=regime_active_frac,
    )


def as_dict(m: Metrics) -> Dict[str, Any]:
    return {
        "steps": m.steps,
        "total_pnl": m.total_pnl,
        "final_inventory": m.final_inventory,
        "final_cash": m.final_cash,
        "mean_step_return": m.mean_step_return,
        "std_step_return": m.std_step_return,
        "sharpe_step": m.sharpe_step,
        "max_drawdown": m.max_drawdown,
        "max_drawdown_duration": m.max_drawdown_duration,
        "inv_mean": m.inv_mean,
        "inv_std": m.inv_std,
        "inv_min": m.inv_min,
        "inv_max": m.inv_max,
        "inv_p05": m.inv_p05,
        "inv_p50": m.inv_p50,
        "inv_p95": m.inv_p95,
        "fill_events": m.fill_events,
        "fill_event_rate": m.fill_event_rate,
        "fill_qty_total": m.fill_qty_total,
        "fill_qty_mean_per_step": m.fill_qty_mean_per_step,
        "adv_score": m.adv_score,
        "adv_hit_rate": m.adv_hit_rate,
        "adv_count": m.adv_count,
        "adv_horizon": m.adv_horizon,
        "avg_spread_ticks": m.avg_spread_ticks,
        "regime_active_frac": m.regime_active_frac,
    }


def print_report(m: Metrics) -> None:
    def f(x: Optional[float], nd: int = 6) -> str:
        if x is None:
            return "NA"
        return f"{x:.{nd}f}"

    print("=== Simulation Metrics ===")
    print(f"Steps: {m.steps}")
    print(f"Total PnL: {f(m.total_pnl, 6)}")
    print(f"Final Inventory: {m.final_inventory}")
    print(f"Final Cash: {f(m.final_cash, 6)}")
    print()
    print("Returns / Risk")
    print(f"Mean step return (ΔPnL): {f(m.mean_step_return, 6)}")
    print(f"Std step return: {f(m.std_step_return, 6)}")
    print(f"Sharpe (per-step): {f(m.sharpe_step, 4)}")
    print(f"Max drawdown: {f(m.max_drawdown, 6)}")
    print(f"Max DD duration (steps): {m.max_drawdown_duration}")
    print()
    print("Inventory")
    print(f"Inv mean: {f(m.inv_mean, 4)} | std: {f(m.inv_std, 4)} | min: {m.inv_min} | max: {m.inv_max}")
    print(f"Inv p05: {f(m.inv_p05, 2)} | p50: {f(m.inv_p50, 2)} | p95: {f(m.inv_p95, 2)}")
    print()
    print("Fills")
    print(f"Fill events: {m.fill_events}")
    print(f"Fill event rate (per step): {f(m.fill_event_rate, 6)}")
    print(f"Total fill qty: {m.fill_qty_total}")
    print(f"Mean fill qty per step: {f(m.fill_qty_mean_per_step, 6)}")
    print()
    print("Adverse Selection (proxy)")
    print(f"Horizon (steps): {m.adv_horizon}")
    print(f"Adv score (avg signed magnitude): {f(m.adv_score, 6)}")
    print(f"Adv hit rate: {f(m.adv_hit_rate, 4)} (count={m.adv_count})")
    print()
    if m.avg_spread_ticks is not None:
        print(f"Avg spread (ticks): {f(m.avg_spread_ticks, 4)}")
    if m.regime_active_frac is not None:
        print(f"Active regime fraction: {f(m.regime_active_frac, 4)}")
