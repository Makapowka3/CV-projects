# src/market_maker.py
"""
Market making strategies for the LOB simulator.

Includes:
- FixedSpreadMarketMaker (baseline)
- AvellanedaStoikovMarketMaker (discrete-tick, practical version)

This file is designed to match the simulator interface:

    bid_quote, ask_quote = mm.quote(state, inventory, active_orders)

Where a quote is (price_ticks, qty) or None.

Notes on the Avellaneda–Stoikov implementation here:
- We use a rolling volatility estimate from mid returns.
- We implement an inventory-tilted reservation price:
      r_t = mid_ticks - inventory * gamma * sigma_ticks^2
  (discrete-time approximation; not a perfect continuous-time derivation, but very usable)
- We set half-spread as:
      half_spread_ticks = base_half_spread + spread_mult * sigma_ticks
  plus an optional inventory widening term.
- We ensure we do NOT cross the book.

This gives you the right behavior:
- In high vol: spread widens (quote less aggressively)
- When inventory is long: reservation price shifts down -> more selling / less buying
- When inventory is short: reservation price shifts up -> more buying / less selling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from collections import deque
import math


Quote = Optional[Tuple[int, int]]  

@dataclass
class FixedSpreadMarketMaker:
    half_spread_ticks: int = 2
    quote_size: int = 1
    max_inventory: int = 25

    def quote(
        self,
        state: Dict[str, Any],
        inventory: int,
        active_orders: Dict[int, Dict[str, Any]],
    ) -> Tuple[Quote, Quote]:
        bb = state.get("best_bid")
        ba = state.get("best_ask")
        if bb is None or ba is None:
            return None, None

        mid_ticks = (bb + ba) // 2

        bid: Quote = None
        ask: Quote = None

        if inventory < self.max_inventory:
            bid = (max(1, mid_ticks - self.half_spread_ticks), self.quote_size)
        if inventory > -self.max_inventory:
            ask = (mid_ticks + self.half_spread_ticks, self.quote_size)

        if bid is not None and bid[0] >= ba:
            bid = (ba - 1, bid[1])
        if ask is not None and ask[0] <= bb:
            ask = (bb + 1, ask[1])

        if bid is not None and bid[0] <= 0:
            bid = (1, bid[1])

        return bid, ask


@dataclass
class AvellanedaStoikovMarketMaker:
    """
    Practical A–S style market maker.

    Parameters you will tune:
    - gamma: risk aversion / inventory penalty (higher => more inventory control)
    - base_half_spread_ticks: minimum half-spread in ticks
    - spread_mult: how much half-spread widens with volatility (sigma in ticks)
    - inv_spread_mult: extra spread widening proportional to |inventory|
    - sigma_window: rolling window for mid returns (in steps)
    - quote_size: order size per side
    - max_inventory: hard inventory clamp (stop quoting one side when breached)
    - min_half_spread_ticks: safety floor (>=1)

    Internal state:
    - keeps a rolling window of mid tick returns to estimate sigma_ticks
    """
    gamma: float = 0.05
    base_half_spread_ticks: int = 1
    spread_mult: float = 1.0
    inv_spread_mult: float = 0.03
    sigma_window: int = 50
    quote_size: int = 1
    max_inventory: int = 25
    min_half_spread_ticks: int = 1

    _last_mid_ticks: Optional[int] = field(default=None, init=False, repr=False)
    _ret_window: deque = field(default_factory=deque, init=False, repr=False)

    def _update_sigma_ticks(self, mid_ticks: int) -> float:
        """
        Update rolling window of mid returns (in ticks) and return sigma (std dev).
        """
        if self._last_mid_ticks is not None:
            r = mid_ticks - self._last_mid_ticks
            self._ret_window.append(float(r))
            if len(self._ret_window) > self.sigma_window:
                self._ret_window.popleft()

        self._last_mid_ticks = mid_ticks

        if len(self._ret_window) < 2:
            return 0.0

        mu = sum(self._ret_window) / len(self._ret_window)
        var = sum((x - mu) ** 2 for x in self._ret_window) / (len(self._ret_window) - 1)
        return math.sqrt(max(0.0, var))

    def quote(
        self,
        state: Dict[str, Any],
        inventory: int,
        active_orders: Dict[int, Dict[str, Any]],
    ) -> Tuple[Quote, Quote]:
        bb = state.get("best_bid")
        ba = state.get("best_ask")
        if bb is None or ba is None:
            return None, None

        mid_ticks = (bb + ba) // 2

        sigma_ticks = self._update_sigma_ticks(mid_ticks)

        r_ticks = mid_ticks - int(round(inventory * self.gamma * (sigma_ticks ** 2)))

        half = (
            self.base_half_spread_ticks
            + int(round(self.spread_mult * sigma_ticks))
            + int(round(self.inv_spread_mult * abs(inventory)))
        )
        half = max(self.min_half_spread_ticks, half)

        bid_px = r_ticks - half
        ask_px = r_ticks + half

        bid: Quote = None
        ask: Quote = None

        if inventory < self.max_inventory:
            bid = (max(1, bid_px), self.quote_size)
        if inventory > -self.max_inventory:
            ask = (ask_px, self.quote_size)

        if bid is not None and bid[0] >= ba:
            bid = (ba - 1, bid[1])
        if ask is not None and ask[0] <= bb:
            ask = (bb + 1, ask[1])

        if bid is not None and ask is not None and bid[0] >= ask[0]:
            return None, None

        if bid is not None and bid[0] <= 0:
            bid = (1, bid[1])

        return bid, ask
