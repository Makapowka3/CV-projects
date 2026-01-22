# src/simulator.py
"""
More realistic event-driven simulator for OrderBook + Market Maker.

1) Quote refresh cadence + TTL
   - Don't cancel/replace every step
   - Orders expire after quote_ttl_steps even if you don't refresh

2) Market regimes (CALM / ACTIVE)
   - Volatility clustering via regime switching
   - Order flow intensity changes by regime

3) Latent "fundamental" mid (ticks) random walk
   - Drives where external liquidity is posted
   - Lets you control volatility independent of book shape

4) Paired limit orders (liquidity provider crowd)
   - Often add both bid & ask around latent mid

5) External cancellations
   - Cancels hit non-MM orders too (keeps churn realistic)

6) Minimum depth injector
   - Prevents book from degenerating (one-sided / too thin)

Important correctness point:
- We process fills from market activity BEFORE we cancel MM orders, otherwise MM fills
  may not be credited.

Run:
    python -m src.simulator
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Protocol, Any
import random
import math

from src.orderbook import OrderBook, Trade

class MarketMaker(Protocol):
    def quote(
        self,
        state: Dict[str, Any],
        inventory: int,
        active_orders: Dict[int, Dict[str, Any]],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Return (bid_quote, ask_quote) where:
        - bid_quote: (price_ticks, qty) or None
        - ask_quote: (price_ticks, qty) or None
        """

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
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        bb = state["best_bid"]
        ba = state["best_ask"]
        if bb is None or ba is None:
            return None, None

        mid_ticks = (bb + ba) // 2

        bid: Optional[Tuple[int, int]] = None
        ask: Optional[Tuple[int, int]] = None

        if inventory < self.max_inventory:
            bid = (mid_ticks - self.half_spread_ticks, self.quote_size)
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
class SimConfig:
    steps: int = 10_000
    seed: int = 42
    tick_size: float = 0.01

    init_mid_ticks: int = 10_000
    init_spread_ticks: int = 2
    init_levels: int = 12
    init_qty: int = 6

    quote_refresh_steps: int = 5
    quote_ttl_steps: int = 25

    regime_switch_p: float = 0.01
    sigma_calm: float = 0.6
    sigma_active: float = 2.2

    lambda_events_calm: float = 1.2
    lambda_events_active: float = 2.5

    p_market_calm: float = 0.45
    p_market_active: float = 0.70

    p_cancel_calm: float = 0.10
    p_cancel_active: float = 0.18

    market_qty_min: int = 1
    market_qty_max_calm: int = 4
    market_qty_max_active: int = 10

    limit_qty_min: int = 1
    limit_qty_max: int = 6

    limit_offset_min: int = 1
    limit_offset_max_calm: int = 12
    limit_offset_max_active: int = 25

    p_paired_limits: float = 0.35

    p_inside_spread: float = 0.12

    p_cancel_hits_external: float = 0.80
    external_id_pool_cap: int = 15_000

    min_top_qty: int = 3
    min_levels_each_side: int = 4
    replenish_levels: int = 8
    replenish_qty: int = 6


@dataclass
class SimResult:
    history: List[Dict[str, Any]]
    trades: List[Trade]

def poisson_sample(lam: float) -> int:
    """
    Knuth algorithm. Fine for small/medium lambdas (<~10).
    """
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

class Simulator:
    def __init__(self, config: SimConfig, mm: MarketMaker) -> None:
        self.cfg = config
        self.mm = mm

        random.seed(self.cfg.seed)

        self.ob = OrderBook(tick_size=self.cfg.tick_size)

        self.inventory: int = 0
        self.cash: float = 0.0

        self.active_orders: Dict[int, Dict[str, Any]] = {}

        self.external_order_ids: List[int] = []

        self.regime: str = "calm"
        self.latent_mid_ticks: int = self.cfg.init_mid_ticks

        self._cursor: int = self.ob.trade_cursor()

        self.history: List[Dict[str, Any]] = []

        self._seed_book()

        state0 = self.ob.get_state(depth_levels=5)
        bid0, ask0 = self.mm.quote(state=state0, inventory=self.inventory, active_orders=self.active_orders)
        self._place_mm_quotes(bid0, ask0)

    def _seed_book(self) -> None:
        mid = self.cfg.init_mid_ticks
        spread = max(1, self.cfg.init_spread_ticks)
        bb = mid - spread // 2
        ba = bb + spread

        for i in range(self.cfg.init_levels):
            self.ob.add_limit("buy", bb - i, self.cfg.init_qty)
            self.ob.add_limit("sell", ba + i, self.cfg.init_qty)

    def _maybe_switch_regime(self) -> None:
        if random.random() < self.cfg.regime_switch_p:
            self.regime = "active" if self.regime == "calm" else "calm"

    def _step_latent_mid(self) -> None:
        sigma = self.cfg.sigma_active if self.regime == "active" else self.cfg.sigma_calm
        delta = int(round(random.gauss(0.0, sigma)))
        self.latent_mid_ticks = max(1, self.latent_mid_ticks + delta)

    def _ensure_min_depth(self) -> None:
        bb, bb_qty, ba, ba_qty = self.ob.top_of_book()
        if bb is None or ba is None or bb_qty < self.cfg.min_top_qty or ba_qty < self.cfg.min_top_qty:
            self._replenish_book()

        snap = self.ob.depth(self.cfg.min_levels_each_side)
        if len(snap["bids"]) < self.cfg.min_levels_each_side or len(snap["asks"]) < self.cfg.min_levels_each_side:
            self._replenish_book()

    def _replenish_book(self) -> None:
        """
        Inject external liquidity around latent_mid_ticks so the sim doesn't die.
        """
        mid = self.latent_mid_ticks
        spread = max(2, self.cfg.init_spread_ticks)
        bb = max(1, mid - spread // 2)
        ba = bb + spread

        for i in range(self.cfg.replenish_levels):
            bid_p = max(1, bb - i)
            ask_p = ba + i
            oid_b = self.ob.add_limit("buy", bid_p, self.cfg.replenish_qty)
            oid_a = self.ob.add_limit("sell", ask_p, self.cfg.replenish_qty)
            self._push_external_id(oid_b)
            self._push_external_id(oid_a)

    def _push_external_id(self, oid: int) -> None:
        self.external_order_ids.append(oid)
        if len(self.external_order_ids) > self.cfg.external_id_pool_cap:
            self.external_order_ids = self.external_order_ids[-self.cfg.external_id_pool_cap :]

    def _event_counts(self) -> int:
        lam = self.cfg.lambda_events_active if self.regime == "active" else self.cfg.lambda_events_calm
        n = poisson_sample(lam)
        return max(1, n)

    def _pick_event_type(self) -> str:
        p_market = self.cfg.p_market_active if self.regime == "active" else self.cfg.p_market_calm
        p_cancel = self.cfg.p_cancel_active if self.regime == "active" else self.cfg.p_cancel_calm
        r = random.random()
        if r < p_market:
            return "market"
        if r < p_market + p_cancel:
            return "cancel"
        return "limit"

    def _random_market_event(self) -> None:
        side = "buy" if random.random() < 0.5 else "sell"
        qty_max = self.cfg.market_qty_max_active if self.regime == "active" else self.cfg.market_qty_max_calm
        qty = random.randint(self.cfg.market_qty_min, qty_max)
        self.ob.add_market(side, qty)

    def _random_limit_event(self) -> None:
        """
        Place external passive liquidity around latent_mid.
        Sometimes places paired leads (bid+ask).
        Sometimes tightens inside-spread.
        """
        bb, ba = self.ob.best_prices()
        if bb is None or ba is None:
            self._replenish_book()
            return

        mid = self.latent_mid_ticks

        offset_max = self.cfg.limit_offset_max_active if self.regime == "active" else self.cfg.limit_offset_max_calm
        offset = random.randint(self.cfg.limit_offset_min, offset_max)
        qty = random.randint(self.cfg.limit_qty_min, self.cfg.limit_qty_max)

        def place_one(side: str) -> None:
            nonlocal offset, qty, mid, bb, ba
            if random.random() < self.cfg.p_inside_spread:
                # tighten inside the spread (if possible)
                if side == "buy":
                    price = bb + 1 if ba - bb >= 2 else bb
                    price = min(price, ba - 1)
                else:
                    price = ba - 1 if ba - bb >= 2 else ba
                    price = max(price, bb + 1)
            else:
                if side == "buy":
                    price = max(1, mid - offset)
                    price = min(price, ba - 1) 
                else:
                    price = mid + offset
                    price = max(price, bb + 1)

            if price <= 0:
                price = 1

            oid = self.ob.add_limit(side, price, qty)
            self._push_external_id(oid)

        if random.random() < self.cfg.p_paired_limits:
            place_one("buy")
            place_one("sell")
        else:
            side = "buy" if random.random() < 0.5 else "sell"
            place_one(side)

    def _random_cancel_event(self) -> None:
        """
        Cancel an order: mostly external, sometimes MM.
        """
        if random.random() < self.cfg.p_cancel_hits_external:
            if not self.external_order_ids:
                return
            oid = random.choice(self.external_order_ids)
            ok = self.ob.cancel(oid)
            try:
                self.external_order_ids.remove(oid)
            except ValueError:
                pass
            return

        # cancel MM sometimes
        if not self.active_orders:
            return
        oid = random.choice(list(self.active_orders.keys()))
        self.ob.cancel(oid)
        self.active_orders.pop(oid, None)

    def _step_synthetic_flow(self) -> None:
        n_events = self._event_counts()
        for _ in range(n_events):
            et = self._pick_event_type()
            if et == "market":
                self._random_market_event()
            elif et == "limit":
                self._random_limit_event()
            else:
                self._random_cancel_event()

    def _cancel_expired_mm_orders(self) -> None:
        """
        Cancel MM quotes older than TTL. Safe if already filled (cancel returns False).
        """
        now = self.ob.t
        ttl = self.cfg.quote_ttl_steps
        for oid, info in list(self.active_orders.items()):
            if now - int(info["placed_t"]) >= ttl:
                self.ob.cancel(oid)
                self.active_orders.pop(oid, None)

    def _refresh_mm_quotes_if_due(self, state: Dict[str, Any]) -> None:
        """
        Refresh quotes on cadence: every quote_refresh_steps steps.
        """
        if self.cfg.quote_refresh_steps <= 0:
            return
        if self.ob.t % self.cfg.quote_refresh_steps != 0:
            return

        for oid in list(self.active_orders.keys()):
            self.ob.cancel(oid)
            self.active_orders.pop(oid, None)

        bid, ask = self.mm.quote(state=state, inventory=self.inventory, active_orders=self.active_orders)
        self._place_mm_quotes(bid, ask)

    def _place_mm_quotes(self, bid: Optional[Tuple[int, int]], ask: Optional[Tuple[int, int]]) -> None:
        now = self.ob.t
        if bid is not None:
            p, q = bid
            oid = self.ob.add_limit("buy", p, q)
            self.active_orders[oid] = {
                "side": "buy",
                "price": p,
                "qty": q,
                "remaining": q,
                "placed_t": now,
            }
        if ask is not None:
            p, q = ask
            oid = self.ob.add_limit("sell", p, q)
            self.active_orders[oid] = {
                "side": "sell",
                "price": p,
                "qty": q,
                "remaining": q,
                "placed_t": now,
            }

    def _process_new_trades(self) -> Tuple[int, int]:
        """
        Credit MM fills (MM is maker). We track remaining qty per MM order_id so we can
        remove fully filled orders from active_orders immediately.
        """
        mm_fill_count = 0
        mm_fill_qty = 0

        new_trades = self.ob.trades_since(self._cursor)
        self._cursor = self.ob.trade_cursor()

        for tr in new_trades:
            info = self.active_orders.get(tr.maker_order_id)
            if info is None:
                continue

            side = str(info["side"])
            px = tr.price * self.cfg.tick_size
            qty = tr.qty

            mm_fill_count += 1
            mm_fill_qty += qty

            if side == "buy":
                self.inventory += qty
                self.cash -= px * qty
            else:
                self.inventory -= qty
                self.cash += px * qty

            info["remaining"] = int(info["remaining"]) - qty
            if info["remaining"] <= 0:
                self.active_orders.pop(tr.maker_order_id, None)

        return mm_fill_count, mm_fill_qty

    def _mark_to_market(self) -> float:
        mid = self.ob.mid()
        if mid is not None:
            return self.cash + self.inventory * mid
        if self.ob.trades:
            last_px = self.ob.trades[-1].price * self.cfg.tick_size
            return self.cash + self.inventory * last_px
        return self.cash

    # ---------- main loop ----------
    def run(self) -> SimResult:
        for _ in range(self.cfg.steps):
            self.ob.increment_time(1)

            # Regime + latent mid
            self._maybe_switch_regime()
            self._step_latent_mid()

            # Ensure book doesn't degenerate
            self._ensure_min_depth()

            # 1) Exogenous flow (hits resting MM quotes)
            self._step_synthetic_flow()

            # 2) Credit MM fills BEFORE any MM cancels/refresh
            fill_count, fill_qty = self._process_new_trades()

            # 3) State snapshot after flow
            state = self.ob.get_state(depth_levels=5)

            # 4) Cancel expired quotes (TTL), then refresh on cadence
            self._cancel_expired_mm_orders()
            self._refresh_mm_quotes_if_due(state=state)

            # 5) Record
            pnl = self._mark_to_market()
            self.history.append(
                {
                    "t": self.ob.t,
                    "regime": self.regime,
                    "latent_mid_ticks": self.latent_mid_ticks,
                    "mid": state["mid"],
                    "best_bid": state["best_bid"],
                    "best_ask": state["best_ask"],
                    "spread_ticks": state["spread_ticks"],
                    "inventory": self.inventory,
                    "cash": self.cash,
                    "pnl": pnl,
                    "mm_fill_count": fill_count,
                    "mm_fill_qty": fill_qty,
                    "n_trades_total": state["n_trades"],
                    "n_mm_active_orders": len(self.active_orders),
                    "n_external_ids": len(self.external_order_ids),
                }
            )

        return SimResult(history=self.history, trades=list(self.ob.trades))


from src.market_maker import AvellanedaStoikovMarketMaker

def run_demo():
    cfg = SimConfig(steps=2000, seed=42)
    mm = AvellanedaStoikovMarketMaker(
        gamma=0.05,
        base_half_spread_ticks=1,
        spread_mult=1.2,
        inv_spread_mult=0.05,
        sigma_window=50,
        quote_size=1,
        max_inventory=25,
    )
    sim = Simulator(config=cfg, mm=mm)
    return sim.run()


if __name__ == "__main__":
    from src.metrics import compute_metrics, print_report

    res = run_demo()
    last = res.history[-1]
    fills = sum(h["mm_fill_count"] for h in res.history)

    print("Done.")
    print(
        f"Final t={last['t']}, pnl={last['pnl']:.6f}, inventory={last['inventory']}, cash={last['cash']:.6f}"
    )
    print(f"Total MM fill events: {fills}")
    print(f"Total trades in tape: {len(res.trades)}")

    print("\n--- METRICS REPORT ---")
    m = compute_metrics(res.history, mid_horizon=10)
    print_report(m)

