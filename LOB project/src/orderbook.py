# src/orderbook.py
"""
Simple Limit Order Book (LOB) with price-time priority (FIFO per price level).

Design goals:
- Clear, modifiable code (not ultra-optimized)
- Integer tick prices (avoid float issues)
- Supports:
  - add_limit(side, price_ticks, qty)
  - add_market(side, qty)  # "buy" hits asks, "sell" hits bids
  - cancel(order_id)
  - best_bid(), best_ask(), mid(), spread_ticks()
  - top-of-book depth + multi-level depth snapshot
  - trade log + helpers to fetch incremental trades

Notes:
- side is "buy" or "sell"
- price is integer ticks, not float
- quantities are positive integers
- Time (self.t) is controlled by the simulator: call increment_time() externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from bisect import bisect_left
from typing import Deque, Dict, List, Optional, Tuple, Any


Side = str  # "buy" or "sell"


@dataclass(frozen=True)
class Order:
    order_id: int
    side: Side
    price: int  # ticks
    qty: int
    t: int      # event-time when created


@dataclass(frozen=True)
class Trade:
    t: int
    price: int  # ticks
    qty: int
    aggressor_side: Side          # side of incoming order ("buy" or "sell")
    maker_order_id: int      
    taker_order_id: Optional[int] 


class OrderBook:
    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size

        self._bids: Dict[int, Deque[Order]] = {}
        self._asks: Dict[int, Deque[Order]] = {}

        self._bid_prices: List[int] = []
        self._ask_prices: List[int] = []

        self._order_index: Dict[int, Tuple[Side, int]] = {}

        self._next_order_id: int = 1
        self.t: int = 0

        self.trades: List[Trade] = []

    def best_bid(self) -> Optional[int]:
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self) -> Optional[int]:
        return self._ask_prices[0] if self._ask_prices else None

    def best_prices(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (best_bid_ticks, best_ask_ticks)."""
        return self.best_bid(), self.best_ask()

    def mid(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0 * self.tick_size

    def spread_ticks(self) -> Optional[int]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def spread(self) -> Optional[float]:
        s = self.spread_ticks()
        return None if s is None else s * self.tick_size

    def qty_at_price(self, side: Side, price: int) -> int:
        """Total resting quantity at a price level."""
        self._validate_side(side)
        self._validate_price(price)
        levels = self._bids if side == "buy" else self._asks
        q = levels.get(price)
        return 0 if q is None else sum(o.qty for o in q)

    def top_of_book(self) -> Tuple[Optional[int], int, Optional[int], int]:
        """
        Return (best_bid_price, best_bid_qty, best_ask_price, best_ask_qty).
        If side is empty, price is None and qty is 0.
        """
        bb = self.best_bid()
        ba = self.best_ask()
        bb_qty = 0 if bb is None else sum(o.qty for o in self._bids[bb])
        ba_qty = 0 if ba is None else sum(o.qty for o in self._asks[ba])
        return bb, bb_qty, ba, ba_qty

    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[int, int]]]:
        """
        Return depth snapshot up to N price levels:
        {
          "bids": [(price_ticks, total_qty), ...] from best -> worse,
          "asks": [(price_ticks, total_qty), ...] from best -> worse
        }
        """
        if levels <= 0:
            raise ValueError("levels must be positive")

        bid_out: List[Tuple[int, int]] = []
        ask_out: List[Tuple[int, int]] = []

        for p in reversed(self._bid_prices[-levels:]):
            bid_out.append((p, self.qty_at_price("buy", p)))

        for p in self._ask_prices[:levels]:
            ask_out.append((p, self.qty_at_price("sell", p)))

        return {"bids": bid_out, "asks": ask_out}

    def get_state(self, depth_levels: int = 5) -> Dict[str, Any]:
        """Convenient state snapshot for simulators/strategies."""
        bb, bb_qty, ba, ba_qty = self.top_of_book()
        return {
            "t": self.t,
            "best_bid": bb,
            "best_bid_qty": bb_qty,
            "best_ask": ba,
            "best_ask_qty": ba_qty,
            "spread_ticks": self.spread_ticks(),
            "mid": self.mid(),
            "depth": self.depth(depth_levels),
            "n_orders": len(self._order_index),
            "n_trades": len(self.trades),
        }

    def increment_time(self, steps: int = 1) -> None:
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.t += steps

    def trade_cursor(self) -> int:
        """Return current end index of the trade tape."""
        return len(self.trades)

    def trades_since(self, cursor: int) -> List[Trade]:
        """Return trades from cursor to end (copy list)."""
        if not isinstance(cursor, int) or cursor < 0 or cursor > len(self.trades):
            raise ValueError("cursor out of range")
        return list(self.trades[cursor:])

    def add_limit(self, side: Side, price: int, qty: int) -> int:
        """
        Add a limit order. If it crosses the book, it executes immediately against resting liquidity
        (price-time priority), and any remainder rests at its limit price.
        Returns the order_id of the incoming order.
        """
        self._validate_side(side)
        self._validate_qty(qty)
        self._validate_price(price)

        order_id = self._next_order_id
        self._next_order_id += 1

        remaining = qty
        if side == "buy":
            remaining = self._match_buy(remaining, limit_price=price, taker_order_id=order_id)
        else:
            remaining = self._match_sell(remaining, limit_price=price, taker_order_id=order_id)

        if remaining > 0:
            o = Order(order_id=order_id, side=side, price=price, qty=remaining, t=self.t)
            self._rest_order(o)

        return order_id

    def add_market(self, side: Side, qty: int) -> int:
        """
        Add a market order. Executes immediately against best available prices until filled
        or the book empties. Unfilled remainder is discarded.
        Returns the order_id of the incoming order.
        """
        self._validate_side(side)
        self._validate_qty(qty)

        order_id = self._next_order_id
        self._next_order_id += 1

        if side == "buy":
            _ = self._match_buy(qty, limit_price=None, taker_order_id=order_id)
        else:
            _ = self._match_sell(qty, limit_price=None, taker_order_id=order_id)

        return order_id

    def cancel(self, order_id: int) -> bool:
        """
        Cancel a resting order by id.
        Returns True if found and removed, False otherwise.

        Implementation detail:
        - Fast lookup via _order_index.
        - Then scan that price level queue to remove the order (O(level_size)).
        """
        if order_id not in self._order_index:
            return False

        side, price = self._order_index[order_id]
        levels = self._bids if side == "buy" else self._asks
        q = levels.get(price)
        if q is None:
            del self._order_index[order_id]
            return False

        new_q: Deque[Order] = deque()
        removed = False
        while q:
            o = q.popleft()
            if o.order_id == order_id and not removed:
                removed = True
                continue
            new_q.append(o)

        if removed:
            levels[price] = new_q
            del self._order_index[order_id]
            if not new_q:
                self._remove_price_level(side=side, price=price)
            return True

        del self._order_index[order_id]
        return False

    def _match_buy(self, qty: int, limit_price: Optional[int], taker_order_id: Optional[int]) -> int:
        """
        Incoming buy consumes asks from best ask upwards.
        If limit_price is not None, stop when best ask > limit_price.
        Returns remaining qty.
        """
        remaining = qty
        while remaining > 0 and self._ask_prices:
            best_ask = self._ask_prices[0]
            if limit_price is not None and best_ask > limit_price:
                break

            ask_queue = self._asks[best_ask]
            while remaining > 0 and ask_queue:
                maker = ask_queue[0]
                traded = min(remaining, maker.qty)

                self.trades.append(
                    Trade(
                        t=self.t,
                        price=best_ask,
                        qty=traded,
                        aggressor_side="buy",
                        maker_order_id=maker.order_id,
                        taker_order_id=taker_order_id,
                    )
                )

                remaining -= traded

                if traded == maker.qty:
                    ask_queue.popleft()
                    self._order_index.pop(maker.order_id, None)
                else:
                    ask_queue[0] = Order(
                        order_id=maker.order_id,
                        side=maker.side,
                        price=maker.price,
                        qty=maker.qty - traded,
                        t=maker.t,
                    )

            if not ask_queue:
                self._remove_price_level(side="sell", price=best_ask)

        return remaining

    def _match_sell(self, qty: int, limit_price: Optional[int], taker_order_id: Optional[int]) -> int:
        """
        Incoming sell consumes bids from best bid downwards.
        If limit_price is not None, stop when best bid < limit_price.
        Returns remaining qty.
        """
        remaining = qty
        while remaining > 0 and self._bid_prices:
            best_bid = self._bid_prices[-1]
            if limit_price is not None and best_bid < limit_price:
                break

            bid_queue = self._bids[best_bid]
            while remaining > 0 and bid_queue:
                maker = bid_queue[0]
                traded = min(remaining, maker.qty)

                self.trades.append(
                    Trade(
                        t=self.t,
                        price=best_bid,
                        qty=traded,
                        aggressor_side="sell",
                        maker_order_id=maker.order_id,
                        taker_order_id=taker_order_id,
                    )
                )

                remaining -= traded

                if traded == maker.qty:
                    bid_queue.popleft()
                    self._order_index.pop(maker.order_id, None)
                else:
                    bid_queue[0] = Order(
                        order_id=maker.order_id,
                        side=maker.side,
                        price=maker.price,
                        qty=maker.qty - traded,
                        t=maker.t,
                    )

            if not bid_queue:
                self._remove_price_level(side="buy", price=best_bid)

        return remaining

    def _rest_order(self, order: Order) -> None:
        """Add an order to the book without matching."""
        levels = self._bids if order.side == "buy" else self._asks
        prices = self._bid_prices if order.side == "buy" else self._ask_prices

        if order.price not in levels:
            levels[order.price] = deque()
            self._insert_price_level(prices, order.price)

        levels[order.price].append(order)
        self._order_index[order.order_id] = (order.side, order.price)

    @staticmethod
    def _insert_price_level(sorted_prices: List[int], price: int) -> None:
        """Insert price into ascending list if not present."""
        i = bisect_left(sorted_prices, price)
        if i == len(sorted_prices) or sorted_prices[i] != price:
            sorted_prices.insert(i, price)

    def _remove_price_level(self, side: Side, price: int) -> None:
        """Remove an empty price level and its price from the sorted list."""
        levels = self._bids if side == "buy" else self._asks
        prices = self._bid_prices if side == "buy" else self._ask_prices

        levels.pop(price, None)

        i = bisect_left(prices, price)
        if i < len(prices) and prices[i] == price:
            prices.pop(i)

    @staticmethod
    def _validate_side(side: Side) -> None:
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

    @staticmethod
    def _validate_qty(qty: int) -> None:
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError("qty must be a positive integer")

    @staticmethod
    def _validate_price(price: int) -> None:
        if not isinstance(price, int) or price <= 0:
            raise ValueError("price must be a positive integer tick value")

    def to_ticks(self, price: float) -> int:
        """Convert float price to integer ticks (rounded to nearest tick)."""
        return int(round(price / self.tick_size))

    def from_ticks(self, price_ticks: int) -> float:
        """Convert integer ticks to float price."""
        return price_ticks * self.tick_size
