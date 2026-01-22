import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.orderbook import OrderBook


def test_best_prices_empty():
    ob = OrderBook()
    assert ob.best_bid() is None
    assert ob.best_ask() is None
    assert ob.mid() is None
    assert ob.spread_ticks() is None
    assert ob.top_of_book() == (None, 0, None, 0)


def test_add_limit_resting_sets_best_prices_and_depth():
    ob = OrderBook()
    ob.add_limit("buy", 99, 10)
    ob.add_limit("sell", 101, 7)

    assert ob.best_bid() == 99
    assert ob.best_ask() == 101
    assert ob.spread_ticks() == 2

    bb, bb_qty, ba, ba_qty = ob.top_of_book()
    assert (bb, bb_qty, ba, ba_qty) == (99, 10, 101, 7)

    d = ob.depth(1)
    assert d["bids"] == [(99, 10)]
    assert d["asks"] == [(101, 7)]


def test_fifo_matching_same_price_level():
    ob = OrderBook()
    id1 = ob.add_limit("sell", 101, 5)
    id2 = ob.add_limit("sell", 101, 6)
    assert ob.qty_at_price("sell", 101) == 11

    cursor = ob.trade_cursor()
    ob.add_market("buy", 7)  # should fill id1 fully (5) then 2 from id2

    new_trades = ob.trades_since(cursor)
    assert len(new_trades) == 2
    assert new_trades[0].maker_order_id == id1 and new_trades[0].qty == 5
    assert new_trades[1].maker_order_id == id2 and new_trades[1].qty == 2

    # remaining at 101 should be 4 from id2
    assert ob.qty_at_price("sell", 101) == 4


def test_aggressive_limit_buy_crosses_and_rests_remainder():
    ob = OrderBook()
    ob.add_limit("sell", 101, 5)
    cursor = ob.trade_cursor()

    buy_id = ob.add_limit("buy", 101, 8)  # crosses 5, rests 3 at 101 as bid
    trades = ob.trades_since(cursor)
    assert sum(t.qty for t in trades) == 5
    assert all(t.aggressor_side == "buy" for t in trades)
    assert all(t.taker_order_id == buy_id for t in trades)

    # ask side empty now; bid at 101 should exist with qty 3
    assert ob.best_ask() is None
    assert ob.best_bid() == 101
    assert ob.qty_at_price("buy", 101) == 3


def test_aggressive_limit_respects_limit_price_and_stops():
    ob = OrderBook()
    ob.add_limit("sell", 101, 5)
    ob.add_limit("sell", 103, 5)

    cursor = ob.trade_cursor()
    ob.add_limit("buy", 101, 10)  # should only fill at 101, not at 103
    trades = ob.trades_since(cursor)

    assert len(trades) == 1
    assert trades[0].price == 101
    assert trades[0].qty == 5

    # remaining should rest at 101 bid
    assert ob.qty_at_price("buy", 101) == 5
    assert ob.qty_at_price("sell", 103) == 5


def test_market_order_discards_unfilled_when_book_empty():
    ob = OrderBook()
    cursor = ob.trade_cursor()
    ob.add_market("buy", 10)
    assert ob.trades_since(cursor) == []
    assert ob.best_bid() is None
    assert ob.best_ask() is None


def test_cancel_existing_order_removes_it_and_price_level_if_empty():
    ob = OrderBook()
    oid = ob.add_limit("buy", 100, 10)
    assert ob.best_bid() == 100
    assert ob.qty_at_price("buy", 100) == 10

    assert ob.cancel(oid) is True
    assert ob.best_bid() is None
    assert ob.qty_at_price("buy", 100) == 0


def test_cancel_nonexistent_returns_false():
    ob = OrderBook()
    ob.add_limit("buy", 100, 10)
    assert ob.cancel(999999) is False


def test_partial_fill_updates_remaining_qty_correctly():
    ob = OrderBook()
    ob.add_limit("sell", 101, 10)
    cursor = ob.trade_cursor()

    ob.add_market("buy", 4)
    trades = ob.trades_since(cursor)
    assert len(trades) == 1
    assert trades[0].qty == 4
    assert ob.qty_at_price("sell", 101) == 6


def test_depth_multiple_levels_ordering():
    ob = OrderBook()
    # bids
    ob.add_limit("buy", 99, 1)
    ob.add_limit("buy", 100, 2)
    ob.add_limit("buy", 98, 3)
    # asks
    ob.add_limit("sell", 101, 4)
    ob.add_limit("sell", 102, 5)

    d = ob.depth(2)
    # bids: best to worse => 100 then 99
    assert d["bids"] == [(100, 2), (99, 1)]
    # asks: best to worse => 101 then 102
    assert d["asks"] == [(101, 4), (102, 5)]


def test_trades_since_and_cursor_validation_and_time_increment():
    ob = OrderBook()
    ob.add_limit("sell", 101, 5)
    c0 = ob.trade_cursor()

    ob.add_market("buy", 2)
    assert ob.trade_cursor() == c0 + 1
    assert len(ob.trades_since(c0)) == 1

    with pytest.raises(ValueError):
        ob.trades_since(-1)
    with pytest.raises(ValueError):
        ob.trades_since(10**9)

    # time increment should validate
    with pytest.raises(ValueError):
        ob.increment_time(0)

    ob.increment_time(3)
    assert ob.t == 3

