"""
signal_agent.py
Evaluates all 7 strategies for a given stock and returns the highest-priority signal.
Priority order (from highest conviction to lowest):
  1. Post-Earnings Move  — confirmed catalyst, CALL or PUT based on direction
  2. Pre-Earnings Drift  — event-driven drift into results
  3. FII Sector Momentum — institutional buying footprint
  4. Bear Momentum Put   — institutional selling footprint (PUT)
  5. 52-Week High        — momentum, no overhead resistance
  6. Breakout Retest     — technical edge, high RR
  7. Fibonacci Pullback  — trend continuation
"""
from __future__ import annotations

import logging
from typing import Optional

from strategies.base import StockSignal
from strategies.post_earnings_move import PostEarningsMoveStrategy
from strategies.pre_earnings_drift import PreEarningsDriftStrategy
from strategies.fii_sector_momentum import FIISectorMomentumStrategy
from strategies.bear_momentum_put import BearMomentumPutStrategy
from strategies.fiftytwo_week_high import FiftyTwoWeekHighStrategy
from strategies.breakout_retest import BreakoutRetestStrategy
from strategies.fib_pullback import FibPullbackStrategy
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class SignalAgent:
    PRIORITY = [
        "Post-Earnings Move",
        "Pre-Earnings Drift",
        "FII Sector Momentum",
        "Bear Momentum Put",
        "52-Week High",
        "Breakout Retest",
        "Fibonacci Pullback",
    ]

    def __init__(self):
        self._strategies = {
            "Post-Earnings Move":  PostEarningsMoveStrategy(),
            "Pre-Earnings Drift":  PreEarningsDriftStrategy(),
            "FII Sector Momentum": FIISectorMomentumStrategy(),
            "Bear Momentum Put":   BearMomentumPutStrategy(),
            "52-Week High":        FiftyTwoWeekHighStrategy(),
            "Breakout Retest":     BreakoutRetestStrategy(),
            "Fibonacci Pullback":  FibPullbackStrategy(),
        }

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        """Run all strategies; return highest-priority signal or None."""
        results: list[StockSignal] = []

        for name in self.PRIORITY:
            strat = self._strategies[name]
            try:
                sig = strat.evaluate(ctx)
            except Exception as e:
                log.exception("Strategy %s raised for %s: %s", name, ctx.symbol, e)
                sig = None

            if sig:
                log.info("[%s] Signal: %s %s @ strike=%.0f | %s",
                         ctx.symbol, name, sig.direction, sig.strike, sig.confidence)
                results.append(sig)

        if not results:
            return None

        chosen = results[0]
        if len(results) > 1:
            others = [s.strategy for s in results[1:]]
            log.info("[%s] Multiple signals %s — choosing: %s", ctx.symbol, others, chosen.strategy)

        return chosen

    def evaluate_all(self, ctx: StockMarketContext) -> list[StockSignal]:
        """Returns all signals for dashboard / analysis."""
        results = []
        for name in self.PRIORITY:
            try:
                sig = self._strategies[name].evaluate(ctx)
                if sig:
                    results.append(sig)
            except Exception as e:
                log.exception("Strategy %s exception: %s", name, e)
        return results
