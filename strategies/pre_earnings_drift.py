"""
pre_earnings_drift.py
Strategy 1 — Pre-Earnings Drift Play
POP: 65-70% | R:R: 1:3 to 1:5 | Frequency: 4-6 per quarter

Enter 10 days before earnings, exit 1-2 days before results.
Rides institutional pre-positioning drift.
Requires: stock above 20-day EMA, RSI > 50, confluence >= 3.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class PreEarningsDriftStrategy(BaseStrategy):
    NAME = "Pre-Earnings Drift"

    DAYS_BEFORE_ENTRY = 10    # Enter this many days before earnings
    DAYS_BEFORE_EXIT = 1      # Exit this many days before earnings
    RSI_MIN = 50
    CONFLUENCE_MIN = 3

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if ctx.earnings_next is None:
            return None

        today = date.today()
        days_to_earnings = (ctx.earnings_next - today).days

        # Entry window: exactly around DAYS_BEFORE_ENTRY before earnings
        if not (self.DAYS_BEFORE_EXIT + 1 <= days_to_earnings <= self.DAYS_BEFORE_ENTRY + 2):
            return None

        # Stock must be in an uptrend
        if ctx.spot <= ctx.ema_20:
            log.debug("[%s] Pre-Earnings: spot below EMA20 — skip", ctx.symbol)
            return None

        if ctx.rsi_14 < self.RSI_MIN:
            log.debug("[%s] Pre-Earnings: RSI %.1f < %d — skip", ctx.symbol, ctx.rsi_14, self.RSI_MIN)
            return None

        # Confluence check
        score, met = self.check_confluence(ctx)
        if score < self.CONFLUENCE_MIN:
            log.debug("[%s] Pre-Earnings: confluence %d/5 — need %d", ctx.symbol, score, self.CONFLUENCE_MIN)
            return None

        # Always buy CALL (riding upward drift)
        direction = "CALL"
        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, itm_strikes=1)
        if sid == 0 or premium <= 0:
            return None

        confidence = "HIGH" if score >= 4 else "MEDIUM"
        return StockSignal(
            strategy=self.NAME,
            symbol=ctx.symbol,
            direction=direction,
            spot=ctx.spot,
            strike=strike,
            expiry=ctx.expiry,
            option_type="CE",
            security_id=sid,
            tradingsymbol=tsym,
            expected_premium=premium,
            confidence=confidence,
            confluence_score=score,
            rationale=(
                f"Pre-earnings drift | {days_to_earnings}d to earnings ({ctx.earnings_next}) | "
                f"RSI={ctx.rsi_14:.0f} | Spot={ctx.spot:.0f} EMA20={ctx.ema_20:.0f} | "
                f"Confluence {score}/5: {','.join(met)}"
            ),
        )
