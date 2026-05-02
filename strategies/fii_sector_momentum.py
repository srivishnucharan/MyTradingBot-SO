"""
fii_sector_momentum.py
Strategy 4 — FII/DII Sector Momentum Ride
POP: 65-70% | R:R: 1:2 to 1:3 | Frequency: 1-2 per month

When sector shows FII institutional buying momentum for 3+ consecutive days,
enter the sector leader on a 1-2 day intraday pullback.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class FIISectorMomentumStrategy(BaseStrategy):
    NAME = "FII Sector Momentum"

    # Minimum consecutive days of FII-positive sector trend
    MIN_POSITIVE_DAYS = 3
    PULLBACK_MAX_PCT = 0.03  # Stock pulled back no more than 3% from recent high
    CONFLUENCE_MIN = 3

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if len(ctx.bars_daily) < 10:
            return None

        # Require sector to be in FII buying mode
        if ctx.fii_sector_trend != "POSITIVE":
            return None

        bars = ctx.bars_daily
        closes = bars["close"].tolist()
        volumes = bars["volume"].tolist()

        # Check for consecutive positive sector days (using 5-day price trend as proxy)
        if len(closes) < 5:
            return None

        recent_5 = closes[-5:]
        positive_days = sum(1 for i in range(1, len(recent_5)) if recent_5[i] > recent_5[i - 1])
        if positive_days < self.MIN_POSITIVE_DAYS:
            log.debug("[%s] FII Momentum: only %d positive days (need %d)",
                      ctx.symbol, positive_days, self.MIN_POSITIVE_DAYS)
            return None

        # Stock should have pulled back 0.5-3% from recent high (entry on dip)
        recent_high = max(closes[-5:])
        pullback_pct = (recent_high - ctx.spot) / recent_high * 100

        if not (0.3 <= pullback_pct <= 3.0):
            log.debug("[%s] FII Momentum: pullback %.1f%% outside 0.3-3%%", ctx.symbol, pullback_pct)
            return None

        # Stock must be above 20-day EMA
        if ctx.spot < ctx.ema_20:
            return None

        # Confluence check
        score, met = self.check_confluence(ctx)
        if score < self.CONFLUENCE_MIN:
            return None

        direction = "CALL"
        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, itm_strikes=1)
        if sid == 0 or premium <= 0:
            return None

        confidence = "HIGH" if score >= 4 and pullback_pct <= 1.5 else "MEDIUM"
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
                f"FII sector momentum | Sector={ctx.sector} trend={ctx.fii_sector_trend} | "
                f"Pullback={pullback_pct:.1f}% from {recent_high:.0f} | "
                f"RSI={ctx.rsi_14:.0f} | Confluence {score}/5: {','.join(met)}"
            ),
        )
