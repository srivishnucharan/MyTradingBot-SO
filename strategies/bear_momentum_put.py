"""
bear_momentum_put.py
Strategy — Bear Momentum Put
POP: 55-65% | R:R: 1:2.5 | Frequency: 1-2 per month

Buys PUT when a stock is in a confirmed downtrend with institutional selling,
entering on a weak dead-cat bounce (0.3-3% rally off the recent low).

Setup:
- FII sector trend is NEGATIVE (price momentum proxy for institutional selling)
- Stock is below 20-day EMA (confirmed downtrend)
- At least 3 of last 5 sessions closed lower (bearish sequence)
- Current price has bounced 0.3-3% off 5-day low (entry on the rally, not the low)
- RSI in 35-60 range (weak, not yet washed out — still room to fall)

This mirrors FII Sector Momentum (which buys dips in uptrends) but applied to
the downside: selling weak bounces in FII-selling environments.
"""
from __future__ import annotations

import logging
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class BearMomentumPutStrategy(BaseStrategy):
    NAME = "Bear Momentum Put"

    MIN_NEGATIVE_DAYS = 3
    BOUNCE_MIN_PCT = 0.3
    BOUNCE_MAX_PCT = 3.0
    RSI_MIN = 35
    RSI_MAX = 60
    CONFLUENCE_MIN = 3

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if len(ctx.bars_daily) < 10:
            return None

        if ctx.fii_sector_trend != "NEGATIVE":
            return None

        bars = ctx.bars_daily
        closes = bars["close"].tolist()

        if len(closes) < 5:
            return None

        # At least MIN_NEGATIVE_DAYS of last 5 sessions must be down
        recent_5 = closes[-5:]
        negative_days = sum(1 for i in range(1, len(recent_5)) if recent_5[i] < recent_5[i - 1])
        if negative_days < self.MIN_NEGATIVE_DAYS:
            log.debug("[%s] BearPut: only %d negative days (need %d)",
                      ctx.symbol, negative_days, self.MIN_NEGATIVE_DAYS)
            return None

        # Stock must be below 20-day EMA (confirmed downtrend)
        if ctx.spot > ctx.ema_20:
            return None

        # Entry on a dead-cat bounce off the 5-day low
        recent_low = min(closes[-5:])
        if recent_low <= 0:
            return None
        bounce_pct = (ctx.spot - recent_low) / recent_low * 100
        if not (self.BOUNCE_MIN_PCT <= bounce_pct <= self.BOUNCE_MAX_PCT):
            log.debug("[%s] BearPut: bounce %.1f%% outside [%.1f-%.1f%%]",
                      ctx.symbol, bounce_pct, self.BOUNCE_MIN_PCT, self.BOUNCE_MAX_PCT)
            return None

        # RSI in weak-rally zone — not washed out, still room to fall
        if not (self.RSI_MIN <= ctx.rsi_14 <= self.RSI_MAX):
            log.debug("[%s] BearPut: RSI %.1f not in [%d-%d]",
                      ctx.symbol, ctx.rsi_14, self.RSI_MIN, self.RSI_MAX)
            return None

        score, met = self.check_bearish_confluence(ctx)
        if score < self.CONFLUENCE_MIN:
            return None

        direction = "PUT"
        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, itm_strikes=1)
        if sid == 0 or premium <= 0:
            return None

        confidence = "HIGH" if score >= 4 and bounce_pct <= 1.5 else "MEDIUM"
        return StockSignal(
            strategy=self.NAME,
            symbol=ctx.symbol,
            direction=direction,
            spot=ctx.spot,
            strike=strike,
            expiry=ctx.expiry,
            option_type="PE",
            security_id=sid,
            tradingsymbol=tsym,
            expected_premium=premium,
            confidence=confidence,
            confluence_score=score,
            rationale=(
                f"Bear momentum PUT | Sector={ctx.sector} FII={ctx.fii_sector_trend} | "
                f"Bounce={bounce_pct:.1f}% from {recent_low:.0f} | BelowEMA20={ctx.ema_20:.0f} | "
                f"RSI={ctx.rsi_14:.0f} | Confluence {score}/5: {','.join(met)}"
            ),
        )
