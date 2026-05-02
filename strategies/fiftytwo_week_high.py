"""
fiftytwo_week_high.py
Strategy 5 — 52-Week High Momentum Play
POP: 60-65% | R:R: 1:3 to 1:5 | Frequency: 1-2 per month

Counter-intuitive but statistically powerful: buy stocks making new 52-week highs
with 2x+ volume. No overhead resistance, institutional momentum behind the move.
Enter after 1-2 day consolidation above the prior high.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class FiftyTwoWeekHighStrategy(BaseStrategy):
    NAME = "52-Week High"

    VOLUME_MULTIPLIER = 2.0     # Breakout day volume must be >= 2x average
    CONSOL_DAYS = 2             # Wait for 1-2 day consolidation above the high
    TOLERANCE_PCT = 0.02        # Allow up to 2% below 52-week high as "still near"
    CONFLUENCE_MIN = 3

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if len(ctx.bars_daily) < 260:  # Need a full year of data
            log.debug("[%s] 52WH: insufficient data (%d bars)", ctx.symbol, len(ctx.bars_daily))
            return None

        bars = ctx.bars_daily
        closes = bars["close"].tolist()
        highs = bars["high"].tolist()
        volumes = bars["volume"].tolist()

        # 52-week high from the last 252 trading days
        high_252 = max(highs[-252:])
        current_close = closes[-1]

        # Stock must be near or at 52-week high (within tolerance)
        if current_close < high_252 * (1 - self.TOLERANCE_PCT):
            return None

        # Did we just break the 52-week high recently (in last 1-3 days)?
        prior_high = max(highs[-255:-3]) if len(highs) >= 255 else max(highs[:-3])
        breakout_day_idx = None

        for i in range(-3, -1):
            if len(closes) < abs(i):
                continue
            if highs[i] >= prior_high and volumes[i] >= ctx.vol_avg_20 * self.VOLUME_MULTIPLIER:
                breakout_day_idx = i
                break

        if breakout_day_idx is None:
            return None  # No recent high-volume breakout

        # Must have consolidated for at least 1 day after breakout
        days_since_breakout = abs(breakout_day_idx) - 1
        if days_since_breakout < 1:
            return None

        # Confirm not extended — should consolidate sideways (within 2% of breakout close)
        breakout_close = closes[breakout_day_idx]
        consolidation_ok = all(
            abs(closes[i] - breakout_close) / breakout_close < 0.02
            for i in range(breakout_day_idx + 1, 0)
            if len(closes) > abs(i)
        )

        if not consolidation_ok:
            return None

        # Stock should be above 20-day EMA (always in uptrend at 52-week high)
        if ctx.spot < ctx.ema_20 * 0.98:
            return None

        # Confluence check
        score, met = self.check_confluence(ctx)
        if score < self.CONFLUENCE_MIN:
            return None

        direction = "CALL"
        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, itm_strikes=0)  # ATM
        if sid == 0 or premium <= 0:
            return None

        breakout_vol = volumes[breakout_day_idx]
        vol_mult = round(breakout_vol / ctx.vol_avg_20, 1) if ctx.vol_avg_20 > 0 else 0

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
                f"52-week high breakout | 52W-High={high_252:.0f} | "
                f"Breakout vol={vol_mult:.1f}x avg | {days_since_breakout}d consolidation | "
                f"RSI={ctx.rsi_14:.0f} | Confluence {score}/5: {','.join(met)}"
            ),
        )
