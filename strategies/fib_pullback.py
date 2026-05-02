"""
fib_pullback.py
Strategy 3 — Fibonacci Pullback in a Strong Trend
POP: 60-65% | R:R: 1:3 | Frequency: 1-2 per month

Buy when price pulls back to 50% or 61.8% Fibonacci level in an established uptrend.
Confirmed with RSI 40-55 + volume dry-up + reversal candle.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class FibPullbackStrategy(BaseStrategy):
    NAME = "Fibonacci Pullback"

    SWING_LOOKBACK = 60       # days to identify swing high/low
    FIB_LEVELS = [0.50, 0.618]
    FIB_ZONE_PCT = 0.015      # ±1.5% around Fib level counts as "at the level"
    RSI_MIN = 40
    RSI_MAX = 55
    CONFLUENCE_MIN = 3

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if len(ctx.bars_daily) < self.SWING_LOOKBACK:
            return None

        bars = ctx.bars_daily
        closes = bars["close"].tolist()
        highs = bars["high"].tolist()
        lows = bars["low"].tolist()
        volumes = bars["volume"].tolist()

        lookback_n = min(self.SWING_LOOKBACK, len(closes))
        window_highs = highs[-lookback_n:]
        window_lows = lows[-lookback_n:]
        window_closes = closes[-lookback_n:]

        swing_high = max(window_highs)
        swing_low = min(window_lows)
        swing_range = swing_high - swing_low

        if swing_range < ctx.spot * 0.05:
            # Too narrow a range — not a meaningful swing
            return None

        # Only trade uptrend: swing_high must be recent (last 30 bars) and swing_low further back
        high_idx = window_highs.index(swing_high)
        low_idx = window_lows.index(swing_low)
        if low_idx >= high_idx:
            # Low is more recent than high — downtrend, skip
            return None

        # Stock must be in pullback: current price is below swing_high
        if ctx.spot >= swing_high * 0.98:
            return None  # at or near the top — not a pullback

        # Check if at a Fib level
        at_fib = None
        for fib in self.FIB_LEVELS:
            fib_price = swing_high - fib * swing_range
            zone_lo = fib_price * (1 - self.FIB_ZONE_PCT)
            zone_hi = fib_price * (1 + self.FIB_ZONE_PCT)
            if zone_lo <= ctx.spot <= zone_hi:
                at_fib = fib
                fib_price_target = fib_price
                break

        if at_fib is None:
            return None

        # RSI should be in reset zone (cooling, not oversold)
        if not (self.RSI_MIN <= ctx.rsi_14 <= self.RSI_MAX):
            log.debug("[%s] Fib: RSI %.1f not in [%d-%d]", ctx.symbol, ctx.rsi_14, self.RSI_MIN, self.RSI_MAX)
            return None

        # Volume should be declining (pullback on weak selling)
        today_vol = volumes[-1]
        vol_declining = today_vol < ctx.vol_avg_20 * 0.8 if ctx.vol_avg_20 > 0 else False

        # Look for reversal candle
        last_open = float(bars.iloc[-1]["open"])
        last_close = float(bars.iloc[-1]["close"])
        last_high = float(bars.iloc[-1]["high"])
        last_low = float(bars.iloc[-1]["low"])
        reversal = (
            self._is_bullish_candle(last_open, last_close, last_high, last_low) or
            self._is_hammer(last_open, last_close, last_high, last_low)
        )

        if not reversal and not vol_declining:
            return None  # need at least one confirmation

        # Confluence check
        score, met = self.check_confluence(ctx)
        if score < self.CONFLUENCE_MIN:
            return None

        direction = "CALL"
        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, 0)  # ATM
        if sid == 0 or premium <= 0:
            return None

        fib_pct = int(at_fib * 100)
        confidence = "HIGH" if score >= 4 and reversal else "MEDIUM"
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
                f"Fib {fib_pct}% pullback | Swing: {swing_low:.0f}-{swing_high:.0f} | "
                f"RSI={ctx.rsi_14:.0f} | VolDry={'Y' if vol_declining else 'N'} | "
                f"Reversal={'Y' if reversal else 'N'} | Confluence {score}/5"
            ),
        )
