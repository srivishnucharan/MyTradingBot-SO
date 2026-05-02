"""
breakout_retest.py
Strategy 2 — Breakout-Retest Entry
POP: 62-68% | R:R: 1:3 to 1:4 | Frequency: 2-3 per month

Wait for stock to break out of multi-week consolidation, then buy the retest.
Never chase the initial breakout — enter on the retest of broken resistance.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class BreakoutRetestStrategy(BaseStrategy):
    NAME = "Breakout Retest"

    CONSOLIDATION_LOOKBACK = 20  # bars for consolidation range
    BREAKOUT_VOL_MULT = 1.5     # breakout day volume must be >= this x average
    RETEST_DAYS = 10             # max days to be in retest mode
    RSI_MIN = 45
    CONFLUENCE_MIN = 3

    def __init__(self):
        # Track per-symbol breakout state
        self._breakout_levels: dict[str, tuple[float, date]] = {}  # symbol → (level, breakout_date)

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if len(ctx.bars_daily) < self.CONSOLIDATION_LOOKBACK + 5:
            return None

        bars = ctx.bars_daily
        closes = bars["close"].tolist()
        highs = bars["high"].tolist()
        lows = bars["low"].tolist()
        volumes = bars["volume"].tolist()

        today = date.today()
        symbol = ctx.symbol

        # Check if we already have a breakout level tracked
        if symbol in self._breakout_levels:
            level, bk_date = self._breakout_levels[symbol]
            days_since = (today - bk_date).days

            if days_since > self.RETEST_DAYS:
                # Expired — clear
                del self._breakout_levels[symbol]
            else:
                # Check if current price is retesting the breakout level (within 1%)
                retest_zone_lo = level * 0.99
                retest_zone_hi = level * 1.015

                if retest_zone_lo <= ctx.spot <= retest_zone_hi:
                    # Volume should be declining (drying up) on pullback
                    recent_vol = volumes[-1]
                    avg_vol = ctx.vol_avg_20
                    volume_dry = recent_vol < avg_vol * 0.8

                    # RSI should be healthy
                    rsi_ok = ctx.rsi_14 >= self.RSI_MIN

                    # Look for bullish reversal candle today
                    last_open = float(bars.iloc[-1]["open"])
                    last_close = float(bars.iloc[-1]["close"])
                    last_high = float(bars.iloc[-1]["high"])
                    last_low = float(bars.iloc[-1]["low"])
                    bullish_candle = (
                        self._is_bullish_candle(last_open, last_close, last_high, last_low) or
                        self._is_hammer(last_open, last_close, last_high, last_low)
                    )

                    score, met = self.check_confluence(ctx)

                    if rsi_ok and score >= self.CONFLUENCE_MIN and (volume_dry or bullish_candle):
                        del self._breakout_levels[symbol]  # consume signal
                        direction = "CALL"
                        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, 1)
                        if sid == 0 or premium <= 0:
                            return None

                        confidence = "HIGH" if score >= 4 and bullish_candle else "MEDIUM"
                        return StockSignal(
                            strategy=self.NAME,
                            symbol=symbol,
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
                                f"Breakout retest of {level:.0f} | {days_since}d since breakout | "
                                f"RSI={ctx.rsi_14:.0f} | VolumeOK={'Y' if volume_dry else 'N'} | "
                                f"BullCandle={'Y' if bullish_candle else 'N'} | Confluence {score}/5"
                            ),
                        )
                return None  # In retest tracking but not at zone yet

        # Look for fresh breakout: close above 20-day high with strong volume
        lookback_closes = closes[-(self.CONSOLIDATION_LOOKBACK + 1):-1]
        if not lookback_closes:
            return None

        resistance = max(lookback_closes)  # highest close in prior CONSOLIDATION_LOOKBACK bars
        today_close = closes[-1]
        today_vol = volumes[-1]

        # Breakout: close above resistance with 1.5x volume
        if today_close > resistance * 1.005 and ctx.vol_avg_20 > 0:
            if today_vol >= ctx.vol_avg_20 * self.BREAKOUT_VOL_MULT:
                # Record breakout — wait for retest next cycle
                self._breakout_levels[symbol] = (resistance, today)
                log.info("[%s] Breakout detected above %.0f — watching for retest", symbol, resistance)

        return None
