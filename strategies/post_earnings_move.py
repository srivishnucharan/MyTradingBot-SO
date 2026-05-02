"""
post_earnings_move.py
Strategy — Post-Earnings Move (Re-Rating / Downgrade)
POP: 60-70% | R:R: 1:2.5 | Frequency: 4-8 per quarter

After a stock makes a large single-day move (>=3%) following results:
- Large UP move  -> buy CALL  (re-rating, institutional accumulation)
- Large DOWN move -> buy PUT  (downgrade, institutional distribution)

Entry: 1-3 days after the event. SL/Target follow standard risk rules.
Rationale: earnings-driven re-ratings and downgrades have strong follow-through
because institutional mandates (index rebalancing, target price changes, analyst
upgrades/downgrades) take 2-10 sessions to fully play out.
"""
from __future__ import annotations

import logging
from typing import Optional

from strategies.base import BaseStrategy, StockSignal
from data.market_data import StockMarketContext

log = logging.getLogger(__name__)


class PostEarningsMoveStrategy(BaseStrategy):
    NAME_CALL = "Post-Earnings Re-Rating"
    NAME_PUT  = "Post-Earnings Downgrade"

    MOVE_THRESHOLD = 0.03   # 3% single-day move triggers the setup
    LOOKBACK_DAYS  = 5      # Look for the event within last 5 sessions
    REVERSAL_TOL   = 0.03   # Ignore if spot has reversed >3% from event close
    CONFLUENCE_MIN = 3

    def evaluate(self, ctx: StockMarketContext) -> Optional[StockSignal]:
        if len(ctx.bars_daily) < self.LOOKBACK_DAYS + 3:
            return None

        closes = ctx.bars_daily["close"].tolist()

        # Find the most recent single-day move >= threshold (within last LOOKBACK_DAYS)
        # lookback=1 → yesterday vs day-before-yesterday
        event_move = None
        event_close = None
        for lookback in range(1, self.LOOKBACK_DAYS + 1):
            idx_curr = -(lookback + 1)
            idx_prev = -(lookback + 2)
            if abs(idx_prev) > len(closes):
                break
            prev = closes[idx_prev]
            curr = closes[idx_curr]
            if prev <= 0:
                continue
            move = (curr - prev) / prev
            if abs(move) >= self.MOVE_THRESHOLD:
                event_move = move
                event_close = curr
                break  # most recent qualifying event

        if event_move is None or event_close is None:
            return None

        is_bullish = event_move > 0

        # Spot must not have fully reversed since the event
        if is_bullish:
            if ctx.spot < event_close * (1 - self.REVERSAL_TOL):
                return None
        else:
            if ctx.spot > event_close * (1 + self.REVERSAL_TOL):
                return None

        # RSI guards: don't chase exhausted moves
        if is_bullish and ctx.rsi_14 > 75:
            log.debug("[%s] PostEarnings: RSI %.1f overbought after up-move", ctx.symbol, ctx.rsi_14)
            return None
        if not is_bullish and ctx.rsi_14 < 25:
            log.debug("[%s] PostEarnings: RSI %.1f oversold after down-move", ctx.symbol, ctx.rsi_14)
            return None

        # Confluence
        if is_bullish:
            score, met = self.check_confluence(ctx)
        else:
            score, met = self.check_bearish_confluence(ctx)

        if score < self.CONFLUENCE_MIN:
            log.debug("[%s] PostEarnings: confluence %d < %d", ctx.symbol, score, self.CONFLUENCE_MIN)
            return None

        direction = "CALL" if is_bullish else "PUT"
        option_type = "CE" if is_bullish else "PE"
        strategy_name = self.NAME_CALL if is_bullish else self.NAME_PUT

        sid, tsym, strike, premium = self.pick_itm_option(ctx, direction, itm_strikes=1)
        if sid == 0 or premium <= 0:
            return None

        move_pct = event_move * 100
        confidence = "HIGH" if abs(event_move) >= 0.05 else "MEDIUM"
        return StockSignal(
            strategy=strategy_name,
            symbol=ctx.symbol,
            direction=direction,
            spot=ctx.spot,
            strike=strike,
            expiry=ctx.expiry,
            option_type=option_type,
            security_id=sid,
            tradingsymbol=tsym,
            expected_premium=premium,
            confidence=confidence,
            confluence_score=score,
            rationale=(
                f"Post-earnings {'re-rating' if is_bullish else 'downgrade'} | "
                f"Event move={move_pct:+.1f}% | EventClose={event_close:.0f} | "
                f"Spot={ctx.spot:.0f} | RSI={ctx.rsi_14:.0f} | Confluence {score}/5"
            ),
        )
