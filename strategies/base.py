"""
base.py
Shared types and confluence checker for Nifty50 stock options strategies.
Confluence rule: need 3 of 5 factors before any entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data.market_data import StockMarketContext


@dataclass
class StockSignal:
    strategy: str
    symbol: str
    direction: str          # "CALL" / "PUT"
    spot: float
    strike: float
    expiry: date
    option_type: str        # "CE" / "PE"
    security_id: int
    tradingsymbol: str
    expected_premium: float
    confidence: str         # "HIGH" / "MEDIUM"
    rationale: str
    confluence_score: int = 0   # how many confluence factors met (5 base + 2 OI when available)
    lots: int = 1


class BaseStrategy:
    NAME: str = "BASE"

    def evaluate(self, ctx: "StockMarketContext") -> Optional[StockSignal]:
        raise NotImplementedError

    # ── confluence check ──────────────────────────────────────────────────────

    @staticmethod
    def check_confluence(ctx: "StockMarketContext", vix_max: float = 18.0) -> tuple[int, list[str]]:
        """
        Returns (score, met_factors).
        Score: count of confluence factors met (0-7).
        Need at least 3 to proceed.

        Factors:
          1. Stock above 20-day EMA
          2. RSI between 50-65 (momentum, not overbought)
          3. Volume on trigger candle >= 1.5x 20-day average
          4. India VIX < vix_max (18)
          5. Sector FII data positive or neutral
          6. PCR < 0.8 (call-heavy chain = bullish OI sentiment)       [skipped if OI unavailable]
          7. ATM CE OI > ATM PE OI (call concentration at ATM)         [skipped if OI unavailable]
          8. ATM CE OI change > 0 (fresh call positions being built)   [skipped if OI unavailable]
        """
        met = []
        closes = ctx.bars_daily["close"].tolist()
        volumes = ctx.bars_daily["volume"].tolist()
        today_vol = volumes[-1] if volumes else 0

        if ctx.spot > ctx.ema_20:
            met.append("above_ema20")

        if 50 <= ctx.rsi_14 <= 65:
            met.append("rsi_50_65")

        if ctx.vol_avg_20 > 0 and today_vol >= ctx.vol_avg_20 * 1.5:
            met.append("volume_confirm")

        if ctx.vix < vix_max:
            met.append("vix_ok")

        if ctx.fii_sector_trend in ("POSITIVE", "NEUTRAL"):
            met.append("fii_ok")

        # OI-based factors — skipped gracefully when OI is unavailable (e.g. synthetic backtest chain)
        total_ce_oi = ctx.chain["ce_oi"].sum()
        total_pe_oi = ctx.chain["pe_oi"].sum()
        if total_ce_oi > 0 and total_pe_oi > 0:
            pcr = total_pe_oi / total_ce_oi
            if pcr < 0.8:
                met.append("pcr_bullish")
            atm_row = ctx.chain[ctx.chain["strike"] == ctx.atm_strike]
            if not atm_row.empty:
                atm = atm_row.iloc[0]
                if float(atm["ce_oi"]) > float(atm["pe_oi"]):
                    met.append("oi_skew_call")
                # CE OI building (day-over-day) = fresh call positions being added
                if "ce_oi_chg" in ctx.chain.columns and float(atm["ce_oi_chg"]) > 0:
                    met.append("oi_build_call")

        return len(met), met

    @staticmethod
    def check_bearish_confluence(ctx: "StockMarketContext", vix_min: float = 12.0) -> tuple[int, list[str]]:
        """
        Bearish confluence (PUT strategies). Need 3 of 8.
          1. Stock below 20-day EMA
          2. RSI 35-55 (weak, not yet oversold)
          3. Volume >= 1.5x 20-day average
          4. VIX > vix_min (some fear in market)
          5. Sector FII trend negative or neutral
          6. PCR > 1.2 (put-heavy chain = bearish OI sentiment)       [skipped if OI unavailable]
          7. ATM PE OI > ATM CE OI (put concentration at ATM)         [skipped if OI unavailable]
          8. ATM PE OI change > 0 (fresh put positions being built)   [skipped if OI unavailable]
        """
        met = []
        volumes = ctx.bars_daily["volume"].tolist()
        today_vol = volumes[-1] if volumes else 0

        if ctx.spot < ctx.ema_20:
            met.append("below_ema20")

        if 35 <= ctx.rsi_14 <= 55:
            met.append("rsi_35_55")

        if ctx.vol_avg_20 > 0 and today_vol >= ctx.vol_avg_20 * 1.5:
            met.append("volume_confirm")

        if ctx.vix > vix_min:
            met.append("vix_elevated")

        if ctx.fii_sector_trend in ("NEGATIVE", "NEUTRAL"):
            met.append("fii_negative")

        # OI-based factors — skipped gracefully when OI is unavailable (e.g. synthetic backtest chain)
        total_ce_oi = ctx.chain["ce_oi"].sum()
        total_pe_oi = ctx.chain["pe_oi"].sum()
        if total_ce_oi > 0 and total_pe_oi > 0:
            pcr = total_pe_oi / total_ce_oi
            if pcr > 1.2:
                met.append("pcr_bearish")
            atm_row = ctx.chain[ctx.chain["strike"] == ctx.atm_strike]
            if not atm_row.empty:
                atm = atm_row.iloc[0]
                if float(atm["pe_oi"]) > float(atm["ce_oi"]):
                    met.append("oi_skew_put")
                # PE OI building (day-over-day) = fresh put positions being added
                if "pe_oi_chg" in ctx.chain.columns and float(atm["pe_oi_chg"]) > 0:
                    met.append("oi_build_put")

        return len(met), met

    # ── option selection helpers ──────────────────────────────────────────────

    @staticmethod
    def pick_itm_option(ctx: "StockMarketContext", direction: str,
                         itm_strikes: int = 1) -> tuple[int, str, float, float]:
        """
        Pick ATM or ITM option from chain.
        itm_strikes=0 → ATM, itm_strikes=1 → 1 strike ITM
        Returns (security_id, tradingsymbol, strike, premium)
        """
        opt = "ce" if direction == "CALL" else "pe"
        sid_col = f"{opt}_sid"
        ltp_col = f"{opt}_ltp"

        atm = ctx.atm_strike
        gap = ctx.strike_gap

        # ITM for CALL means lower strike, for PUT means higher strike
        if direction == "CALL":
            target_strike = atm - gap * itm_strikes
        else:
            target_strike = atm + gap * itm_strikes

        chain = ctx.chain
        row = chain[chain["strike"] == target_strike]
        if row.empty:
            idx = (chain["strike"] - target_strike).abs().idxmin()
            row = chain.iloc[[idx]]

        if row.empty or row.iloc[0][sid_col] is None:
            # Fall back to ATM
            row = chain[chain["strike"] == atm]
            if row.empty:
                return 0, "", atm, 0.0

        sid = int(row.iloc[0][sid_col]) if row.iloc[0][sid_col] else 0
        ltp = float(row.iloc[0][ltp_col])
        strike = float(row.iloc[0]["strike"])
        return sid, "", strike, ltp

    # ── technical helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _ema(closes: list[float], period: int) -> float:
        if len(closes) < period:
            return closes[-1] if closes else 0.0
        k = 2 / (period + 1)
        val = sum(closes[:period]) / period
        for c in closes[period:]:
            val = c * k + val * (1 - k)
        return val

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(-period, 0):
            d = closes[i] - closes[i - 1]
            (gains if d > 0 else losses).append(abs(d))
        ag = sum(gains) / period if gains else 0.0
        al = sum(losses) / period if losses else 0.0
        if al == 0:
            return 100.0
        return 100 - 100 / (1 + ag / al)

    @staticmethod
    def _highest_close(closes: list[float], lookback: int) -> float:
        window = closes[-lookback:] if len(closes) >= lookback else closes
        return max(window) if window else 0.0

    @staticmethod
    def _lowest_close(closes: list[float], lookback: int) -> float:
        window = closes[-lookback:] if len(closes) >= lookback else closes
        return min(window) if window else 0.0

    @staticmethod
    def _swing_high(highs: list[float], lookback: int) -> float:
        window = highs[-lookback:] if len(highs) >= lookback else highs
        return max(window) if window else 0.0

    @staticmethod
    def _swing_low(lows: list[float], lookback: int) -> float:
        window = lows[-lookback:] if len(lows) >= lookback else lows
        return min(window) if window else 0.0

    @staticmethod
    def _is_bullish_candle(open_: float, close: float, high: float, low: float) -> bool:
        body = abs(close - open_)
        total_range = high - low
        return close > open_ and body >= 0.5 * total_range

    @staticmethod
    def _is_hammer(open_: float, close: float, high: float, low: float) -> bool:
        body = abs(close - open_)
        total_range = high - low
        lower_wick = min(open_, close) - low
        return lower_wick >= 2 * body and total_range > 0
