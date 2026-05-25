"""
market_data.py
Fetches and structures market data for Nifty50 stock options swing trading.
Provides: daily OHLCV bars, EMA, RSI, volume metrics, option chain, VIX.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import exp, log as mlog, sqrt, erf
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from data.dhan_client import DhanClient
from data.security_master import SecurityMaster

log = logging.getLogger(__name__)

# Forward reference for type hint (avoids circular import)
_MacroSentiment = None


@dataclass
class StockMarketContext:
    """All data needed by all strategies for one stock per cycle."""
    symbol: str
    security_id: int
    sector: str
    spot: float
    vix: float
    expiry: date
    atm_strike: float
    strike_gap: int
    bars_daily: pd.DataFrame     # Daily OHLCV, columns: open,high,low,close,volume,date
    chain: pd.DataFrame          # Option chain: strike,ce_ltp,pe_ltp,ce_oi,pe_oi,ce_sid,pe_sid
    ema_20: float
    ema_50: float
    rsi_14: float
    vol_avg_20: float            # 20-day average volume
    hist_vol_30: float           # 30-day historical volatility (annualised)
    earnings_next: Optional[date]
    fii_sector_trend: str        # "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    macro_sentiment: Optional[object] = None  # MacroSentiment — set by orchestrator/backtest
    timestamp: datetime = field(default_factory=datetime.now)


class MarketData:
    def __init__(self, dhan: Optional[DhanClient] = None):
        self.dhan = dhan or DhanClient()
        self.master = SecurityMaster()

    # ── public ────────────────────────────────────────────────────────────────

    def build_context(self, instr: dict, expiry: date,
                       earnings_next: Optional[date] = None) -> Optional[StockMarketContext]:
        symbol = instr["symbol"]
        strike_gap = instr.get("strike_gap", 10)

        try:
            sec_id, eq_seg = self.master.equity_info(symbol)
        except KeyError as e:
            log.warning("%s", e)
            return None

        sector = self.master.sector(symbol)
        today = date.today()
        hist_from = (today - timedelta(days=400)).strftime("%Y-%m-%d")  # ~15 months
        today_str = today.strftime("%Y-%m-%d")

        # Fetch daily bars via yfinance (primary) or Dhan historical API
        try:
            bars_daily = self._fetch_daily_bars(str(sec_id), eq_seg, hist_from, today_str, symbol=symbol)
        except Exception as e:
            log.warning("Daily bars failed for %s: %s", symbol, e)
            return None

        if bars_daily.empty or len(bars_daily) < 30:
            log.warning("Insufficient daily bars for %s (%d)", symbol, len(bars_daily))
            return None

        spot = float(bars_daily.iloc[-1]["close"])

        # Derived indicators
        closes = bars_daily["close"].tolist()
        volumes = bars_daily["volume"].tolist()
        ema_20 = self._ema(closes, 20)
        ema_50 = self._ema(closes, 50) if len(closes) >= 50 else ema_20
        rsi_14 = self._rsi(closes, 14)
        vol_avg_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        hist_vol = self._hist_vol(closes, 30)

        # ATM strike
        atm = round(spot / strike_gap) * strike_gap

        # Option chain
        try:
            chain = self._fetch_chain(sec_id, eq_seg, expiry, symbol)
        except Exception as e:
            log.warning("Option chain failed for %s: %s", symbol, e)
            return None

        # VIX
        vix = self._fetch_vix()

        # FII sector trend approximation (price momentum proxy)
        fii_trend = self._estimate_fii_trend(bars_daily, sector)

        return StockMarketContext(
            symbol=symbol,
            security_id=sec_id,
            sector=sector,
            spot=spot,
            vix=vix,
            expiry=expiry,
            atm_strike=atm,
            strike_gap=strike_gap,
            bars_daily=bars_daily,
            chain=chain,
            ema_20=ema_20,
            ema_50=ema_50,
            rsi_14=rsi_14,
            vol_avg_20=vol_avg_20,
            hist_vol_30=hist_vol,
            earnings_next=earnings_next,
            fii_sector_trend=fii_trend,
        )

    def fetch_vix(self) -> float:
        return self._fetch_vix()

    def fetch_stock_ltp(self, security_id: int) -> float:
        try:
            resp = self.dhan.ltp({"NSE_EQ": [security_id]})
            data = resp.get("data", {})
            if isinstance(data, dict):
                return float(data.get("NSE_EQ", {}).get(str(security_id), {}).get("last_price", 0))
        except Exception as e:
            log.debug("Stock LTP failed (sid=%s): %s", security_id, e)
        return 0.0

    def fetch_option_ltp(self, security_id: int, symbol: str) -> float:
        try:
            resp = self.dhan.ltp({"NSE_FNO": [security_id]})
            data = resp.get("data", {})
            if isinstance(data, dict):
                return float(data.get("NSE_FNO", {}).get(str(security_id), {}).get("last_price", 0))
        except Exception as e:
            log.debug("Option LTP failed (sid=%s): %s", security_id, e)
        return 0.0

    # ── private ───────────────────────────────────────────────────────────────

    def _fetch_daily_bars(self, sec_id: str, segment: str,
                           from_date: str, to_date: str,
                           symbol: Optional[str] = None) -> pd.DataFrame:
        if _YF_AVAILABLE and symbol:
            df = self._fetch_bars_yf(symbol, from_date, to_date)
            if df is not None and len(df) >= 30:
                return df

        # Dhan historical_daily_data fallback
        try:
            data = self.dhan.historical_daily_data(
                security_id=sec_id,
                exchange_segment=segment,
                instrument="EQUITY",
                from_date=from_date,
                to_date=to_date,
            )
            if data and data.get("open"):
                df = self._to_df(data)
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                df = df.sort_values("timestamp").drop_duplicates("date").reset_index(drop=True)
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            log.warning("Dhan historical_daily_data failed for sec_id=%s: %s", sec_id, e)

        return pd.DataFrame()

    @staticmethod
    def _fetch_bars_yf(symbol: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(start=from_date, end=to_date, auto_adjust=True)
            if hist.empty:
                return None
            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
            hist = hist.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                        "Close": "close", "Volume": "volume"})
            return hist[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception as e:
            log.debug("yfinance fetch failed for %s: %s", symbol, e)
            return None

    def _to_df(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame({
            "open":      data.get("open", []),
            "high":      data.get("high", []),
            "low":       data.get("low", []),
            "close":     data.get("close", []),
            "volume":    data.get("volume", []),
            "timestamp": data.get("timestamp", []),
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _fetch_chain(self, sec_id: int, segment: str,
                      expiry: date, symbol: str) -> pd.DataFrame:
        raw = self.dhan.option_chain(sec_id, segment, expiry)
        if raw is None:
            raise RuntimeError("Option chain returned None")
        oc = raw.get("oc", {})
        rows = []
        for strike_str, legs in oc.items():
            ce = legs.get("ce", {}) or {}
            pe = legs.get("pe", {}) or {}
            rows.append({
                "strike":    float(strike_str),
                "ce_ltp":    ce.get("last_price", 0),
                "pe_ltp":    pe.get("last_price", 0),
                "ce_oi":     ce.get("oi", 0),
                "pe_oi":     pe.get("oi", 0),
                "ce_volume": ce.get("volume", 0),
                "pe_volume": pe.get("volume", 0),
                "ce_iv":     ce.get("implied_volatility", 0),
                "pe_iv":     pe.get("implied_volatility", 0),
                "ce_sid":    ce.get("security_id"),
                "pe_sid":    pe.get("security_id"),
            })
        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        return df

    def _fetch_vix(self) -> float:
        vix_id, vix_seg = self.master.vix_info()
        try:
            resp = self.dhan.ltp({vix_seg: [vix_id]})
            data = resp.get("data", {})
            if isinstance(data, dict):
                return float(data.get(vix_seg, {}).get(str(vix_id), {}).get("last_price", 0))
        except Exception:
            pass
        return 15.0  # fallback if VIX unavailable

    @staticmethod
    def _ema(closes: list[float], period: int) -> float:
        if len(closes) < period:
            return closes[-1] if closes else 0.0
        k = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for c in closes[period:]:
            ema = c * k + ema * (1 - k)
        return ema

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
    def _hist_vol(closes: list[float], period: int = 30) -> float:
        """Annualised historical volatility from daily log returns."""
        if len(closes) < period + 1:
            return 0.20
        returns = [mlog(closes[i] / closes[i - 1]) for i in range(-period, 0)]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return sqrt(variance * 252)

    @staticmethod
    def _estimate_fii_trend(bars: pd.DataFrame, sector: str) -> str:
        """Approximate FII sector trend from stock price momentum (5-day vs 20-day)."""
        if len(bars) < 20:
            return "NEUTRAL"
        closes = bars["close"].tolist()
        ret_5 = (closes[-1] - closes[-5]) / closes[-5] * 100
        ret_20 = (closes[-1] - closes[-20]) / closes[-20] * 100
        if ret_5 > 1.5 and ret_20 > 3.0:
            return "POSITIVE"
        if ret_5 < -1.5 and ret_20 < -3.0:
            return "NEGATIVE"
        return "NEUTRAL"

    # ── Black-Scholes for paper/backtest ──────────────────────────────────────

    @staticmethod
    def bs_price(spot: float, strike: float, dte: int, vol: float,
                  option_type: str = "CE", r: float = 0.07) -> float:
        """Black-Scholes option price. option_type: CE or PE."""
        T = max(dte, 1) / 365
        if spot <= 0 or strike <= 0 or vol <= 0:
            return 0.05
        d1 = (mlog(spot / strike) + (r + vol ** 2 / 2) * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
        if option_type.upper() == "CE":
            return max(spot * N(d1) - strike * exp(-r * T) * N(d2), 0.05)
        return max(strike * exp(-r * T) * N(-d2) - spot * N(-d1), 0.05)
