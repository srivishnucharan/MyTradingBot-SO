"""
sentiment_engine.py
Global macro sentiment engine for pre-trade filtering.

Fetches daily global macro signals and translates them into per-sector bias
and an overall market sentiment score used to filter or adjust trade decisions.

Data sources (via yfinance, free):
  - US Indices : S&P 500, NASDAQ Composite
  - Crude Oil  : WTI Crude (CL=F), Brent (BZ=F)
  - Currencies : USD Index (DXY), USD/INR spot
  - Fear gauge : US VIX
  - Safe-haven : Gold futures
  - Indian ADRs: INFY, HDB (HDFC Bank), IBN (ICICI Bank), WIT (Wipro)

Macro-to-sector rules (hard-coded domain knowledge):
  Crude HIGH   -> ENERGY+, AUTO-, CONSUMER-(paints/tyres/pipes)
  Crude LOW    -> ENERGY-, AUTO+, CONSUMER+
  US down 1%+  -> risk-off, BANKING/IT headwind
  ADRs up 1.5% -> IT+, BANKING+ (pre-market signal for Indian stocks)
  DXY up 0.4%  -> EM outflows (score-), IT revenue boost (IT+)
  USD/INR up   -> rupee weakening, IT exporter boost, import costs rise
  US VIX > 25  -> global fear, broad caution
  Gold up 1.2% -> safe-haven demand, risk-off signal
  Bear market  -> defensive rotation: PHARMA+
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

log = logging.getLogger(__name__)

# ── Macro ticker map ───────────────────────────────────────────────────────────

MACRO_TICKERS: dict[str, str] = {
    "sp500":    "^GSPC",      # S&P 500
    "nasdaq":   "^IXIC",      # NASDAQ Composite
    "crude":    "CL=F",       # WTI Crude Oil (USD/bbl)
    "dxy":      "DX-Y.NYB",   # US Dollar Index
    "usdinr":   "USDINR=X",   # USD/INR spot rate
    "us_vix":   "^VIX",       # CBOE VIX (fear gauge)
    "gold":     "GC=F",       # Gold futures
    "infy_adr": "INFY",       # Infosys ADR (IT sector proxy)
    "hdb_adr":  "HDB",        # HDFC Bank ADR (banking proxy)
    "ibn_adr":  "IBN",        # ICICI Bank ADR (banking proxy)
    "wit_adr":  "WIT",        # Wipro ADR (IT sector)
}

# ── Thresholds ────────────────────────────────────────────────────────────────

CRUDE_HIGH_LEVEL  = 88.0    # USD/bbl — structurally high crude
CRUDE_LOW_LEVEL   = 65.0    # USD/bbl — structurally low crude
CRUDE_MOVE_PCT    = 2.5     # Daily % move that counts as a meaningful signal

SP500_MOVE_PCT    = 1.0     # % threshold for US index signal
NASDAQ_MOVE_PCT   = 1.5
ADR_MOVE_PCT      = 1.5     # % avg move across Indian ADRs
DXY_MOVE_PCT      = 0.4     # % move in dollar index
USDINR_MOVE_PCT   = 0.5     # % move in USD/INR
GOLD_MOVE_PCT     = 1.2     # % move in gold
US_VIX_HIGH       = 25.0
US_VIX_ELEVATED   = 20.0
US_VIX_BENIGN     = 16.0

ALL_SECTORS = [
    "ENERGY", "BANKING", "IT", "AUTO", "PHARMA",
    "CONSUMER", "TELECOM", "FINANCE", "INFRASTRUCTURE",
]


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class MacroSentiment:
    """
    Output of SentimentEngine.compute() for one trading day.

    overall_score : -100 (extreme bear) to +100 (extreme bull).
    sector_bias   : per-sector string — "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    call_blocked_sectors : sectors where macro is a headwind for CALL buyers
    put_blocked_sectors  : sectors where macro is a tailwind (headwind for PUT buyers)
    signals       : human-readable list of active macro signals (for Telegram/log)
    """
    as_of: date

    # Raw readings stored for dashboard / debugging
    sp500_chg:   float = 0.0
    nasdaq_chg:  float = 0.0
    crude_level: float = 0.0
    crude_chg:   float = 0.0
    dxy_chg:     float = 0.0
    usdinr:      float = 0.0
    usdinr_chg:  float = 0.0
    us_vix:      float = 16.0
    gold_chg:    float = 0.0
    adr_chg:     float = 0.0     # avg of Indian ADRs

    overall_score: int = 0

    sector_bias:           dict = field(default_factory=dict)
    call_blocked_sectors:  set  = field(default_factory=set)
    put_blocked_sectors:   set  = field(default_factory=set)
    signals:               list = field(default_factory=list)

    # ── convenience predicates ────────────────────────────────────────────────

    def is_bearish(self) -> bool:
        return self.overall_score < -20

    def is_bullish(self) -> bool:
        return self.overall_score > 20

    def allows_call(self, sector: str) -> bool:
        """Return False if macro is a headwind for CALL in this sector."""
        return sector not in self.call_blocked_sectors

    def allows_put(self, sector: str) -> bool:
        """Return False if macro is a tailwind (headwind for PUT) in this sector."""
        return sector not in self.put_blocked_sectors

    def summary(self) -> str:
        lines = [f"MacroSentiment {self.as_of} | Score={self.overall_score:+d}"]
        if self.signals:
            lines.append("  " + " | ".join(self.signals))
        biased = {s: v for s, v in self.sector_bias.items() if v != "NEUTRAL"}
        if biased:
            lines.append("  Sector: " + ", ".join(f"{s}={v}" for s, v in sorted(biased.items())))
        return "\n".join(lines)


# ── Engine ─────────────────────────────────────────────────────────────────────

class SentimentEngine:
    """
    Two-phase usage:
      Phase 1 (startup) : bars = SentimentEngine.fetch_macro_bars(start, end)
      Phase 2 (per-day) : sentiment = SentimentEngine.compute(date, bars)

    For live trading, use fetch_live() which combines both phases.
    """

    @staticmethod
    def fetch_macro_bars(from_date: date, to_date: date) -> dict[str, pd.DataFrame]:
        """
        Download historical daily close data for all macro tickers.
        Returns a dict keyed by the friendly names in MACRO_TICKERS.
        Silently skips any ticker that fails — compute() handles missing data gracefully.
        """
        if not _YF_AVAILABLE:
            log.warning("yfinance unavailable — macro sentiment will use neutral defaults")
            return {}

        result: dict[str, pd.DataFrame] = {}
        # Fetch a few extra days before from_date to ensure we have a prev-day close
        fetch_from = from_date - timedelta(days=10)
        fetch_to   = to_date + timedelta(days=1)

        for key, sym in MACRO_TICKERS.items():
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(
                    start=fetch_from.isoformat(),
                    end=fetch_to.isoformat(),
                    auto_adjust=True,
                )
                if hist.empty:
                    log.debug("No macro data for %s (%s)", key, sym)
                    continue
                hist = hist.reset_index()
                hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
                hist = hist.rename(columns={"Close": "close"})
                hist = hist[["date", "close"]].sort_values("date").reset_index(drop=True)
                result[key] = hist
            except Exception as e:
                log.debug("Macro fetch error %s (%s): %s", key, sym, e)

        log.info("Macro bars loaded: %d/%d tickers", len(result), len(MACRO_TICKERS))
        return result

    @staticmethod
    def fetch_live(lookback_days: int = 10) -> "MacroSentiment":
        """Fetch recent macro data and return today's sentiment. Convenience for live mode."""
        today = date.today()
        bars = SentimentEngine.fetch_macro_bars(today - timedelta(days=lookback_days), today)
        return SentimentEngine.compute(today, bars)

    @staticmethod
    def compute(as_of: date, macro_bars: dict[str, pd.DataFrame]) -> "MacroSentiment":
        """
        Compute MacroSentiment for as_of from pre-loaded historical bars.
        Returns neutral defaults when data is unavailable — never raises.
        """
        s = MacroSentiment(as_of=as_of)

        if not macro_bars:
            return s

        ts = pd.Timestamp(as_of)

        def _chg(key: str) -> float:
            """1-day % change for key, as-of as_of. Returns 0 if data missing."""
            bars = macro_bars.get(key)
            if bars is None or len(bars) < 2:
                return 0.0
            past = bars[bars["date"] <= ts]
            if len(past) < 2:
                return 0.0
            prev = float(past.iloc[-2]["close"])
            curr = float(past.iloc[-1]["close"])
            return (curr - prev) / prev * 100 if prev != 0 else 0.0

        def _level(key: str) -> float:
            bars = macro_bars.get(key)
            if bars is None or bars.empty:
                return 0.0
            past = bars[bars["date"] <= ts]
            return float(past.iloc[-1]["close"]) if not past.empty else 0.0

        # ── Raw readings ───────────────────────────────────────────────────────

        s.sp500_chg   = _chg("sp500")
        s.nasdaq_chg  = _chg("nasdaq")
        s.crude_level = _level("crude")
        s.crude_chg   = _chg("crude")
        s.dxy_chg     = _chg("dxy")
        s.usdinr      = _level("usdinr")
        s.usdinr_chg  = _chg("usdinr")
        s.us_vix      = _level("us_vix") or 16.0
        s.gold_chg    = _chg("gold")

        adr_keys = [k for k in ("infy_adr", "hdb_adr", "ibn_adr", "wit_adr") if k in macro_bars]
        adr_changes = [_chg(k) for k in adr_keys]
        s.adr_chg = sum(adr_changes) / len(adr_changes) if adr_changes else 0.0

        # ── Rules engine ───────────────────────────────────────────────────────

        bias = {sec: "NEUTRAL" for sec in ALL_SECTORS}
        score = 0
        signals: list[str] = []

        # ── 1. US S&P 500 ──────────────────────────────────────────────────────
        sp = s.sp500_chg
        if sp < -SP500_MOVE_PCT:
            score -= 20
            # CALL headwind on rate-sensitive sectors when US is down
            bias["BANKING"] = "NEGATIVE"
            signals.append(f"SP500 {sp:+.1f}% risk-off")
        elif sp > SP500_MOVE_PCT:
            score += 15
            signals.append(f"SP500 {sp:+.1f}% positive global cues")

        # ── 2. NASDAQ (IT proxy) ───────────────────────────────────────────────
        nq = s.nasdaq_chg
        if nq < -NASDAQ_MOVE_PCT:
            score -= 10
            bias["IT"] = "NEGATIVE"
            signals.append(f"NASDAQ {nq:+.1f}% IT headwind")
        elif nq > NASDAQ_MOVE_PCT:
            score += 8
            if bias["IT"] != "NEGATIVE":
                bias["IT"] = "POSITIVE"
            signals.append(f"NASDAQ {nq:+.1f}% IT tailwind")

        # ── 3. Crude Oil ──────────────────────────────────────────────────────
        cr_lv  = s.crude_level
        cr_chg = s.crude_chg
        if cr_lv > CRUDE_HIGH_LEVEL or cr_chg > CRUDE_MOVE_PCT:
            bias["ENERGY"] = "POSITIVE"
            # High crude hurts: Auto (fuel/input cost), Consumer (paints-solvents,
            # tyres-rubber-crude, pipes-PVC), downstream oil (BPCL/HPCL/IOC—margins
            # compress when crude rises faster than retail prices)
            bias["AUTO"]           = "NEGATIVE"
            bias["CONSUMER"]       = "NEGATIVE"
            bias["INFRASTRUCTURE"] = "NEGATIVE"
            score -= 5  # net negative for broad market (input cost inflation)
            signals.append(f"Crude ${cr_lv:.0f}/bbl ({cr_chg:+.1f}%) -> Energy+, Auto/Consumer-")
        elif cr_lv < CRUDE_LOW_LEVEL or cr_chg < -CRUDE_MOVE_PCT:
            bias["ENERGY"]   = "NEGATIVE"
            bias["AUTO"]     = "POSITIVE"
            bias["CONSUMER"] = "POSITIVE"
            score += 8  # input cost relief, margins expand
            signals.append(f"Crude ${cr_lv:.0f}/bbl ({cr_chg:+.1f}%) -> Auto/Consumer+, Energy-")

        # ── 4. Indian ADRs ────────────────────────────────────────────────────
        adr = s.adr_chg
        if adr > ADR_MOVE_PCT:
            if bias["IT"] != "NEGATIVE":
                bias["IT"] = "POSITIVE"
            if bias["BANKING"] != "NEGATIVE":
                bias["BANKING"] = "POSITIVE"
            score += 10
            signals.append(f"ADRs {adr:+.1f}% -> IT/Banking gap-up signal")
        elif adr < -ADR_MOVE_PCT:
            bias["IT"]      = "NEGATIVE"
            bias["BANKING"] = "NEGATIVE"
            score -= 10
            signals.append(f"ADRs {adr:+.1f}% -> IT/Banking headwind")

        # ── 5. Dollar Index (DXY) ─────────────────────────────────────────────
        dxy = s.dxy_chg
        if dxy > DXY_MOVE_PCT:
            # Strong dollar -> EM capital outflows; but IT earns in USD
            score -= 8
            if bias["IT"] not in ("NEGATIVE", "POSITIVE"):
                bias["IT"] = "POSITIVE"
            signals.append(f"DXY {dxy:+.1f}% USD strong -> FII outflow risk, IT revenue+")
        elif dxy < -DXY_MOVE_PCT:
            score += 8
            signals.append(f"DXY {dxy:+.1f}% USD weak -> FII inflows favorable")

        # ── 6. USD/INR (Rupee) ────────────────────────────────────────────────
        inr = s.usdinr_chg
        if inr > USDINR_MOVE_PCT:      # Rupee weakening
            score -= 5
            if bias["IT"] not in ("NEGATIVE",):
                bias["IT"] = "POSITIVE"  # IT export revenue in USD
            signals.append(f"USD/INR {s.usdinr:.1f} ({inr:+.1f}%) rupee weak -> imports costly, IT+")
        elif inr < -USDINR_MOVE_PCT:   # Rupee strengthening
            score += 5
            signals.append(f"USD/INR {s.usdinr:.1f} ({inr:+.1f}%) rupee strong -> import relief")

        # ── 7. US VIX (global fear gauge) ────────────────────────────────────
        vix = s.us_vix
        if vix > US_VIX_HIGH:
            score -= 25
            signals.append(f"US VIX {vix:.1f} extreme fear -> broad caution")
        elif vix > US_VIX_ELEVATED:
            score -= 12
            signals.append(f"US VIX {vix:.1f} elevated fear")
        elif vix < US_VIX_BENIGN:
            score += 5
            signals.append(f"US VIX {vix:.1f} benign -> risk-on environment")

        # ── 8. Gold (safe-haven demand) ───────────────────────────────────────
        gld = s.gold_chg
        if gld > GOLD_MOVE_PCT:
            score -= 8
            signals.append(f"Gold {gld:+.1f}% safe-haven buying -> risk-off")
        elif gld < -GOLD_MOVE_PCT:
            score += 5
            signals.append(f"Gold {gld:+.1f}% safe-haven selling -> risk appetite improving")

        # ── 9. Derived: defensive rotation in bear mode ───────────────────────
        if score < -20:
            bias["PHARMA"] = "POSITIVE"
            signals.append("Macro bearish -> defensive tilt, Pharma favored")

        # ── 10. Derived: cyclical boost in bull mode ──────────────────────────
        if score > 30:
            if bias["BANKING"] != "NEGATIVE":
                bias["BANKING"] = "POSITIVE"
            if bias["FINANCE"] != "NEGATIVE":
                bias["FINANCE"] = "POSITIVE"

        # ── 11. Extreme bearish: block banking calls ──────────────────────────
        if score < -30 and bias["BANKING"] != "POSITIVE":
            bias["BANKING"] = "NEGATIVE"

        # ── Finalise ──────────────────────────────────────────────────────────
        s.overall_score          = max(-100, min(100, score))
        s.sector_bias            = bias
        s.call_blocked_sectors   = {sec for sec, b in bias.items() if b == "NEGATIVE"}
        s.put_blocked_sectors    = {sec for sec, b in bias.items() if b == "POSITIVE"}
        s.signals                = signals

        return s
