"""
risk_agent.py
Entry gating and position sizing for Nifty50 stock options swing trading.
Gates:
  1. Entry time window
  2. VIX range (hard ceiling 18)
  3. Max concurrent open positions (5)
  4. Lot sizing: keep premium risk <= 3% of capital per trade
  5. Confluence minimum (3 of 5 factors)
  6. Macro sentiment filter (blocks counter-macro trades)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Optional

from strategies.base import StockSignal

log = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    sized_lots: int = 0
    sl_price: float = 0.0
    target_price: float = 0.0


class RiskAgent:
    VIX_MAX = 18.0
    VIX_MIN = 10.0
    MAX_POSITIONS = 5
    MAX_RISK_PCT = 2.0
    DEFAULT_CAPITAL = 500_000
    ENTRY_START = dtime(9, 45)
    ENTRY_END = dtime(14, 0)
    CONFLUENCE_MIN = 3
    SL_PCT = 0.30
    TARGET_PCT = 1.50    # 2.5x premium — swing trades have higher R:R

    def __init__(self, config: Optional[dict] = None, vix_at_open: float = 0.0):
        cfg = config or {}
        risk = cfg.get("risk", {})
        mon = cfg.get("monitoring", {})

        self.capital = risk.get("capital", self.DEFAULT_CAPITAL)
        self.max_risk_pct = risk.get("max_risk_per_trade_pct", self.MAX_RISK_PCT)
        self.max_positions = risk.get("max_concurrent_positions", self.MAX_POSITIONS)
        self.vix_max = risk.get("vix_max", self.VIX_MAX)
        self.vix_min = risk.get("vix_min", self.VIX_MIN)
        self.sl_pct = risk.get("sl_pct", self.SL_PCT)
        self.target_pct = risk.get("target_pct", self.TARGET_PCT)
        self.confluence_min = risk.get("confluence_min", self.CONFLUENCE_MIN)

        entry_start = mon.get("entry_window_start", "09:45")
        entry_end = mon.get("entry_window_end", "14:00")
        self.entry_start = self._parse_time(entry_start)
        self.entry_end = self._parse_time(entry_end)
        self.vix_at_open = vix_at_open

    def evaluate(self, signal: StockSignal, vix: float,
                  open_positions: int, lot_size: int,
                  now: Optional[dtime] = None,
                  sector: str = "",
                  macro: Optional[object] = None) -> RiskDecision:
        """
        Gate and size a signal.

        sector : the stock's sector string (e.g. "BANKING", "IT") — used for macro filter.
        macro  : MacroSentiment from SentimentEngine — if None, macro filter is skipped.
        """
        now = now or datetime.now().time()

        # 1. Time window
        if now < self.entry_start:
            return RiskDecision(False, f"Pre-entry window ({self.entry_start})")
        if now > self.entry_end:
            return RiskDecision(False, f"Past entry window ({self.entry_end})")

        # 2. VIX range
        if vix > self.vix_max:
            return RiskDecision(False, f"VIX {vix:.1f} > max {self.vix_max}")
        if vix < self.vix_min:
            log.warning("VIX %.1f < min %.1f — premium cheap", vix, self.vix_min)

        # 3. Max positions
        if open_positions >= self.max_positions:
            return RiskDecision(False, f"Max {self.max_positions} positions reached")

        # 4. Macro sentiment filter
        if macro is not None and sector:
            if signal.direction == "CALL" and not macro.allows_call(sector):
                return RiskDecision(
                    False,
                    f"Macro veto: {sector} has NEGATIVE bias (score={macro.overall_score:+d}) — avoid CALL",
                )
            if signal.direction == "PUT" and not macro.allows_put(sector):
                return RiskDecision(
                    False,
                    f"Macro veto: {sector} has POSITIVE bias (score={macro.overall_score:+d}) — avoid PUT",
                )
            # Require stronger confluence when macro is broadly bearish (calls need more proof)
            effective_confluence_min = self.confluence_min
            if macro.is_bearish() and signal.direction == "CALL":
                effective_confluence_min = self.confluence_min + 1
        else:
            effective_confluence_min = self.confluence_min

        # 5. Confluence minimum
        if signal.confluence_score < effective_confluence_min:
            return RiskDecision(
                False,
                f"Confluence {signal.confluence_score}/5 < min {effective_confluence_min}")

        # 6. Position sizing (risk <= max_risk_pct% of capital on premium)
        premium = signal.expected_premium
        if premium <= 0:
            return RiskDecision(False, "Expected premium is 0 — option data missing")

        max_risk_inr = self.capital * self.max_risk_pct / 100
        max_lots = int(max_risk_inr / (premium * lot_size)) if (premium * lot_size) > 0 else 0

        if max_lots < 1:
            return RiskDecision(
                False,
                f"Risk cap: Rs{max_risk_inr:.0f} / (Rs{premium:.2f}x{lot_size}) = 0 lots",
            )

        sized_lots = min(max_lots, signal.lots if signal.lots > 0 else max_lots)
        sl_price = round(premium * (1 - self.sl_pct), 2)
        target_price = round(premium * (1 + self.target_pct), 2)

        macro_note = f" | Macro={macro.overall_score:+d}" if macro is not None else ""
        return RiskDecision(
            approved=True,
            reason=f"Approved{macro_note}",
            sized_lots=sized_lots,
            sl_price=sl_price,
            target_price=target_price,
        )

    @staticmethod
    def _parse_time(s: str) -> dtime:
        h, m = s.split(":")
        return dtime(int(h), int(m))
