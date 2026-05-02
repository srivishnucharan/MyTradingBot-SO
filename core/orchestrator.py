"""
orchestrator.py
Main agent loop for MyTradingBot-SO (Nifty50 Stock Options).
Modes: BACKTEST | PAPER | LIVE
Cycle (every hour during market hours for PAPER/LIVE):
  1. Monitor open positions (SL / target / expiry / VIX spike)
  2. For each enabled stock:
     a. Fetch StockMarketContext (daily bars, EMA, RSI, option chain, VIX)
     b. Select next monthly expiry with 20-35 DTE
     c. If no open position for that stock: evaluate all 5 strategies
     d. If signal passes RiskAgent: execute via ExecutionAgent
"""
from __future__ import annotations

import logging
import time
from calendar import monthrange
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from data.dhan_client import DhanClient
from data.market_data import MarketData
from data.security_master import SecurityMaster
from data.sentiment_engine import SentimentEngine, MacroSentiment
from data import store
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.monitor_agent import MonitorAgent
from notifications.telegram_bot import TelegramBot, build_from_env

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config_path: str = "config/config.yaml",
                  mode: Optional[str] = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.mode = mode or self.cfg["execution"]["mode"]
        self.cfg["execution"]["mode"] = self.mode

        store.init_db()

        self.dhan = DhanClient()
        self.md = MarketData(self.dhan)
        self.master = SecurityMaster()
        self.notifier: Optional[TelegramBot] = build_from_env()

        self.signal_agent = SignalAgent()
        self.execution_agent = ExecutionAgent(
            dhan=self.dhan,
            mode=self.mode,
            slippage_pct=self.cfg["execution"].get("slippage_buffer_pct", 0.5),
            notifier=self.notifier,
        )
        self.monitor_agent = MonitorAgent(
            dhan=self.dhan,
            market_data=self.md,
            mode=self.mode,
            notifier=self.notifier,
        )

        mon = self.cfg["monitoring"]
        self.refresh_sec = mon.get("refresh_interval_seconds", 3600)
        self.market_open = self._parse_time(mon.get("market_open_time", "09:15"))
        self.market_close = self._parse_time(mon.get("market_close_time", "15:30"))
        self._vix_at_open: float = 0.0
        self._vix_set = False
        self._macro: Optional[MacroSentiment] = None
        self._macro_date: Optional[date] = None

    # ── public ─────────────────────────────────────────────────────────────────

    def run_forever(self):
        log.info("Orchestrator starting — mode=%s", self.mode)
        if self.notifier:
            self.notifier.start(self)
        try:
            self._run_loop()
        finally:
            if self.notifier:
                self.notifier.stop()

    def _run_loop(self):
        while True:
            try:
                if not self._market_open():
                    log.debug("Market closed — sleeping 60s")
                    time.sleep(60)
                    continue

                self._capture_vix()
                self._capture_macro()
                self.monitor_agent.check_all()

                for instr in self.cfg["instruments"]:
                    if not instr.get("enabled", True):
                        continue
                    self._try_entry(instr)

                time.sleep(self.refresh_sec)

            except KeyboardInterrupt:
                log.info("Shutting down on user interrupt")
                break
            except Exception as e:
                log.exception("Loop error: %s", e)
                time.sleep(self.refresh_sec)

    def run_once(self) -> dict:
        if not self._market_open():
            return {"market_status": "CLOSED"}
        self._capture_vix()
        self._capture_macro()
        self.monitor_agent.check_all()
        results = []
        for instr in self.cfg["instruments"]:
            if not instr.get("enabled", True):
                continue
            r = self._try_entry(instr)
            if r:
                results.append(r)
        return {"proposals": results}

    # ── internals ──────────────────────────────────────────────────────────────

    def _try_entry(self, instr: dict) -> Optional[dict]:
        symbol = instr["symbol"]

        open_trades = store.get_open_trades(self.mode)
        # One position per symbol at a time
        if any(t["symbol"] == symbol for t in open_trades):
            return None

        expiry = self._select_expiry(instr)
        if expiry is None:
            return None

        # Compute next earnings date from config
        earnings_next = self._next_earnings(instr)

        ctx = self.md.build_context(instr, expiry, earnings_next)
        if ctx is None:
            return None

        signal = self.signal_agent.evaluate(ctx)
        if signal is None:
            return None

        risk = RiskAgent(config=self.cfg, vix_at_open=self._vix_at_open)
        decision = risk.evaluate(
            signal=signal,
            vix=ctx.vix,
            open_positions=len(open_trades),
            lot_size=instr["lot_size"],
            sector=ctx.sector,
            macro=self._macro,
        )

        if not decision.approved:
            log.info("[%s] %s rejected: %s", symbol, signal.strategy, decision.reason)
            self.execution_agent.log_rejected_signal(signal, decision.reason)
            return None

        return self.execution_agent.execute(signal, decision, instr["lot_size"])

    def _select_expiry(self, instr: dict) -> Optional[date]:
        """Select monthly expiry with 20-35 DTE. Stock options expire last Thursday of month."""
        symbol = instr["symbol"]
        min_dte = self.cfg["risk"].get("min_dte", 20)
        max_dte = self.cfg["risk"].get("max_dte", 35)

        try:
            sec_id, seg = self.master.equity_info(symbol)
            expiries = self.dhan.expiry_list(sec_id, seg)
            today = date.today()
            for e in sorted(expiries):
                dte = (e - today).days
                if min_dte <= dte <= max_dte:
                    return e
        except Exception:
            pass

        # Fallback: compute last Thursday of next 1-2 months
        return self._last_thursday_in_range(date.today(), min_dte, max_dte)

    @staticmethod
    def _last_thursday_in_range(today: date, min_dte: int, max_dte: int) -> Optional[date]:
        for months_ahead in range(1, 4):
            year = today.year
            month = today.month + months_ahead
            if month > 12:
                month -= 12
                year += 1
            last_day = monthrange(year, month)[1]
            # Find last Thursday
            candidate = date(year, month, last_day)
            while candidate.weekday() != 3:  # Thursday = 3
                candidate -= timedelta(days=1)
            dte = (candidate - today).days
            if min_dte <= dte <= max_dte:
                return candidate
        return None

    @staticmethod
    def _next_earnings(instr: dict) -> Optional[date]:
        """Estimate next earnings date from earnings_months in config."""
        months = instr.get("earnings_months", [])
        if not months:
            return None
        today = date.today()
        # Approximate: results typically announced in 2nd week of the month
        for ahead in range(0, 5):
            m = (today.month + ahead - 1) % 12 + 1
            y = today.year + (today.month + ahead - 1) // 12
            if m in months:
                candidate = date(y, m, 15)  # approx mid-month
                if candidate > today + timedelta(days=5):
                    return candidate
        return None

    def _capture_vix(self):
        if not self._vix_set:
            try:
                self._vix_at_open = self.md.fetch_vix()
                self.monitor_agent.set_vix_at_open(self._vix_at_open)
                self._vix_set = True
                log.info("VIX at open: %.1f", self._vix_at_open)
            except Exception as e:
                log.warning("VIX capture failed: %s", e)

    def _capture_macro(self):
        """Fetch global macro sentiment once per trading day."""
        today = date.today()
        if self._macro_date == today:
            return  # already fetched today
        try:
            self._macro = SentimentEngine.fetch_live()
            self._macro_date = today
            log.info("Macro sentiment: score=%+d | %s",
                     self._macro.overall_score, " | ".join(self._macro.signals[:3]))
            if self._macro.signals:
                log.info("Sector bias: %s", {
                    s: v for s, v in self._macro.sector_bias.items() if v != "NEUTRAL"
                })
        except Exception as e:
            log.warning("Macro sentiment fetch failed: %s — using neutral", e)
            self._macro = None

    def _market_open(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        return self.market_open <= now.time() <= self.market_close

    @staticmethod
    def _parse_time(s: str) -> dtime:
        h, m = s.split(":")
        return dtime(int(h), int(m))
