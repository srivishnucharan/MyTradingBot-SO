"""
orchestrator.py
Main agent loop for MyTradingBot-SO (Nifty50 Stock Options).
Modes: BACKTEST | PAPER | LIVE
Cycle (every hour during market hours for PAPER/LIVE):
  1. Monitor open positions (SL / target / expiry / VIX spike)
  2. For each enabled stock:
     a. Fetch StockMarketContext (daily bars, EMA, RSI, option chain, VIX)
     b. Select next monthly expiry with 20-35 DTE
     c. If no open position for that stock: evaluate all strategies
     d. If signal passes RiskAgent: execute via ExecutionAgent
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

_RESTART_FLAG = Path("logs/restart.flag")

import yaml

from data.dhan_client import DhanClient
from data.market_data import MarketData
from data.security_master import SecurityMaster
from data.sentiment_engine import SentimentEngine, MacroSentiment
from data.features import build_features
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
        self.eod_exit_time = self._parse_time(mon.get("eod_exit_time", "15:15"))
        self._vix_at_open: float = 0.0
        self._vix_set = False
        self._vix_date: Optional[date] = None
        self._macro: Optional[MacroSentiment] = None
        self._macro_date: Optional[date] = None
        self._eod_exited_date: Optional[date] = None

    # ── public ─────────────────────────────────────────────────────────────────

    def run_forever(self):
        log.info("Orchestrator starting — mode=%s", self.mode)
        Path("logs").mkdir(exist_ok=True)
        Path("logs/bot.pid").write_text(str(os.getpid()))
        if self.notifier:
            self.notifier.start(self)
        try:
            self._run_loop()
        finally:
            if self.notifier:
                self.notifier.stop()
            Path("logs/bot.pid").unlink(missing_ok=True)

    def _run_loop(self):
        while True:
            try:
                if _RESTART_FLAG.exists():
                    _RESTART_FLAG.unlink()
                    log.info("Restart flag detected — reloading Dhan token")
                    self.dhan.reload_token()

                if not self._market_open():
                    log.debug("Market closed — sleeping 60s")
                    time.sleep(60)
                    continue

                self._capture_vix()
                self._capture_macro()
                self.monitor_agent.check_all()
                self.monitor_agent.check_shadows()

                # EOD exit: squareoff all open positions by 15:15 IST
                now_ist = datetime.now(self._IST)
                today = now_ist.date()
                if now_ist.time() >= self.eod_exit_time and self._eod_exited_date != today:
                    self._eod_exited_date = today
                    results = self.monitor_agent.squareoff_all("EOD_CLOSE")
                    log.info("EOD squareoff at %s — closed %d positions", self.eod_exit_time, len(results))
                    time.sleep(60)
                    continue

                for instr in self.cfg["instruments"]:
                    if not instr.get("enabled", True):
                        continue
                    self._try_entry(instr)

                time.sleep(self._sleep_seconds())

            except KeyboardInterrupt:
                log.info("Shutting down on user interrupt")
                break
            except Exception as e:
                log.exception("Loop error: %s", e)
                time.sleep(self._sleep_seconds())

    def run_once(self) -> dict:
        if not self._market_open():
            return {"market_status": "CLOSED"}
        self._capture_vix()
        self._capture_macro()
        self.monitor_agent.check_all()
        self.monitor_agent.check_shadows()
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

        signals = self.signal_agent.evaluate_all(ctx)
        if not signals:
            return None

        # Highest-priority signal is the candidate; the rest become shadow
        # trades so we still learn what the discarded strategies would have done
        signal = signals[0]
        features = build_features(ctx, signal, self._macro)
        for other in signals[1:]:
            other_feats = build_features(ctx, other, self._macro)
            store.log_signal(
                symbol=other.symbol, strategy=other.strategy,
                direction=other.direction, confidence=other.confidence,
                rationale=other.rationale, acted=False,
                reject_reason="LOWER_PRIORITY", mode=self.mode,
                features=other_feats,
            )
            self._open_shadow(other, "LOWER_PRIORITY", other_feats)

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
            self.execution_agent.log_rejected_signal(signal, decision.reason, features)
            self._open_shadow(signal, f"REJECTED: {decision.reason}", features)
            return None

        return self.execution_agent.execute(signal, decision, instr["lot_size"], features)

    def _open_shadow(self, signal, reason: str, features: str):
        """Record an untaken signal as a counterfactual to track forward."""
        if not signal.security_id or signal.expected_premium <= 0:
            return
        if store.has_open_shadow(signal.symbol, signal.strategy,
                                 signal.direction, self.mode):
            return
        risk = self.cfg.get("risk", {})
        sl_pct = risk.get("sl_pct", 0.30)
        target_pct = risk.get("target_pct", 1.50)
        premium = signal.expected_premium
        store.open_shadow({
            "symbol": signal.symbol,
            "strategy": signal.strategy,
            "direction": signal.direction,
            "strike": signal.strike,
            "option_type": signal.option_type,
            "expiry": signal.expiry.isoformat(),
            "security_id": signal.security_id,
            "entry_price": premium,
            "sl_price": round(premium * (1 - sl_pct), 2),
            "target_price": round(premium * (1 + target_pct), 2),
            "reason_not_taken": reason,
            "mode": self.mode,
            "features": features,
        })

    def _select_expiry(self, instr: dict) -> Optional[date]:
        """Select expiry from Dhan's live list, targeting 20-35 DTE.
        Falls back to the nearest valid expiry >= min_dte if nothing fits exactly."""
        symbol = instr["symbol"]
        min_dte = self.cfg["risk"].get("min_dte", 20)
        max_dte = self.cfg["risk"].get("max_dte", 35)

        try:
            sec_id, seg = self.master.equity_info(symbol)
            expiries = sorted(self.dhan.expiry_list(sec_id, seg))
            today = date.today()
            # Prefer expiry strictly in window
            for e in expiries:
                dte = (e - today).days
                if min_dte <= dte <= max_dte:
                    return e
            # Accept nearest expiry >= min_dte (handles when expiry falls just outside window)
            candidates = [e for e in expiries if (e - today).days >= min_dte]
            if candidates:
                target = (min_dte + max_dte) // 2
                return min(candidates, key=lambda e: abs((e - today).days - target))
        except Exception:
            pass

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

    def _sleep_seconds(self) -> float:
        """Cap the refresh sleep so the loop wakes inside the EOD exit window
        (15:15-15:30) instead of sleeping past market close and skipping the
        squareoff."""
        now = datetime.now(self._IST)
        if now.time() < self.eod_exit_time:
            eod = datetime.combine(now.date(), self.eod_exit_time, tzinfo=self._IST)
            until_eod = (eod - now).total_seconds() + 5
            return min(self.refresh_sec, max(60.0, until_eod))
        return self.refresh_sec

    def _capture_vix(self):
        # Reset baseline each trading day so spike detection compares
        # against today's open, not the first day the process started
        today = datetime.now(self._IST).date()
        if self._vix_date != today:
            self._vix_set = False
            self._vix_date = today
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

    _IST = timezone(timedelta(hours=5, minutes=30))

    def _market_open(self) -> bool:
        now = datetime.now(self._IST)
        if now.weekday() >= 5:
            return False
        return self.market_open <= now.time() <= self.market_close

    @staticmethod
    def _parse_time(s: str) -> dtime:
        h, m = s.split(":")
        return dtime(int(h), int(m))
