"""
monitor_agent.py
Monitors all open swing trades each cycle.
Swing trades: held for 7-21 days, not intraday.
Exits: TARGET | STOPLOSS | EXPIRY_APPROACHING | EMERGENCY (VIX spike) | MANUAL
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dtime
from math import log as mlog, sqrt, exp, erf
from typing import Optional, TYPE_CHECKING

from data.dhan_client import DhanClient
from data.market_data import MarketData
from data.security_master import SecurityMaster
from data import store

if TYPE_CHECKING:
    from notifications.telegram_bot import TelegramBot

log = logging.getLogger(__name__)

EXPIRY_EXIT_DTE = 7      # Exit if ≤ 7 DTE remaining (avoid theta cliff)
VIX_SPIKE_PCT = 20.0     # Emergency exit if VIX spikes 20% from session open


class MonitorAgent:
    def __init__(self, dhan: Optional[DhanClient] = None,
                  market_data: Optional[MarketData] = None,
                  mode: str = "PAPER",
                  vix_spike_pct: float = VIX_SPIKE_PCT,
                  notifier: Optional["TelegramBot"] = None):
        self.dhan = dhan or DhanClient()
        self.md = market_data or MarketData(self.dhan)
        self.mode = mode
        self.vix_spike_pct = vix_spike_pct
        self.notifier = notifier
        self._vix_at_open: float = 0.0
        self.master = SecurityMaster()

    def set_vix_at_open(self, vix: float):
        self._vix_at_open = vix

    def check_all(self) -> list[dict]:
        trades = store.get_open_trades(self.mode)
        results = []
        for trade in trades:
            try:
                results.append(self._check_one(trade))
            except Exception as e:
                log.exception("Monitor failed for %s: %s", trade.get("trade_id"), e)
        return results

    def _check_one(self, trade: dict) -> dict:
        trade_id = trade["trade_id"]
        symbol = trade["symbol"]
        security_id = int(trade["security_id"])
        fill_price = float(trade.get("fill_price") or 0)
        sl_price = float(trade.get("sl_price") or fill_price * 0.70)
        target_price = float(trade.get("target_price") or fill_price * 2.50)
        qty = int(trade["lots"]) * int(trade["lot_size"])
        expiry = date.fromisoformat(trade["expiry"])
        dte = (expiry - date.today()).days

        # Exit if too close to expiry (theta crush)
        if dte <= EXPIRY_EXIT_DTE:
            ltp = self._get_option_ltp(security_id, symbol)
            pnl = (ltp - fill_price) * qty
            self._close(trade_id, ltp, pnl, "EXPIRY_EXIT")
            log.info("[%s] %s EXPIRY EXIT (%d DTE) @ %.2f | PnL=%.0f",
                     self.mode, symbol, dte, ltp, pnl)
            if self.notifier:
                self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                          trade["direction"], ltp, pnl, f"EXPIRY EXIT ({dte} DTE)")
            return {"trade_id": trade_id, "action": "EXIT_EXPIRY", "pnl": pnl}

        # VIX spike emergency
        vix_spiked, vix_now = self._vix_spike_detail()
        if vix_spiked:
            ltp = self._get_option_ltp(security_id, symbol)
            pnl = (ltp - fill_price) * qty
            self._close(trade_id, ltp, pnl, "VIX_EMERGENCY")
            log.warning("[%s] %s VIX spike emergency exit @ %.2f", self.mode, symbol, ltp)
            if self.notifier:
                self.notifier.notify_vix_spike(vix_now, self._vix_at_open)
                self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                          trade["direction"], ltp, pnl, "VIX SPIKE EMERGENCY")
            return {"trade_id": trade_id, "action": "EXIT_EMERGENCY", "pnl": pnl}

        # In LIVE mode, Super Order handles SL/target broker-side
        if self.mode == "LIVE":
            return {"trade_id": trade_id, "action": "HOLD", "dte": dte}

        # PAPER: simulate option LTP
        ltp = self._get_option_ltp(security_id, symbol)
        if ltp <= 0:
            ltp = self._bs_estimate(trade)
        if ltp <= 0:
            return {"trade_id": trade_id, "action": "HOLD"}

        pnl = (ltp - fill_price) * qty

        if ltp >= target_price:
            self._close(trade_id, ltp, pnl, "TARGET")
            log.info("[PAPER] %s %s TARGET @ %.2f | PnL=+%.0f", symbol, trade["strategy"], ltp, pnl)
            if self.notifier:
                self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                          trade["direction"], ltp, pnl, "TARGET HIT")
            return {"trade_id": trade_id, "action": "EXIT_TARGET", "pnl": pnl}

        if ltp <= sl_price:
            self._close(trade_id, ltp, pnl, "STOPLOSS")
            log.info("[PAPER] %s %s STOPLOSS @ %.2f | PnL=%.0f", symbol, trade["strategy"], ltp, pnl)
            if self.notifier:
                self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                          trade["direction"], ltp, pnl, "STOPLOSS HIT")
            return {"trade_id": trade_id, "action": "EXIT_STOPLOSS", "pnl": pnl}

        log.debug("[PAPER] %s HOLD | LTP=%.2f | PnL=%.0f", trade_id[:12], ltp, pnl)
        return {"trade_id": trade_id, "action": "HOLD", "ltp": ltp, "pnl": pnl}

    def squareoff_all(self, reason: str = "MANUAL") -> list[dict]:
        trades = store.get_open_trades(self.mode)
        results = []
        for trade in trades:
            try:
                security_id = int(trade["security_id"])
                fill_price = float(trade.get("fill_price") or 0)
                qty = int(trade["lots"]) * int(trade["lot_size"])
                ltp = self._get_option_ltp(security_id, trade["symbol"])
                pnl = (ltp - fill_price) * qty
                self._close(trade["trade_id"], ltp, pnl, reason)
                if self.notifier:
                    self.notifier.notify_exit(trade["trade_id"], trade["symbol"],
                                              trade["strategy"], trade["direction"],
                                              ltp, pnl, reason)
                results.append({"trade_id": trade["trade_id"], "pnl": pnl})
            except Exception as e:
                log.exception("squareoff_all failed for %s: %s", trade.get("trade_id"), e)
        return results

    def _close(self, trade_id: str, exit_price: float, pnl: float, reason: str):
        store.close_trade(trade_id, exit_price, pnl, reason)

    def _get_option_ltp(self, security_id: int, symbol: str) -> float:
        try:
            return self.md.fetch_option_ltp(security_id, symbol)
        except Exception:
            return 0.0

    def _bs_estimate(self, trade: dict) -> float:
        expiry = date.fromisoformat(trade["expiry"])
        dte = max((expiry - date.today()).days, 0)
        if dte == 0:
            return 0.05
        strike = float(trade["strike"])
        # Use fill_price as proxy for spot (crude approximation)
        spot_approx = strike * 1.01 if trade["option_type"] == "CE" else strike * 0.99
        vol = 0.30  # typical stock volatility
        T = dte / 365
        r = 0.07
        d1 = (mlog(spot_approx / strike) + (r + vol ** 2 / 2) * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
        opt = trade["option_type"]
        if opt == "CE":
            return max(spot_approx * N(d1) - strike * exp(-r * T) * N(d2), 0.05)
        return max(strike * exp(-r * T) * N(-d2) - spot_approx * N(-d1), 0.05)

    def _vix_spike_detail(self) -> tuple[bool, float]:
        if self._vix_at_open <= 0:
            return False, 0.0
        try:
            current_vix = self.md.fetch_vix()
        except Exception:
            return False, 0.0
        if current_vix <= 0:
            return False, 0.0
        change_pct = (current_vix - self._vix_at_open) / self._vix_at_open * 100
        return change_pct > self.vix_spike_pct, current_vix
