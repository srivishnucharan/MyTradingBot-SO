"""
monitor_agent.py
Monitors all open swing trades each cycle.
Swing trades: held for 7-21 days, not intraday.
Exits: TARGET | STOPLOSS | EXPIRY_APPROACHING | EMERGENCY (VIX spike) | MANUAL
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dtime, timedelta, timezone
from math import log as mlog, sqrt, exp, erf
from typing import Optional, TYPE_CHECKING

from data.dhan_client import DhanClient, FNO_PRODUCT_TYPE
from data.market_data import MarketData
from data.security_master import SecurityMaster
from data import store

if TYPE_CHECKING:
    from notifications.telegram_bot import TelegramBot

log = logging.getLogger(__name__)

EXPIRY_EXIT_DTE = 7      # Exit if ≤ 7 DTE remaining (avoid theta cliff)
VIX_SPIKE_PCT = 20.0     # Emergency exit if VIX spikes 20% from session open
SHADOW_TIMEOUT_DAYS = 15  # Close counterfactuals after this many calendar days
_IST = timezone(timedelta(hours=5, minutes=30))
_EOD_TIME = dtime(15, 15)  # matches monitoring.eod_exit_time default


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

        # One LTP fetch per cycle: reused by all exit checks below and
        # recorded as peak/trough excursion (MFE/MAE source)
        ltp_now = self._get_option_ltp(security_id, symbol)
        if ltp_now > 0:
            store.update_trade_excursion(trade_id, ltp_now)

        # Exit if too close to expiry (theta crush)
        if dte <= EXPIRY_EXIT_DTE:
            ltp = ltp_now
            pnl = (ltp - fill_price) * qty
            self._exit_trade(trade, ltp, pnl, "EXPIRY_EXIT")
            log.info("[%s] %s EXPIRY EXIT (%d DTE) @ %.2f | PnL=%.0f",
                     self.mode, symbol, dte, ltp, pnl)
            if self.notifier:
                self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                          trade["direction"], ltp, pnl, f"EXPIRY EXIT ({dte} DTE)")
            return {"trade_id": trade_id, "action": "EXIT_EXPIRY", "pnl": pnl}

        # VIX spike emergency
        vix_spiked, vix_now = self._vix_spike_detail()
        if vix_spiked:
            ltp = ltp_now
            pnl = (ltp - fill_price) * qty
            self._exit_trade(trade, ltp, pnl, "VIX_EMERGENCY")
            log.warning("[%s] %s VIX spike emergency exit @ %.2f", self.mode, symbol, ltp)
            if self.notifier:
                self.notifier.notify_vix_spike(vix_now, self._vix_at_open)
                self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                          trade["direction"], ltp, pnl, "VIX SPIKE EMERGENCY")
            return {"trade_id": trade_id, "action": "EXIT_EMERGENCY", "pnl": pnl}

        # In LIVE mode, Super Order handles SL/target broker-side.
        # Reconcile: if the broker position is already flat (SL/target leg fired),
        # close the DB trade so the symbol slot is freed.
        if self.mode == "LIVE":
            net_qty = self._broker_net_qty(trade["security_id"])
            if net_qty == 0:
                ltp = ltp_now
                pnl = (ltp - fill_price) * qty  # approximation — actual fill is broker-side
                self._close(trade_id, ltp, pnl, "BROKER_CLOSED")
                log.info("[LIVE] %s broker position flat — reconciled DB close | PnL~%.0f",
                         symbol, pnl)
                if self.notifier:
                    self.notifier.notify_exit(trade_id, symbol, trade["strategy"],
                                              trade["direction"], ltp, pnl, "BROKER SL/TARGET")
                return {"trade_id": trade_id, "action": "EXIT_BROKER", "pnl": pnl}
            return {"trade_id": trade_id, "action": "HOLD", "dte": dte}

        # PAPER: simulate option LTP
        ltp = ltp_now
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
                if ltp > 0:
                    store.update_trade_excursion(trade["trade_id"], ltp)
                pnl = (ltp - fill_price) * qty
                self._exit_trade(trade, ltp, pnl, reason)
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

    def _exit_trade(self, trade: dict, ltp: float, pnl: float, reason: str):
        """Close a position. In LIVE mode the broker position is actually sold
        before the DB trade is closed; raises if the broker exit cannot be
        confirmed so the exit is retried next cycle instead of orphaning the
        position."""
        if self.mode == "LIVE":
            self._live_exit(trade)
        self._close(trade["trade_id"], ltp, pnl, reason)

    def _live_exit(self, trade: dict):
        # Cancel pending super-order legs so a leg can't fire after our exit
        order_id = str(trade.get("super_order_id") or "")
        if order_id:
            for leg in ("TARGET_LEG", "STOP_LOSS_LEG", "ENTRY_LEG"):
                try:
                    self.dhan.cancel_super_order(order_id, leg)
                except Exception as e:
                    log.debug("cancel_super_order %s %s: %s", order_id, leg, e)

        qty = int(trade["lots"]) * int(trade["lot_size"])
        net_qty = self._broker_net_qty(trade["security_id"])
        if net_qty is None:
            raise RuntimeError(
                f"Cannot verify broker position for {trade['symbol']} — exit deferred")
        sell_qty = min(qty, net_qty)
        if sell_qty <= 0:
            log.info("[LIVE] %s already flat at broker — closing DB only", trade["symbol"])
            return

        segment = self.master.option_segment(trade["symbol"])
        resp = self.dhan.place_order(
            security_id=str(trade["security_id"]),
            exchange_segment=segment,
            transaction_type="SELL",
            quantity=sell_qty,
            order_type="MARKET",
            product_type=FNO_PRODUCT_TYPE,
            price=0,
        )
        if resp.get("status") != "success":
            raise RuntimeError(f"LIVE exit SELL rejected for {trade['symbol']}: {resp}")
        log.info("[LIVE] SELL %d %s placed for exit", sell_qty, trade["symbol"])

    def _broker_net_qty(self, security_id) -> Optional[int]:
        """Net broker quantity for a security. None = unknown (API error or
        position not found in the book)."""
        try:
            resp = self.dhan.positions()
            data = resp.get("data") if isinstance(resp, dict) else None
            if not isinstance(data, list):
                return None
            for pos in data:
                sid = str(pos.get("securityId") or pos.get("security_id") or "")
                if sid == str(security_id):
                    return int(pos.get("netQty") or pos.get("net_qty") or 0)
        except Exception as e:
            log.warning("Broker positions fetch failed: %s", e)
        return None

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

    # ── shadow trades (counterfactuals) ───────────────────────────────────────

    def check_shadows(self) -> None:
        """Advance open shadow trades: update excursions, capture first-day EOD
        price, and close on target/SL/expiry/timeout using the same rules a
        real swing position would follow."""
        shadows = store.get_open_shadows(self.mode)
        if not shadows:
            return
        prices = self._batch_ltp(
            [int(s["security_id"]) for s in shadows if s["security_id"]])
        now = datetime.now(_IST)
        today = now.date()

        for s in shadows:
            try:
                ltp = prices.get(str(s["security_id"]), 0.0)
                if ltp <= 0:
                    continue
                peak = max(ltp, float(s["peak_ltp"] or ltp))
                trough = min(ltp, float(s["trough_ltp"] or ltp))
                open_date = date.fromisoformat(s["ts_open"][:10])
                first_eod = None
                if s["first_eod_price"] is None and (
                        today > open_date or now.time() >= _EOD_TIME):
                    first_eod = ltp
                store.update_shadow(s["id"], peak, trough, first_eod)

                target = float(s["target_price"] or 0)
                sl = float(s["sl_price"] or 0)
                dte = (date.fromisoformat(s["expiry"]) - today).days
                age = (today - open_date).days
                if target > 0 and ltp >= target:
                    store.close_shadow(s["id"], ltp, "TARGET")
                elif sl > 0 and ltp <= sl:
                    store.close_shadow(s["id"], ltp, "STOPLOSS")
                elif dte <= EXPIRY_EXIT_DTE:
                    store.close_shadow(s["id"], ltp, "EXPIRY_EXIT")
                elif age >= SHADOW_TIMEOUT_DAYS:
                    store.close_shadow(s["id"], ltp, "TIMEOUT")
            except Exception as e:
                log.warning("Shadow check failed for id=%s: %s", s.get("id"), e)

    def _batch_ltp(self, security_ids: list[int]) -> dict[str, float]:
        """One quote call for many FNO securities: {security_id: ltp}."""
        out: dict[str, float] = {}
        if not security_ids:
            return out
        try:
            resp = self.dhan.ltp({"NSE_FNO": security_ids})
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            for sid, row in (data.get("NSE_FNO", {}) or {}).items():
                try:
                    out[str(sid)] = float(row.get("last_price", 0))
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            log.warning("Shadow batch LTP failed: %s", e)
        return out

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
