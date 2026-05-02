"""
telegram_bot.py
Telegram alerts and remote commands for MyTradingBot-SO.

Bot commands:
  /start /status  — mode, market status, open positions, VIX
  /positions       — all open swing positions
  /pnl             — all-time closed P&L summary
  /squareoff       — emergency close all positions
  /pass            — trigger one strategy evaluation cycle
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime
from typing import Optional, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from core.orchestrator import Orchestrator

log = logging.getLogger(__name__)
_API = "https://api.telegram.org/bot{token}/{method}"


def get_chat_id():
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / "config" / "secrets.env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in config/secrets.env first.")
        return
    r = requests.get(_API.format(token=token, method="getUpdates"), timeout=10)
    updates = r.json().get("result", [])
    if not updates:
        print("No messages found. Send any message to your bot then re-run.")
        return
    for u in updates:
        chat = u.get("message", {}).get("chat", {})
        print(f"chat_id={chat.get('id')}  name={chat.get('first_name','')}")


class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = str(chat_id)
        self._orch: Optional["Orchestrator"] = None
        self._running = False
        self._offset = 0
        self._thread: Optional[threading.Thread] = None

    def start(self, orch: "Orchestrator"):
        self._orch = orch
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="tg-poll")
        self._thread.start()
        enabled = [i["symbol"] for i in orch.cfg["instruments"] if i.get("enabled", True)]
        self.send(
            f"<b>MyTradingBot-SO started</b>\n"
            f"Mode: {orch.mode}\n"
            f"Stocks: {', '.join(enabled[:10])}{'...' if len(enabled) > 10 else ''}\n"
            f"Commands: /status /positions /pnl /pass /squareoff"
        )

    def stop(self):
        self._running = False
        self.send("MyTradingBot-SO stopped.")

    def send(self, text: str):
        try:
            requests.post(
                _API.format(token=self._token, method="sendMessage"),
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            log.warning("Telegram send failed: %s", e)

    def notify_entry(self, trade: dict):
        fill = trade.get("fill_price", 0)
        sl = trade.get("sl_price", 0)
        tgt = trade.get("target_price", 0)
        qty = int(trade.get("lots", 1)) * int(trade.get("lot_size", 1))
        try:
            expiry_date = date.fromisoformat(trade["expiry"])
            dte = (expiry_date - date.today()).days
        except Exception:
            dte = 0
        self.send(
            f"<b>ENTRY [{trade['mode']}]</b>\n"
            f"{trade['symbol']} | {trade['strategy']}\n"
            f"Direction: {trade['direction']} | Strike: {trade['strike']:.0f} {trade['option_type']}\n"
            f"Expiry: {trade['expiry']} ({dte} DTE)\n"
            f"Fill: {fill:.2f} | Qty: {qty}\n"
            f"SL: {sl:.2f} | Target: {tgt:.2f}\n"
            f"<i>{trade.get('rationale','')[:120]}</i>"
        )

    def notify_exit(self, trade_id: str, symbol: str, strategy: str,
                     direction: str, exit_price: float, pnl: float, reason: str):
        sign = "+" if pnl >= 0 else ""
        self.send(
            f"<b>EXIT — {reason}</b>\n"
            f"{symbol} | {strategy} | {direction}\n"
            f"Exit: {exit_price:.2f} | P&L: {sign}₹{pnl:,.0f}"
        )

    def notify_vix_spike(self, vix_now: float, vix_open: float):
        pct = (vix_now - vix_open) / vix_open * 100
        self.send(
            f"<b>VIX SPIKE ALERT</b>\n"
            f"VIX {vix_open:.1f} → {vix_now:.1f} (+{pct:.1f}%)\n"
            f"Emergency squareoff triggered."
        )

    def _poll_loop(self):
        while self._running:
            try:
                r = requests.get(
                    _API.format(token=self._token, method="getUpdates"),
                    params={"offset": self._offset, "timeout": 20},
                    timeout=30,
                )
                for update in r.json().get("result", []):
                    self._offset = update["update_id"] + 1
                    text = update.get("message", {}).get("text", "").strip()
                    if text:
                        self._dispatch(text)
            except Exception as e:
                log.warning("Telegram poll error: %s", e)
                time.sleep(5)

    def _dispatch(self, text: str):
        cmd = text.split()[0].lower().split("@")[0]
        handlers = {
            "/start": self._cmd_status, "/status": self._cmd_status,
            "/positions": self._cmd_positions, "/pnl": self._cmd_pnl,
            "/pass": self._cmd_pass, "/squareoff": self._cmd_squareoff,
        }
        fn = handlers.get(cmd)
        if fn:
            fn()
        else:
            self.send(f"Unknown: {cmd}\nTry /status /positions /pnl /pass /squareoff")

    def _cmd_status(self):
        if not self._orch:
            self.send("Orchestrator not ready.")
            return
        from data import store
        open_trades = store.get_open_trades(self._orch.mode)
        market = "OPEN" if self._orch._market_open() else "CLOSED"
        vix = 0.0
        try:
            vix = self._orch.md.fetch_vix()
        except Exception:
            pass
        enabled = [i["symbol"] for i in self._orch.cfg["instruments"] if i.get("enabled", True)]
        self.send(
            f"<b>Status</b>\n"
            f"Mode: {self._orch.mode} | Market: {market}\n"
            f"VIX: {vix:.1f}\n"
            f"Stocks: {len(enabled)} enabled\n"
            f"Open positions: {len(open_trades)}"
        )

    def _cmd_positions(self):
        if not self._orch:
            self.send("Orchestrator not ready.")
            return
        from data import store
        trades = store.get_open_trades(self._orch.mode)
        if not trades:
            self.send("No open positions.")
            return
        lines = ["<b>Open Positions</b>"]
        for t in trades:
            fill = float(t.get("fill_price") or 0)
            sl = float(t.get("sl_price") or 0)
            tgt = float(t.get("target_price") or 0)
            ltp = 0.0
            try:
                ltp = self._orch.md.fetch_option_ltp(int(t["security_id"]), t["symbol"])
            except Exception:
                pass
            qty = int(t["lots"]) * int(t["lot_size"])
            upnl = (ltp - fill) * qty if ltp > 0 else 0
            sign = "+" if upnl >= 0 else ""
            try:
                dte = (date.fromisoformat(t["expiry"]) - date.today()).days
            except Exception:
                dte = 0
            lines.append(
                f"\n{t['symbol']} {t['strategy']} {t['direction']}\n"
                f"  {t['strike']:.0f}{t['option_type']} | Exp:{t['expiry']} ({dte}DTE)\n"
                f"  Fill:{fill:.2f} LTP:{ltp:.2f} SL:{sl:.2f} T:{tgt:.2f}\n"
                f"  uP&L: {sign}₹{upnl:,.0f}"
            )
        self.send("\n".join(lines))

    def _cmd_pnl(self):
        if not self._orch:
            self.send("Orchestrator not ready.")
            return
        from data import store
        trades = store.get_trades(self._orch.mode)
        closed = [t for t in trades if t.get("ts_close")]
        if not closed:
            self.send("No closed trades.")
            return
        total = sum(float(t.get("realised_pnl") or 0) for t in closed)
        wins = [t for t in closed if float(t.get("realised_pnl") or 0) > 0]
        sign = "+" if total >= 0 else ""
        self.send(
            f"<b>All-Time P&L</b>\n"
            f"Trades: {len(closed)} | Wins: {len(wins)} | WR: {len(wins)/len(closed)*100:.0f}%\n"
            f"Total: {sign}₹{total:,.0f}"
        )

    def _cmd_pass(self):
        if not self._orch:
            self.send("Orchestrator not ready.")
            return
        self.send("Running evaluation pass…")
        try:
            result = self._orch.run_once()
            if result.get("market_status") == "CLOSED":
                self.send("Market is closed.")
                return
            proposals = result.get("proposals", [])
            self.send(f"Pass complete.\nSignals acted: {len(proposals)}")
        except Exception as e:
            self.send(f"Pass failed: {e}")

    def _cmd_squareoff(self):
        if not self._orch:
            self.send("Orchestrator not ready.")
            return
        self.send("<b>MANUAL SQUAREOFF</b>\nClosing all positions…")
        try:
            from data import store
            if not store.get_open_trades(self._orch.mode):
                self.send("No open positions.")
                return
            results = self._orch.monitor_agent.squareoff_all(reason="MANUAL_TELEGRAM")
            self.send(f"Squared off {len(results)} position(s).")
        except Exception as e:
            self.send(f"Squareoff failed: {e}")


def build_from_env() -> Optional[TelegramBot]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.info("Telegram not configured")
        return None
    return TelegramBot(token, chat_id)
