"""
execution_agent.py
Places stock option buys on Dhan.
PAPER — simulates fill at expected_premium + slippage, stores in DB.
LIVE  — places real Super Order; SL + target managed broker-side.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from data.dhan_client import DhanClient
from data.security_master import SecurityMaster
from data import store
from strategies.base import StockSignal
from agents.risk_agent import RiskDecision

if TYPE_CHECKING:
    from notifications.telegram_bot import TelegramBot

log = logging.getLogger(__name__)


class ExecutionAgent:
    def __init__(self, dhan: Optional[DhanClient] = None,
                  mode: str = "PAPER",
                  slippage_pct: float = 0.5,
                  notifier: Optional["TelegramBot"] = None):
        self.dhan = dhan or DhanClient()
        self.mode = mode
        self.slippage = slippage_pct / 100
        self.master = SecurityMaster()
        self.notifier = notifier

    def execute(self, signal: StockSignal, decision: RiskDecision, lot_size: int) -> dict:
        trade_id = f"T-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        qty = decision.sized_lots * lot_size

        trade = {
            "trade_id": trade_id,
            "ts_open": datetime.now().isoformat(),
            "ts_close": None,
            "symbol": signal.symbol,
            "strategy": signal.strategy,
            "direction": signal.direction,
            "strike": signal.strike,
            "option_type": signal.option_type,
            "expiry": signal.expiry.isoformat(),
            "security_id": str(signal.security_id),
            "tradingsymbol": signal.tradingsymbol,
            "lots": decision.sized_lots,
            "lot_size": lot_size,
            "sl_price": decision.sl_price,
            "target_price": decision.target_price,
            "mode": self.mode,
            "rationale": signal.rationale,
        }

        if self.mode == "PAPER":
            return self._paper_execute(trade, signal.expected_premium)
        return self._live_execute(trade, signal, decision, qty)

    def _paper_execute(self, trade: dict, expected_premium: float) -> dict:
        fill_price = round(expected_premium * (1 + self.slippage), 2)
        trade["fill_price"] = fill_price
        trade["sl_price"] = round(fill_price * 0.70, 2)          # -30%
        trade["target_price"] = round(fill_price * 2.50, 2)       # +150%
        store.save_trade(trade)
        store.log_signal(
            symbol=trade["symbol"], strategy=trade["strategy"],
            direction=trade["direction"], confidence="",
            rationale=trade.get("rationale", ""), acted=True,
            reject_reason="", mode="PAPER",
        )
        log.info("[PAPER] %s %s %s @ %.2f | SL=%.2f | T=%.2f | %dx%d lots",
                 trade["symbol"], trade["strategy"], trade["direction"],
                 fill_price, trade["sl_price"], trade["target_price"],
                 trade["lots"], trade["lot_size"])
        if self.notifier:
            self.notifier.notify_entry(trade)
        return {"status": "success", "trade_id": trade["trade_id"], "mode": "PAPER",
                "fill_price": fill_price}

    def _live_execute(self, trade: dict, signal: StockSignal,
                       decision: RiskDecision, qty: int) -> dict:
        segment = self.master.option_segment(signal.symbol)
        entry_price = signal.expected_premium * (1 + self.slippage)

        try:
            resp = self.dhan.place_super_order(
                security_id=str(signal.security_id),
                exchange_segment=segment,
                transaction_type="BUY",
                quantity=qty,
                order_type="MARKET",
                product_type="CNC",         # Delivery for swing trades
                price=0,
                target_price=decision.target_price,
                stoploss_price=decision.sl_price,
            )
        except Exception as e:
            log.error("Super order placement failed: %s", e)
            return {"status": "failed", "error": str(e)}

        if resp.get("status") != "success":
            log.error("Super order rejected: %s", resp)
            return {"status": "failed", "response": resp}

        order_id = resp.get("orderId") or resp.get("data", {}).get("orderId", "")
        trade["fill_price"] = entry_price
        trade["super_order_id"] = str(order_id)
        store.save_trade(trade)
        store.save_order(trade["trade_id"], str(order_id))
        store.log_signal(
            symbol=trade["symbol"], strategy=trade["strategy"],
            direction=trade["direction"], confidence="",
            rationale=trade.get("rationale", ""), acted=True,
            reject_reason="", mode="LIVE",
        )
        log.info("[LIVE] %s %s %s | Order %s | SL=%.2f | T=%.2f",
                 trade["symbol"], trade["strategy"], trade["direction"],
                 order_id, decision.sl_price, decision.target_price)
        if self.notifier:
            self.notifier.notify_entry(trade)
        return {"status": "success", "trade_id": trade["trade_id"],
                "order_id": order_id, "mode": "LIVE"}

    def log_rejected_signal(self, signal: StockSignal, reason: str):
        store.log_signal(
            symbol=signal.symbol, strategy=signal.strategy,
            direction=signal.direction, confidence=signal.confidence,
            rationale=signal.rationale, acted=False,
            reject_reason=reason, mode=self.mode,
        )
