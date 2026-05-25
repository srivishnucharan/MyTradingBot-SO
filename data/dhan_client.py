"""
dhan_client.py
Thin singleton wrapper around DhanHQ SDK for stock options.
Supports equity data (NSE_EQ) and FNO option chains.
"""
from __future__ import annotations

import os
import time
import logging
from datetime import date, datetime
from typing import Optional
from threading import Lock
from pathlib import Path

from dotenv import load_dotenv

try:
    from dhanhq import DhanContext, dhanhq
except ImportError:
    DhanContext = None
    dhanhq = None

log = logging.getLogger(__name__)


class DhanClient:
    _instance: Optional["DhanClient"] = None
    _lock = Lock()

    OPTION_CHAIN_RATE_LIMIT_SEC = 3.0
    HISTORICAL_RATE_LIMIT_SEC = 1.0

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            return cls._instance

    def __init__(self):
        if self._initialised:
            return
        _env = Path(__file__).parent.parent / "config" / "secrets.env"
        load_dotenv(_env, override=True)
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN")
        if not self.client_id or not self.access_token:
            raise RuntimeError("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in config/secrets.env")
        if dhanhq is None:
            raise ImportError("Run: pip install dhanhq")

        self.ctx = DhanContext(self.client_id, self.access_token)
        self.dhan = dhanhq(self.ctx)
        self._last_chain_call: float = 0.0
        self._last_hist_call: float = 0.0
        self._initialised = True
        log.info("Dhan client initialised — client_id=%s***", self.client_id[:4])

    def reload_token(self):
        from dotenv import load_dotenv
        _env = Path(__file__).parent.parent / "config" / "secrets.env"
        load_dotenv(_env, override=True)
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN")
        self.ctx = DhanContext(self.client_id, self.access_token)
        self.dhan = dhanhq(self.ctx)
        log.info("Dhan token reloaded — client_id=%s***", (self.client_id or "")[:4])

    # ── option chain ──────────────────────────────────────────────────────────

    def option_chain(self, security_id: int, segment: str, expiry: date) -> Optional[dict]:
        elapsed = time.time() - self._last_chain_call
        if elapsed < self.OPTION_CHAIN_RATE_LIMIT_SEC:
            time.sleep(self.OPTION_CHAIN_RATE_LIMIT_SEC - elapsed)

        for attempt in range(1, 4):
            try:
                resp = self.dhan.option_chain(
                    under_security_id=security_id,
                    under_exchange_segment=segment,
                    expiry=expiry.strftime("%Y-%m-%d"),
                )
                self._last_chain_call = time.time()
                if resp.get("status") == "success":
                    return resp["data"]["data"]
                log.warning("option_chain attempt %d/3 — %s", attempt, resp)
            except Exception as e:
                log.warning("option_chain attempt %d/3 exception: %s", attempt, e)
            if attempt < 3:
                time.sleep(2)

        log.error("option_chain gave up — sec_id=%s expiry=%s", security_id, expiry)
        return None

    def expiry_list(self, security_id: int, segment: str) -> list[date]:
        resp = self.dhan.expiry_list(
            under_security_id=security_id,
            under_exchange_segment=segment,
        )
        if resp.get("status") != "success":
            raise RuntimeError(f"expiry_list failed: {resp}")
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in resp["data"]["data"]]

    # ── historical OHLCV ──────────────────────────────────────────────────────

    def historical_daily_data(self, security_id: str, exchange_segment: str,
                               instrument: str, from_date: str, to_date: str,
                               expiry_code: int = 0) -> dict:
        """Fetch daily OHLCV bars (years of history). Use this for strategy data, not intraday."""
        elapsed = time.time() - self._last_hist_call
        if elapsed < self.HISTORICAL_RATE_LIMIT_SEC:
            time.sleep(self.HISTORICAL_RATE_LIMIT_SEC - elapsed)

        resp = self.dhan.historical_daily_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument,
            expiry_code=expiry_code,
            from_date=from_date,
            to_date=to_date,
        )
        self._last_hist_call = time.time()
        if resp.get("status") != "success":
            raise RuntimeError(f"Historical daily data failed: {resp}")
        return resp["data"]

    def intraday_minute_data(self, security_id: str, exchange_segment: str,
                              instrument: str, from_date: str, to_date: str,
                              interval: int = 60, oi: bool = False) -> dict:
        elapsed = time.time() - self._last_hist_call
        if elapsed < self.HISTORICAL_RATE_LIMIT_SEC:
            time.sleep(self.HISTORICAL_RATE_LIMIT_SEC - elapsed)

        resp = self.dhan.intraday_minute_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            oi=oi,
        )
        self._last_hist_call = time.time()
        if resp.get("status") != "success":
            raise RuntimeError(f"Historical fetch failed: {resp}")
        return resp["data"]

    # ── quotes ─────────────────────────────────────────────────────────────────

    def ltp(self, securities: dict[str, list[int]]) -> dict:
        import json
        resp = self.dhan.ticker_data(securities=securities)
        if isinstance(resp, str):
            resp = json.loads(resp)
        return resp

    # ── orders ─────────────────────────────────────────────────────────────────

    def place_super_order(self, security_id: str, exchange_segment: str,
                           transaction_type: str, quantity: int,
                           order_type: str, product_type: str,
                           price: float, target_price: float,
                           stoploss_price: float,
                           trailing_jump: float = 0.0) -> dict:
        params = dict(
            security_id=security_id,
            exchange_segment=exchange_segment,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            targetPrice=target_price,
            stopLossPrice=stoploss_price,
        )
        if trailing_jump > 0:
            params["trailingJump"] = trailing_jump
        return self.dhan.place_super_order(**params)

    def place_order(self, **kwargs) -> dict:
        return self.dhan.place_order(**kwargs)

    def cancel_order(self, order_id: str) -> dict:
        return self.dhan.cancel_order(order_id)

    def positions(self) -> dict:
        return self.dhan.get_positions()

    def funds(self) -> dict:
        return self.dhan.get_fund_limits()
