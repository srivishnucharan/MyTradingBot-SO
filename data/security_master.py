"""
security_master.py
Maps Nifty 50 stock symbols -> Dhan security IDs.
Downloads and caches Dhan scrip master CSV daily.

CSV column names (as of 2025 Dhan format):
  EXCH_ID, SEGMENT, SECURITY_ID, INSTRUMENT, UNDERLYING_SECURITY_ID,
  UNDERLYING_SYMBOL, SYMBOL_NAME, SM_EXPIRY_DATE, STRIKE_PRICE, OPTION_TYPE
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from threading import Lock
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

DHAN_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
LOCAL_PATH = Path("data/scrip_master.csv")
MAX_AGE_HOURS = 24

# Nifty 50 stock symbols — options trade on NSE_FNO, underlying on NSE_EQ
NIFTY50_SYMBOLS = {
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "BAJFINANCE",
    "BHARTIARTL", "TATAMOTORS", "WIPRO", "AXISBANK", "KOTAKBANK", "SBIN",
    "LT", "HCLTECH", "SUNPHARMA", "MARUTI", "TITAN", "ASIANPAINT",
    "BAJAJFINSV", "NESTLEIND", "ULTRACEMCO", "POWERGRID", "NTPC", "TECHM",
    "HINDALCO", "TATASTEEL", "JSWSTEEL", "COALINDIA", "GRASIM", "INDUSINDBK",
    "CIPLA", "DRREDDY", "DIVISLAB", "ADANIENT", "ADANIPORTS", "APOLLOHOSP",
    "BPCL", "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "M&M", "ONGC",
    "SBILIFE", "SHRIRAMFIN", "TATACONSUM", "TRENT", "ETERNAL",
}

# Sector mapping for confluence FII checks
SECTOR_MAP = {
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "AXISBANK": "BANKING",
    "KOTAKBANK": "BANKING", "SBIN": "BANKING", "INDUSINDBK": "BANKING",
    "INFY": "IT", "TCS": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "COALINDIA": "ENERGY",
    "TATAMOTORS": "AUTO", "MARUTI": "AUTO", "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE", "SBILIFE": "FINANCE", "SHRIRAMFIN": "FINANCE",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA", "DIVISLAB": "PHARMA",
    "LT": "INFRASTRUCTURE", "ADANIPORTS": "INFRASTRUCTURE", "POWERGRID": "UTILITIES",
    "NTPC": "UTILITIES", "BHARTIARTL": "TELECOM", "TITAN": "CONSUMER",
    "ASIANPAINT": "CONSUMER", "NESTLEIND": "CONSUMER", "BRITANNIA": "CONSUMER",
    "TATACONSUM": "CONSUMER", "TRENT": "CONSUMER", "ETERNAL": "CONSUMER",
    "HINDALCO": "METALS", "TATASTEEL": "METALS", "JSWSTEEL": "METALS",
    "GRASIM": "MATERIALS", "ULTRACEMCO": "MATERIALS", "ADANIENT": "CONGLOMERATE",
    "APOLLOHOSP": "HEALTHCARE", "M&M": "AUTO",
}

# India VIX security ID
VIX_INFO = (264969, "IDX_I")


class SecurityMaster:
    _instance: Optional["SecurityMaster"] = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
            return cls._instance

    def __init__(self):
        if self._loaded:
            return
        self._equity_ids: dict[str, int] = {}     # symbol -> equity security_id (underlying)
        self._option_rows: list[dict] = []          # NSE OPTSTK rows for find_option()
        self._ensure_fresh()
        self._load()
        self._loaded = True

    def _ensure_fresh(self):
        needs_refresh = True
        if LOCAL_PATH.exists():
            mtime = datetime.fromtimestamp(LOCAL_PATH.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=MAX_AGE_HOURS):
                needs_refresh = False
        if needs_refresh:
            log.info("Downloading Dhan scrip master...")
            LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(DHAN_MASTER_URL, timeout=60)
            r.raise_for_status()
            LOCAL_PATH.write_bytes(r.content)
            log.info("Scrip master saved to %s", LOCAL_PATH)

    def _load(self):
        """
        Build equity ID map from NSE OPTSTK rows:
          UNDERLYING_SYMBOL -> UNDERLYING_SECURITY_ID (the equity security ID for market data)
        """
        with LOCAL_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exch = row.get("EXCH_ID", "").strip()
                instr = row.get("INSTRUMENT", "").strip()
                if exch != "NSE" or instr != "OPTSTK":
                    continue

                und_sym = row.get("UNDERLYING_SYMBOL", "").strip().upper()
                und_sec_id = row.get("UNDERLYING_SECURITY_ID", "").strip()

                # Store equity underlying ID (first occurrence per symbol)
                if und_sym and und_sym not in self._equity_ids:
                    try:
                        self._equity_ids[und_sym] = int(und_sec_id)
                    except (ValueError, TypeError):
                        pass

                # Keep option rows for find_option() — only for tracked symbols
                if und_sym in NIFTY50_SYMBOLS:
                    self._option_rows.append(row)

        log.info("Loaded %d Nifty50 equity IDs, %d option rows",
                 len(self._equity_ids), len(self._option_rows))
        missing = NIFTY50_SYMBOLS - set(self._equity_ids.keys())
        if missing:
            log.warning("Missing equity IDs for: %s", sorted(missing))

    def equity_info(self, symbol: str) -> tuple[int, str]:
        """Returns (security_id, segment) for an NSE equity stock."""
        sid = self._equity_ids.get(symbol.upper())
        if sid is None:
            raise KeyError(f"Equity not found in scrip master: {symbol}")
        return sid, "NSE_EQ"

    def vix_info(self) -> tuple[int, str]:
        return VIX_INFO

    def option_segment(self, symbol: str) -> str:
        return "NSE_FNO"

    def sector(self, symbol: str) -> str:
        return SECTOR_MAP.get(symbol.upper(), "OTHER")

    def find_option(self, symbol: str, expiry: "date",
                     strike: float, option_type: str) -> tuple[int, str]:
        """Returns (security_id, tradingsymbol) for a specific option contract."""
        expiry_str = expiry.strftime("%Y-%m-%d")
        sym_upper = symbol.upper()
        for row in self._option_rows:
            if row.get("UNDERLYING_SYMBOL", "").upper() != sym_upper:
                continue
            row_expiry = row.get("SM_EXPIRY_DATE", "")[:10]
            if row_expiry != expiry_str:
                continue
            try:
                row_strike = float(row.get("STRIKE_PRICE", 0))
            except (ValueError, TypeError):
                continue
            if abs(row_strike - strike) < 0.01:
                if row.get("OPTION_TYPE", "").upper() == option_type.upper():
                    try:
                        sec_id = int(row["SECURITY_ID"])
                    except (ValueError, TypeError):
                        continue
                    return sec_id, row.get("SYMBOL_NAME", "")
        raise KeyError(f"Option not found: {symbol} {expiry} {strike} {option_type}")

    def all_equity_ids(self) -> dict[str, int]:
        return dict(self._equity_ids)
