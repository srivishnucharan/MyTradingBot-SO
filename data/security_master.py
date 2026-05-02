"""
security_master.py
Maps Nifty 50 stock symbols → Dhan security IDs.
Downloads and caches Dhan scrip master CSV daily.
Supports equity (NSE_EQ) stocks and their FNO option chains.
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
    "SBILIFE", "SHRIRAMFIN", "TATACONSUM", "TRENT", "ZOMATO",
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
    "TATACONSUM": "CONSUMER", "TRENT": "CONSUMER", "ZOMATO": "CONSUMER",
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
        self._equity_ids: dict[str, int] = {}     # symbol → equity security_id
        self._row_by_id: dict[int, dict] = {}
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
        with LOCAL_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sec_id = int(row["SEM_SMST_SECURITY_ID"])
                except (KeyError, ValueError):
                    continue
                tsym = row.get("SEM_TRADING_SYMBOL", "").strip().upper()
                exch = row.get("SEM_EXM_EXCH_ID", "").strip().upper()
                instr = row.get("SEM_INSTRUMENT_NAME", "").strip().upper()
                self._row_by_id[sec_id] = row

                # Equity NSE: look for EQUITY instruments in NSE exchange
                if exch == "NSE" and instr in ("EQUITY",) and tsym in NIFTY50_SYMBOLS:
                    if tsym not in self._equity_ids:
                        self._equity_ids[tsym] = sec_id

        log.info("Loaded %d instruments; %d Nifty50 equity IDs mapped",
                 len(self._row_by_id), len(self._equity_ids))
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
        from datetime import date as _date
        for sec_id, row in self._row_by_id.items():
            if row.get("SEM_INSTRUMENT_NAME", "").upper() not in ("OPTSTK",):
                continue
            tsym = row.get("SEM_TRADING_SYMBOL", "").upper()
            if not tsym.startswith(symbol.upper()):
                continue
            row_expiry = row.get("SEM_EXPIRY_DATE", "")[:10]
            if row_expiry != expiry.strftime("%Y-%m-%d"):
                continue
            try:
                row_strike = float(row.get("SEM_STRIKE_PRICE", 0))
            except ValueError:
                continue
            if abs(row_strike - strike) < 0.01:
                if row.get("SEM_OPTION_TYPE", "").upper() == option_type.upper():
                    return sec_id, row["SEM_TRADING_SYMBOL"]
        raise KeyError(f"Option not found: {symbol} {expiry} {strike} {option_type}")

    def all_equity_ids(self) -> dict[str, int]:
        return dict(self._equity_ids)
