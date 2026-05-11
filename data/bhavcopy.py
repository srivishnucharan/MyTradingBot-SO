"""
bhavcopy.py
Downloads and caches NSE F&O EOD Bhavcopy data for historical OI in backtest.

Source: https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
Cache:  data/bhavcopy_cache/{YYYY-MM-DD}.csv.gz  (filtered to configured symbols, ~50-150KB each)

Two CSV formats handled automatically:
  New UDiFF (post July 2024): FinInstrmTp / TckrSymb / XpryDt / StrkPric / OptnTp / OpnIntrst / ChngInOpnIntrst
  Old format (pre July 2024): INSTRUMENT / SYMBOL / EXPIRY_DT / STRIKE_PR / OPTION_TYP / OPEN_INT / CHG_IN_OI
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/bhavcopy_cache")

_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.nseindia.com",
}

# Sentinel file written when a date is confirmed unavailable (holiday/weekend)
_MISS = ".miss"


class BhavCopyLoader:
    """
    Loads NSE F&O EOD Bhavcopy for historical OI per trading day.
    On first access per date: downloads, filters to configured symbols, writes cache.
    Subsequent runs: reads from cache — no network required.
    """

    def __init__(self, symbols: list[str], cache_dir: Path = CACHE_DIR):
        self.symbols = set(symbols)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, Optional[pd.DataFrame]] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def prefetch(self, from_date: date, to_date: date) -> None:
        """Pre-download all trading days in range. Safe to call repeatedly (skips cached)."""
        d = from_date
        loaded = skipped = 0
        while d <= to_date:
            if d.weekday() < 5:
                df = self._load(d)
                if df is not None:
                    loaded += 1
                else:
                    skipped += 1
            d += timedelta(days=1)
        print(f"  Bhavcopy OI: {loaded} days loaded, {skipped} skipped (holidays/unavailable)")

    def get_chain_oi(self, symbol: str, expiry: date, as_of: date) -> dict[float, dict]:
        """
        Returns {strike: {ce_oi, pe_oi, ce_oi_chg, pe_oi_chg}} for symbol/expiry on as_of.
        Empty dict when Bhavcopy data is unavailable for that date.
        """
        df = self._load(as_of)
        if df is None or df.empty:
            return {}

        mask = (df["symbol"] == symbol) & (df["expiry"] == pd.Timestamp(expiry))
        rows = df[mask]
        if rows.empty:
            return {}

        result: dict[float, dict] = {}
        for _, row in rows.iterrows():
            k = float(row["strike"])
            opt = str(row["option_type"]).upper().strip()
            if k not in result:
                result[k] = {"ce_oi": 0, "pe_oi": 0, "ce_oi_chg": 0, "pe_oi_chg": 0}
            if opt == "CE":
                result[k]["ce_oi"]     = int(row["oi"])
                result[k]["ce_oi_chg"] = int(row["oi_chg"])
            elif opt == "PE":
                result[k]["pe_oi"]     = int(row["oi"])
                result[k]["pe_oi_chg"] = int(row["oi_chg"])

        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self, d: date) -> Optional[pd.DataFrame]:
        key = d.isoformat()
        if key in self._mem:
            return self._mem[key]

        cache_csv = self.cache_dir / f"{key}.csv.gz"
        miss_file = self.cache_dir / f"{key}{_MISS}"

        if miss_file.exists():
            self._mem[key] = None
            return None

        if cache_csv.exists():
            try:
                df = pd.read_csv(cache_csv, compression="gzip")
                df["expiry"] = pd.to_datetime(df["expiry"])
                self._mem[key] = df
                return df
            except Exception as e:
                log.debug("Bhavcopy cache read error %s: %s", key, e)
                cache_csv.unlink(missing_ok=True)

        df = self._download(d)
        if df is not None and not df.empty:
            df.to_csv(cache_csv, index=False, compression="gzip")
        else:
            miss_file.touch()  # mark as tried so we don't retry on every run
            df = None
        self._mem[key] = df
        return df

    def _download(self, d: date) -> Optional[pd.DataFrame]:
        url = _URL.format(date=d.strftime("%Y%m%d"))
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code != 200:
                log.debug("Bhavcopy HTTP %s for %s", resp.status_code, d)
                return None
            return self._parse(resp.content)
        except Exception as e:
            log.debug("Bhavcopy download error %s: %s", d, e)
            return None

    def _parse(self, content: bytes) -> Optional[pd.DataFrame]:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                with z.open(csv_name) as f:
                    raw = pd.read_csv(f, low_memory=False)
        except Exception as e:
            log.debug("Bhavcopy zip/csv error: %s", e)
            return None

        cols = set(raw.columns)
        if "FinInstrmTp" in cols:
            return self._norm_new(raw)
        if "INSTRUMENT" in cols:
            return self._norm_old(raw)

        log.debug("Bhavcopy: unrecognised columns: %s", list(raw.columns)[:8])
        return None

    def _norm_new(self, raw: pd.DataFrame) -> pd.DataFrame:
        """New UDiFF format — active since July 2024."""
        mask = (raw["FinInstrmTp"] == "OPTSTK") & (raw["TckrSymb"].isin(self.symbols))
        sub = raw.loc[mask, ["TckrSymb", "XpryDt", "StrkPric", "OptnTp",
                              "OpnIntrst", "ChngInOpnIntrst"]].copy()
        sub.columns = ["symbol", "expiry", "strike", "option_type", "oi", "oi_chg"]
        return self._coerce(sub)

    def _norm_old(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Old format — for backtest dates before July 2024."""
        mask = (raw["INSTRUMENT"] == "OPTSTK") & (raw["SYMBOL"].isin(self.symbols))
        sub = raw.loc[mask, ["SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP",
                              "OPEN_INT", "CHG_IN_OI"]].copy()
        sub.columns = ["symbol", "expiry", "strike", "option_type", "oi", "oi_chg"]
        sub["expiry"] = pd.to_datetime(sub["expiry"], format="%d-%b-%Y", errors="coerce")
        return self._coerce(sub, parse_expiry=False)

    @staticmethod
    def _coerce(df: pd.DataFrame, parse_expiry: bool = True) -> pd.DataFrame:
        if parse_expiry:
            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df["strike"]     = pd.to_numeric(df["strike"],  errors="coerce")
        df["oi"]         = pd.to_numeric(df["oi"],      errors="coerce").fillna(0).astype(int)
        df["oi_chg"]     = pd.to_numeric(df["oi_chg"],  errors="coerce").fillna(0).astype(int)
        return df.dropna(subset=["symbol", "expiry", "strike"]).reset_index(drop=True)
