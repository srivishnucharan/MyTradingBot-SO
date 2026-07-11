"""
features.py
Feature-vector snapshot for every generated signal (taken or not).
Stored as JSON in signals_log.features / shadow_trades.features so a
meta-labeling model can later be trained on signal -> outcome pairs.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Optional

log = logging.getLogger(__name__)


def build_features(ctx, signal, macro=None) -> str:
    """Serialise the market context at signal time to a JSON feature dict.

    ctx    : StockMarketContext
    signal : StockSignal
    macro  : MacroSentiment or None
    """
    try:
        feats = {
            "strategy": signal.strategy,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "confluence_score": signal.confluence_score,
            "sector": ctx.sector,
            "vix": round(ctx.vix, 2),
            "rsi_14": round(ctx.rsi_14, 2),
            "hist_vol_30": round(ctx.hist_vol_30, 4),
            "ema20_dist_pct": _pct(ctx.spot, ctx.ema_20),
            "ema50_dist_pct": _pct(ctx.spot, ctx.ema_50),
            "vol_ratio": _vol_ratio(ctx),
            "fii_trend": ctx.fii_sector_trend,
            "macro_score": getattr(macro, "overall_score", None) if macro else None,
            "dte": (signal.expiry - date.today()).days,
            "expected_premium": round(signal.expected_premium, 2),
            "premium_pct_spot": _pct_of(signal.expected_premium, ctx.spot),
            "strike_dist_gaps": _strike_dist(signal, ctx),
            "pcr": _pcr(ctx),
            "atm_iv": _atm_iv(ctx, signal.option_type),
        }
        return json.dumps(feats)
    except Exception as e:
        log.warning("build_features failed for %s: %s", getattr(signal, "symbol", "?"), e)
        return ""


def _pct(a: float, b: float) -> Optional[float]:
    return round((a - b) / b * 100, 3) if b else None


def _pct_of(a: float, b: float) -> Optional[float]:
    return round(a / b * 100, 3) if b else None


def _vol_ratio(ctx) -> Optional[float]:
    try:
        today_vol = float(ctx.bars_daily["volume"].iloc[-1])
        return round(today_vol / ctx.vol_avg_20, 3) if ctx.vol_avg_20 else None
    except Exception:
        return None


def _strike_dist(signal, ctx) -> Optional[float]:
    if not ctx.strike_gap:
        return None
    return round((signal.strike - ctx.atm_strike) / ctx.strike_gap, 1)


def _pcr(ctx) -> Optional[float]:
    try:
        ce = float(ctx.chain["ce_oi"].sum())
        pe = float(ctx.chain["pe_oi"].sum())
        return round(pe / ce, 3) if ce > 0 else None
    except Exception:
        return None


def _atm_iv(ctx, option_type: str) -> Optional[float]:
    try:
        row = ctx.chain[ctx.chain["strike"] == ctx.atm_strike]
        if row.empty:
            return None
        col = "ce_iv" if option_type == "CE" else "pe_iv"
        iv = float(row.iloc[0][col])
        return round(iv, 2) if iv > 0 else None
    except Exception:
        return None
