"""
analytics.py
Phase 2: per-strategy expectancy from real trades and shadow counterfactuals.
All returns are R-multiples: R = (exit - entry) / (entry - SL), so strategies
with different premium sizes are directly comparable. Rows with fewer than
MIN_SAMPLE closed samples are flagged unreliable.
"""
from __future__ import annotations

import logging
from statistics import median
from typing import Iterator

from data import store

log = logging.getLogger(__name__)

MIN_SAMPLE = 30
_FALLBACK_SL_PCT = 0.30   # risk basis when a row has no valid SL


def expectancy_report(mode: str) -> dict:
    groups: dict[tuple[str, str], dict] = {}
    for source, records in (("REAL", _real_records(mode)),
                            ("SHADOW", _shadow_records(mode))):
        for rec in records:
            g = groups.setdefault((rec["strategy"], source), {
                "strategy": rec["strategy"], "source": source,
                "n_closed": 0, "n_open": 0, "r": [], "mfe": [], "mae": [],
                "eod_r": [], "hold_r": [], "exits": {},
            })
            if rec["open"]:
                g["n_open"] += 1
                continue
            g["n_closed"] += 1
            g["r"].append(rec["r"])
            if rec["mfe_r"] is not None:
                g["mfe"].append(rec["mfe_r"])
            if rec["mae_r"] is not None:
                g["mae"].append(rec["mae_r"])
            if rec["eod_r"] is not None:
                # paired sample: same trade valued at first EOD vs actual exit
                g["eod_r"].append(rec["eod_r"])
                g["hold_r"].append(rec["r"])
            reason = rec["exit_reason"] or "?"
            g["exits"][reason] = g["exits"].get(reason, 0) + 1

    rows = []
    for g in groups.values():
        r = g["r"]
        wins = [x for x in r if x > 0]
        losses = [x for x in r if x <= 0]
        rows.append({
            "strategy": g["strategy"],
            "source": g["source"],
            "n_closed": g["n_closed"],
            "n_open": g["n_open"],
            "win_rate": round(len(wins) / len(r) * 100, 1) if r else None,
            "expectancy_r": _avg(r),
            "avg_win_r": _avg(wins),
            "avg_loss_r": _avg(losses),
            "median_mfe_r": round(median(g["mfe"]), 3) if g["mfe"] else None,
            "median_mae_r": round(median(g["mae"]), 3) if g["mae"] else None,
            "n_eod": len(g["eod_r"]),
            "avg_eod_r": _avg(g["eod_r"]),
            "avg_hold_r": _avg(g["hold_r"]),
            "exits": g["exits"],
            "reliable": g["n_closed"] >= MIN_SAMPLE,
        })
    rows.sort(key=lambda x: (x["strategy"], x["source"]))
    return {"mode": mode, "min_sample": MIN_SAMPLE, "rows": rows}


def _avg(xs: list) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _risk(entry: float, sl) -> float:
    sl = float(sl or 0)
    if 0 < sl < entry:
        return entry - sl
    return entry * _FALLBACK_SL_PCT


def _real_records(mode: str) -> Iterator[dict]:
    with store.connect() as con:
        rows = con.execute(
            """SELECT strategy, fill_price, sl_price, exit_price, exit_reason,
                      peak_ltp, trough_ltp, ts_close
               FROM trades WHERE mode=?""", (mode,)).fetchall()
    for t in rows:
        entry = float(t["fill_price"] or 0)
        if entry <= 0:
            continue
        risk = _risk(entry, t["sl_price"])
        is_open = t["ts_close"] is None
        if not is_open and t["exit_price"] is None:
            continue  # closed without an exit price — nothing to score
        yield {
            "strategy": t["strategy"],
            "open": is_open,
            "r": (float(t["exit_price"]) - entry) / risk if not is_open else None,
            "mfe_r": (float(t["peak_ltp"]) - entry) / risk if t["peak_ltp"] else None,
            "mae_r": (float(t["trough_ltp"]) - entry) / risk if t["trough_ltp"] else None,
            "eod_r": None,  # real trades have no first-day EOD snapshot
            "exit_reason": t["exit_reason"],
        }


def _shadow_records(mode: str) -> Iterator[dict]:
    with store.connect() as con:
        rows = con.execute(
            """SELECT strategy, entry_price, sl_price, exit_price, exit_reason,
                      peak_ltp, trough_ltp, first_eod_price, ts_close
               FROM shadow_trades WHERE mode=?""", (mode,)).fetchall()
    for t in rows:
        entry = float(t["entry_price"] or 0)
        if entry <= 0:
            continue
        risk = _risk(entry, t["sl_price"])
        is_open = t["ts_close"] is None
        if not is_open and t["exit_price"] is None:
            continue
        eod = t["first_eod_price"]
        yield {
            "strategy": t["strategy"],
            "open": is_open,
            "r": (float(t["exit_price"]) - entry) / risk if not is_open else None,
            "mfe_r": (float(t["peak_ltp"]) - entry) / risk if t["peak_ltp"] else None,
            "mae_r": (float(t["trough_ltp"]) - entry) / risk if t["trough_ltp"] else None,
            "eod_r": (float(eod) - entry) / risk if (eod and not is_open) else None,
            "exit_reason": t["exit_reason"],
        }
