"""
api.py — FastAPI dashboard backend for MyTradingBot-SO.
Port 8002.

Endpoints:
  GET  /                       dashboard HTML
  GET  /api/status             mode, market open, positions, daily P&L
  GET  /api/positions          open swing trades
  GET  /api/trades             closed trades (all time or date-filtered)
  GET  /api/signals            recent signal log
  GET  /api/metrics/{mode}     performance summary
  GET  /api/backtest           latest backtest summary
  GET  /api/token-info         masked current token
  GET  /api/risk-status        live risk meter values
  GET  /api/regime             latest regime per symbol
  POST /api/set-mode           write PAPER/LIVE to ui_mode.txt
  POST /api/squareoff          emergency close all positions
  POST /api/update-token       update Dhan access token
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from data import store

log = logging.getLogger(__name__)

app = FastAPI(title="MyTradingBot-SO Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_PROJECT_ROOT = Path(__file__).parent.parent
_LOGS_DIR = Path("/app/logs") if Path("/app/logs").exists() else _PROJECT_ROOT / "logs"
_ENV_PATH = (Path("/app/config/secrets.env")
             if Path("/app/config/secrets.env").exists()
             else _PROJECT_ROOT / "config" / "secrets.env")
_UI_MODE_FILE = _LOGS_DIR / "ui_mode.txt"
_RESTART_FLAG = _LOGS_DIR / "restart.flag"

_VALID_MODES = ("PAPER", "LIVE")

_cfg_cache: dict | None = None
_orch = None


def _get_cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        with open(_PROJECT_ROOT / "config" / "config.yaml") as f:
            _cfg_cache = yaml.safe_load(f)
    return _cfg_cache


def _get_orch():
    global _orch
    if _orch is None:
        from core.orchestrator import Orchestrator
        _orch = Orchestrator()
    return _orch


def _detect_mode() -> str:
    if _UI_MODE_FILE.exists():
        try:
            m = _UI_MODE_FILE.read_text().strip().upper()
            if m in _VALID_MODES:
                return m
        except Exception:
            pass
    for mode in ("LIVE", "PAPER"):
        try:
            if store.get_open_trades(mode):
                return mode
        except Exception:
            pass
    return "PAPER"


def _is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


def _compute_metrics(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("ts_close")]
    n_open = sum(1 for t in trades if not t.get("ts_close"))
    if not closed:
        return {"n_trades": 0, "n_open_trades": n_open, "win_rate": 0.0,
                "total_pnl": 0.0, "avg_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0}
    pnls = [float(t.get("realised_pnl") or 0) for t in closed]
    wins = [p for p in pnls if p > 0]
    return {
        "n_trades": len(closed),
        "n_open_trades": n_open,
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(sum(pnls), 0),
        "avg_pnl": round(sum(pnls) / len(closed), 0),
        "best_trade": round(max(pnls), 0),
        "worst_trade": round(min(pnls), 0),
    }


def _write_token(path: Path, token: str) -> bool:
    lines = path.read_text().splitlines() if path.exists() else []
    new_lines = []
    found = False
    for line in lines:
        if line.startswith("DHAN_ACCESS_TOKEN="):
            new_lines.append(f"DHAN_ACCESS_TOKEN={token}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"DHAN_ACCESS_TOKEN={token}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(new_lines) + "\n")
    return True


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    mode = _detect_mode()
    open_trades = store.get_open_trades(mode)
    today_trades = store.get_trades(mode, since=date.today())
    closed_today = [t for t in today_trades if t.get("ts_close")]
    daily_pnl = sum(float(t.get("realised_pnl") or 0) for t in closed_today)
    return {
        "mode": mode,
        "market_open": _is_market_open(),
        "now": datetime.now().isoformat(),
        "open_positions": len(open_trades),
        "daily_pnl": round(daily_pnl, 0),
    }


@app.get("/api/positions")
def positions(mode: str = "PAPER"):
    try:
        trades = store.get_open_trades(mode)
    except Exception as e:
        log.warning("positions DB error: %s", e)
        return []
    out = []
    for t in trades:
        try:
            fill = float(t.get("fill_price") or 0)
            sl = float(t.get("sl_price") or 0)
            tgt = float(t.get("target_price") or 0)
            qty = int(t["lots"]) * int(t["lot_size"])
            sl_dist = round((fill - sl) / fill * 100, 1) if fill else 0
            tgt_dist = round((tgt - fill) / fill * 100, 1) if fill else 0
            try:
                expiry_date = date.fromisoformat(t["expiry"])
                dte = (expiry_date - date.today()).days
            except Exception:
                dte = 0
            out.append({
                "trade_id": t["trade_id"],
                "symbol": t["symbol"],
                "strategy": t["strategy"],
                "direction": t["direction"],
                "strike": t["strike"],
                "option_type": t["option_type"],
                "expiry": t["expiry"],
                "dte": dte,
                "lots": t["lots"],
                "lot_size": t["lot_size"],
                "qty": qty,
                "fill_price": fill,
                "sl_price": sl,
                "target_price": tgt,
                "sl_dist_pct": sl_dist,
                "tgt_dist_pct": tgt_dist,
                "ts_open": t.get("ts_open", ""),
                "rationale": t.get("rationale", ""),
                "pnl_note": None,
            })
        except Exception as e:
            log.warning("positions row error trade_id=%s: %s", t.get("trade_id"), e)
    return out


@app.get("/api/trades")
def trades(mode: str = "PAPER", limit: int = 50, since: str = None):
    since_date = date.fromisoformat(since) if since else None
    all_trades = store.get_trades(mode, since=since_date)
    closed = [t for t in all_trades if t.get("ts_close")]
    return closed[:limit]


@app.get("/api/signals")
def signals(mode: str = "PAPER", limit: int = 30):
    with store.connect() as con:
        rows = con.execute(
            "SELECT * FROM signals_log WHERE mode=? ORDER BY ts DESC LIMIT ?",
            (mode, limit),
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/metrics/{mode}")
def metrics(mode: str):
    all_trades = store.get_trades(mode)
    return _compute_metrics(all_trades)


@app.get("/api/backtest")
def backtest_summary():
    try:
        with store.connect() as con:
            rows = con.execute(
                """SELECT run_id, COUNT(*) as trades,
                   SUM(realised_pnl) as total_pnl,
                   SUM(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) as wins,
                   MIN(trade_date) as from_date, MAX(trade_date) as to_date
                   FROM backtest_trades GROUP BY run_id ORDER BY run_id DESC LIMIT 5"""
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


@app.get("/api/backtest/{run_id}")
def backtest_detail(run_id: str):
    return store.get_backtest_trades(run_id)


@app.get("/api/token-info")
def token_info():
    if not _ENV_PATH.exists():
        return {"token": None, "masked": "—"}
    for line in _ENV_PATH.read_text().splitlines():
        if line.startswith("DHAN_ACCESS_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if not token:
                return {"token": None, "masked": "not set"}
            masked = token[:10] + "…" + token[-4:] if len(token) > 14 else token[:4] + "…"
            return {"token": token, "masked": masked}
    return {"token": None, "masked": "not set"}


@app.post("/api/set-mode")
def set_mode(body: dict):
    mode = (body.get("mode") or "").strip().upper()
    if mode not in _VALID_MODES:
        raise HTTPException(400, f"mode must be one of {_VALID_MODES}")
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _UI_MODE_FILE.write_text(mode)
    return {"status": "ok", "mode": mode}


@app.get("/api/risk-status")
def risk_status(mode: str = "PAPER"):
    risk = _get_cfg().get("risk", {})
    capital  = float(risk.get("capital", 500000))
    max_pos  = int(risk.get("max_concurrent_positions", 3))
    loss_pct = float(risk.get("max_risk_per_trade_pct", 2.0))
    sl_pct   = float(risk.get("sl_pct", 0.30))

    open_trades = store.get_open_trades(mode)
    n_open = len(open_trades)
    slots_pct = min(100.0, n_open / max_pos * 100) if max_pos else 0.0

    today_trades = store.get_trades(mode, since=date.today())
    closed_today = [t for t in today_trades if t.get("ts_close")]
    daily_pnl    = sum(float(t.get("realised_pnl") or 0) for t in closed_today)
    daily_loss   = max(0.0, -daily_pnl)
    daily_cap    = capital * loss_pct * max_pos / 100
    daily_loss_pct = min(100.0, daily_loss / daily_cap * 100) if daily_cap else 0.0

    margin_used = sum(
        float(t.get("fill_price") or 0) * sl_pct
        * int(t.get("lots", 1)) * int(t.get("lot_size", 1))
        for t in open_trades
    )
    margin_pct = min(100.0, margin_used / daily_cap * 100) if daily_cap else 0.0

    return {
        "position_slots_pct":  round(slots_pct, 1),
        "position_slots_used": n_open,
        "position_slots_max":  max_pos,
        "daily_loss_cap_pct":  round(daily_loss_pct, 1),
        "daily_pnl":           round(daily_pnl, 0),
        "daily_cap":           round(daily_cap, 0),
        "margin_used_pct":     round(margin_pct, 1),
        "margin_used":         round(margin_used, 0),
        "margin_cap":          round(daily_cap, 0),
    }


@app.get("/api/regime")
def regime(mode: str = "PAPER"):
    with store.connect() as con:
        try:
            rows = con.execute(
                """SELECT underlying, MAX(ts) AS ts, regime_json
                   FROM decisions WHERE mode=? GROUP BY underlying
                   ORDER BY underlying""", (mode,)
            ).fetchall()
        except Exception:
            rows = []

    if rows:
        result = []
        for r in rows:
            try:
                rj = json.loads(r["regime_json"]) if r["regime_json"] else {}
            except Exception:
                rj = {}
            rj["underlying"] = r["underlying"]
            rj["ts"] = r["ts"]
            result.append(rj)
        return result

    with store.connect() as con:
        sig_rows = con.execute(
            """SELECT symbol AS underlying, MAX(ts) AS ts, strategy, direction,
                      confidence, rationale, acted, reject_reason
               FROM signals_log WHERE mode=?
               GROUP BY symbol ORDER BY symbol""", (mode,)
        ).fetchall()
        today_rows = con.execute(
            """SELECT symbol AS underlying, COUNT(*) AS cnt
               FROM signals_log WHERE mode=? AND acted=1 AND ts >= ?
               GROUP BY symbol""",
            (mode, date.today().isoformat())
        ).fetchall()

    if not sig_rows:
        return []

    today_counts = {r["underlying"]: r["cnt"] for r in today_rows}
    return [
        {
            "underlying":    r["underlying"],
            "ts":            r["ts"],
            "direction":     r["direction"],
            "strategy":      r["strategy"],
            "confidence":    r["confidence"],
            "rationale":     r["rationale"],
            "acted":         bool(r["acted"]),
            "reject_reason": r["reject_reason"],
            "trades_today":  today_counts.get(r["underlying"], 0),
        }
        for r in sig_rows
    ]


@app.post("/api/update-token")
def update_token(body: dict):
    token = (body.get("access_token") or "").strip()
    if not token:
        raise HTTPException(400, "access_token required")
    updated = _write_token(_ENV_PATH, token)
    try:
        import docker as docker_sdk
        client = docker_sdk.from_env()
        containers = client.containers.list(filters={"label": "com.docker.compose.service=bot"})
        for c in containers:
            c.restart()
        return {"status": "ok", "updated": updated, "restarted": [c.name for c in containers]}
    except Exception:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _RESTART_FLAG.write_text("1")
        return {"status": "ok", "updated": updated, "restarted": []}


@app.post("/api/squareoff")
def squareoff():
    try:
        orch = _get_orch()
        results = orch.monitor_agent.squareoff_all(reason="MANUAL_DASHBOARD")
        return {"status": "ok", "closed": len(results)}
    except Exception as e:
        log.exception("Squareoff failed: %s", e)
        raise HTTPException(500, str(e))


DASHBOARD_DIR = Path(__file__).parent


@app.get("/")
def index():
    html = DASHBOARD_DIR / "index.html"
    return FileResponse(html) if html.exists() else HTMLResponse("<h1>index.html not found</h1>")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    store.init_db()
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
