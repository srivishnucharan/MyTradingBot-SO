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
  POST /api/squareoff          emergency close all positions
  POST /api/update-token       update Dhan access token
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from data import store

log = logging.getLogger(__name__)

app = FastAPI(title="MyTradingBot-SO Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_orch = None


def _get_orch():
    global _orch
    if _orch is None:
        from core.orchestrator import Orchestrator
        _orch = Orchestrator()
    return _orch


def _detect_mode() -> str:
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
    if not closed:
        return {"n_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0}
    pnls = [float(t.get("realised_pnl") or 0) for t in closed]
    wins = [p for p in pnls if p > 0]
    return {
        "n_trades": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(sum(pnls), 0),
        "avg_pnl": round(sum(pnls) / len(closed), 0),
        "best_trade": round(max(pnls), 0),
        "worst_trade": round(min(pnls), 0),
    }


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
    trades = store.get_open_trades(mode)
    out = []
    for t in trades:
        fill = float(t.get("fill_price") or 0)
        sl = float(t.get("sl_price") or 0)
        tgt = float(t.get("target_price") or 0)
        qty = int(t["lots"]) * int(t["lot_size"])
        sl_dist = round((fill - sl) / fill * 100, 1) if fill else 0
        tgt_dist = round((tgt - fill) / fill * 100, 1) if fill else 0
        # Days to expiry
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
        })
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
    except Exception as e:
        return []


@app.get("/api/backtest/{run_id}")
def backtest_detail(run_id: str):
    trades = store.get_backtest_trades(run_id)
    return trades


def _write_token(path: Path, token: str) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
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
    path.write_text("\n".join(new_lines) + "\n")
    return True


@app.post("/api/update-token")
def update_token(body: dict):
    token = (body.get("access_token") or "").strip()
    if not token:
        raise HTTPException(400, "access_token required")
    so_path = Path("/app/config/secrets.env")
    if not so_path.exists():
        raise HTTPException(500, "secrets.env not found")
    updated = _write_token(so_path, token)
    try:
        import docker as docker_sdk
        client = docker_sdk.from_env()
        containers = client.containers.list(filters={"label": "com.docker.compose.service=bot"})
        for c in containers:
            c.restart()
        return {"status": "ok", "updated": updated, "restarted": [c.name for c in containers]}
    except Exception as e:
        return {"status": "ok", "updated": updated, "warning": str(e)}


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
