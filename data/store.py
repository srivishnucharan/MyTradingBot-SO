"""
store.py
SQLite persistence for MyTradingBot-SO.
Tables: trades, orders, signals_log, backtest_trades.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, date
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path("logs/so_bot.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id       TEXT UNIQUE NOT NULL,
    ts_open        TEXT NOT NULL,
    ts_close       TEXT,
    symbol         TEXT NOT NULL,
    strategy       TEXT NOT NULL,
    direction      TEXT NOT NULL,
    strike         REAL NOT NULL,
    option_type    TEXT NOT NULL,
    expiry         TEXT NOT NULL,
    security_id    TEXT NOT NULL,
    tradingsymbol  TEXT NOT NULL,
    lots           INTEGER NOT NULL,
    lot_size       INTEGER NOT NULL,
    fill_price     REAL,
    sl_price       REAL,
    target_price   REAL,
    exit_price     REAL,
    realised_pnl   REAL,
    exit_reason    TEXT,
    mode           TEXT NOT NULL,
    super_order_id TEXT,
    rationale      TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode, ts_open);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(ts_close);

CREATE TABLE IF NOT EXISTS orders (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id       TEXT NOT NULL,
    dhan_order_id  TEXT,
    status         TEXT DEFAULT 'PENDING',
    updated_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_trade ON orders(trade_id);

CREATE TABLE IF NOT EXISTS signals_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    direction     TEXT NOT NULL,
    confidence    TEXT,
    rationale     TEXT,
    acted         INTEGER DEFAULT 0,
    reject_reason TEXT,
    mode          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals_log(ts, symbol);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    trade_date    TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    direction     TEXT NOT NULL,
    strike        REAL NOT NULL,
    option_type   TEXT NOT NULL,
    expiry        TEXT NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL,
    exit_date     TEXT,
    realised_pnl  REAL,
    exit_reason   TEXT,
    lots          INTEGER NOT NULL,
    lot_size      INTEGER NOT NULL,
    sl_price      REAL,
    target_price  REAL
);
CREATE INDEX IF NOT EXISTS idx_bt_run ON backtest_trades(run_id, symbol);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    underlying  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    regime_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions ON decisions(underlying, mode, ts);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


# ── trades ─────────────────────────────────────────────────────────────────────

def save_trade(trade: dict):
    with connect() as con:
        con.execute(
            """INSERT OR REPLACE INTO trades
               (trade_id, ts_open, ts_close, symbol, strategy, direction,
                strike, option_type, expiry, security_id, tradingsymbol, lots,
                lot_size, fill_price, sl_price, target_price, exit_price,
                realised_pnl, exit_reason, mode, super_order_id, rationale)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade["trade_id"], trade["ts_open"], trade.get("ts_close"),
                trade["symbol"], trade["strategy"], trade["direction"],
                trade["strike"], trade["option_type"], trade["expiry"],
                str(trade["security_id"]), trade["tradingsymbol"],
                trade["lots"], trade["lot_size"],
                trade.get("fill_price"), trade.get("sl_price"), trade.get("target_price"),
                trade.get("exit_price"), trade.get("realised_pnl"), trade.get("exit_reason"),
                trade["mode"], trade.get("super_order_id"), trade.get("rationale"),
            ),
        )


def get_open_trades(mode: str) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM trades WHERE mode=? AND ts_close IS NULL ORDER BY ts_open DESC",
            (mode,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_trades(mode: str, since: Optional[date] = None) -> list[dict]:
    sql = "SELECT * FROM trades WHERE mode=?"
    params: list = [mode]
    if since:
        sql += " AND ts_open >= ?"
        params.append(since.isoformat())
    sql += " ORDER BY ts_open DESC"
    with connect() as con:
        rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def close_trade(trade_id: str, exit_price: float, realised_pnl: float, reason: str):
    with connect() as con:
        con.execute(
            """UPDATE trades SET ts_close=?, exit_price=?, realised_pnl=?, exit_reason=?
               WHERE trade_id=?""",
            (datetime.now().isoformat(), exit_price, realised_pnl, reason, trade_id),
        )


def save_order(trade_id: str, dhan_order_id: str):
    with connect() as con:
        con.execute(
            """INSERT OR IGNORE INTO orders (trade_id, dhan_order_id, status, updated_at)
               VALUES (?,?,?,?)""",
            (trade_id, dhan_order_id, "PENDING", datetime.now().isoformat()),
        )


def log_signal(symbol: str, strategy: str, direction: str,
                confidence: str, rationale: str, acted: bool,
                reject_reason: str, mode: str):
    with connect() as con:
        con.execute(
            """INSERT INTO signals_log
               (ts, symbol, strategy, direction, confidence, rationale,
                acted, reject_reason, mode)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().isoformat(), symbol, strategy, direction,
                confidence, rationale, 1 if acted else 0, reject_reason, mode,
            ),
        )


# ── backtest ───────────────────────────────────────────────────────────────────

def save_backtest_trade(run_id: str, trade: dict):
    with connect() as con:
        con.execute(
            """INSERT INTO backtest_trades
               (run_id, trade_date, symbol, strategy, direction, strike, option_type,
                expiry, entry_price, exit_price, exit_date, realised_pnl, exit_reason,
                lots, lot_size, sl_price, target_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, trade["trade_date"], trade["symbol"], trade["strategy"],
                trade["direction"], trade["strike"], trade["option_type"],
                trade["expiry"], trade["entry_price"],
                trade.get("exit_price"), trade.get("exit_date"),
                trade.get("realised_pnl"), trade.get("exit_reason"),
                trade["lots"], trade["lot_size"],
                trade.get("sl_price"), trade.get("target_price"),
            ),
        )


def get_backtest_trades(run_id: str) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY trade_date",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
