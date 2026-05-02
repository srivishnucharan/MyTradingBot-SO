"""
backtesting/engine.py
12-month backtest engine for Nifty50 stock options strategies.
Uses Dhan historical daily data + Black-Scholes option pricing.
No look-ahead bias: signals computed only from bars available on that date.
"""
from __future__ import annotations

import logging
import uuid
from calendar import monthrange
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from math import exp, log as mlog, sqrt, erf
from typing import Optional

import pandas as pd
import yaml

from data.dhan_client import DhanClient
from data.security_master import SecurityMaster
from data.market_data import MarketData, StockMarketContext
from data import store
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent

log = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    run_id: str
    trade_date: str
    symbol: str
    strategy: str
    direction: str
    strike: float
    option_type: str
    expiry: str
    entry_price: float
    lots: int
    lot_size: int
    sl_price: float
    target_price: float
    exit_price: float = 0.0
    exit_date: str = ""
    realised_pnl: float = 0.0
    exit_reason: str = "OPEN"


class BacktestEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.dhan = DhanClient()
        self.md = MarketData(self.dhan)
        self.master = SecurityMaster()
        self.signal_agent = SignalAgent()
        self.run_id = f"BT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def run(self, months: int = 12) -> list[BacktestTrade]:
        store.init_db()
        end_date = date.today()
        start_date = date(end_date.year - 1, end_date.month, end_date.day)

        log.info("Backtest run_id=%s | %s → %s", self.run_id, start_date, end_date)

        all_bars: dict[str, pd.DataFrame] = {}
        instruments = [i for i in self.cfg["instruments"] if i.get("enabled", True)]

        # Pre-fetch 15 months of daily bars for each stock
        print(f"\nFetching historical data for {len(instruments)} stocks...")
        for instr in instruments:
            symbol = instr["symbol"]
            print(f"  Loading {symbol}...", end=" ", flush=True)
            try:
                bars = self._fetch_bars(instr, start_date - timedelta(days=120), end_date)
                if bars is not None and len(bars) >= 30:
                    all_bars[symbol] = bars
                    print(f"OK ({len(bars)} days)")
                else:
                    print("SKIP (insufficient data)")
            except Exception as e:
                print(f"ERROR: {e}")

        # Walk forward through trading days
        trades: list[BacktestTrade] = []
        open_trades: list[BacktestTrade] = []
        current = start_date

        print(f"\nSimulating trades ({start_date} → {end_date})...")

        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            # Check exits for open trades first
            still_open = []
            for t in open_trades:
                closed = self._check_exit(t, current, all_bars)
                if closed:
                    trades.append(t)
                    store.save_backtest_trade(self.run_id, self._bt_to_dict(t))
                else:
                    still_open.append(t)
            open_trades = still_open

            # Evaluate signals for each stock (limit to 5 concurrent)
            if len(open_trades) < self.cfg["risk"].get("max_concurrent_positions", 5):
                for instr in instruments:
                    symbol = instr["symbol"]
                    if symbol not in all_bars:
                        continue
                    if any(t.symbol == symbol for t in open_trades):
                        continue

                    bars = all_bars[symbol]
                    ctx = self._build_backtest_ctx(instr, bars, current)
                    if ctx is None:
                        continue

                    signal = self.signal_agent.evaluate(ctx)
                    if signal is None:
                        continue

                    risk = RiskAgent(config=self.cfg)
                    decision = risk.evaluate(
                        signal=signal,
                        vix=ctx.vix,
                        open_positions=len(open_trades),
                        lot_size=instr["lot_size"],
                    )

                    if not decision.approved:
                        continue

                    # Calculate entry premium using Black-Scholes
                    expiry = self._select_expiry_for_date(current)
                    dte = (expiry - current).days
                    entry_premium = self.md.bs_price(
                        spot=ctx.spot,
                        strike=signal.strike,
                        dte=dte,
                        vol=ctx.hist_vol_30,
                        option_type=signal.option_type,
                    )

                    if entry_premium < 1.0:
                        continue

                    sl_price = round(entry_premium * 0.70, 2)
                    target_price = round(entry_premium * 2.50, 2)

                    bt = BacktestTrade(
                        run_id=self.run_id,
                        trade_date=current.isoformat(),
                        symbol=symbol,
                        strategy=signal.strategy,
                        direction=signal.direction,
                        strike=signal.strike,
                        option_type=signal.option_type,
                        expiry=expiry.isoformat(),
                        entry_price=entry_premium,
                        lots=decision.sized_lots,
                        lot_size=instr["lot_size"],
                        sl_price=sl_price,
                        target_price=target_price,
                    )
                    open_trades.append(bt)

                    if len(open_trades) >= self.cfg["risk"].get("max_concurrent_positions", 5):
                        break

            current += timedelta(days=1)

        # Force-close any remaining open trades at end date
        for t in open_trades:
            bars = all_bars.get(t.symbol)
            if bars is not None:
                last_bars = bars[bars["date"] <= pd.Timestamp(end_date)]
                if not last_bars.empty:
                    spot = float(last_bars.iloc[-1]["close"])
                    expiry = date.fromisoformat(t.expiry)
                    dte = max((expiry - end_date).days, 1)
                    bars_list = last_bars["close"].tolist()
                    vol = self.md._hist_vol(bars_list, 30)
                    ltp = self.md.bs_price(spot, t.strike, dte, vol, t.option_type)
                    t.exit_price = ltp
                    t.exit_date = end_date.isoformat()
                    t.realised_pnl = (ltp - t.entry_price) * t.lots * t.lot_size
                    t.exit_reason = "END_OF_PERIOD"
            trades.append(t)
            store.save_backtest_trade(self.run_id, self._bt_to_dict(t))

        print(f"Backtest complete: {len(trades)} trades simulated.")
        return trades

    # ── internals ──────────────────────────────────────────────────────────────

    def _check_exit(self, trade: BacktestTrade, today: date,
                     all_bars: dict[str, pd.DataFrame]) -> bool:
        """Update trade with exit if conditions met. Returns True if closed."""
        bars = all_bars.get(trade.symbol)
        if bars is None:
            return False

        today_bars = bars[bars["date"] <= pd.Timestamp(today)]
        if today_bars.empty:
            return False

        spot = float(today_bars.iloc[-1]["close"])
        expiry = date.fromisoformat(trade.expiry)
        dte = (expiry - today).days

        # Force exit at 7 DTE
        if dte <= 7:
            closes = today_bars["close"].tolist()
            vol = self.md._hist_vol(closes, 30)
            ltp = self.md.bs_price(spot, trade.strike, max(dte, 1), vol, trade.option_type)
            trade.exit_price = ltp
            trade.exit_date = today.isoformat()
            trade.realised_pnl = (ltp - trade.entry_price) * trade.lots * trade.lot_size
            trade.exit_reason = "EXPIRY_EXIT"
            return True

        # Simulate daily option price using BS
        closes = today_bars["close"].tolist()
        vol = self.md._hist_vol(closes, 30)
        ltp = self.md.bs_price(spot, trade.strike, dte, vol, trade.option_type)

        if ltp >= trade.target_price:
            trade.exit_price = ltp
            trade.exit_date = today.isoformat()
            trade.realised_pnl = (ltp - trade.entry_price) * trade.lots * trade.lot_size
            trade.exit_reason = "TARGET"
            return True

        if ltp <= trade.sl_price:
            trade.exit_price = ltp
            trade.exit_date = today.isoformat()
            trade.realised_pnl = (ltp - trade.entry_price) * trade.lots * trade.lot_size
            trade.exit_reason = "STOPLOSS"
            return True

        # Pre-earnings exit (exit 1 day before earnings if this was a pre-earnings drift trade)
        if trade.strategy == "Pre-Earnings Drift":
            # Approximate: exit if DTE < 20 (earnings likely imminent)
            if dte < 20:
                trade.exit_price = ltp
                trade.exit_date = today.isoformat()
                trade.realised_pnl = (ltp - trade.entry_price) * trade.lots * trade.lot_size
                trade.exit_reason = "PRE_EARNINGS_EXIT"
                return True

        return False

    def _build_backtest_ctx(self, instr: dict, bars: pd.DataFrame,
                              as_of: date) -> Optional[StockMarketContext]:
        """Build a StockMarketContext using only data available as of `as_of`."""
        symbol = instr["symbol"]
        past_bars = bars[bars["date"] <= pd.Timestamp(as_of)].copy()

        if len(past_bars) < 30:
            return None

        closes = past_bars["close"].tolist()
        volumes = past_bars["volume"].tolist()
        spot = closes[-1]

        ema_20 = self.md._ema(closes, 20)
        ema_50 = self.md._ema(closes, 50) if len(closes) >= 50 else ema_20
        rsi = self.md._rsi(closes, 14)
        vol_avg_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / max(len(volumes), 1)
        hist_vol = self.md._hist_vol(closes, 30)
        fii_trend = self.md._estimate_fii_trend(past_bars, self.master.sector(symbol))

        # Minimal option chain (empty — strategies adapt when chain is unavailable)
        # Build a synthetic chain for backtest pricing
        strike_gap = instr.get("strike_gap", 10)
        atm = round(spot / strike_gap) * strike_gap
        chain_rows = []
        for offset in range(-5, 6):
            k = atm + offset * strike_gap
            dte_approx = 25  # typical DTE during backtest
            ce_ltp = self.md.bs_price(spot, k, dte_approx, hist_vol, "CE")
            pe_ltp = self.md.bs_price(spot, k, dte_approx, hist_vol, "PE")
            chain_rows.append({
                "strike": k, "ce_ltp": ce_ltp, "pe_ltp": pe_ltp,
                "ce_oi": 100000, "pe_oi": 100000,
                "ce_sid": int(hash(f"{symbol}{k}CE") % 1_000_000 + 1),
                "pe_sid": int(hash(f"{symbol}{k}PE") % 1_000_000 + 1),
                "ce_iv": hist_vol * 100, "pe_iv": hist_vol * 100,
                "ce_volume": 50000, "pe_volume": 50000,
            })

        chain_df = pd.DataFrame(chain_rows)

        earnings_next = self._get_earnings_for_date(instr, as_of)

        return StockMarketContext(
            symbol=symbol,
            security_id=0,
            sector=self.master.sector(symbol),
            spot=spot,
            vix=15.0,  # approximate neutral VIX for backtest
            expiry=self._select_expiry_for_date(as_of),
            atm_strike=atm,
            strike_gap=strike_gap,
            bars_daily=past_bars,
            chain=chain_df,
            ema_20=ema_20,
            ema_50=ema_50,
            rsi_14=rsi,
            vol_avg_20=vol_avg_20,
            hist_vol_30=hist_vol,
            earnings_next=earnings_next,
            fii_sector_trend=fii_trend,
        )

    def _fetch_bars(self, instr: dict, from_date: date, to_date: date) -> Optional[pd.DataFrame]:
        try:
            sec_id, seg = self.master.equity_info(instr["symbol"])
        except KeyError:
            return None
        return self.md._fetch_daily_bars(
            str(sec_id), seg,
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
        )

    @staticmethod
    def _select_expiry_for_date(as_of: date) -> date:
        """Select monthly expiry (last Thursday) with ~25 DTE from the given date."""
        for months_ahead in range(1, 4):
            year = as_of.year
            month = as_of.month + months_ahead
            if month > 12:
                month -= 12
                year += 1
            last_day = monthrange(year, month)[1]
            candidate = date(year, month, last_day)
            while candidate.weekday() != 3:
                candidate -= timedelta(days=1)
            dte = (candidate - as_of).days
            if 20 <= dte <= 40:
                return candidate
        # Default: 2 months out
        year = as_of.year
        month = as_of.month + 2
        if month > 12:
            month -= 12
            year += 1
        last_day = monthrange(year, month)[1]
        candidate = date(year, month, last_day)
        while candidate.weekday() != 3:
            candidate -= timedelta(days=1)
        return candidate

    @staticmethod
    def _get_earnings_for_date(instr: dict, as_of: date) -> Optional[date]:
        months = instr.get("earnings_months", [])
        for ahead in range(0, 4):
            m = (as_of.month + ahead - 1) % 12 + 1
            y = as_of.year + (as_of.month + ahead - 1) // 12
            if m in months:
                candidate = date(y, m, 15)
                if candidate > as_of + timedelta(days=5):
                    return candidate
        return None

    @staticmethod
    def _bt_to_dict(t: BacktestTrade) -> dict:
        return {
            "trade_date": t.trade_date, "symbol": t.symbol,
            "strategy": t.strategy, "direction": t.direction,
            "strike": t.strike, "option_type": t.option_type,
            "expiry": t.expiry, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "exit_date": t.exit_date,
            "realised_pnl": t.realised_pnl, "exit_reason": t.exit_reason,
            "lots": t.lots, "lot_size": t.lot_size,
            "sl_price": t.sl_price, "target_price": t.target_price,
        }
