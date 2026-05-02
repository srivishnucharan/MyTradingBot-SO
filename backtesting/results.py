"""
backtesting/results.py
Calculates and pretty-prints backtest performance metrics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from backtesting.engine import BacktestTrade


def compute_results(trades: List[BacktestTrade]) -> dict:
    closed = [t for t in trades if t.exit_reason != "OPEN"]
    if not closed:
        return {}

    pnls = [t.realised_pnl for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(closed) * 100
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Strategy breakdown
    by_strategy = defaultdict(list)
    for t in closed:
        by_strategy[t.strategy].append(t.realised_pnl)

    strat_stats = {}
    for strat, ps in sorted(by_strategy.items()):
        ws = [p for p in ps if p > 0]
        strat_stats[strat] = {
            "trades": len(ps),
            "wins": len(ws),
            "win_rate": round(len(ws) / len(ps) * 100, 1),
            "total_pnl": round(sum(ps), 0),
            "avg_pnl": round(sum(ps) / len(ps), 0),
            "best": round(max(ps), 0),
            "worst": round(min(ps), 0),
        }

    # Max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in sorted(pnls):
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    # By exit reason
    by_exit = defaultdict(int)
    for t in closed:
        by_exit[t.exit_reason] += 1

    return {
        "run_id": closed[0].run_id if closed else "",
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 0),
        "avg_pnl": round(total_pnl / len(closed), 0),
        "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0),
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        "best_trade": round(max(pnls), 0),
        "worst_trade": round(min(pnls), 0),
        "max_drawdown": round(max_dd, 0),
        "by_strategy": strat_stats,
        "by_exit": dict(by_exit),
    }


def print_results(results: dict):
    if not results:
        print("No closed trades to report.")
        return

    SEP = "═" * 72
    sep = "─" * 72

    print(f"\n{SEP}")
    print("  MyTradingBot-SO │ Nifty50 Stock Options │ 12-Month Backtest Results")
    print(SEP)
    print(f"  Run ID : {results['run_id']}")
    print(f"  Trades : {results['total_trades']}  │  Wins: {results['wins']}  Losses: {results['losses']}")
    print(f"  Win Rate         : {results['win_rate']}%")
    print(f"  Total P&L        : ₹{results['total_pnl']:>+,.0f}")
    print(f"  Avg P&L/Trade    : ₹{results['avg_pnl']:>+,.0f}")
    print(f"  Avg Win          : ₹{results['avg_win']:>+,.0f}")
    print(f"  Avg Loss         : ₹{results['avg_loss']:>+,.0f}")
    print(f"  Profit Factor    : {results['profit_factor']:.2f}x")
    print(f"  Best Trade       : ₹{results['best_trade']:>+,.0f}")
    print(f"  Worst Trade      : ₹{results['worst_trade']:>+,.0f}")
    print(f"  Max Drawdown     : ₹{results['max_drawdown']:>,.0f}")

    print(f"\n{sep}")
    print("  BY STRATEGY")
    print(sep)
    header = f"  {'Strategy':<22} {'Trades':>6} {'Win%':>6} {'TotalP&L':>12} {'AvgP&L':>10} {'Best':>10} {'Worst':>10}"
    print(header)
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for strat, s in results["by_strategy"].items():
        print(
            f"  {strat:<22} {s['trades']:>6} {s['win_rate']:>5.1f}%"
            f" {s['total_pnl']:>+12,.0f} {s['avg_pnl']:>+10,.0f}"
            f" {s['best']:>+10,.0f} {s['worst']:>+10,.0f}"
        )

    print(f"\n{sep}")
    print("  BY EXIT REASON")
    print(sep)
    for reason, count in sorted(results["by_exit"].items()):
        print(f"  {reason:<25} {count:>5} trades")

    print(SEP)
    pnl = results["total_pnl"]
    verdict = "PROFITABLE ✓" if pnl > 0 else "LOSS-MAKING ✗"
    print(f"  VERDICT: {verdict} │ Net P&L: ₹{pnl:>+,.0f}")
    print(SEP)
    print()
