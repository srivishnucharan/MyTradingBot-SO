"""
run_backtest.py — Run a 12-month backtest on all enabled Nifty50 stock options strategies.

Usage:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --months 6
"""
import sys, logging
from pathlib import Path

sys.path.insert(0, ".")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/backtest.log")],
)

if __name__ == "__main__":
    months = 12
    for i, arg in enumerate(sys.argv):
        if arg == "--months" and i + 1 < len(sys.argv):
            months = int(sys.argv[i + 1])

    print(f"\nRunning {months}-month backtest for Nifty50 Stock Options strategies...")

    from backtesting.engine import BacktestEngine
    from backtesting.results import compute_results, print_results

    engine = BacktestEngine(config_path="config/config.yaml")
    trades = engine.run(months=months)
    results = compute_results(trades)
    print_results(results)
