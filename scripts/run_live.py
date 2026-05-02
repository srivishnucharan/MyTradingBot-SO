"""run_live.py — Launch the bot in LIVE mode (real orders on Dhan)."""
import sys, logging
from pathlib import Path

sys.path.insert(0, ".")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/live.log")],
)

log = logging.getLogger(__name__)

if "--no-confirm" not in sys.argv:
    print("\n" + "=" * 60)
    print("  MyTradingBot-SO — LIVE MODE")
    print("=" * 60)
    print("This will place REAL options orders on your Dhan account.")
    print("Capital at risk: see config/config.yaml → risk.capital")
    confirm = input("\nType 'YES' to proceed: ").strip()
    if confirm != "YES":
        print("Aborted.")
        sys.exit(0)

from core.orchestrator import Orchestrator

if __name__ == "__main__":
    log.info("Starting orchestrator in LIVE mode")
    orch = Orchestrator(config_path="config/config.yaml", mode="LIVE")
    orch.run_forever()
