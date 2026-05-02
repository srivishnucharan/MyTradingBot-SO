"""run_paper.py — Launch the bot in PAPER mode (simulated fills, real market data)."""
import sys, logging
from pathlib import Path

sys.path.insert(0, ".")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/paper.log")],
)

from core.orchestrator import Orchestrator

if __name__ == "__main__":
    orch = Orchestrator(config_path="config/config.yaml", mode="PAPER")
    orch.run_forever()
