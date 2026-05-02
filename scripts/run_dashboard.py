"""run_dashboard.py — Start the web dashboard on port 8002."""
import sys, logging
from pathlib import Path

sys.path.insert(0, ".")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from data import store
store.init_db()

import uvicorn
uvicorn.run("dashboard.api:app", host="0.0.0.0", port=8002, reload=False)
