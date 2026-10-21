"""Send log output from the test run to logs/test_run/ so the evidence logs stay clean."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PA3_LOG_DIR", str(ROOT / "logs" / "test_run"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(ROOT))
