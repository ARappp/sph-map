import sys; from pathlib import Path
_root = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(_root), str(_root / "lib")]
from config import INPUT, WORK, step
