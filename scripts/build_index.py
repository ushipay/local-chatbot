"""FAISSインデックスを構築するスクリプト。

Usage:
    uv run python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import build_index

if __name__ == "__main__":
    build_index()
