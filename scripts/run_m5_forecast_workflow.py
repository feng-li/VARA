#!/usr/bin/env python3
"""Compatibility entry point for the generic M5 forecast workflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_forecast_workflow import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv[1:1] = ["--dataset", "m5"]
    main()
