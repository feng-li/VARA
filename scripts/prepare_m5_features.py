#!/usr/bin/env python3
"""Compatibility wrapper for preparing M5 VARA attention features."""

from __future__ import annotations

try:
    from prepare_m5_attention_features import main
except ModuleNotFoundError:
    from scripts.prepare_m5_attention_features import main


if __name__ == "__main__":
    main()
