"""Parameter block helpers for the VARA model."""

from __future__ import annotations

from typing import Dict, List


def make_block_order(p: int) -> Dict[str, List[int]]:
    """Return theta block indices for ``(eta_a, gamma_a, beta, sigma2)``."""
    return {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }
