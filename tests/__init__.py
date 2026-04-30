"""Test package for VARA.

Keep JAX on CPU in tests so discovery does not try to initialize unavailable
GPU plugins on shared machines.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
