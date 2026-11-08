"""Remaining Useful Life prediction on NASA C-MAPSS."""

import os
import sys

# On macOS the PyTorch and XGBoost wheels each bring their own OpenMP runtime.
# When both run multi-threaded in one process (the test suite, the two-stage
# hybrid), whichever starts second segfaults ("Python quit unexpectedly").
# One OpenMP thread avoids it and costs little for these small models (an LSTM
# epoch on FD001 takes about 1.1 s instead of 1.0 s). To override, set
# OMP_NUM_THREADS yourself before importing rul.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
