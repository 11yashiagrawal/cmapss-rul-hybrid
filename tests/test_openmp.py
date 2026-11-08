"""PyTorch and XGBoost must be able to train in the same process."""

import os
import subprocess
import sys

SCRIPT = """
import rul
import numpy as np
import torch
import xgboost
from torch import nn

lstm = nn.LSTM(15, 64, num_layers=2, batch_first=True)
x = torch.randn(512, 30, 15)
lstm(x)[0].sum().backward()
xgboost.XGBRegressor(n_estimators=20).fit(np.random.rand(3000, 20), np.random.rand(3000))
lstm(x)[0].sum().backward()
print("ok")
"""


def test_torch_and_xgboost_train_in_one_process():
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"
