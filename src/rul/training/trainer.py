"""Shared PyTorch training loop for every Phase 2 sequence model.

Every model is trained the same way: MSE on the (capped) RUL, Adam, gradient
clipping and a learning-rate schedule that does not depend on validation
data, so a refit for a fixed number of epochs follows exactly the same
schedule as the cross-validation run it copies.

Targets are divided by ``target_scale`` during training (so outputs stay near
1); predictions and validation RMSE are always reported in cycles.
"""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCHEDULERS = ("cosine", "none")
LOSSES = {"mse": nn.MSELoss}

Inputs = np.ndarray | Sequence[np.ndarray]


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings shared by every sequence model.

    Args:
        epochs: Maximum epochs (early stopping may end sooner).
        batch_size: Windows per optimiser step.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 penalty.
        input_noise_std: Gaussian noise standard deviation in scaled input units,
            added to training batches only. Zero disables augmentation.
        grad_clip: Max gradient norm, or ``None`` for no clipping.
        scheduler: ``"cosine"`` (annealed over ``epochs``) or ``"none"``.
        patience: Epochs without a better validation RMSE before stopping.
        loss: Training loss; ``"mse"``.
        device: Torch device. Reported runs use ``"cpu"``.
        seed: Seed for weight initialisation, dropout and batch order.
    """

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    scheduler: str = "cosine"
    patience: int = 10
    loss: str = "mse"
    device: str = "cpu"
    seed: int = 42
    input_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {tuple(LOSSES)}, got {self.loss!r}")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if not np.isfinite(self.input_noise_std) or self.input_noise_std < 0:
            raise ValueError("input_noise_std must be finite and non-negative")

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        """Build from a config mapping; unknown keys raise ``TypeError``."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise TypeError(f"unknown training settings: {unknown}")
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitResult:
    """What happened during one call to :func:`fit`.

    ``history`` has one entry per epoch: ``epoch``, ``train_loss`` (on the
    scaled target), ``val_rmse`` (cycles, or ``None`` without validation) and
    ``lr``. ``best_epoch`` is 1-based.
    """

    best_epoch: int
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. Call it before building a model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_tensors(inputs: Inputs) -> list[torch.Tensor]:
    arrays = [inputs] if isinstance(inputs, np.ndarray) else list(inputs)
    return [torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in arrays]


def _check_rows(tensors: list[torch.Tensor], y: torch.Tensor | None = None) -> int:
    n = len(tensors[0])
    if any(len(t) != n for t in tensors) or (y is not None and len(y) != n):
        lengths = [len(t) for t in tensors] + ([len(y)] if y is not None else [])
        raise ValueError(f"inputs and targets must have the same rows, got {lengths}")
    return n


def predict(
    model: nn.Module,
    inputs: Inputs,
    *,
    target_scale: float = 1.0,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """Model outputs for ``inputs``, in cycles (``output * target_scale``)."""
    tensors = _as_tensors(inputs)
    n = _check_rows(tensors)
    model.to(device).eval()
    with torch.no_grad():
        out = [
            model(*(t[i : i + batch_size].to(device) for t in tensors)).cpu()
            for i in range(0, n, batch_size)
        ]
    return torch.cat(out).numpy().astype(float) * target_scale


def fit(
    model: nn.Module,
    inputs: Inputs,
    y: np.ndarray,
    config: TrainConfig,
    *,
    validation: tuple[Inputs, np.ndarray] | None = None,
    epochs: int | None = None,
    target_scale: float = 1.0,
) -> FitResult:
    """Train ``model`` in place and return its history.

    With ``validation``, training stops ``config.patience`` epochs after the
    best validation RMSE and the model keeps the best epoch's weights. Without
    it, the model trains for exactly ``epochs`` (default ``config.epochs``).

    Args:
        model: Module whose ``forward`` takes one tensor per input array and
            returns ``[batch]``.
        inputs: One array, or several with the same number of rows.
        y: Targets in cycles.
        config: Optimisation settings.
        validation: ``(inputs, y)`` used for early stopping.
        epochs: Overrides ``config.epochs`` (used by the final refit).
        target_scale: Targets are divided by this during training.
    """
    tensors = _as_tensors(inputs)
    targets = torch.as_tensor(np.asarray(y, dtype=np.float32)) / target_scale
    _check_rows(tensors, targets)
    n_epochs = config.epochs if epochs is None else epochs
    device = torch.device(config.device)
    model.to(device)

    generator = torch.Generator().manual_seed(config.seed)
    noise_generator = (
        torch.Generator(device=device).manual_seed(config.seed)
        if config.input_noise_std else None
    )
    loader = DataLoader(
        TensorDataset(*tensors, targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine"
        else None
    )
    loss_fn = LOSSES[config.loss]()

    start = time.perf_counter()
    history: list[dict] = []
    best_rmse, best_epoch, best_state = float("inf"), n_epochs, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for *batch, target in loader:
            batch = [b.to(device) for b in batch]
            if noise_generator is not None:
                batch = [
                    b + config.input_noise_std * torch.randn(
                        b.shape, dtype=b.dtype, device=device, generator=noise_generator
                    )
                    for b in batch
                ]
            target = target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(*batch), target)
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total += loss.item() * len(target)
            seen += len(target)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        val_rmse = None
        if validation is not None:
            val_pred = predict(
                model, validation[0], target_scale=target_scale, device=config.device
            )
            val_rmse = float(np.sqrt(np.mean((val_pred - np.asarray(validation[1])) ** 2)))
        history.append(
            {"epoch": epoch, "train_loss": total / seen, "val_rmse": val_rmse, "lr": lr}
        )

        if validation is not None:
            if val_rmse < best_rmse:
                best_rmse, best_epoch = val_rmse, epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(best_epoch=best_epoch, history=history, seconds=time.perf_counter() - start)
