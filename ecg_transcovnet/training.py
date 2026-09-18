"""Training utilities for ECG-TransCovNet."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .constants import CLASS_NAMES
from .evaluation import compute_metrics


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, loss_fn, device, return_predictions: bool = False):
    """Validation loss and accuracy; optionally also ``(labels, predictions)``."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    labels, preds = [], []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if return_predictions:
            labels.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
    if return_predictions:
        return total_loss / total, correct / total, np.concatenate(labels), np.concatenate(preds)
    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels over a loader, in loader order."""
    model.eval()
    logits, labels = [], []
    for X, y in loader:
        logits.append(model(X.to(device, non_blocking=True)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


@torch.no_grad()
def evaluate_detailed(model, loader, device, class_names: Sequence[str] | None = None):
    """Per-class precision / recall / specificity / F1 + confusion matrix.

    The confusion matrix is sized to *class_names* (default: the 16-class
    simulator head).  Macro averages use classes present in the data.
    """
    names = list(class_names) if class_names is not None else CLASS_NAMES
    logits, labels = predict(model, loader, device)
    m = compute_metrics(labels, logits.argmax(1), names)
    macro = dict(m["macro"])
    per_class = {n: dict(m["per_class"][n]) for n in names}
    return macro, per_class, np.asarray(m["confusion_matrix"])
