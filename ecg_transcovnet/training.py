"""Training utilities for ECG-TransCovNet."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .constants import NUM_CLASSES, CLASS_NAMES


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
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """Per-class precision / recall / specificity / F1 + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        preds = model(X).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro = defaultdict(float)
    per_class = {}
    for i in range(NUM_CLASSES):
        tp = int(((all_preds == i) & (all_labels == i)).sum())
        fp = int(((all_preds == i) & (all_labels != i)).sum())
        fn = int(((all_preds != i) & (all_labels == i)).sum())
        tn = int(((all_preds != i) & (all_labels != i)).sum())

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        per_class[CLASS_NAMES[i]] = dict(
            precision=prec, recall=rec, specificity=spec, f1=f1, support=tp + fn,
        )
        for k, v in [("precision", prec), ("recall", rec), ("specificity", spec), ("f1", f1)]:
            macro[k] += v

    macro = {k: v / NUM_CLASSES for k, v in macro.items()}
    macro["accuracy"] = float((all_preds == all_labels).mean())

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    return dict(macro), per_class, cm
