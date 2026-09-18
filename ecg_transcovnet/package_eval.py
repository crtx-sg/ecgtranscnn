"""Evaluate one or more models on a split of an ``ecgpkg`` package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluation import build_report, map_probs_to_head
from .package import LengthBucketSampler, PackageDataset


@dataclass
class EvalModel:
    model: torch.nn.Module
    class_names: list[str]
    name: str = "model"


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    dataset: PackageDataset,
    device: torch.device,
    batch_size: int = 128,
    workers: int = 4,
) -> np.ndarray:
    """Softmax outputs in dataset index order (batches grouped by length)."""
    model.eval()
    sampler = LengthBucketSampler(dataset, batch_size)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    for X, _ in loader:
        chunks.append(F.softmax(model(X.to(device, non_blocking=True)).float(), dim=-1).cpu().numpy())
    order = np.fromiter((i for b in sampler.batches for i in b), dtype=np.int64)
    probs = np.concatenate(chunks)
    out = np.empty_like(probs)
    out[order] = probs
    return out


def combine_predictions(
    member_probs: Sequence[tuple[np.ndarray, Sequence[str]]], head: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Average ensemble members on the package head.

    A single member with a different head is mapped by class name
    (baseline mode; predictions outside the head become ``-1``).
    Returns ``(probs, pred, baseline_mode)``.
    """
    head = list(head)
    if len(member_probs) == 1 and list(member_probs[0][1]) != head:
        probs, pred = map_probs_to_head(member_probs[0][0], member_probs[0][1], head)
        return probs, pred, True
    for _, names in member_probs:
        if list(names) != head:
            raise ValueError("ensemble members must all use the package head")
    probs = np.mean([p for p, _ in member_probs], axis=0)
    return probs, probs.argmax(axis=1), False


def evaluate_split(
    models: Sequence[EvalModel],
    dataset: PackageDataset,
    device: torch.device,
    batch_size: int = 128,
    workers: int = 4,
    bias: np.ndarray | None = None,
    rows_mask: np.ndarray | None = None,
) -> dict:
    """Predict with *models* on *dataset* and build the grouped report.

    *bias* is added to ``log(probs)`` before the arg-max (ignored in baseline
    mode).  *rows_mask* restricts the report to a subset of rows.
    """
    head = list(dataset.package.spec.names)
    member_probs = [
        (predict_probs(m.model, dataset, device, batch_size, workers), m.class_names) for m in models
    ]
    probs, pred, baseline = combine_predictions(member_probs, head)
    if bias is not None and not baseline:
        pred = (np.log(probs + 1e-9) + bias).argmax(axis=1)
    labels = np.asarray(dataset.labels)
    sel = np.arange(len(labels)) if rows_mask is None else np.flatnonzero(rows_mask)
    report = build_report(labels[sel], pred[sel], head, [dataset.rows[i] for i in sel], probs[sel],
                          eval_flags=dataset.package.eval_flags,
                          reporting=dataset.package.reporting)
    return {"report": report, "probs": probs, "pred": pred, "labels": labels, "baseline_mode": baseline}
