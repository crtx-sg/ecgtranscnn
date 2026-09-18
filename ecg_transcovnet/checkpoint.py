"""Checkpoint loading, model construction and warm start shared by all scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .classes import ClassSpec
from .constants import ALL_LEADS, SIGNAL_LENGTH
from .model import ECGTransCovNet

_ARCH_DEFAULTS = {
    "embed_dim": 128,
    "nhead": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,
}


def arch_kwargs(saved_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Architecture hyper-parameters from saved args, with model defaults."""
    saved_args = saved_args or {}
    return {
        k: saved_args[k] if saved_args.get(k) is not None else default
        for k, default in _ARCH_DEFAULTS.items()
    }


def build_model(
    spec: ClassSpec, in_channels: int, saved_args: Mapping[str, Any] | None = None,
) -> ECGTransCovNet:
    return ECGTransCovNet(
        num_classes=len(spec),
        in_channels=in_channels,
        signal_length=SIGNAL_LENGTH,
        **arch_kwargs(saved_args),
    )


def checkpoint_filter_preset(ckpt: Mapping[str, Any]) -> str:
    """Filter preset a checkpoint was trained with (legacy checkpoints: ``none``)."""
    preset = ckpt.get("filter_preset") or (ckpt.get("args") or {}).get("filter_preset")
    return preset or "none"


@dataclass
class LoadedModel:
    model: ECGTransCovNet
    class_spec: ClassSpec
    leads: list[str]
    filter_preset: str
    checkpoint: dict


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, weights_only=False, map_location=device)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> LoadedModel:
    """Build a model sized to the checkpoint's head and load its weights."""
    ckpt = load_checkpoint(path, device)
    spec = ClassSpec.from_checkpoint(ckpt)
    leads = list(ckpt.get("leads") or ALL_LEADS)
    model = build_model(spec, len(leads), ckpt.get("args")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return LoadedModel(model, spec, leads, checkpoint_filter_preset(ckpt), ckpt)


class EnsembleModel(torch.nn.Module):
    """Averages the softmax outputs of several models that share one head.

    Returns log-probabilities, so a caller applying ``softmax`` to the output
    recovers the averaged probabilities unchanged — the same combination rule
    as :func:`ecg_transcovnet.package_eval.combine_predictions`.
    """

    def __init__(self, models: Sequence[ECGTransCovNet]):
        super().__init__()
        self.members = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([F.softmax(m(x).float(), dim=-1) for m in self.members])
        return probs.mean(0).clamp_min(torch.finfo(torch.float32).tiny).log()


def load_models(
    paths: Sequence[str | Path], device: torch.device | str = "cpu",
) -> LoadedModel:
    """Load one checkpoint, or several as a softmax-averaging ensemble.

    Every member must share the class head, leads and filter preset; the
    returned :class:`LoadedModel` carries the first member's metadata.
    """
    loaded = [load_model(p, device) for p in paths]
    ref = loaded[0]
    for path, lm in zip(paths[1:], loaded[1:]):
        if list(lm.class_spec.names) != list(ref.class_spec.names):
            raise ValueError(f"{path}: class head differs from {paths[0]}")
        if lm.leads != ref.leads:
            raise ValueError(f"{path}: leads differ from {paths[0]}")
        if lm.filter_preset != ref.filter_preset:
            raise ValueError(f"{path}: filter preset differs from {paths[0]}")
    if len(loaded) == 1:
        return ref
    model = EnsembleModel([lm.model for lm in loaded]).to(device)
    model.eval()
    return LoadedModel(model, ref.class_spec, ref.leads, ref.filter_preset, ref.checkpoint)


def warm_start(
    model: ECGTransCovNet,
    source_ckpt: Mapping[str, Any],
    target_spec: ClassSpec,
    init_queries: str = "by_name",
) -> dict[str, Any]:
    """Load compatible weights from *source_ckpt* into *model*.

    Every tensor whose name and shape match is copied.  Object queries are
    per class: with ``init_queries="by_name"`` the rows of classes present in
    both heads are copied and the rest keep their fresh initialisation.
    With ``"reinit"`` the queries and the FFN head are re-initialised when
    the heads differ.
    """
    if init_queries not in ("by_name", "reinit"):
        raise ValueError(f"init_queries must be 'by_name' or 'reinit', got {init_queries!r}")
    src_state = source_ckpt["model_state_dict"]
    src_spec = ClassSpec.from_checkpoint(source_ckpt)
    same_head = src_spec.names == target_spec.names
    dst_state = model.state_dict()

    new_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in src_state.items():
        if key == "object_queries":
            continue
        if not same_head and init_queries == "reinit" and key.startswith("ffn_head."):
            skipped.append(key)
            continue
        if key in dst_state and dst_state[key].shape == value.shape:
            new_state[key] = value
        else:
            skipped.append(key)

    copied: list[str] = []
    src_q = src_state.get("object_queries")
    if src_q is not None:
        queries = dst_state["object_queries"].clone()
        if same_head and src_q.shape == queries.shape:
            queries = src_q.clone()
            copied = list(target_spec.names)
        elif init_queries == "by_name" and src_q.shape[2] == queries.shape[2]:
            for i, name in enumerate(target_spec.names):
                j = src_spec.get_index(name)
                if j is not None:
                    queries[0, i] = src_q[0, j]
                    copied.append(name)
        new_state["object_queries"] = queries

    result = model.load_state_dict(new_state, strict=False)
    return {
        "loaded_tensors": len(new_state),
        "skipped": skipped,
        "queries_copied": copied,
        "missing": list(result.missing_keys),
    }
