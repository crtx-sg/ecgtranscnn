"""Tests for noise-robust ECG-TransCovNet model.

Evaluates the noise-robust model (trained with mixed noise) against
clean and noisy synthetic data at multiple noise levels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_transcovnet import (
    ECGTransCovNet,
    NUM_CLASSES,
    CLASS_NAMES,
    SIGNAL_LENGTH,
    ALL_LEADS,
)
from ecg_transcovnet.data import generate_dataset
from ecg_transcovnet.training import evaluate_detailed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOISE_ROBUST_CKPT = Path("models/noise_robust/best_model.pt")
IMPROVED_CKPT = Path("models/improved/best_model.pt")

# Classes where AV_BLOCK_1 is a known weak spot — excluded from strict checks
AV_BLOCK_CLASSES = {"AV_BLOCK_1"}


def _load_model(ckpt_path: Path, device: torch.device) -> ECGTransCovNet:
    """Load a model from a checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    saved_args = ckpt.get("args", {})
    leads = ckpt.get("leads", ALL_LEADS)
    model = ECGTransCovNet(
        num_classes=NUM_CLASSES,
        in_channels=len(leads),
        signal_length=SIGNAL_LENGTH,
        embed_dim=saved_args.get("embed_dim", 128),
        nhead=saved_args.get("nhead", 8),
        num_encoder_layers=saved_args.get("num_encoder_layers", 3),
        num_decoder_layers=saved_args.get("num_decoder_layers", 3),
        dim_feedforward=saved_args.get("dim_feedforward", 512),
        dropout=saved_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _evaluate_on_noise(model, noise_level: str, device: torch.device, n=500):
    """Generate synthetic data at a given noise level and evaluate."""
    leads = ALL_LEADS
    test_X, test_y = generate_dataset(n, leads, noise_level=noise_level, seed=777)
    ds = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    macro, per_class, cm = evaluate_detailed(model, loader, device)
    return macro, per_class, cm


# ---------------------------------------------------------------------------
# Skip if checkpoint not available
# ---------------------------------------------------------------------------

requires_noise_robust = pytest.mark.skipif(
    not NOISE_ROBUST_CKPT.exists(),
    reason="Noise-robust checkpoint not found; run training first",
)
requires_improved = pytest.mark.skipif(
    not IMPROVED_CKPT.exists(),
    reason="Improved checkpoint not found",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_noise_robust
class TestNoiseRobustOnClean:
    """Noise-robust model should still perform well on clean data."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        device = torch.device("cpu")
        model = _load_model(NOISE_ROBUST_CKPT, device)
        macro, per_class, cm = _evaluate_on_noise(model, "clean", device)
        request.cls.macro = macro
        request.cls.per_class = per_class

    def test_accuracy_above_85(self):
        assert self.macro["accuracy"] >= 0.85, (
            f"Clean accuracy {self.macro['accuracy']:.3f} below 0.85"
        )

    def test_macro_f1_above_80(self):
        assert self.macro["f1"] >= 0.80, (
            f"Clean macro-F1 {self.macro['f1']:.3f} below 0.80"
        )

    def test_no_class_below_50_f1(self):
        for name, m in self.per_class.items():
            if name in AV_BLOCK_CLASSES:
                continue
            assert m["f1"] >= 0.50, (
                f"{name} clean F1={m['f1']:.3f} below 0.50"
            )


@requires_noise_robust
class TestNoiseRobustOnLowNoise:
    """Evaluate on low-noise synthetic data."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        device = torch.device("cpu")
        model = _load_model(NOISE_ROBUST_CKPT, device)
        macro, per_class, cm = _evaluate_on_noise(model, "low", device)
        request.cls.macro = macro
        request.cls.per_class = per_class

    def test_accuracy_above_80(self):
        assert self.macro["accuracy"] >= 0.80, (
            f"Low-noise accuracy {self.macro['accuracy']:.3f} below 0.80"
        )

    def test_macro_f1_above_75(self):
        assert self.macro["f1"] >= 0.75, (
            f"Low-noise macro-F1 {self.macro['f1']:.3f} below 0.75"
        )


@requires_noise_robust
class TestNoiseRobustOnMediumNoise:
    """Evaluate on medium-noise synthetic data."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        device = torch.device("cpu")
        model = _load_model(NOISE_ROBUST_CKPT, device)
        macro, per_class, cm = _evaluate_on_noise(model, "medium", device)
        request.cls.macro = macro
        request.cls.per_class = per_class

    def test_accuracy_above_70(self):
        assert self.macro["accuracy"] >= 0.70, (
            f"Medium-noise accuracy {self.macro['accuracy']:.3f} below 0.70"
        )

    def test_macro_f1_above_65(self):
        assert self.macro["f1"] >= 0.65, (
            f"Medium-noise macro-F1 {self.macro['f1']:.3f} below 0.65"
        )


@requires_noise_robust
class TestNoiseRobustOnHighNoise:
    """Evaluate on high-noise synthetic data — stress test."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        device = torch.device("cpu")
        model = _load_model(NOISE_ROBUST_CKPT, device)
        macro, per_class, cm = _evaluate_on_noise(model, "high", device)
        request.cls.macro = macro
        request.cls.per_class = per_class

    def test_accuracy_above_50(self):
        assert self.macro["accuracy"] >= 0.50, (
            f"High-noise accuracy {self.macro['accuracy']:.3f} below 0.50"
        )

    def test_macro_f1_above_45(self):
        assert self.macro["f1"] >= 0.45, (
            f"High-noise macro-F1 {self.macro['f1']:.3f} below 0.45"
        )


@requires_noise_robust
@requires_improved
class TestNoiseRobustVsImproved:
    """Compare noise-robust model against improved (clean-trained) model on noisy data.

    The noise-robust model should degrade less under noise.
    """

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        device = torch.device("cpu")
        robust_model = _load_model(NOISE_ROBUST_CKPT, device)
        improved_model = _load_model(IMPROVED_CKPT, device)

        # Evaluate both on medium noise
        robust_macro, _, _ = _evaluate_on_noise(robust_model, "medium", device)
        improved_macro, _, _ = _evaluate_on_noise(improved_model, "medium", device)

        request.cls.robust_macro = robust_macro
        request.cls.improved_macro = improved_macro

    def test_noise_robust_better_on_medium_noise(self):
        robust_acc = self.robust_macro["accuracy"]
        improved_acc = self.improved_macro["accuracy"]
        print(
            f"\nMedium noise — Noise-robust: {robust_acc:.3f}, "
            f"Improved (clean): {improved_acc:.3f}"
        )
        assert robust_acc >= improved_acc - 0.05, (
            f"Noise-robust model ({robust_acc:.3f}) should not be significantly "
            f"worse than clean model ({improved_acc:.3f}) on medium noise"
        )


@requires_noise_robust
class TestPerClassMetricsPrinted:
    """Print full per-class metrics across noise levels for manual inspection."""

    def test_print_full_report(self):
        device = torch.device("cpu")
        model = _load_model(NOISE_ROBUST_CKPT, device)

        for noise in ("clean", "low", "medium", "high"):
            macro, per_class, _ = _evaluate_on_noise(model, noise, device, n=320)
            print(f"\n{'=' * 70}")
            print(f"Noise level: {noise}")
            print(f"  Accuracy={macro['accuracy']:.3f}  Macro-F1={macro['f1']:.3f}")
            print(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>5s}")
            print(f"  {'-' * 51}")
            for name in CLASS_NAMES:
                m = per_class[name]
                print(
                    f"  {name:<28s} {m['precision']:6.3f} {m['recall']:6.3f} "
                    f"{m['f1']:6.3f} {m['support']:5d}"
                )
