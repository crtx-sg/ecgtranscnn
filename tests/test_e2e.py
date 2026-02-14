"""End-to-end smoke test: generate data, train briefly, evaluate."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_transcovnet import (
    ECGTransCovNet,
    FocalLoss,
    NUM_CLASSES,
    SIGNAL_LENGTH,
    ALL_LEADS,
)
from ecg_transcovnet.data import generate_dataset, AugmentedECGDataset
from ecg_transcovnet.training import train_one_epoch, validate, evaluate_detailed


def test_e2e_pipeline():
    """Smoke test: generate small dataset, train 2 epochs, evaluate."""
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    leads = ALL_LEADS
    in_channels = len(leads)

    # Generate tiny dataset
    train_X, train_y = generate_dataset(32, leads, noise_level="clean", seed=42)
    val_X, val_y = generate_dataset(16, leads, noise_level="clean", seed=99)

    assert train_X.shape == (32, in_channels, SIGNAL_LENGTH)
    assert train_y.shape == (32,)

    # Use augmented dataset for training
    train_ds_raw = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    train_ds = AugmentedECGDataset(train_ds_raw)
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Build model
    model = ECGTransCovNet(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        signal_length=SIGNAL_LENGTH,
        embed_dim=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    # Per-class alpha weights
    class_counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.mean()
    alpha_tensor = torch.from_numpy(class_weights)
    loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train 2 epochs
    for _ in range(2):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler=None)
        assert tr_loss >= 0
        assert 0 <= tr_acc <= 1

    # Validate
    vl_loss, vl_acc = validate(model, val_loader, loss_fn, device)
    assert vl_loss >= 0
    assert 0 <= vl_acc <= 1

    # Detailed evaluation
    macro, per_class, cm = evaluate_detailed(model, val_loader, device)
    assert "accuracy" in macro
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert len(per_class) == NUM_CLASSES


def test_augmented_dataset():
    """Test that AugmentedECGDataset produces correct shapes and differs from original."""
    torch.manual_seed(0)
    signals = torch.randn(10, 7, SIGNAL_LENGTH)
    labels = torch.randint(0, NUM_CLASSES, (10,))
    raw_ds = TensorDataset(signals, labels)
    aug_ds = AugmentedECGDataset(raw_ds)

    assert len(aug_ds) == 10

    sig_aug, lbl_aug = aug_ds[0]
    sig_raw, lbl_raw = raw_ds[0]

    # Shape preserved
    assert sig_aug.shape == sig_raw.shape
    # Label unchanged
    assert lbl_aug == lbl_raw
    # Signal should differ (augmentation applied)
    assert not torch.allclose(sig_aug, sig_raw, atol=1e-6)
