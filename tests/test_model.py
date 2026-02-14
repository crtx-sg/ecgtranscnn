"""Tests for ECG-TransCovNet model architecture."""

import torch
import pytest

from ecg_transcovnet.model import (
    SKConv,
    CNNBackbone,
    ResidualBlock,
    ECGTransCovNet,
    FocalLoss,
    SinusoidalPositionalEncoding,
)
from ecg_transcovnet.constants import NUM_CLASSES, SIGNAL_LENGTH


class TestSKConv:
    def test_output_shape(self):
        block = SKConv(in_ch=32, out_ch=64, M=2)
        x = torch.randn(4, 32, 600)
        out = block(x)
        assert out.shape == (4, 64, 600)

    def test_single_branch(self):
        block = SKConv(in_ch=16, out_ch=32, M=1)
        x = torch.randn(2, 16, 100)
        out = block(x)
        assert out.shape == (2, 32, 100)


class TestResidualBlock:
    def test_output_shape(self):
        block = ResidualBlock(in_ch=32, out_ch=64, kernel_size=5, stride=2, pool=True)
        x = torch.randn(2, 32, 600)
        out = block(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64
        # stride=2 + pool stride=2 => ~600/4 = 150
        assert out.shape[2] == 150

    def test_no_pool(self):
        block = ResidualBlock(in_ch=16, out_ch=32, kernel_size=5, stride=2, pool=False)
        x = torch.randn(2, 16, 200)
        out = block(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 32
        assert out.shape[2] == 100  # stride=2 only


class TestCNNBackbone:
    def test_output_shape(self):
        backbone = CNNBackbone(in_channels=7, embed_dim=128)
        x = torch.randn(2, 7, SIGNAL_LENGTH)
        out = backbone(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 128
        assert out.shape[2] > 0  # sequence length depends on input

    def test_single_channel(self):
        backbone = CNNBackbone(in_channels=1, embed_dim=64)
        x = torch.randn(1, 1, SIGNAL_LENGTH)
        out = backbone(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 64


class TestSinusoidalPositionalEncoding:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=128, max_len=200)
        x = torch.randn(2, 50, 128)
        out = pe(x)
        assert out.shape == (2, 50, 128)

    def test_deterministic(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        x = torch.zeros(1, 30, 64)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)


class TestECGTransCovNet:
    def test_forward_shape(self):
        model = ECGTransCovNet(
            num_classes=NUM_CLASSES,
            in_channels=7,
            signal_length=SIGNAL_LENGTH,
        )
        x = torch.randn(2, 7, SIGNAL_LENGTH)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_single_lead(self):
        model = ECGTransCovNet(
            num_classes=NUM_CLASSES,
            in_channels=1,
            signal_length=SIGNAL_LENGTH,
        )
        x = torch.randn(2, 1, SIGNAL_LENGTH)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_with_attention(self):
        model = ECGTransCovNet(
            num_classes=NUM_CLASSES,
            in_channels=7,
            signal_length=SIGNAL_LENGTH,
        )
        x = torch.randn(1, 7, SIGNAL_LENGTH)
        logits, attn = model.forward_with_attention(x)
        assert logits.shape == (1, NUM_CLASSES)
        assert attn is not None
        assert attn.shape[0] == 1  # batch
        assert attn.shape[1] == NUM_CLASSES  # queries

    def test_seq_len_attribute(self):
        model = ECGTransCovNet(
            num_classes=16,
            in_channels=7,
            signal_length=SIGNAL_LENGTH,
        )
        assert model.seq_len > 0
        assert isinstance(model.seq_len, int)


class TestFocalLoss:
    def test_loss_value(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(4, 16)
        targets = torch.randint(0, 16, (4,))
        loss = loss_fn(inputs, targets)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_perfect_prediction(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        # Create inputs where prediction matches target with high confidence
        inputs = torch.zeros(2, 4)
        inputs[0, 0] = 10.0
        inputs[1, 2] = 10.0
        targets = torch.tensor([0, 2])
        loss = loss_fn(inputs, targets)
        assert loss.item() < 0.01  # should be very small

    def test_per_class_alpha(self):
        alpha = torch.tensor([1.0, 2.0, 0.5, 1.5])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        inputs = torch.randn(4, 4)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(inputs, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_per_class_alpha_device_move(self):
        alpha = torch.tensor([1.0, 2.0, 0.5, 1.5])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        # Verify alpha is a buffer (moves with .to())
        assert "alpha" in dict(loss_fn.named_buffers())
