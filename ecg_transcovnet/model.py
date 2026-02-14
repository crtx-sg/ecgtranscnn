"""ECG-TransCovNet model architecture.

Hybrid CNN-Transformer for ECG arrhythmia classification, based on:
  Shah et al., "ECG-TransCovNet: A hybrid transformer model for accurate
  arrhythmia detection using Electrocardiogram signals", IET CIT 2024.

Components:
  - SKConv: Selective Kernel convolution block
  - CNNBackbone: 1D CNN feature extractor with residual connections
  - CustomTransformerDecoderLayer: decoder layer returning cross-attention weights
  - ECGTransCovNet: full hybrid model
  - FocalLoss: class-imbalanced focal loss with per-class weights
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SKConv(nn.Module):
    """Selective Kernel convolution block (paper Figure 3a).

    Uses *M* parallel conv branches with different kernel sizes and an
    attention mechanism to dynamically weight them.
    """

    def __init__(self, in_ch: int, out_ch: int, M: int = 2, r: int = 16):
        super().__init__()
        d = max(in_ch // r, 32)
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3 + i * 2, padding=1 + i),
                nn.BatchNorm1d(out_ch),
                nn.SiLU(inplace=True),
            )
            for i in range(M)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_ch, d)
        self.fcs = nn.ModuleList(nn.Linear(d, out_ch) for _ in range(M))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.convs]
        feats_cat = torch.stack(feats, dim=1)          # (B, M, C, L)
        s = self.gap(sum(feats)).squeeze(-1)            # (B, C)
        z = self.fc(s)                                  # (B, d)
        weights = torch.stack([fc(z) for fc in self.fcs], dim=1)  # (B, M, C)
        attn = self.softmax(weights).unsqueeze(-1)      # (B, M, C, 1)
        return (feats_cat * attn).sum(dim=1)            # (B, C, L)


class ResidualBlock(nn.Module):
    """Conv block with a residual (skip) connection.

    Uses a 1x1 conv for channel alignment when in_ch != out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5,
                 stride: int = 2, pool: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if pool else nn.Identity()

        # Skip connection: match channels and spatial dims
        total_stride = stride * (2 if pool else 1)
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=total_stride),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(self.conv(x))
        return self.act(out + self.skip(x))


class CNNBackbone(nn.Module):
    """CNN feature extractor with residual connections.

    Down-sampling path:
        2400 -> [ResidualBlock s2 + pool s2] -> 600
             -> [SK block]                   -> 600
             -> [ResidualBlock s2 + pool s2] -> 150
             -> [ResidualBlock s2 + pool s2] -> 38
             -> [1x1 bottleneck]             -> 38  (seq_len for transformer)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.stage1 = ResidualBlock(in_channels, 32, kernel_size=7, stride=2, pool=True)
        self.sk_block = SKConv(32, 64)
        self.stage2 = ResidualBlock(64, 128, kernel_size=5, stride=2, pool=True)
        self.stage3 = ResidualBlock(128, 256, kernel_size=5, stride=2, pool=True)
        self.bottleneck = nn.Conv1d(256, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.sk_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.bottleneck(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (max_len, 1, d_model) for batch_first=False or (1, max_len, d_model) for batch_first=True
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, D) with batch_first=True."""
        return x + self.pe[:, :x.size(1)]


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder layer that also returns cross-attention weights."""

    def forward(
        self, tgt, memory,
        tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        tgt_is_causal=False, memory_is_causal=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal,
            )
            cross_out, cross_attn = self._mha_custom(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask,
            )
            x = x + cross_out
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal)
            )
            cross_out, cross_attn = self._mha_custom(
                x, memory, memory_mask, memory_key_padding_mask,
            )
            x = self.norm2(x + cross_out)
            x = self.norm3(x + self._ff_block(x))
        return x, cross_attn

    def _mha_custom(self, x, mem, attn_mask, key_padding_mask):
        out, attn = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=True,
        )
        return self.dropout2(out), attn


class ECGTransCovNet(nn.Module):
    """Hybrid CNN-Transformer for ECG arrhythmia classification.

    Pipeline:
        raw signal  ->  CNN backbone (local features)
                    ->  Transformer encoder (global context)
                    ->  Transformer decoder with DETR object queries
                    ->  per-query FFN  ->  class logits
    """

    def __init__(
        self,
        num_classes: int = 16,
        in_channels: int = 1,
        signal_length: int = 2400,
        embed_dim: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # CNN backbone
        self.cnn_backbone = CNNBackbone(in_channels, embed_dim)

        # Determine the sequence length after CNN (data-driven)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_length)
            seq_len = self.cnn_backbone(dummy).shape[2]
        self.seq_len = seq_len

        # Sinusoidal positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=max(seq_len, 512))

        # Transformer encoder (batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # Transformer decoder (custom layers for attention extraction, batch_first=True)
        self.decoder_layers = nn.ModuleList(
            CustomTransformerDecoderLayer(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Learnable object queries: (1, num_classes, embed_dim) for batch_first
        self.object_queries = nn.Parameter(
            torch.randn(1, num_classes, embed_dim) * 0.02
        )

        # Classification head (per-query)
        self.ffn_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_backbone(x)                        # (B, D, S)
        features = features.permute(0, 2, 1)                  # (B, S, D)
        features = self.positional_encoding(features)

        memory = self.encoder(features)                        # (B, S, D)

        B = x.shape[0]
        queries = self.object_queries.expand(B, -1, -1)        # (B, C, D)
        dec = queries
        for layer in self.decoder_layers:
            dec, _ = layer(dec, memory)
        dec = self.decoder_norm(dec)                           # (B, C, D)

        return self.ffn_head(dec).squeeze(-1)                  # (B, C)

    def forward_with_attention(
        self, x: torch.Tensor, layer_idx: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass that also returns cross-attention weights."""
        features = self.cnn_backbone(x).permute(0, 2, 1)      # (B, S, D)
        features = self.positional_encoding(features)
        memory = self.encoder(features)

        B = x.shape[0]
        dec = self.object_queries.expand(B, -1, -1)
        captured = None
        for i, layer in enumerate(self.decoder_layers):
            dec, attn = layer(dec, memory)
            target = layer_idx if layer_idx >= 0 else len(self.decoder_layers) - 1
            if i == target:
                captured = attn

        dec = self.decoder_norm(dec)
        logits = self.ffn_head(dec).squeeze(-1)
        return logits, captured


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification (paper Section 3.5).

    Args:
        alpha: Per-class weight tensor of shape (num_classes,), or a scalar
               float applied uniformly. When a tensor is provided, each class
               gets its own weight (higher = more importance).
        gamma: Focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha: Union[float, torch.Tensor] = 0.25, gamma: float = 2.0):
        super().__init__()
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)

        if self.alpha.ndim == 0:
            # Scalar alpha
            alpha_t = self.alpha
        else:
            # Per-class alpha: gather weights for each target
            alpha_t = self.alpha[targets]

        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()
