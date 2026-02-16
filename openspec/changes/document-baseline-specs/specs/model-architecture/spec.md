## ADDED Requirements

### Requirement: CNN Feature Extraction Backbone
The system SHALL extract local features from 7-lead ECG signals using a CNN backbone with Selective Kernel Convolution (SKConv) and residual blocks. The backbone SHALL progressively expand channels (7→32→64→128) while reducing sequence length (2400→600→150→38) through three stages of SKConv + ResidualBlock pairs with max pooling.

#### Scenario: Forward pass through CNN backbone
- **WHEN** a batch of raw ECG signals of shape `(B, 7, 2400)` is passed to the CNN backbone
- **THEN** the output SHALL have shape `(B, 128, 38)` representing 128-dimensional features at 38 temporal positions

#### Scenario: Selective Kernel Convolution
- **WHEN** the SKConv module processes input features
- **THEN** it SHALL apply multi-branch convolution with kernel sizes 3, 5, and 7, and use channel-wise attention to adaptively select the optimal receptive field

### Requirement: Transformer Encoder-Decoder
The system SHALL use a Transformer encoder (3 layers, 8 heads, dim=128, ff=512) for global temporal context extraction via self-attention on CNN features, followed by a Transformer decoder (3 layers, 8 heads) with 16 learnable object queries (one per cardiac condition) in DETR-style cross-attention with encoder memory.

#### Scenario: Encoder processes CNN features
- **WHEN** CNN backbone output `(B, 128, 38)` is passed through sinusoidal positional encoding and the Transformer encoder
- **THEN** the encoder SHALL produce memory of shape `(B, 38, 128)` capturing global temporal dependencies

#### Scenario: Decoder with object queries
- **WHEN** the 16 learnable object queries attend to encoder memory via cross-attention
- **THEN** each query SHALL specialize to detect its assigned cardiac condition, producing output of shape `(B, 16, 128)`

#### Scenario: Cross-attention weight extraction
- **WHEN** the model is run with attention extraction enabled
- **THEN** the custom decoder layer SHALL expose cross-attention weights for interpretability

### Requirement: Multi-Label Classification Head
The system SHALL classify ECG signals into 16 independent cardiac conditions using a per-query FFN classification head (128→64→1) with sigmoid activation. Predictions SHALL be thresholded at 0.5 for binary decisions.

#### Scenario: Classification output
- **WHEN** the decoder outputs 16 query embeddings of shape `(B, 16, 128)`
- **THEN** the classification head SHALL produce logits of shape `(B, 16)` with sigmoid activation for multi-label binary classification

### Requirement: Focal Loss Training Objective
The system SHALL use Focal Loss with γ=2.0 and per-class α weights based on MIT-BIH prevalence proportions to address class imbalance across 16 conditions during training.

#### Scenario: Loss computation with class imbalance
- **WHEN** computing loss on a batch with imbalanced condition distribution
- **THEN** Focal Loss SHALL reduce the contribution of well-classified examples (via γ=2.0) and apply per-class α weights to prioritize underrepresented conditions

### Requirement: Training Pipeline
The system SHALL train using AdamW optimizer (lr=5e-4, weight_decay=1e-4) with mixed precision (AMP), gradient clipping (max_norm=1.0), linear warmup (5 epochs), and early stopping (patience=20 epochs on validation loss). Checkpoints SHALL save both the best model (by validation loss) and the final model, including hyperparameters for reconstruction.

#### Scenario: Training with early stopping
- **WHEN** validation loss does not improve for 20 consecutive epochs
- **THEN** training SHALL stop early and the best checkpoint SHALL be preserved

#### Scenario: Checkpoint contents
- **WHEN** a model checkpoint is saved
- **THEN** it SHALL contain `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_loss`, and architecture hyperparameters (`embed_dim`, `nhead`, `num_encoder_layers`, `num_decoder_layers`)
