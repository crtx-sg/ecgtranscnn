"""ECG-TransCovNet: Hybrid CNN-Transformer for ECG arrhythmia classification."""

__version__ = "0.1.0"

from .model import ECGTransCovNet, SKConv, CNNBackbone, FocalLoss
from .constants import (
    NUM_CLASSES,
    CLASS_NAMES,
    CONDITION_LIST,
    CONDITION_TO_IDX,
    SIGNAL_LENGTH,
    ALL_LEADS,
    MIT_BIH_PROPORTIONS,
)
from .data import generate_dataset, load_or_generate_data, AugmentedECGDataset
from .training import train_one_epoch, validate, evaluate_detailed
from .visualization import (
    save_training_curves,
    save_confusion_matrix,
    plot_ecg_waveform,
    plot_predictions,
    plot_attention_map,
)

__all__ = [
    # Model
    "ECGTransCovNet",
    "SKConv",
    "CNNBackbone",
    "FocalLoss",
    # Constants
    "NUM_CLASSES",
    "CLASS_NAMES",
    "CONDITION_LIST",
    "CONDITION_TO_IDX",
    "SIGNAL_LENGTH",
    "ALL_LEADS",
    "MIT_BIH_PROPORTIONS",
    # Data
    "generate_dataset",
    "load_or_generate_data",
    "AugmentedECGDataset",
    # Training
    "train_one_epoch",
    "validate",
    "evaluate_detailed",
    # Visualization
    "save_training_curves",
    "save_confusion_matrix",
    "plot_ecg_waveform",
    "plot_predictions",
    "plot_attention_map",
]
