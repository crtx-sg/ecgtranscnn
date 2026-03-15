"""Visualization utilities for ECG-TransCovNet."""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import CLASS_NAMES


def save_training_curves(history: dict, path: str):
    """Save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, names: list[str], path: str):
    """Save confusion matrix as an image."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(cm.shape[1]), yticks=range(cm.shape[0]),
        xticklabels=names, yticklabels=names,
        title="Confusion Matrix", ylabel="True", xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=6,
            )
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_ecg_waveform(
    signal: np.ndarray,
    lead_names: list[str] | None = None,
    title: str = "ECG Signal",
    path: str | None = None,
    fs: float = 200.0,
    pacer_time: float | None = None,
):
    """Plot a multi-lead ECG waveform.

    Args:
        signal: (num_leads, signal_length) array.
        lead_names: names for each lead channel.
        title: plot title.
        path: if given, save to file; otherwise show.
        fs: sampling frequency in Hz.
    """
    num_leads = signal.shape[0]
    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(num_leads)]

    time = np.arange(signal.shape[1]) / fs

    fig, axes = plt.subplots(num_leads, 1, figsize=(14, 2 * num_leads), sharex=True)
    if num_leads == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, lead_names)):
        ax.plot(time, signal[i], linewidth=0.5)
        if pacer_time is not None:
            ax.axvline(pacer_time, color="red", ls="--", alpha=0.6, label="Pacer")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_predictions(
    probs: np.ndarray,
    true_idx: int | None = None,
    path: str | None = None,
):
    """Bar chart of class prediction probabilities.

    Args:
        probs: (num_classes,) probability array.
        true_idx: index of true class (highlighted in green).
        path: if given, save to file; otherwise show.
    """
    num_classes = len(probs)
    names = CLASS_NAMES[:num_classes]
    colors = ["#4CAF50" if i == true_idx else "#2196F3" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(num_classes), probs, color=colors)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Probability")
    ax.set_title("Class Prediction Probabilities")
    ax.invert_yaxis()
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_attention_map(
    attention: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Decoder Cross-Attention",
    path: str | None = None,
):
    """Plot attention heatmap from the decoder.

    Args:
        attention: (num_queries, seq_len) attention weights.
        class_names: labels for the query axis.
        title: plot title.
        path: if given, save to file; otherwise show.
    """
    if class_names is None:
        class_names = CLASS_NAMES[: attention.shape[0]]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(attention, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("CNN Feature Sequence Position")
    ax.set_ylabel("Class Query")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()
