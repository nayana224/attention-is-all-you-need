import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


STUDY_DIR = Path(__file__).resolve().parents[1]

VISUAL_DIR = (
    STUDY_DIR
    / "02_visual_intuition"
)

sys.path.insert(
    0,
    str(VISUAL_DIR),
)

from plot_utils import (
    save_figure,
)


HISTORY_PATH = (
    STUDY_DIR
    / "outputs"
    / "csv"
    / "full_training_history.csv"
)

OUTPUT_DIR = (
    STUDY_DIR
    / "outputs"
    / "training_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


epochs = []

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_epoch = None
best_val_loss = None


with HISTORY_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    reader = csv.DictReader(
        file
    )

    for row in reader:
        epoch = int(
            row[
                "epoch"
            ]
        )

        train_loss = float(
            row[
                "train_eval_loss"
            ]
        )

        val_loss = float(
            row[
                "val_loss"
            ]
        )

        train_accuracy = float(
            row[
                "train_token_accuracy"
            ]
        )

        val_accuracy = float(
            row[
                "val_token_accuracy"
            ]
        )

        is_best = int(
            row[
                "is_best"
            ]
        )

        epochs.append(
            epoch
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )

        if is_best == 1:
            best_epoch = epoch
            best_val_loss = val_loss


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    constrained_layout=True,
)


axes[0].plot(
    epochs,
    train_losses,
    marker="o",
    label="Train",
)

axes[0].plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation",
)

axes[0].set_title(
    "Loss vs Epoch"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Cross Entropy Loss"
)

axes[0].grid(
    alpha=0.3,
)

axes[0].legend()


axes[1].plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Train",
)

axes[1].plot(
    epochs,
    val_accuracies,
    marker="o",
    label="Validation",
)

axes[1].set_title(
    "Token Accuracy vs Epoch"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Token Accuracy"
)

axes[1].set_ylim(
    0.0,
    1.05,
)

axes[1].grid(
    alpha=0.3,
)

axes[1].legend()


if best_epoch is not None:
    for axis in axes:
        axis.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.5,
            label="Best Validation Epoch",
        )

    axes[0].text(
        best_epoch,
        best_val_loss,
        "  best epoch = "
        + str(
            best_epoch
        ),
        va="bottom",
    )


figure_path = (
    OUTPUT_DIR
    / "15_full_training_curves.png"
)

save_figure(
    fig,
    figure_path,
)


print("=" * 72)
print("Full Training Curves")
print("=" * 72)

print(
    "history:"
)

print(
    HISTORY_PATH
)

print(
    "best validation epoch:",
    best_epoch,
)

print(
    "saved:"
)

print(
    figure_path
)
