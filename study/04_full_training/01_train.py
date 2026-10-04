import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn


STUDY_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(STUDY_DIR),
)

from src.data import (
    load_translation_pairs,
    make_example,
)

from src.factory import (
    build_study_objects,
)


TRAIN_PATH = (
    STUDY_DIR
    / "data"
    / "en_ko_train.csv"
)

VAL_PATH = (
    STUDY_DIR
    / "data"
    / "en_ko_val.csv"
)

CHECKPOINT_DIR = (
    STUDY_DIR
    / "outputs"
    / "checkpoints"
)

CSV_DIR = (
    STUDY_DIR
    / "outputs"
    / "csv"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEVICE = "cpu"
SEED = 42

D_MODEL = 8
NUM_HEADS = 2
D_FF = 16
NUM_LAYERS = 1

LEARNING_RATE = 0.01

MAX_EPOCHS = 300
EVAL_EVERY = 10

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 0.0001


(
    train_pairs,
    src_vocab,
    tgt_vocab,
    model,
) = build_study_objects(
    TRAIN_PATH,
    seed=SEED,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    device=DEVICE,
)

val_pairs = load_translation_pairs(
    VAL_PATH
)


train_examples = []

for pair in train_pairs:
    example = make_example(
        pair,
        src_vocab,
        tgt_vocab,
        device=DEVICE,
    )

    train_examples.append(
        example
    )


val_examples = []

for pair in val_pairs:
    example = make_example(
        pair,
        src_vocab,
        tgt_vocab,
        device=DEVICE,
    )

    val_examples.append(
        example
    )


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


def evaluate_examples(
    examples,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for example in examples:
            trace = model(
                example[
                    "src_ids"
                ],
                example[
                    "decoder_input_ids"
                ],
            )

            logits = trace[
                "logits"
            ]

            loss = criterion(
                logits,
                example[
                    "gt_ids"
                ],
            )

            total_loss = (
                total_loss
                + loss.item()
            )

            pred_ids = logits.argmax(
                dim=-1
            )

            correct = (
                pred_ids
                == example[
                    "gt_ids"
                ]
            ).sum().item()

            total_correct = (
                total_correct
                + correct
            )

            total_tokens = (
                total_tokens
                + len(
                    example[
                        "gt_ids"
                    ]
                )
            )

    mean_loss = (
        total_loss
        / len(
            examples
        )
    )

    token_accuracy = (
        total_correct
        / total_tokens
    )

    return (
        mean_loss,
        token_accuracy,
    )


history = []

best_val_loss = float(
    "inf"
)

best_val_accuracy = 0.0
best_epoch = 0

evaluations_without_improvement = 0
stopped_epoch = MAX_EPOCHS


print("=" * 72)
print("Train Tiny Transformer")
print("=" * 72)

print(
    "train pairs:",
    len(
        train_examples
    ),
)

print(
    "val pairs  :",
    len(
        val_examples
    ),
)

print(
    "max epochs :",
    MAX_EPOCHS,
)

print(
    "optimizer  : Adam"
)

print(
    "lr         :",
    LEARNING_RATE,
)

print(
    "early stop :",
    EARLY_STOPPING_PATIENCE,
    "evaluations",
)

print()

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):
    model.train()

    epoch_loss_sum = 0.0

    for example in train_examples:
        optimizer.zero_grad()

        trace = model(
            example[
                "src_ids"
            ],
            example[
                "decoder_input_ids"
            ],
        )

        loss = criterion(
            trace[
                "logits"
            ],
            example[
                "gt_ids"
            ],
        )

        loss.backward()

        optimizer.step()

        epoch_loss_sum = (
            epoch_loss_sum
            + loss.item()
        )

    mean_epoch_loss = (
        epoch_loss_sum
        / len(
            train_examples
        )
    )

    should_evaluate = False

    if epoch == 1:
        should_evaluate = True

    if epoch % EVAL_EVERY == 0:
        should_evaluate = True

    if epoch == MAX_EPOCHS:
        should_evaluate = True

    if not should_evaluate:
        continue

    (
        train_eval_loss,
        train_accuracy,
    ) = evaluate_examples(
        train_examples
    )

    (
        val_loss,
        val_accuracy,
    ) = evaluate_examples(
        val_examples
    )

    improved = False

    if val_loss < (
        best_val_loss
        - MIN_DELTA
    ):
        improved = True

    row = {
        "epoch":
            epoch,
        "train_epoch_loss":
            mean_epoch_loss,
        "train_eval_loss":
            train_eval_loss,
        "train_token_accuracy":
            train_accuracy,
        "val_loss":
            val_loss,
        "val_token_accuracy":
            val_accuracy,
        "is_best":
            int(
                improved
            ),
    }

    history.append(
        row
    )

    print(
        "epoch",
        epoch,
        "| train loss",
        f"{train_eval_loss:.4f}",
        "| train acc",
        f"{train_accuracy:.3f}",
        "| val loss",
        f"{val_loss:.4f}",
        "| val acc",
        f"{val_accuracy:.3f}",
    )

    if improved:
        best_val_loss = (
            val_loss
        )

        best_val_accuracy = (
            val_accuracy
        )

        best_epoch = epoch

        evaluations_without_improvement = 0

        checkpoint = {
            "model_state_dict":
                model.state_dict(),
            "src_vocab":
                src_vocab,
            "tgt_vocab":
                tgt_vocab,
            "config": {
                "d_model":
                    D_MODEL,
                "num_heads":
                    NUM_HEADS,
                "d_ff":
                    D_FF,
                "num_layers":
                    NUM_LAYERS,
                "seed":
                    SEED,
            },
            "training": {
                "epoch":
                    epoch,
                "learning_rate":
                    LEARNING_RATE,
                "optimizer":
                    "Adam",
                "selection_split":
                    "validation",
                "best_val_loss":
                    best_val_loss,
                "val_token_accuracy":
                    best_val_accuracy,
            },
        }

        best_path = (
            CHECKPOINT_DIR
            / "tiny_transformer_best.pt"
        )

        torch.save(
            checkpoint,
            best_path,
        )

    else:
        evaluations_without_improvement = (
            evaluations_without_improvement
            + 1
        )

    if evaluations_without_improvement >= (
        EARLY_STOPPING_PATIENCE
    ):
        stopped_epoch = epoch

        print()
        print(
            "Early stopping:",
            "validation loss did not improve for",
            EARLY_STOPPING_PATIENCE,
            "evaluations.",
        )

        break


final_checkpoint = {
    "model_state_dict":
        model.state_dict(),
    "src_vocab":
        src_vocab,
    "tgt_vocab":
        tgt_vocab,
    "config": {
        "d_model":
            D_MODEL,
        "num_heads":
            NUM_HEADS,
        "d_ff":
            D_FF,
        "num_layers":
            NUM_LAYERS,
        "seed":
            SEED,
    },
    "training": {
        "epoch":
            stopped_epoch,
        "learning_rate":
            LEARNING_RATE,
        "optimizer":
            "Adam",
        "selection_split":
            "validation",
    },
}

final_path = (
    CHECKPOINT_DIR
    / "tiny_transformer_final.pt"
)

torch.save(
    final_checkpoint,
    final_path,
)


history_path = (
    CSV_DIR
    / "full_training_history.csv"
)

with history_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    fieldnames = [
        "epoch",
        "train_epoch_loss",
        "train_eval_loss",
        "train_token_accuracy",
        "val_loss",
        "val_token_accuracy",
        "is_best",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in history:
        writer.writerow(
            row
        )


print()
print("=" * 72)
print("Training finished")
print("=" * 72)

print(
    "best epoch:",
    best_epoch,
)

print(
    "best val loss:",
    best_val_loss,
)

print(
    "best val accuracy:",
    best_val_accuracy,
)

print(
    "stopped epoch:",
    stopped_epoch,
)

print(
    "best checkpoint:"
)

print(
    CHECKPOINT_DIR
    / "tiny_transformer_best.pt"
)

print(
    "final checkpoint:"
)

print(
    final_path
)

print(
    "history:"
)

print(
    history_path
)

print()
print(
    "Test data was NOT used for checkpoint selection."
)
