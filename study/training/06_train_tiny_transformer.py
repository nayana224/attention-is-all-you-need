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

TEST_PATH = (
    STUDY_DIR
    / "data"
    / "en_ko_test.csv"
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
NUM_EPOCHS = 300
EVAL_EVERY = 10


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

test_pairs = load_translation_pairs(
    TEST_PATH
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


test_examples = []

for pair in test_pairs:
    example = make_example(
        pair,
        src_vocab,
        tgt_vocab,
        device=DEVICE,
    )

    test_examples.append(
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
                example["src_ids"],
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
        / len(examples)
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

best_test_loss = float(
    "inf"
)

best_epoch = 0


print("=" * 72)
print("Train Tiny Transformer")
print("=" * 72)
print("train pairs:", len(train_examples))
print("test pairs :", len(test_examples))
print("epochs     :", NUM_EPOCHS)
print("optimizer  : Adam")
print("lr         :", LEARNING_RATE)
print()


for epoch in range(
    1,
    NUM_EPOCHS + 1,
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

    if epoch == NUM_EPOCHS:
        should_evaluate = True

    if should_evaluate:
        (
            train_eval_loss,
            train_accuracy,
        ) = evaluate_examples(
            train_examples
        )

        (
            test_loss,
            test_accuracy,
        ) = evaluate_examples(
            test_examples
        )

        row = {
            "epoch":
                epoch,
            "train_epoch_loss":
                mean_epoch_loss,
            "train_eval_loss":
                train_eval_loss,
            "train_token_accuracy":
                train_accuracy,
            "test_loss":
                test_loss,
            "test_token_accuracy":
                test_accuracy,
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
            "| test loss",
            f"{test_loss:.4f}",
            "| test acc",
            f"{test_accuracy:.3f}",
        )

        if test_loss < best_test_loss:
            best_test_loss = (
                test_loss
            )

            best_epoch = epoch

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
                    "best_test_loss":
                        best_test_loss,
                    "test_token_accuracy":
                        test_accuracy,
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
            NUM_EPOCHS,
        "learning_rate":
            LEARNING_RATE,
        "optimizer":
            "Adam",
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
) as f:
    fieldnames = [
        "epoch",
        "train_epoch_loss",
        "train_eval_loss",
        "train_token_accuracy",
        "test_loss",
        "test_token_accuracy",
    ]

    writer = csv.DictWriter(
        f,
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
print("best epoch:", best_epoch)
print("best test loss:", best_test_loss)
print("best checkpoint:")
print(
    CHECKPOINT_DIR
    / "tiny_transformer_best.pt"
)
print("final checkpoint:")
print(final_path)
print("history:")
print(history_path)
