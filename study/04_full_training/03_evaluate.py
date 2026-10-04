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

from src.tokenization import (
    encode_source,
    invert_vocab,
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

TEST_PATH = (
    STUDY_DIR
    / "data"
    / "en_ko_test.csv"
)

CHECKPOINT_PATH = (
    STUDY_DIR
    / "outputs"
    / "checkpoints"
    / "tiny_transformer_best.pt"
)

OUTPUT_PATH = (
    STUDY_DIR
    / "outputs"
    / "csv"
    / "trained_model_evaluation.csv"
)

DEVICE = "cpu"


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

config = checkpoint[
    "config"
]

(
    train_pairs,
    src_vocab,
    tgt_vocab,
    model,
) = build_study_objects(
    TRAIN_PATH,
    seed=config[
        "seed"
    ],
    d_model=config[
        "d_model"
    ],
    num_heads=config[
        "num_heads"
    ],
    d_ff=config[
        "d_ff"
    ],
    num_layers=config[
        "num_layers"
    ],
    device=DEVICE,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

id_to_target = invert_vocab(
    tgt_vocab
)

criterion = nn.CrossEntropyLoss()


def teacher_forced_prediction(
    example,
):
    with torch.no_grad():
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
        ).item()

        pred_ids = logits.argmax(
            dim=-1
        )

    pred_tokens = []

    for token_id in pred_ids:
        token = id_to_target[
            token_id.item()
        ]

        pred_tokens.append(
            token
        )

    correct = (
        pred_ids
        == example[
            "gt_ids"
        ]
    ).sum().item()

    accuracy = (
        correct
        / len(
            example[
                "gt_ids"
            ]
        )
    )

    return (
        pred_tokens,
        loss,
        accuracy,
    )


def greedy_decode(
    source_text,
    max_length=12,
):
    (
        src_tokens,
        src_ids,
    ) = encode_source(
        source_text,
        src_vocab,
    )

    src_ids_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    decoder_ids = [
        tgt_vocab[
            "<SOS>"
        ]
    ]

    generated_tokens = []

    for step in range(
        max_length
    ):
        decoder_input_ids = torch.tensor(
            decoder_ids,
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.no_grad():
            trace = model(
                src_ids_tensor,
                decoder_input_ids,
            )

        last_logits = trace[
            "logits"
        ][
            -1
        ]

        next_id = last_logits.argmax(
            dim=-1
        ).item()

        next_token = id_to_target[
            next_id
        ]

        generated_tokens.append(
            next_token
        )

        if next_token == "<EOS>":
            break

        decoder_ids.append(
            next_id
        )

    return generated_tokens


def evaluate_split(
    split_name,
    pairs,
):
    rows = []

    total_teacher_correct = 0
    total_teacher_tokens = 0

    greedy_exact_count = 0

    loss_sum = 0.0

    for pair in pairs:
        example = make_example(
            pair,
            src_vocab,
            tgt_vocab,
            device=DEVICE,
        )

        (
            teacher_tokens,
            teacher_loss,
            teacher_accuracy,
        ) = teacher_forced_prediction(
            example
        )

        greedy_tokens = greedy_decode(
            pair[
                "source"
            ]
        )

        gt_tokens = example[
            "gt_tokens"
        ]

        greedy_exact = (
            greedy_tokens
            == gt_tokens
        )

        if greedy_exact:
            greedy_exact_count = (
                greedy_exact_count
                + 1
            )

        total_teacher_correct = (
            total_teacher_correct
            + teacher_accuracy
            * len(
                gt_tokens
            )
        )

        total_teacher_tokens = (
            total_teacher_tokens
            + len(
                gt_tokens
            )
        )

        loss_sum = (
            loss_sum
            + teacher_loss
        )

        row = {
            "split":
                split_name,
            "source":
                pair[
                    "source"
                ],
            "ground_truth":
                " ".join(
                    gt_tokens
                ),
            "teacher_prediction":
                " ".join(
                    teacher_tokens
                ),
            "teacher_loss":
                teacher_loss,
            "teacher_token_accuracy":
                teacher_accuracy,
            "greedy_prediction":
                " ".join(
                    greedy_tokens
                ),
            "greedy_exact_match":
                int(
                    greedy_exact
                ),
        }

        rows.append(
            row
        )

    mean_teacher_accuracy = (
        total_teacher_correct
        / total_teacher_tokens
    )

    greedy_exact_rate = (
        greedy_exact_count
        / len(
            pairs
        )
    )

    mean_loss = (
        loss_sum
        / len(
            pairs
        )
    )

    print()
    print(
        split_name,
        "loss:",
        f"{mean_loss:.4f}",
    )

    print(
        split_name,
        "teacher-forced token accuracy:",
        f"{mean_teacher_accuracy:.3f}",
    )

    print(
        split_name,
        "greedy exact-match:",
        f"{greedy_exact_rate:.3f}",
    )

    return rows


print("=" * 72)
print("Evaluate Trained Tiny Transformer")
print("=" * 72)

print(
    "checkpoint:"
)

print(
    CHECKPOINT_PATH
)

print(
    "selected at epoch:",
    checkpoint[
        "training"
    ][
        "epoch"
    ],
)

print(
    "selection split:",
    checkpoint[
        "training"
    ].get(
        "selection_split",
        "unknown",
    ),
)


train_rows = evaluate_split(
    "train",
    train_pairs,
)

val_pairs = load_translation_pairs(
    VAL_PATH
)

val_rows = evaluate_split(
    "validation",
    val_pairs,
)

test_pairs = load_translation_pairs(
    TEST_PATH
)

test_rows = evaluate_split(
    "test",
    test_pairs,
)


all_rows = []

for row in train_rows:
    all_rows.append(
        row
    )

for row in val_rows:
    all_rows.append(
        row
    )

for row in test_rows:
    all_rows.append(
        row
    )


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    fieldnames = [
        "split",
        "source",
        "ground_truth",
        "teacher_prediction",
        "teacher_loss",
        "teacher_token_accuracy",
        "greedy_prediction",
        "greedy_exact_match",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in all_rows:
        writer.writerow(
            row
        )


print()
print("Final test predictions")
print("-" * 72)

for row in test_rows:
    print()

    print(
        "Source:"
    )

    print(
        row[
            "source"
        ]
    )

    print(
        "GT:"
    )

    print(
        row[
            "ground_truth"
        ]
    )

    print(
        "Teacher-forced:"
    )

    print(
        row[
            "teacher_prediction"
        ]
    )

    print(
        "Greedy:"
    )

    print(
        row[
            "greedy_prediction"
        ]
    )

print()
print(
    "saved:"
)

print(
    OUTPUT_PATH
)
