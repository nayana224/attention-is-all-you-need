import sys
from pathlib import Path

import torch
import torch.nn as nn

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from src.data import load_translation_pairs, make_example
from src.factory import build_study_objects
from src.tracing import (
    all_indices,
    append_attention_trace,
    append_tensor_trace,
    index_to_text,
    snapshot_parameters,
    write_csv,
)


TRAIN_PATH = STUDY_DIR / "data" / "en_ko_train.csv"
TEST_PATH = STUDY_DIR / "data" / "en_ko_test.csv"
OUTPUT_DIR = STUDY_DIR / "outputs" / "csv"

DEVICE = "cpu"
SEED = 42
LEARNING_RATE = 0.05
STEPS = 120
PROBE_TEST_INDEX = 0


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


(
    train_pairs,
    src_vocab,
    tgt_vocab,
    model,
) = build_study_objects(
    TRAIN_PATH,
    seed=SEED,
    d_model=8,
    num_heads=2,
    d_ff=16,
    num_layers=1,
    device=DEVICE,
)

test_pairs = load_translation_pairs(TEST_PATH)

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

probe_example = test_examples[
    PROBE_TEST_INDEX
]

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)


def evaluate_example(example):
    trace = model(
        example["src_ids"],
        example["decoder_input_ids"],
    )

    logits = trace["logits"]
    probabilities = trace["probabilities"]

    loss = criterion(
        logits,
        example["gt_ids"],
    )

    predictions = probabilities.argmax(
        dim=-1
    )

    correct = predictions.eq(
        example["gt_ids"]
    )

    positions = torch.arange(
        len(example["gt_ids"]),
        device=DEVICE,
    )

    gt_probabilities = probabilities[
        positions,
        example["gt_ids"],
    ]

    return {
        "trace": trace,
        "loss": loss,
        "correct_tokens":
            int(correct.sum().item()),
        "total_tokens":
            len(example["gt_ids"]),
        "accuracy":
            correct.float().mean().item(),
        "gt_probabilities":
            gt_probabilities,
        "mean_gt_probability":
            gt_probabilities.mean().item(),
    }


def evaluate_test_set():
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for example in test_examples:
            result = evaluate_example(example)

            total_loss += result[
                "loss"
            ].item()

            total_correct += result[
                "correct_tokens"
            ]

            total_tokens += result[
                "total_tokens"
            ]

    return {
        "mean_loss":
            total_loss / len(test_examples),
        "token_accuracy":
            total_correct / total_tokens,
    }


training_rows = []
tensor_rows = []
attention_rows = []
parameter_rows = []


print("=" * 78)
print("EN-KO Training Trace")
print("=" * 78)

print(f"Train pairs : {len(train_examples)}")
print(f"Test pairs  : {len(test_examples)}")
print(f"Steps       : {STEPS}")
print(f"Learning rate: {LEARNING_RATE}")

print("\nFixed probe test sentence:")
print(
    f"{probe_example['source_text']} "
    f"-> {probe_example['target_text']}"
)

print(
    "\n각 optimizer step은 train pair 하나를 순서대로 사용한다."
)

print(
    "매 step 뒤에 고정된 test probe와 전체 test set을 "
    "다시 평가해 변화 과정을 저장한다."
)


for step in range(STEPS):
    train_index = step % len(
        train_examples
    )

    train_example = train_examples[
        train_index
    ]

    # ----------------------------------------------
    # 1. 학습 문장 하나 업데이트
    # ----------------------------------------------

    optimizer.zero_grad()

    parameter_before = snapshot_parameters(
        model
    )

    train_trace = model(
        train_example["src_ids"],
        train_example["decoder_input_ids"],
    )

    train_loss = criterion(
        train_trace["logits"],
        train_example["gt_ids"],
    )

    train_loss.backward()

    gradients = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach().cpu()

        for index in all_indices(
            grad.shape
        ):
            gradients[(name, index)] = (
                grad[index].item()
            )

    optimizer.step()

    parameter_after = snapshot_parameters(
        model
    )

    for key, value_before in (
        parameter_before.items()
    ):
        parameter_name, index = key

        grad = gradients.get(
            key,
            0.0,
        )

        value_after = parameter_after[
            key
        ]

        parameter_rows.append({
            "step": step,
            "parameter":
                parameter_name,
            "index":
                index_to_text(index),
            "value_before":
                value_before,
            "gradient":
                grad,
            "value_after":
                value_after,
            "delta":
                value_after
                - value_before,
        })

    # ----------------------------------------------
    # 2. 업데이트 후 고정 probe 확인
    # ----------------------------------------------

    with torch.no_grad():
        probe_result = evaluate_example(
            probe_example
        )

    probe_trace = probe_result[
        "trace"
    ]

    forward_tensors = {
        "src_embedding":
            probe_trace["src_embedding"],
        "src_pe":
            probe_trace["src_pe"],
        "encoder_input":
            probe_trace["encoder_input"],
        "encoder_output":
            probe_trace["encoder_output"],
        "tgt_embedding":
            probe_trace["tgt_embedding"],
        "tgt_pe":
            probe_trace["tgt_pe"],
        "decoder_input":
            probe_trace["decoder_input"],
        "decoder_output":
            probe_trace["decoder_output"],
        "logits":
            probe_trace["logits"],
        "probabilities":
            probe_trace["probabilities"],
    }

    for tensor_name, tensor in (
        forward_tensors.items()
    ):
        append_tensor_trace(
            tensor_rows,
            step,
            tensor_name,
            tensor,
        )

    append_attention_trace(
        attention_rows,
        step,
        "encoder_self",
        probe_trace[
            "encoder_attention"
        ][0],
        probe_example["src_tokens"],
        probe_example["src_tokens"],
    )

    append_attention_trace(
        attention_rows,
        step,
        "decoder_masked",
        probe_trace[
            "masked_attention"
        ][0],
        probe_example[
            "decoder_input_tokens"
        ],
        probe_example[
            "decoder_input_tokens"
        ],
    )

    append_attention_trace(
        attention_rows,
        step,
        "decoder_cross",
        probe_trace[
            "cross_attention"
        ][0],
        probe_example[
            "decoder_input_tokens"
        ],
        probe_example["src_tokens"],
    )

    # ----------------------------------------------
    # 3. 업데이트 후 전체 test set 확인
    # ----------------------------------------------

    test_result = evaluate_test_set()

    training_row = {
        "step": step,
        "train_pair_index":
            train_index,
        "train_source":
            train_example["source_text"],
        "train_target":
            train_example["target_text"],
        "train_loss":
            train_loss.item(),
        "probe_loss":
            probe_result["loss"].item(),
        "probe_accuracy":
            probe_result["accuracy"],
        "probe_mean_gt_probability":
            probe_result[
                "mean_gt_probability"
            ],
        "test_mean_loss":
            test_result["mean_loss"],
        "test_token_accuracy":
            test_result[
                "token_accuracy"
            ],
    }

    for i, probability in enumerate(
        probe_result[
            "gt_probabilities"
        ]
    ):
        training_row[
            f"probe_gt_probability_{i}"
        ] = probability.item()

    training_rows.append(
        training_row
    )

    print(
        f"step={step:03d}  "
        f"train={train_index:02d}  "
        f"train_loss="
        f"{train_loss.item():.4f}  "
        f"probe_loss="
        f"{probe_result['loss'].item():.4f}  "
        f"test_acc="
        f"{test_result['token_accuracy']:.3f}"
    )


training_fields = [
    "step",
    "train_pair_index",
    "train_source",
    "train_target",
    "train_loss",
    "probe_loss",
    "probe_accuracy",
    "probe_mean_gt_probability",
    "test_mean_loss",
    "test_token_accuracy",
]

num_probe_positions = len(
    probe_example["gt_ids"]
)

for i in range(
    num_probe_positions
):
    field_name = (
        f"probe_gt_probability_{i}"
    )

    training_fields.append(
        field_name
    )


write_csv(
    OUTPUT_DIR
    / "training_trace.csv",
    training_rows,
    training_fields,
)

write_csv(
    OUTPUT_DIR
    / "tensor_trace.csv",
    tensor_rows,
    [
        "step",
        "tensor",
        "index",
        "value",
    ],
)

write_csv(
    OUTPUT_DIR
    / "attention_trace.csv",
    attention_rows,
    [
        "step",
        "attention_type",
        "head",
        "query_index",
        "query_token",
        "key_index",
        "key_token",
        "weight",
    ],
)

write_csv(
    OUTPUT_DIR
    / "parameter_trace.csv",
    parameter_rows,
    [
        "step",
        "parameter",
        "index",
        "value_before",
        "gradient",
        "value_after",
        "delta",
    ],
)


print("\n" + "=" * 78)
print("Saved")
print("=" * 78)

for filename in [
    "training_trace.csv",
    "tensor_trace.csv",
    "attention_trace.csv",
    "parameter_trace.csv",
]:
    print(OUTPUT_DIR / filename)

print(
    "\ntraining_trace.csv는 train sample loss뿐 아니라 "
    "고정 test probe와 전체 test set의 변화를 함께 기록한다."
)

print(
    "복잡한 소수점 계산은 손으로 전개하지 않고 "
    "PyTorch의 실제 계산값을 step별로 관찰한다."
)
