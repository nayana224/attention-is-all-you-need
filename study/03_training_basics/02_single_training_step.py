import sys
from pathlib import Path

import torch
import torch.nn as nn

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from src.data import make_example
from src.factory import build_study_objects
from src.tokenization import invert_vocab


TRAIN_PATH = STUDY_DIR / "data" / "en_ko_train.csv"

DEVICE = "cpu"
SEED = 42
LEARNING_RATE = 0.05
SAMPLE_INDEX = 0


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==================================================
# 0. Model / Data 준비
# ==================================================

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

sample_pair = train_pairs[
    SAMPLE_INDEX
]

example = make_example(
    sample_pair,
    src_vocab,
    tgt_vocab,
    device=DEVICE,
)

id_to_target = invert_vocab(
    tgt_vocab
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)


print_section("0. Input / Decoder Input / GT")

print("Encoder Input (English):")
print(example["src_tokens"])
print(example["src_ids"])

print("\nDecoder Input (Korean):")
print(example["decoder_input_tokens"])
print(example["decoder_input_ids"])

print("\nGround Truth:")
print(example["gt_tokens"])
print(example["gt_ids"])


# ==================================================
# 추적할 parameter 하나 선택
#
# Decoder masked self-attention
# Head 1의 W_Q[0, 0] 하나를 추적한다.
#
# 목적:
# loss.backward() 이후 이 값에도 gradient가 생기고,
# optimizer.step() 이후 실제 값이 바뀌는지 확인한다.
# ==================================================

decoder_layer = model.decoder_layers[0]

masked_attention = (
    decoder_layer.masked_self_attention
)

head_1_W_Q = masked_attention.W_Q[0]

tracked_weight_matrix = (
    head_1_W_Q.weight
)

weight_before = (
    tracked_weight_matrix[0, 0]
    .detach()
    .item()
)


# ==================================================
# 1. Forward
#
# Encoder Input + Decoder Input
# -> Transformer
# -> logits
# -> probability
# ==================================================

optimizer.zero_grad()

trace = model(
    example["src_ids"],
    example["decoder_input_ids"],
)

logits = trace["logits"]
probabilities = trace["probabilities"]

pred_ids = probabilities.argmax(
    dim=-1
)

pred_tokens = []

for token_id in pred_ids:
    token_id_number = token_id.item()

    token = id_to_target[
        token_id_number
    ]

    pred_tokens.append(
        token
    )


print_section("1. Forward")

print("Encoder output shape:")
print(
    trace["encoder_output"].shape
)

print("\nDecoder output shape:")
print(
    trace["decoder_output"].shape
)

print("\nLogits shape:")
print(
    logits.shape
)

print("\nPrediction:")
print(
    pred_tokens
)

print("\nGround Truth:")
print(
    example["gt_tokens"]
)


# ==================================================
# 2. Loss
#
# 각 position에서 GT token의 probability를 확인하고
# 전체 position의 Cross Entropy Loss를 계산한다.
# ==================================================

gt_probabilities = []

num_positions = len(
    example["gt_ids"]
)

for position in range(
    num_positions
):
    gt_id_tensor = example[
        "gt_ids"
    ][position]

    gt_id = gt_id_tensor.item()

    gt_probability_tensor = (
        probabilities[
            position,
            gt_id,
        ]
    )

    gt_probability = (
        gt_probability_tensor.item()
    )

    gt_probabilities.append(
        gt_probability
    )


loss = criterion(
    logits,
    example["gt_ids"],
)


print_section("2. Ground Truth Probability / Loss")

for position in range(
    num_positions
):
    gt_token = example[
        "gt_tokens"
    ][position]

    gt_probability = (
        gt_probabilities[position]
    )

    print(
        "position",
        position,
        "| GT =",
        gt_token,
        "| probability =",
        f"{gt_probability:.6f}",
    )

print("\nCross Entropy Loss:")
print(
    loss.item()
)


# ==================================================
# 3. Backward
#
# loss.backward()
# -> 각 learnable parameter에 gradient가 계산된다.
# ==================================================

loss.backward()

tracked_gradient = (
    tracked_weight_matrix.grad[
        0,
        0,
    ]
    .detach()
    .item()
)


print_section("3. Backward -> Gradient")

print("Tracked parameter:")
print(
    "Decoder Layer 1 / "
    "Masked Self-Attention / "
    "Head 1 / W_Q[0,0]"
)

print("\nParameter before update:")
print(
    weight_before
)

print("\nGradient dLoss/dW:")
print(
    tracked_gradient
)


# ==================================================
# 4. Optimizer Step
#
# SGD:
# W_new = W_old - learning_rate * gradient
# ==================================================

expected_weight_after = (
    weight_before
    - LEARNING_RATE
    * tracked_gradient
)

optimizer.step()

weight_after = (
    tracked_weight_matrix[0, 0]
    .detach()
    .item()
)


print_section("4. Optimizer Step -> Parameter Update")

print("Weight before:")
print(
    weight_before
)

print("\nLearning rate:")
print(
    LEARNING_RATE
)

print("\nGradient:")
print(
    tracked_gradient
)

print("\nExpected by SGD formula:")
print(
    expected_weight_after
)

print("\nActual weight after optimizer.step():")
print(
    weight_after
)


# ==================================================
# 5. Update 후 Loss 다시 계산
#
# optimizer.step() 후 같은 example을 다시 forward해서
# loss가 어떻게 변했는지 확인한다.
# ==================================================

with torch.no_grad():
    trace_after = model(
        example["src_ids"],
        example["decoder_input_ids"],
    )

    logits_after = trace_after[
        "logits"
    ]

    loss_after = criterion(
        logits_after,
        example["gt_ids"],
    )


print_section("5. Loss Before / After One Update")

print("Loss before:")
print(
    loss.item()
)

print("\nLoss after:")
print(
    loss_after.item()
)


print_section("6. Final Flow")

print(
    """
Encoder Input + Decoder Input
        ↓
      model()
        ↓
      logits
        ↓
CrossEntropy(logits, GT)
        ↓
       loss
        ↓
  loss.backward()
        ↓
     gradient
        ↓
 optimizer.step()
        ↓
parameter update
"""
)

print(
    "핵심: forward에서 만든 loss가 backward를 통해 "
    "Attention의 learnable parameter까지 gradient를 전달하고, "
    "optimizer.step()이 그 parameter를 실제로 바꾼다."
)
