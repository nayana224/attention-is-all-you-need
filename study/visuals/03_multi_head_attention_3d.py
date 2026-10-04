import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


STUDY_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = STUDY_DIR / "outputs" / "visuals"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

torch.manual_seed(42)


# ==================================================
# 0. Multi-Head Visualization Setup
# ==================================================
#
# Head 하나를 3D로 직접 볼 수 있게:
#
# d_model = 6
# num_heads = 2
# d_k = d_v = 3
#
# 로 구성한다.
# ==================================================

tokens = [
    "I",
    "love",
    "robots",
]

x = torch.tensor([
    [1.0, 0.2, 0.1, 0.4, 0.0, 0.2],
    [0.2, 1.0, 0.3, 0.1, 0.7, 0.1],
    [0.3, 0.2, 1.0, 0.2, 0.1, 0.8],
])

d_model = 6
num_heads = 2

d_k = (
    d_model
    // num_heads
)

d_v = d_k


# ==================================================
# 1. Head layers
# ==================================================

head_1_W_Q = nn.Linear(
    d_model,
    d_k,
    bias=False,
)

head_1_W_K = nn.Linear(
    d_model,
    d_k,
    bias=False,
)

head_1_W_V = nn.Linear(
    d_model,
    d_v,
    bias=False,
)

head_2_W_Q = nn.Linear(
    d_model,
    d_k,
    bias=False,
)

head_2_W_K = nn.Linear(
    d_model,
    d_k,
    bias=False,
)

head_2_W_V = nn.Linear(
    d_model,
    d_v,
    bias=False,
)


def run_head(
    input_x,
    W_Q,
    W_K,
    W_V,
):
    Q = W_Q(
        input_x
    )

    K = W_K(
        input_x
    )

    V = W_V(
        input_x
    )

    scores = (
        Q @ K.T
    )

    scaled_scores = (
        scores
        / math.sqrt(d_k)
    )

    weights = F.softmax(
        scaled_scores,
        dim=-1,
    )

    output = (
        weights @ V
    )

    return (
        Q,
        K,
        V,
        weights,
        output,
    )


(
    Q1,
    K1,
    V1,
    weights1,
    output1,
) = run_head(
    x,
    head_1_W_Q,
    head_1_W_K,
    head_1_W_V,
)

(
    Q2,
    K2,
    V2,
    weights2,
    output2,
) = run_head(
    x,
    head_2_W_Q,
    head_2_W_K,
    head_2_W_V,
)


# ==================================================
# 2. Concatenate and W_O
# ==================================================

concat = torch.cat(
    [
        output1,
        output2,
    ],
    dim=-1,
)

W_O = nn.Linear(
    d_model,
    d_model,
    bias=False,
)

final_output = W_O(
    concat
)


print("=" * 70)
print("Multi-Head Attention Visualization")
print("=" * 70)

print()
print("Head 1 weights:")
print(weights1)

print()
print("Head 2 weights:")
print(weights2)

print()
print("Head 1 output:")
print(output1)

print()
print("Head 2 output:")
print(output2)

print()
print("Concatenated:")
print(concat)

print()
print("Final output:")
print(final_output)


# ==================================================
# 3. 3D vector helper
# ==================================================

def draw_vectors(
    axis,
    vectors,
    labels,
    title,
):
    for i in range(
        len(labels)
    ):
        vector = vectors[i]

        x_value = vector[0].item()
        y_value = vector[1].item()
        z_value = vector[2].item()

        axis.quiver(
            0.0,
            0.0,
            0.0,
            x_value,
            y_value,
            z_value,
            arrow_length_ratio=0.08,
        )

        axis.text(
            x_value,
            y_value,
            z_value,
            labels[i],
        )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "dim 0"
    )

    axis.set_ylabel(
        "dim 1"
    )

    axis.set_zlabel(
        "dim 2"
    )


# ==================================================
# 4. Compare Head 1 / Head 2 Q spaces
# ==================================================

fig = plt.figure(
    figsize=(12, 5)
)

ax1 = fig.add_subplot(
    121,
    projection="3d",
)

ax2 = fig.add_subplot(
    122,
    projection="3d",
)

draw_vectors(
    ax1,
    Q1,
    tokens,
    "Head 1 - Query Space",
)

draw_vectors(
    ax2,
    Q2,
    tokens,
    "Head 2 - Query Space",
)

query_path = (
    OUTPUT_DIR
    / "05_multihead_query_spaces_3d.png"
)

plt.tight_layout()

plt.savefig(
    query_path,
    dpi=160,
)

plt.close()


# ==================================================
# 5. Compare Attention Weights
# ==================================================

query_index = 1

x_positions = torch.arange(
    len(tokens)
).numpy()

bar_width = 0.35

plt.figure(
    figsize=(8, 4)
)

plt.bar(
    x_positions - bar_width / 2,
    weights1[
        query_index
    ].detach().numpy(),
    width=bar_width,
    label="Head 1",
)

plt.bar(
    x_positions + bar_width / 2,
    weights2[
        query_index
    ].detach().numpy(),
    width=bar_width,
    label="Head 2",
)

plt.xticks(
    x_positions,
    tokens,
)

plt.ylim(
    0.0,
    1.0,
)

plt.title(
    "Different Heads, Different Attention Weights"
)

plt.xlabel(
    "Key Token"
)

plt.ylabel(
    "Attention Weight"
)

plt.legend()
plt.grid(
    axis="y"
)

weight_path = (
    OUTPUT_DIR
    / "06_multihead_weights_love.png"
)

plt.tight_layout()

plt.savefig(
    weight_path,
    dpi=160,
)

plt.close()


# ==================================================
# 6. Compare Head outputs in each 3D space
# ==================================================

fig = plt.figure(
    figsize=(12, 5)
)

ax1 = fig.add_subplot(
    121,
    projection="3d",
)

ax2 = fig.add_subplot(
    122,
    projection="3d",
)

draw_vectors(
    ax1,
    output1,
    tokens,
    "Head 1 Output",
)

draw_vectors(
    ax2,
    output2,
    tokens,
    "Head 2 Output",
)

output_path = (
    OUTPUT_DIR
    / "07_multihead_outputs_3d.png"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=160,
)

plt.close()


print()
print("saved:")
print(query_path)
print(weight_path)
print(output_path)
