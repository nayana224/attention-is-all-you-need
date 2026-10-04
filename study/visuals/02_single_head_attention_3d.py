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
# 0. Input x
# ==================================================
#
# 실제 Transformer는 고차원 vector를 사용한다.
#
# 여기서는 사람이 직접 볼 수 있도록
# d_model = 3 으로 줄인다.
# ==================================================

tokens = [
    "I",
    "love",
    "robots",
]

x = torch.tensor([
    [1.0, 0.2, 0.1],
    [0.2, 1.0, 0.3],
    [0.3, 0.2, 1.0],
])

d_model = 3


# ==================================================
# 1. Q / K / V Projection
# ==================================================

W_Q = nn.Linear(
    d_model,
    d_model,
    bias=False,
)

W_K = nn.Linear(
    d_model,
    d_model,
    bias=False,
)

W_V = nn.Linear(
    d_model,
    d_model,
    bias=False,
)

Q = W_Q(
    x
)

K = W_K(
    x
)

V = W_V(
    x
)


# ==================================================
# 2. Q K^T
# ==================================================

scores = (
    Q @ K.T
)

scaled_scores = (
    scores
    / math.sqrt(d_model)
)

attention_weights = F.softmax(
    scaled_scores,
    dim=-1,
)

output = (
    attention_weights
    @ V
)


# ==================================================
# 3. Focus on query = love
# ==================================================

query_index = 1

query_token = tokens[
    query_index
]

query_weights = attention_weights[
    query_index
]

weighted_values = []

for i in range(
    len(tokens)
):
    weighted_value = (
        query_weights[i]
        * V[i]
    )

    weighted_values.append(
        weighted_value
    )

weighted_values = torch.stack(
    weighted_values
)


print("=" * 70)
print("Single-Head Attention in 3D")
print("=" * 70)

print()
print("Input x:")
print(x)

print()
print("Q:")
print(Q)

print()
print("K:")
print(K)

print()
print("V:")
print(V)

print()
print("Q @ K^T:")
print(scores)

print()
print("Attention weights:")
print(attention_weights)

print()
print("Focused query:")
print(query_token)

for i in range(
    len(tokens)
):
    print(
        query_token,
        "->",
        tokens[i],
        ":",
        query_weights[i].item(),
    )

print()
print("Weighted values:")

for i in range(
    len(tokens)
):
    print(
        tokens[i],
        ":",
        weighted_values[i],
    )

print()
print("Attention output for query =", query_token)
print(
    output[query_index]
)


# ==================================================
# 4. Plot helpers
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
# 5. Plot x / Q / K / V
# ==================================================

fig = plt.figure(
    figsize=(12, 10)
)

ax1 = fig.add_subplot(
    221,
    projection="3d",
)

ax2 = fig.add_subplot(
    222,
    projection="3d",
)

ax3 = fig.add_subplot(
    223,
    projection="3d",
)

ax4 = fig.add_subplot(
    224,
    projection="3d",
)

draw_vectors(
    ax1,
    x,
    tokens,
    "Input x",
)

draw_vectors(
    ax2,
    Q,
    tokens,
    "Query Space",
)

draw_vectors(
    ax3,
    K,
    tokens,
    "Key Space",
)

draw_vectors(
    ax4,
    V,
    tokens,
    "Value Space",
)

qkv_path = (
    OUTPUT_DIR
    / "02_qkv_spaces_3d.png"
)

plt.tight_layout()

plt.savefig(
    qkv_path,
    dpi=160,
)

plt.close()


# ==================================================
# 6. Attention weight bar chart
# ==================================================

plt.figure(
    figsize=(7, 4)
)

plt.bar(
    tokens,
    query_weights.detach().numpy(),
)

plt.ylim(
    0.0,
    1.0,
)

plt.title(
    "Attention Weights for Query = love"
)

plt.xlabel(
    "Key Token"
)

plt.ylabel(
    "Attention Weight"
)

plt.grid(
    axis="y"
)

weight_path = (
    OUTPUT_DIR
    / "03_attention_weights_love.png"
)

plt.tight_layout()

plt.savefig(
    weight_path,
    dpi=160,
)

plt.close()


# ==================================================
# 7. Weighted V and final output
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

weighted_labels = []

for i in range(
    len(tokens)
):
    label = (
        "w * V_"
        + tokens[i]
    )

    weighted_labels.append(
        label
    )

draw_vectors(
    ax1,
    weighted_values,
    weighted_labels,
    "Weighted Value Vectors",
)

final_output = output[
    query_index
].unsqueeze(0)

draw_vectors(
    ax2,
    final_output,
    [
        "Attention Output",
    ],
    "Weighted Sum of V",
)

output_path = (
    OUTPUT_DIR
    / "04_weighted_values_and_output_3d.png"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=160,
)

plt.close()


print()
print("saved:")
print(qkv_path)
print(weight_path)
print(output_path)
