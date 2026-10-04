import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from plot_utils import (
    draw_vector,
    save_figure,
    set_equal_axes,
    style_3d_axis,
)


STUDY_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = STUDY_DIR / "outputs" / "visuals"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

tokens = [
    "I",
    "love",
    "robots",
]

token_colors = {
    "I": "tab:blue",
    "love": "tab:orange",
    "robots": "tab:green",
}

origin = torch.zeros(3)

x = torch.tensor([
    [1.0, 0.2, 0.1],
    [0.2, 1.0, 0.3],
    [0.3, 0.2, 1.0],
], dtype=torch.float32)

# 개념 시각화를 위해 해석하기 쉬운 projection을 사용한다.
W_Q = torch.eye(3)
W_K = torch.eye(3)

W_V = torch.tensor([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.3, 0.0, 1.0],
], dtype=torch.float32)

Q = x @ W_Q.T
K = x @ W_K.T
V = x @ W_V.T

scores = (
    Q @ K.T
)

scaled_scores = (
    scores
    / math.sqrt(3)
)

attention_weights = F.softmax(
    scaled_scores,
    dim=-1,
)

output = (
    attention_weights
    @ V
)

query_index = 1
query_token = tokens[
    query_index
]

query_vector = Q[
    query_index
]

query_scores = scaled_scores[
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


# ==================================================
# 1. 같은 토큰의 서로 다른 역할
# ==================================================

fig = plt.figure(
    figsize=(8, 7),
    constrained_layout=True,
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

draw_vector(
    ax,
    origin,
    x[query_index],
    "black",
    "x_love",
    label_offset=(0.05, -0.05, -0.04),
)

draw_vector(
    ax,
    origin,
    Q[query_index],
    "tab:red",
    "Q_love",
    label_offset=(0.06, 0.04, 0.05),
)

draw_vector(
    ax,
    origin,
    K[query_index],
    "tab:purple",
    "K_love",
    label_offset=(-0.16, 0.08, 0.02),
)

draw_vector(
    ax,
    origin,
    V[query_index],
    "tab:brown",
    "V_love",
    label_offset=(0.03, -0.08, 0.06),
)

style_3d_axis(
    ax,
    "Same token, different roles",
)

role_points = torch.stack(
    [
        origin,
        x[query_index],
        Q[query_index],
        K[query_index],
        V[query_index],
    ],
    dim=0,
)

set_equal_axes(
    ax,
    role_points,
    margin_ratio=0.4,
)

path_1 = (
    OUTPUT_DIR
    / "03_same_token_different_roles.png"
)

save_figure(
    fig,
    path_1,
)


# ==================================================
# 2. Query와 모든 Key 비교
# ==================================================

fig = plt.figure(
    figsize=(8, 7),
    constrained_layout=True,
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

draw_vector(
    ax,
    origin,
    query_vector,
    "tab:red",
    "Q_love",
    linewidth=3.2,
)

for i in range(
    len(tokens)
):
    token = tokens[i]

    label = (
        "K_"
        + token
    )

    label_offsets = {
        "I": (0.05, -0.08, -0.03),
        "love": (0.06, 0.02, 0.03),
        "robots": (-0.10, 0.06, 0.06),
    }

    draw_vector(
        ax,
        origin,
        K[i],
        token_colors[token],
        label,
        label_offset=label_offsets[token],
    )

style_3d_axis(
    ax,
    "Q_love compared with all Keys",
)

comparison_points = torch.cat(
    [
        origin.unsqueeze(0),
        query_vector.unsqueeze(0),
        K,
    ],
    dim=0,
)

set_equal_axes(
    ax,
    comparison_points,
    margin_ratio=0.4,
)

path_2 = (
    OUTPUT_DIR
    / "04_query_vs_all_keys.png"
)

save_figure(
    fig,
    path_2,
)


# ==================================================
# 3. Score에서 Softmax Weight로 변환
# ==================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
    constrained_layout=True,
)

bars_1 = axes[0].bar(
    tokens,
    query_scores.detach().numpy(),
)

axes[0].set_title(
    "Scaled QK Scores"
)

axes[0].set_ylabel(
    "Score"
)

axes[0].grid(
    axis="y",
    alpha=0.35,
)

bars_2 = axes[1].bar(
    tokens,
    query_weights.detach().numpy(),
)

axes[1].set_title(
    "Softmax Attention Weights"
)

axes[1].set_ylabel(
    "Weight"
)

axes[1].set_ylim(
    0.0,
    1.0,
)

axes[1].grid(
    axis="y",
    alpha=0.35,
)

for bar in bars_1:
    height = bar.get_height()

    axes[0].text(
        bar.get_x()
        + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

for bar in bars_2:
    height = bar.get_height()

    axes[1].text(
        bar.get_x()
        + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

fig.suptitle(
    "Query = love: score -> softmax -> attention weight"
)

path_3 = (
    OUTPUT_DIR
    / "05_scores_and_attention_weights.png"
)

save_figure(
    fig,
    path_3,
)


# ==================================================
# 4. V의 가중합
# ==================================================

fig = plt.figure(
    figsize=(8, 7),
    constrained_layout=True,
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

current = origin.clone()

path_points = [
    origin.clone()
]

for i in range(
    len(tokens)
):
    next_point = (
        current
        + weighted_values[i]
    )

    token = tokens[i]

    label = (
        f"{query_weights[i].item():.2f}"
        + " * V_"
        + token
    )

    weighted_offsets = {
        "I": (-0.02, -0.08, -0.02),
        "love": (0.04, 0.04, 0.03),
        "robots": (0.05, -0.02, 0.06),
    }

    draw_vector(
        ax,
        current,
        next_point,
        token_colors[token],
        label,
        label_offset=weighted_offsets[token],
    )

    current = next_point

    path_points.append(
        current.clone()
    )

draw_vector(
    ax,
    origin,
    current,
    "tab:red",
    "Attention Output",
    linewidth=3.4,
    label_offset=(-0.18, 0.08, 0.08),
)

style_3d_axis(
    ax,
    "Weighted sum of V",
)

path_points_tensor = torch.stack(
    path_points,
    dim=0,
)

set_equal_axes(
    ax,
    path_points_tensor,
    margin_ratio=0.45,
)

path_4 = (
    OUTPUT_DIR
    / "06_weighted_sum_of_v.png"
)

save_figure(
    fig,
    path_4,
)

print("saved:")
print(path_1)
print(path_2)
print(path_3)
print(path_4)
