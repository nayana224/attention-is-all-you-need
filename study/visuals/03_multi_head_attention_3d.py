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
query_index = 1

# 설명을 위한 toy projected vectors.
# 실제 학습 결과는 training/05_visualize_learned_vectors.py에서 본다.
Q1 = torch.tensor([
    [0.2, 0.0, 0.1],
    [0.2, 0.1, 1.0],
    [0.1, 0.0, 1.1],
])

K1 = torch.tensor([
    [0.8, 0.0, 0.1],
    [0.1, 0.1, 0.2],
    [0.0, 0.0, 1.0],
])

V1 = torch.tensor([
    [-0.6, 0.2, 0.0],
    [0.2, 0.8, 0.2],
    [0.8, 0.1, 1.0],
])

Q2 = torch.tensor([
    [1.0, 0.2, 0.0],
    [1.1, 0.1, 0.0],
    [0.8, 0.1, 0.1],
])

K2 = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.2, 0.1, 0.0],
    [0.0, 0.2, 0.1],
])

V2 = torch.tensor([
    [-0.9, 0.0, 0.1],
    [0.2, 0.6, 0.2],
    [0.4, 0.1, 0.8],
])


def run_attention(
    Q,
    K,
    V,
):
    scores = (
        Q @ K.T
    )

    scaled_scores = (
        scores
        / math.sqrt(3)
    )

    weights = F.softmax(
        scaled_scores,
        dim=-1,
    )

    output = (
        weights @ V
    )

    return (
        scaled_scores,
        weights,
        output,
    )


scores1, weights1, output1 = run_attention(
    Q1,
    K1,
    V1,
)

scores2, weights2, output2 = run_attention(
    Q2,
    K2,
    V2,
)


def draw_relation_space(
    Q,
    K,
    scores,
    title,
    filename,
):
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
        Q[query_index],
        "tab:red",
        "Q_love",
        linewidth=3.2,
        label_offset=(0.05, 0.06, 0.06),
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
            "robots": (-0.12, 0.06, 0.06),
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
        title,
    )

    points = torch.cat(
        [
            origin.unsqueeze(0),
            Q[query_index].unsqueeze(0),
            K,
        ],
        dim=0,
    )

    set_equal_axes(
        ax,
        points,
        margin_ratio=0.42,
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    save_figure(
        fig,
        path,
    )

    return path


path_1 = draw_relation_space(
    Q1,
    K1,
    scores1,
    "Head 1 relation space",
    "07_head1_relation_space.png",
)

path_2 = draw_relation_space(
    Q2,
    K2,
    scores2,
    "Head 2 relation space",
    "08_head2_relation_space.png",
)


fig, ax = plt.subplots(
    figsize=(8.5, 4.8),
    constrained_layout=True,
)

x_positions = torch.arange(
    len(tokens)
).numpy()

bar_width = 0.35

bars_1 = ax.bar(
    x_positions
    - bar_width / 2.0,
    weights1[
        query_index
    ].numpy(),
    width=bar_width,
    label="Head 1",
)

bars_2 = ax.bar(
    x_positions
    + bar_width / 2.0,
    weights2[
        query_index
    ].numpy(),
    width=bar_width,
    label="Head 2",
)

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    tokens
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.set_title(
    "Different heads, different attention weights"
)

ax.set_xlabel(
    "Key Token"
)

ax.set_ylabel(
    "Attention Weight"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.35,
)

for bar in list(bars_1) + list(bars_2):
    height = bar.get_height()

    ax.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

path_3 = (
    OUTPUT_DIR
    / "09_multihead_attention_weights.png"
)

save_figure(
    fig,
    path_3,
)


fig = plt.figure(
    figsize=(12, 5.8),
    constrained_layout=True,
)

ax1 = fig.add_subplot(
    121,
    projection="3d",
)

ax2 = fig.add_subplot(
    122,
    projection="3d",
)

draw_vector(
    ax1,
    origin,
    output1[query_index],
    "tab:purple",
    "Head 1 output",
    linewidth=3.0,
)

draw_vector(
    ax2,
    origin,
    output2[query_index],
    "tab:brown",
    "Head 2 output",
    linewidth=3.0,
)

style_3d_axis(
    ax1,
    "Head 1 output",
)

style_3d_axis(
    ax2,
    "Head 2 output",
)

output_points = torch.stack(
    [
        origin,
        output1[query_index],
        output2[query_index],
    ],
    dim=0,
)

set_equal_axes(
    ax1,
    output_points,
    margin_ratio=0.5,
)

set_equal_axes(
    ax2,
    output_points,
    margin_ratio=0.5,
)

path_4 = (
    OUTPUT_DIR
    / "10_multihead_output_vectors.png"
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
