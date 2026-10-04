from pathlib import Path

import matplotlib.pyplot as plt
import torch

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

embedding = torch.tensor([
    [1.0, 0.2, 0.1],
    [0.2, 1.0, 0.3],
    [0.3, 0.2, 1.0],
], dtype=torch.float32)

positions = torch.arange(
    len(tokens),
    dtype=torch.float32,
)

pe = torch.zeros(
    len(tokens),
    3,
)

for i in range(
    len(tokens)
):
    position = positions[i]

    pe[i, 0] = torch.sin(
        position
    )

    pe[i, 1] = torch.cos(
        position
    )

    pe[i, 2] = torch.sin(
        position / 10.0
    )

encoder_input = (
    embedding
    + pe
)

origin = torch.zeros(3)


# ==================================================
# 1. Three spaces
# ==================================================

fig = plt.figure(
    figsize=(15, 5.8),
    constrained_layout=True,
)

axes = []

for index in range(3):
    axis = fig.add_subplot(
        1,
        3,
        index + 1,
        projection="3d",
    )

    axes.append(
        axis
    )

for i in range(
    len(tokens)
):
    token = tokens[i]
    color = token_colors[token]

    draw_vector(
        axes[0],
        origin,
        embedding[i],
        color,
        token,
    )

    draw_vector(
        axes[1],
        origin,
        pe[i],
        color,
        token,
    )

    draw_vector(
        axes[2],
        origin,
        encoder_input[i],
        color,
        token,
    )

style_3d_axis(
    axes[0],
    "Token Embedding",
)

style_3d_axis(
    axes[1],
    "Positional Encoding",
)

style_3d_axis(
    axes[2],
    "Embedding + PE",
)

all_points = torch.cat(
    [
        origin.unsqueeze(0),
        embedding,
        pe,
        encoder_input,
    ],
    dim=0,
)

for axis in axes:
    set_equal_axes(
        axis,
        all_points,
        margin_ratio=0.28,
    )

path_1 = (
    OUTPUT_DIR
    / "01_embedding_pe_spaces_3d.png"
)

save_figure(
    fig,
    path_1,
)


# ==================================================
# 2. Vector addition for one token
# ==================================================

focus_index = 1
focus_token = tokens[
    focus_index
]

E = embedding[
    focus_index
]

P = pe[
    focus_index
]

EP = encoder_input[
    focus_index
]

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
    E,
    "tab:orange",
    "Embedding",
)

draw_vector(
    ax,
    origin,
    P,
    "tab:red",
    "PE",
)

draw_vector(
    ax,
    origin,
    EP,
    "tab:purple",
    "Embedding + PE",
    linewidth=3.2,
)

# PE를 Embedding 끝점에서 다시 그려서
# E + PE의 실제 벡터 덧셈을 보여준다.
draw_vector(
    ax,
    E,
    EP,
    "gray",
    "+ PE",
    linewidth=2.0,
)

style_3d_axis(
    ax,
    "Vector Addition: love",
)

focus_points = torch.stack(
    [
        origin,
        E,
        P,
        EP,
    ],
    dim=0,
)

set_equal_axes(
    ax,
    focus_points,
    margin_ratio=0.35,
)

path_2 = (
    OUTPUT_DIR
    / "02_embedding_plus_pe_vector_addition.png"
)

save_figure(
    fig,
    path_2,
)

print("saved:")
print(path_1)
print(path_2)
