import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(STUDY_DIR),
)

VISUAL_DIR = (
    STUDY_DIR
    / "visuals"
)

sys.path.insert(
    0,
    str(VISUAL_DIR),
)

from plot_utils import (
    draw_vector,
    pca_project_to_3d,
    save_figure,
    set_equal_axes,
    style_3d_axis,
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

OUTPUT_DIR = (
    STUDY_DIR
    / "outputs"
    / "training_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEVICE = "cpu"
SEED = 42
LEARNING_RATE = 0.05
STEPS = 120
PROBE_TEST_INDEX = 0

CHECKPOINT_STEPS = [
    0,
    1,
    5,
    20,
    60,
    120,
]


# ==================================================
# 0. Build the same real toy Transformer
# ==================================================
#
# 이 파일은 visuals/의 설명용 toy vector와 다르다.
#
# 실제 study/src/ TinyTransformer를 만들고,
# 실제 EN-KO train pair를 사용해서
# SGD로 parameter를 업데이트한다.
#
# d_model = 8
# num_heads = 2
# -> 각 attention head의 Q/K/V 차원은 4
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


probe_example = test_examples[
    PROBE_TEST_INDEX
]


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# 1. Probe helper
# ==================================================

def get_probe_snapshot(
    step,
):
    with torch.no_grad():
        trace = model(
            probe_example["src_ids"],
            probe_example[
                "decoder_input_ids"
            ],
        )

        logits = trace[
            "logits"
        ]

        probabilities = trace[
            "probabilities"
        ]

        probe_loss = criterion(
            logits,
            probe_example[
                "gt_ids"
            ],
        ).item()

        positions = torch.arange(
            len(
                probe_example[
                    "gt_ids"
                ]
            )
        )

        gt_probabilities = probabilities[
            positions,
            probe_example[
                "gt_ids"
            ],
        ]

        mean_gt_probability = (
            gt_probabilities
            .mean()
            .item()
        )

        # ------------------------------------------
        # Decoder cross-attention 직전 representation
        # 실제 DecoderLayer의 첫 번째 masked
        # self-attention과 norm1을 다시 계산한다.
        # ------------------------------------------

        decoder_layer = model.decoder_layers[
            0
        ]

        decoder_input = trace[
            "decoder_input"
        ]

        masked_output, masked_weights = (
            decoder_layer
            .masked_self_attention(
                query_input=decoder_input,
                key_value_input=decoder_input,
                mask=trace[
                    "causal_mask"
                ],
            )
        )

        cross_query_input = (
            decoder_layer.norm1(
                decoder_input
                + masked_output
            )
        )

        encoder_output = trace[
            "encoder_output"
        ]

        cross_attention = (
            decoder_layer
            .cross_attention
        )

        head_index = 0

        Q = cross_attention.W_Q[
            head_index
        ](
            cross_query_input
        )

        K = cross_attention.W_K[
            head_index
        ](
            encoder_output
        )

        V = cross_attention.W_V[
            head_index
        ](
            encoder_output
        )

        scale = (
            cross_attention.d_k
            ** 0.5
        )

        scores = (
            Q
            @ K.T
        ) / scale

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        last_query_index = (
            Q.shape[0]
            - 1
        )

        snapshot = {
            "step":
                step,
            "probe_loss":
                probe_loss,
            "mean_gt_probability":
                mean_gt_probability,
            "src_embedding":
                trace[
                    "src_embedding"
                ].detach().cpu(),
            "encoder_output":
                encoder_output
                .detach()
                .cpu(),
            "cross_Q":
                Q.detach().cpu(),
            "cross_K":
                K.detach().cpu(),
            "cross_V":
                V.detach().cpu(),
            "cross_scores":
                scores.detach().cpu(),
            "cross_weights":
                weights.detach().cpu(),
            "last_query_index":
                last_query_index,
        }

        return snapshot


# ==================================================
# 2. Training loop
# ==================================================

train_loss_steps = []
train_losses = []

probe_steps = []
probe_losses = []
probe_gt_probabilities = []

snapshots = []


# step 0 = 학습 전
initial_snapshot = get_probe_snapshot(
    0
)

snapshots.append(
    initial_snapshot
)

probe_steps.append(
    0
)

probe_losses.append(
    initial_snapshot[
        "probe_loss"
    ]
)

probe_gt_probabilities.append(
    initial_snapshot[
        "mean_gt_probability"
    ]
)


for step in range(
    1,
    STEPS + 1,
):
    train_index = (
        (step - 1)
        % len(
            train_examples
        )
    )

    train_example = train_examples[
        train_index
    ]

    optimizer.zero_grad()

    train_trace = model(
        train_example[
            "src_ids"
        ],
        train_example[
            "decoder_input_ids"
        ],
    )

    train_loss = criterion(
        train_trace[
            "logits"
        ],
        train_example[
            "gt_ids"
        ],
    )

    train_loss.backward()

    optimizer.step()

    train_loss_steps.append(
        step
    )

    train_losses.append(
        train_loss.item()
    )

    if step in CHECKPOINT_STEPS:
        snapshot = get_probe_snapshot(
            step
        )

        snapshots.append(
            snapshot
        )

        probe_steps.append(
            step
        )

        probe_losses.append(
            snapshot[
                "probe_loss"
            ]
        )

        probe_gt_probabilities.append(
            snapshot[
                "mean_gt_probability"
            ]
        )


# ==================================================
# 3. Figure: actual training loss
# ==================================================

fig, ax = plt.subplots(
    figsize=(8.5, 4.8),
    constrained_layout=True,
)

ax.plot(
    train_loss_steps,
    train_losses,
    linewidth=1.7,
)

ax.set_title(
    "Actual EN-KO Training Loss"
)

ax.set_xlabel(
    "Optimizer Step"
)

ax.set_ylabel(
    "Cross Entropy Loss"
)

ax.grid(
    alpha=0.35,
)

loss_path = (
    OUTPUT_DIR
    / "06_actual_training_loss.png"
)

save_figure(
    fig,
    loss_path,
)


# ==================================================
# 4. Figure: held-out probe quality
# ==================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
    constrained_layout=True,
)

axes[0].plot(
    probe_steps,
    probe_losses,
    marker="o",
)

axes[0].set_title(
    "Held-out Probe Loss"
)

axes[0].set_xlabel(
    "Optimizer Step"
)

axes[0].set_ylabel(
    "Cross Entropy Loss"
)

axes[0].grid(
    alpha=0.35,
)

axes[1].plot(
    probe_steps,
    probe_gt_probabilities,
    marker="o",
)

axes[1].set_title(
    "Probe Mean GT Probability"
)

axes[1].set_xlabel(
    "Optimizer Step"
)

axes[1].set_ylabel(
    "Probability"
)

axes[1].set_ylim(
    0.0,
    1.0,
)

axes[1].grid(
    alpha=0.35,
)

probe_path = (
    OUTPUT_DIR
    / "07_actual_probe_quality.png"
)

save_figure(
    fig,
    probe_path,
)


# ==================================================
# 5. PCA: source embedding trajectory
# ==================================================
#
# 실제 source embedding은 8차원이다.
# 사람이 볼 수 있게 모든 checkpoint의 embedding을
# 한꺼번에 모아 같은 PCA basis로 3차원에 투영한다.
#
# 주의:
# PCA 3D는 시각화를 위한 projection일 뿐
# 실제 모델이 3차원에서 학습된다는 뜻이 아니다.
# ==================================================

all_embedding_vectors = []

for snapshot in snapshots:
    embedding_matrix = snapshot[
        "src_embedding"
    ]

    for row in embedding_matrix:
        all_embedding_vectors.append(
            row
        )

all_embedding_vectors = torch.stack(
    all_embedding_vectors,
    dim=0,
)

projected_embeddings = pca_project_to_3d(
    all_embedding_vectors
)


num_source_tokens = len(
    probe_example[
        "src_tokens"
    ]
)

fig = plt.figure(
    figsize=(9, 7.5),
    constrained_layout=True,
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

projected_index = 0

token_trajectories = []

for token_index in range(
    num_source_tokens
):
    token_trajectory = []

    for snapshot_index in range(
        len(snapshots)
    ):
        flat_index = (
            snapshot_index
            * num_source_tokens
            + token_index
        )

        point = projected_embeddings[
            flat_index
        ]

        token_trajectory.append(
            point
        )

    token_trajectory = torch.stack(
        token_trajectory,
        dim=0,
    )

    token_trajectories.append(
        token_trajectory
    )


for token_index in range(
    num_source_tokens
):
    token = probe_example[
        "src_tokens"
    ][token_index]

    trajectory = token_trajectories[
        token_index
    ]

    ax.plot(
        trajectory[
            :,
            0
        ].numpy(),
        trajectory[
            :,
            1
        ].numpy(),
        trajectory[
            :,
            2
        ].numpy(),
        marker="o",
        label=token,
    )

    first_point = trajectory[
        0
    ]

    last_point = trajectory[
        -1
    ]

    ax.text(
        first_point[0].item(),
        first_point[1].item(),
        first_point[2].item(),
        " start",
        fontsize=8,
    )

    ax.text(
        last_point[0].item(),
        last_point[1].item(),
        last_point[2].item(),
        " "
        + token
        + " end",
        fontsize=9,
    )


style_3d_axis(
    ax,
    "Actual Source Embedding Trajectory (8D -> PCA 3D)",
)

set_equal_axes(
    ax,
    projected_embeddings,
    margin_ratio=0.35,
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(
        1.02,
        1.0,
    ),
)

embedding_path = (
    OUTPUT_DIR
    / "08_actual_embedding_trajectory_pca3d.png"
)

save_figure(
    fig,
    embedding_path,
)


# ==================================================
# 6. Actual cross-attention weight change
# ==================================================

last_query_weights = []

for snapshot in snapshots:
    last_query_index = snapshot[
        "last_query_index"
    ]

    weights = snapshot[
        "cross_weights"
    ][
        last_query_index
    ]

    last_query_weights.append(
        weights
    )

last_query_weights = torch.stack(
    last_query_weights,
    dim=0,
)

fig, ax = plt.subplots(
    figsize=(9, 5),
    constrained_layout=True,
)

for key_index in range(
    num_source_tokens
):
    token = probe_example[
        "src_tokens"
    ][key_index]

    values = last_query_weights[
        :,
        key_index
    ]

    ax.plot(
        probe_steps,
        values.numpy(),
        marker="o",
        label=token,
    )

ax.set_title(
    "Actual Decoder Cross-Attention Change"
)

ax.set_xlabel(
    "Optimizer Step"
)

ax.set_ylabel(
    "Attention Weight"
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.grid(
    alpha=0.35,
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(
        1.02,
        1.0,
    ),
)

attention_path = (
    OUTPUT_DIR
    / "09_actual_cross_attention_vs_step.png"
)

save_figure(
    fig,
    attention_path,
)


# ==================================================
# 7. Final actual Q/K relation
# ==================================================
#
# Head 1의 실제 Q/K는 4차원이다.
# Q와 K를 같은 PCA basis로 3차원에 투영한다.
#
# 표시되는 score 숫자는 PCA 후 3D score가 아니라
# 실제 4D Q/K로 계산된 score다.
# ==================================================

final_snapshot = snapshots[
    -1
]

last_query_index = final_snapshot[
    "last_query_index"
]

final_Q = final_snapshot[
    "cross_Q"
][
    last_query_index
]

final_K = final_snapshot[
    "cross_K"
]

final_scores = final_snapshot[
    "cross_scores"
][
    last_query_index
]

qk_vectors = torch.cat(
    [
        final_Q.unsqueeze(0),
        final_K,
    ],
    dim=0,
)

qk_projected = pca_project_to_3d(
    qk_vectors,
    center=False,
)

projected_Q = qk_projected[
    0
]

projected_K = qk_projected[
    1:
]

origin = torch.zeros(3)

fig = plt.figure(
    figsize=(9, 7.5),
    constrained_layout=True,
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

draw_vector(
    ax,
    origin,
    projected_Q,
    "tab:red",
    "Decoder Q (last position)",
    linewidth=3.2,
)

for key_index in range(
    num_source_tokens
):
    token = probe_example[
        "src_tokens"
    ][key_index]

    label = (
        "K_"
        + token
        + "  true score="
        + f"{final_scores[key_index].item():.2f}"
    )

    draw_vector(
        ax,
        origin,
        projected_K[
            key_index
        ],
        "tab:blue",
        label,
    )


style_3d_axis(
    ax,
    "Actual Cross-Attention Q/K (4D -> PCA 3D)",
)

set_equal_axes(
    ax,
    qk_projected,
    margin_ratio=0.5,
)

qk_path = (
    OUTPUT_DIR
    / "10_actual_final_cross_qk_pca3d.png"
)

save_figure(
    fig,
    qk_path,
)


print("=" * 72)
print("Actual training visualization saved")
print("=" * 72)

for path in [
    loss_path,
    probe_path,
    embedding_path,
    attention_path,
    qk_path,
]:
    print(path)

print()
print("Probe:")
print(
    probe_example[
        "source_text"
    ],
    "->",
    probe_example[
        "target_text"
    ],
)

print()
print(
    "주의: PCA 3D 그림은 실제 8D/4D representation을 "
    "보기 좋게 투영한 시각화이다."
)
