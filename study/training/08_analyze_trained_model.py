import csv
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
    pca_project_to_3d,
    save_figure,
)

from src.data import (
    load_translation_pairs,
    make_example,
)

from src.factory import (
    build_study_objects,
)

from src.tokenization import (
    invert_vocab,
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

CHECKPOINT_PATH = (
    STUDY_DIR
    / "outputs"
    / "checkpoints"
    / "tiny_transformer_best.pt"
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
PROBE_INDEX = 0


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
    initial_model,
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


(
    _,
    _,
    _,
    trained_model,
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

trained_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

initial_model.eval()
trained_model.eval()

test_pairs = load_translation_pairs(
    TEST_PATH
)

probe_pair = test_pairs[
    PROBE_INDEX
]

probe_example = make_example(
    probe_pair,
    src_vocab,
    tgt_vocab,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss()

id_to_target = invert_vocab(
    tgt_vocab
)


def get_snapshot(
    model,
):
    with torch.no_grad():
        trace = model(
            probe_example[
                "src_ids"
            ],
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

        loss = criterion(
            logits,
            probe_example[
                "gt_ids"
            ],
        ).item()

        pred_ids = logits.argmax(
            dim=-1
        )

    pred_tokens = []

    for pred_id in pred_ids:
        pred_tokens.append(
            id_to_target[
                pred_id.item()
            ]
        )

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

    snapshot = {
        "trace":
            trace,
        "loss":
            loss,
        "pred_tokens":
            pred_tokens,
        "gt_probabilities":
            gt_probabilities
            .detach()
            .cpu(),
    }

    return snapshot


before = get_snapshot(
    initial_model
)

after = get_snapshot(
    trained_model
)


print("=" * 72)
print("Analyze Well-Trained Tiny Transformer")
print("=" * 72)
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
print("Before loss:")
print(
    before[
        "loss"
    ]
)
print("After loss:")
print(
    after[
        "loss"
    ]
)
print()
print("Before prediction:")
print(
    before[
        "pred_tokens"
    ]
)
print("After prediction:")
print(
    after[
        "pred_tokens"
    ]
)
print("GT:")
print(
    probe_example[
        "gt_tokens"
    ]
)


# ==================================================
# 1. GT probability before / after
# ==================================================

positions = list(
    range(
        len(
            probe_example[
                "gt_tokens"
            ]
        )
    )
)

bar_width = 0.35

fig, ax = plt.subplots(
    figsize=(10, 5),
    constrained_layout=True,
)

x = torch.arange(
    len(
        positions
    )
).numpy()

ax.bar(
    x - bar_width / 2.0,
    before[
        "gt_probabilities"
    ].numpy(),
    width=bar_width,
    label="Before Training",
)

ax.bar(
    x + bar_width / 2.0,
    after[
        "gt_probabilities"
    ].numpy(),
    width=bar_width,
    label="After Training",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    probe_example[
        "gt_tokens"
    ]
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.set_title(
    "GT Token Probability: Before vs After"
)

ax.set_xlabel(
    "Ground Truth Token"
)

ax.set_ylabel(
    "Probability"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3,
)

probability_path = (
    OUTPUT_DIR
    / "12_trained_gt_probability_before_after.png"
)

save_figure(
    fig,
    probability_path,
)


# ==================================================
# 2. Cross-Attention before / after
# ==================================================

before_cross = before[
    "trace"
][
    "cross_attention"
][
    0
][
    0
].detach().cpu()

after_cross = after[
    "trace"
][
    "cross_attention"
][
    0
][
    0
].detach().cpu()

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.5),
    constrained_layout=True,
)

for axis_index in range(2):
    if axis_index == 0:
        matrix = before_cross
        title = "Before Training"
    else:
        matrix = after_cross
        title = "After Training"

    image = axes[
        axis_index
    ].imshow(
        matrix.numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    axes[
        axis_index
    ].set_title(
        title
    )

    axes[
        axis_index
    ].set_xticks(
        range(
            len(
                probe_example[
                    "src_tokens"
                ]
            )
        )
    )

    axes[
        axis_index
    ].set_xticklabels(
        probe_example[
            "src_tokens"
        ],
        rotation=25,
        ha="right",
    )

    axes[
        axis_index
    ].set_yticks(
        range(
            len(
                probe_example[
                    "decoder_input_tokens"
                ]
            )
        )
    )

    axes[
        axis_index
    ].set_yticklabels(
        [
            "q0",
            "q1",
            "q2",
            "q3",
        ]
    )

    axes[
        axis_index
    ].set_xlabel(
        "Encoder Key Token"
    )

    axes[
        axis_index
    ].set_ylabel(
        "Decoder Query Position"
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ].item()

            axes[
                axis_index
            ].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

fig.suptitle(
    "Cross-Attention Head 1: Before vs After"
)

fig.colorbar(
    image,
    ax=axes,
    shrink=0.82,
    label="Attention Weight",
)

attention_path = (
    OUTPUT_DIR
    / "13_trained_cross_attention_before_after.png"
)

save_figure(
    fig,
    attention_path,
)


# ==================================================
# 3. Source embedding before / after PCA 3D
# ==================================================

before_embedding = before[
    "trace"
][
    "src_embedding"
].detach().cpu()

after_embedding = after[
    "trace"
][
    "src_embedding"
].detach().cpu()

combined_embeddings = torch.cat(
    [
        before_embedding,
        after_embedding,
    ],
    dim=0,
)

projected = pca_project_to_3d(
    combined_embeddings
)

num_tokens = before_embedding.shape[
    0
]

before_projected = projected[
    :num_tokens
]

after_projected = projected[
    num_tokens:
]

fig = plt.figure(
    figsize=(9, 7.5),
    constrained_layout=True,
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

for token_index in range(
    num_tokens
):
    token = probe_example[
        "src_tokens"
    ][
        token_index
    ]

    start = before_projected[
        token_index
    ]

    end = after_projected[
        token_index
    ]

    ax.plot(
        [
            start[0].item(),
            end[0].item(),
        ],
        [
            start[1].item(),
            end[1].item(),
        ],
        [
            start[2].item(),
            end[2].item(),
        ],
        marker="o",
        label=token,
    )

ax.set_title(
    "Source Embedding Shift: Before -> After (8D -> PCA 3D)"
)

ax.set_xlabel(
    "PCA dim 0"
)

ax.set_ylabel(
    "PCA dim 1"
)

ax.set_zlabel(
    "PCA dim 2"
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
    / "14_trained_embedding_before_after_pca3d.png"
)

save_figure(
    fig,
    embedding_path,
)


print()
print("saved:")
print(probability_path)
print(attention_path)
print(embedding_path)
