from pathlib import Path

import matplotlib.pyplot as plt
import torch


STUDY_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = STUDY_DIR / "outputs" / "visuals"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

torch.manual_seed(42)


# ==================================================
# 0. Visualization-only setup
# ==================================================
#
# 실제 Transformer는 훨씬 높은 차원의 embedding을 사용한다.
#
# 여기서는 사람이 공간으로 볼 수 있도록
# d_model = 3 으로 줄여서 관찰한다.
# ==================================================

tokens = [
    "I",
    "love",
    "robots",
]

d_model = 3


# ==================================================
# 1. Toy Embedding Table
# ==================================================
#
# 각 token은 3차원 embedding vector 하나를 가진다.
#
# 실제 코드에서는:
# nn.Embedding(vocab_size, d_model)
#
# 여기서는 시각화에 집중하기 위해
# 직접 3차원 값을 고정한다.
# ==================================================

embedding = torch.tensor([
    [1.0, 0.2, 0.1],
    [0.2, 1.0, 0.3],
    [0.3, 0.2, 1.0],
])


# ==================================================
# 2. 3D Positional Encoding
# ==================================================
#
# 논문의 sin / cos positional encoding을
# 3차원으로 잘라서 보는 visualization용 예시이다.
#
# dim 0 = sin(position)
# dim 1 = cos(position)
# dim 2 = sin(position / 10)
#
# 실제 논문 구현은 더 많은 차원과
# 서로 다른 frequency를 사용한다.
# ==================================================

positions = torch.arange(
    len(tokens),
    dtype=torch.float32,
)

pe = torch.zeros(
    len(tokens),
    d_model,
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


# ==================================================
# 3. Print values
# ==================================================

print("=" * 70)
print("3D Embedding + Positional Encoding")
print("=" * 70)

for i in range(
    len(tokens)
):
    print()
    print("token:", tokens[i])
    print("embedding:")
    print(embedding[i])

    print("positional encoding:")
    print(pe[i])

    print("embedding + PE:")
    print(encoder_input[i])


# ==================================================
# 4. Plot helper
# ==================================================

def draw_vector(
    axis,
    vector,
    label,
):
    x = vector[0].item()
    y = vector[1].item()
    z = vector[2].item()

    axis.quiver(
        0.0,
        0.0,
        0.0,
        x,
        y,
        z,
        arrow_length_ratio=0.08,
    )

    axis.text(
        x,
        y,
        z,
        label,
    )


# ==================================================
# 5. Plot each stage
# ==================================================

fig = plt.figure(
    figsize=(15, 5)
)

ax1 = fig.add_subplot(
    131,
    projection="3d",
)

ax2 = fig.add_subplot(
    132,
    projection="3d",
)

ax3 = fig.add_subplot(
    133,
    projection="3d",
)


for i in range(
    len(tokens)
):
    draw_vector(
        ax1,
        embedding[i],
        tokens[i],
    )

    draw_vector(
        ax2,
        pe[i],
        tokens[i],
    )

    draw_vector(
        ax3,
        encoder_input[i],
        tokens[i],
    )


ax1.set_title(
    "Token Embedding"
)

ax2.set_title(
    "Positional Encoding"
)

ax3.set_title(
    "Embedding + PE"
)


for axis in [
    ax1,
    ax2,
    ax3,
]:
    axis.set_xlabel("dim 0")
    axis.set_ylabel("dim 1")
    axis.set_zlabel("dim 2")


output_path = (
    OUTPUT_DIR
    / "01_embedding_pe_3d.png"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=160,
)

print()
print("saved:")
print(output_path)

plt.show()
