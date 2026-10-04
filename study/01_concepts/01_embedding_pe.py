import math

import torch
import torch.nn as nn


# ==================================================
# 1. 어휘 사전
# ==================================================
#
# token ID
#
# 0 -> <PAD>
# 1 -> I
# 2 -> love
# 3 -> robotics
#

vocab_size = 4
d_model = 4


# ==================================================
# 2. 토큰 ID
# ==================================================
#
# "I love robotics"
#
# tokenization이 이미 끝났다고 가정한다.
#
# I         -> 1
# love      -> 2
# robotics  -> 3
#

token_ids = torch.tensor([
    1,
    2,
    3
])

print("======================================")
print("Token IDs")
print("======================================")

print(token_ids)
print("shape:", token_ids.shape)


# ==================================================
# 3. 임베딩 테이블
# ==================================================
#
# 실제 Transformer에서는 nn.Embedding의 값이
# 학습되는 learnable parameter이다.
#
# 여기서는 손으로 연산을 확인하기 쉽도록
# embedding 값을 직접 고정한다.
# ==================================================

embedding = nn.Embedding(
    vocab_size,
    d_model
)


with torch.no_grad():

    embedding.weight.copy_(

        torch.tensor([

            # <PAD>
            [0.0, 0.0, 0.0, 0.0],

            # I
            [1.0, 0.0, 1.0, 0.0],

            # love
            [0.0, 1.0, 0.0, 1.0],

            # robotics
            [1.0, 1.0, 1.0, 1.0],

        ])
    )


print("\n======================================")
print("Embedding Table")
print("======================================")

print(embedding.weight)


# ==================================================
# 4. 임베딩 조회
# ==================================================
#
# token ID를 embedding table의 row index로 사용한다.
#
# [1, 2, 3]
#
# ↓
#
# row 1
# row 2
# row 3
#
# 를 가져온다.
# ==================================================

embedded = embedding(
    token_ids
)


print("\n======================================")
print("Raw Embedding")
print("======================================")

print(embedded)

print(
    "shape:",
    embedded.shape
)


# ==================================================
# 5. 임베딩 스케일링
# ==================================================
#
# Attention Is All You Need:
#
# Embedding * sqrt(d_model)
#
# 현재:
#
# d_model = 4
#
# sqrt(4) = 2
# ==================================================

scale = math.sqrt(
    d_model
)


print("\nscale:")
print(scale)


scaled_embedding = (
    embedded * scale
)


print("\n======================================")
print("Scaled Embedding")
print("======================================")

print(
    scaled_embedding
)


# ==================================================
# 6. 위치 정보
# ==================================================
#
# token의 위치:
#
# I         -> position 0
# love      -> position 1
# robotics  -> position 2
# ==================================================

seq_len = token_ids.shape[0]


position = torch.arange(
    seq_len,
    dtype=torch.float32
).unsqueeze(1)


print("\n======================================")
print("Position")
print("======================================")

print(position)


# ==================================================
# 7. 위치 인코딩
# ==================================================
#
# PE(pos, 2i)
#   = sin(pos / 10000^(2i / d_model))
#
# PE(pos, 2i+1)
#   = cos(pos / 10000^(2i / d_model))
#
# 코드에서는 동일한 식을 계산하기 위해
# div_term을 먼저 만든다.
# div_term = 1 / 10000^(2i / d_model)
# ==================================================

div_term = torch.exp(

    torch.arange(
        0,
        d_model,
        2,
        dtype=torch.float32
    )

    * (
        -math.log(10000.0)
        / d_model
    )
)


print("\n======================================")
print("div_term")
print("======================================")

print(div_term)


# position * div_term
#
# sin / cos 안에 들어가는 실제 값

angles = (
    position * div_term
)


print("\n======================================")
print("Position × div_term")
print("======================================")

print(angles)


pe = torch.zeros(
    seq_len,
    d_model
)


# even dimension
pe[:, 0::2] = torch.sin(
    angles
)


# odd dimension
pe[:, 1::2] = torch.cos(
    angles
)


print("\n======================================")
print("Positional Encoding")
print("======================================")

print(pe)

print(
    "shape:",
    pe.shape
)


# ==================================================
# 8. Transformer 입력
# ==================================================
#
# x =
#
# scaled embedding
# +
# positional encoding
# ==================================================

x = (
    scaled_embedding
    + pe
)


print("\n======================================")
print("Transformer Input x")
print("======================================")

print(x)

print(
    "shape:",
    x.shape
)