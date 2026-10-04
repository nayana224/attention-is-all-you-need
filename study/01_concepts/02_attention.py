import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================
# 0. 재현성
# ==================================================

torch.manual_seed(42)

# ==================================================
# 1. 입력 x
# ==================================================
#
# 이전 단계:
#
# token IDs
# -> Embedding
# -> Embedding + Positional Encoding
# -> x
#
# 여기서는 Attention 계산 자체에 집중하기 위해
# x를 간단한 toy 값으로 고정한다.
#
# row 0 = I
# row 1 = love
# row 2 = robotics
#
# shape:
# [seq_len, d_model]
# = [3, 4]
# ==================================================

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # I
    [0.0, 2.0, 0.0, 2.0],   # love
    [1.0, 1.0, 1.0, 1.0],   # robotics
])


print("======================================")
print("Input x")
print("======================================")

print(x)
print("shape:", x.shape)


# ==================================================
# 2. Q, K, V 선형 투영
# ==================================================
#
# 같은 x로부터
#
# Q = x W_Q
# K = x W_K
# V = x W_V
#
# 를 만든다.
#
# 각 Linear layer는 서로 다른 learnable parameter를 가진다.
# ==================================================

d_model = 4
d_k = 4
d_v = 4


W_Q = nn.Linear(
    d_model,
    d_k,
    bias=False
)

W_K = nn.Linear(
    d_model,
    d_k,
    bias=False
)

W_V = nn.Linear(
    d_model,
    d_v,
    bias=False
)


print("\n======================================")
print("W_Q")
print("======================================")
print(W_Q.weight)


print("\n======================================")
print("W_K")
print("======================================")
print(W_K.weight)


print("\n======================================")
print("W_V")
print("======================================")
print(W_V.weight)

Q = W_Q(x)
K = W_K(x)
V = W_V(x)


print("\n======================================")
print("Q")
print("======================================")

print(Q)
print("shape:", Q.shape)


print("\n======================================")
print("K")
print("======================================")

print(K)
print("shape:", K.shape)


print("\n======================================")
print("V")
print("======================================")

print(V)
print("shape:", V.shape)


# ==================================================
# 3. Q K^T 내적
# ==================================================
#
# Q:
#
# [Q_I
#  Q_love
#  Q_robotics]
#
#
# K^T:
#
# [K_I^T, K_love^T, K_robotics^T]
#
#
# 따라서:
#
# scores[i, j]
#
# =
#
# Q_i dot K_j
#
#
# 예:
#
# scores[0, 0]
# = Q_I dot K_I
#
# scores[0, 1]
# = Q_I dot K_love
#
# scores[0, 2]
# = Q_I dot K_robotics
# ==================================================

scores = Q @ K.T


print("\n======================================")
print("Q @ K^T")
print("======================================")

print(scores)

print(
    "shape:",
    scores.shape
)


# ==================================================
# 4. 첫 번째 Query를 따로 확인
# ==================================================
#
# 첫 번째 row:
#
# "I"라는 query가
#
# I / love / robotics
#
# 각각의 key와 얼마나 compatible한지 계산한 값
# ==================================================

print("\n======================================")
print("Scores for query = I")
print("======================================")

print(
    "Q_I dot K_I        :",
    scores[0, 0]
)

print(
    "Q_I dot K_love     :",
    scores[0, 1]
)

print(
    "Q_I dot K_robotics :",
    scores[0, 2]
)


# ==================================================
# 5. 스케일링
# ==================================================
#
# Scaled Dot-Product Attention:
#
# QK^T / sqrt(d_k)
#
# 현재:
#
# d_k = 4
# sqrt(4) = 2
# ==================================================

scale = math.sqrt(
    d_k
)

scaled_scores = (
    scores / scale
)


print("\n======================================")
print("Scaled Scores")
print("======================================")

print(
    "sqrt(d_k):",
    scale
)

print(
    scaled_scores
)


# ==================================================
# 6. Softmax 정규화
# ==================================================
#
# 각 query row마다 softmax를 적용한다.
#
# 결과:
#
# 각 query가
# I / love / robotics의 V를
# 각각 얼마나 사용할 것인지 나타내는 weight
#
# 각 row의 합은 1.
# ==================================================

attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)


print("\n======================================")
print("Attention Weights")
print("======================================")

print(
    attention_weights
)


print("\nRow sums:")

print(
    attention_weights.sum(
        dim=-1
    )
)


# ==================================================
# 7. 첫 번째 Query의 Attention Weight
# ==================================================

print("\n======================================")
print("Attention Weights for query = I")
print("======================================")

print(
    "I -> I        :",
    attention_weights[0, 0]
)

print(
    "I -> love     :",
    attention_weights[0, 1]
)

print(
    "I -> robotics :",
    attention_weights[0, 2]
)


# ==================================================
# 8. V의 가중합
# ==================================================
#
# Attention(Q, K, V)
#
# =
#
# softmax(
#     QK^T / sqrt(d_k)
# ) V
#
#
# 첫 번째 output은:
#
# output_I
#
# =
#
# weight(I -> I)        * V_I
# +
# weight(I -> love)     * V_love
# +
# weight(I -> robotics) * V_robotics
# ==================================================

output = (
    attention_weights @ V
)


print("\n======================================")
print("Attention Output")
print("======================================")

print(output)

print(
    "shape:",
    output.shape
)


# ==================================================
# 9. 첫 번째 Output을 직접 계산해서 확인
# ==================================================

output_I_manual = (

    attention_weights[0, 0] * V[0]

    +

    attention_weights[0, 1] * V[1]

    +

    attention_weights[0, 2] * V[2]
)


print("\n======================================")
print("Manual Output for I")
print("======================================")

print(
    output_I_manual
)


print("\nMatrix Output for I:")

print(
    output[0]
)