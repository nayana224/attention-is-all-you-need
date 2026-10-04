import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 0. 재현성
# ==================================================
#
# nn.Linear의 랜덤 초기화를
# 매 실행마다 동일하게 만든다.
# ==================================================

torch.manual_seed(42)


# ==================================================
# 1. Transformer Input x
# ==================================================
#
# 이전 단계와 동일한 예제 입력을 사용한다.
#
# row 0 = I
# 행 1 = love
# 행 2 = robotics
#
# 텐서 크기:
#
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
# 2. 멀티헤드 설정
# ==================================================
#
# d_model = 4
# num_heads = 2
#
# 논문의 base model처럼:
#
# d_k = d_model / num_heads
# d_v = d_model / num_heads
#
# 여기서는:
#
# d_k = 2
# d_v = 2
# ==================================================

d_model = 4
num_heads = 2

d_k = d_model // num_heads
d_v = d_model // num_heads


print("\n======================================")
print("Multi-Head Configuration")
print("======================================")

print("d_model   :", d_model)
print("num_heads :", num_heads)
print("d_k       :", d_k)
print("d_v       :", d_v)


# ==================================================
# 3. Head 1 투영
# ==================================================
#
# 같은 x를:
#
# Q1 = x W_Q1
# K1 = x W_K1
# V1 = x W_V1
#
# 로 선형 투영한다.
#
# PyTorch nn.Linear 계산:
#
# y = x @ weight.T
#
# ==================================================

W_Q1 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

W_K1 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

W_V1 = nn.Linear(
    d_model,
    d_v,
    bias=False
)


print("\n======================================")
print("Head 1 Weight Matrices")
print("======================================")

print("\nW_Q1.weight:")
print(W_Q1.weight)

print("\nW_K1.weight:")
print(W_K1.weight)

print("\nW_V1.weight:")
print(W_V1.weight)


Q1 = W_Q1(x)
K1 = W_K1(x)
V1 = W_V1(x)


print("\n======================================")
print("Head 1 - Q, K, V")
print("======================================")

print("\nQ1:")
print(Q1)
print("shape:", Q1.shape)

print("\nK1:")
print(K1)
print("shape:", K1.shape)

print("\nV1:")
print(V1)
print("shape:", V1.shape)


# ==================================================
# 4. Head 2 투영
# ==================================================
#
# Head 2는 같은 x를 입력으로 받지만
# 완전히 다른 파라미터를 사용한다.
#
# Q2 = x W_Q2
# K2 = x W_K2
# V2 = x W_V2
# ==================================================

W_Q2 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

W_K2 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

W_V2 = nn.Linear(
    d_model,
    d_v,
    bias=False
)


print("\n======================================")
print("Head 2 Weight Matrices")
print("======================================")

print("\nW_Q2.weight:")
print(W_Q2.weight)

print("\nW_K2.weight:")
print(W_K2.weight)

print("\nW_V2.weight:")
print(W_V2.weight)


Q2 = W_Q2(x)
K2 = W_K2(x)
V2 = W_V2(x)


print("\n======================================")
print("Head 2 - Q, K, V")
print("======================================")

print("\nQ2:")
print(Q2)
print("shape:", Q2.shape)

print("\nK2:")
print(K2)
print("shape:", K2.shape)

print("\nV2:")
print(V2)
print("shape:", V2.shape)


# ==================================================
# 5. 스케일드 닷프로덕트 어텐션
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
# ==================================================

def scaled_dot_product_attention(Q, K, V):

    # ----------------------------------------------
    # 1) Query-Key 관계 점수
    # ----------------------------------------------

    scores = Q @ K.T


    # ----------------------------------------------
    # 2) 스케일링
    # ----------------------------------------------

    scaled_scores = (
        scores / math.sqrt(Q.shape[-1])
    )


    # ----------------------------------------------
    # 3) Softmax 정규화
    # ----------------------------------------------

    attention_weights = F.softmax(
        scaled_scores,
        dim=-1
    )


    # ----------------------------------------------
    # 4) V의 가중합
    # ----------------------------------------------

    output = (
        attention_weights @ V
    )


    return (
        output,
        scores,
        scaled_scores,
        attention_weights
    )


# ==================================================
# 6. Head 1 Attention
# ==================================================

(
    head1,
    scores1,
    scaled_scores1,
    weights1
) = scaled_dot_product_attention(
    Q1,
    K1,
    V1
)


print("\n======================================")
print("Head 1 Attention")
print("======================================")

print("\nQ1 @ K1^T:")
print(scores1)

print("\nScaled Scores:")
print(scaled_scores1)

print("\nAttention Weights:")
print(weights1)

print("\nRow sums:")
print(weights1.sum(dim=-1))

print("\nHead 1 Output:")
print(head1)

print(
    "shape:",
    head1.shape
)


# ==================================================
# 7. Head 1에서 I token을 직접 해석
# ==================================================
#
# 첫 번째 Query = I
#
# weights1[0]:
#
# I -> I
# I -> love
# I -> robotics
#
#
# 그 가중치로:
#
# V1_I
# V1_love
# V1_robotics
#
# 를 가중합한다.
# ==================================================

print("\n======================================")
print("Head 1 - Query I")
print("======================================")

print("\nAttention weights:")

print(
    "I -> I        :",
    weights1[0, 0]
)

print(
    "I -> love     :",
    weights1[0, 1]
)

print(
    "I -> robotics :",
    weights1[0, 2]
)


head1_I_manual = (

    weights1[0, 0] * V1[0]

    +

    weights1[0, 1] * V1[1]

    +

    weights1[0, 2] * V1[2]
)


print("\nManual Head 1 output for I:")
print(head1_I_manual)

print("\nMatrix Head 1 output for I:")
print(head1[0])


# ==================================================
# 8. Head 2 Attention
# ==================================================

(
    head2,
    scores2,
    scaled_scores2,
    weights2
) = scaled_dot_product_attention(
    Q2,
    K2,
    V2
)


print("\n======================================")
print("Head 2 Attention")
print("======================================")

print("\nQ2 @ K2^T:")
print(scores2)

print("\nScaled Scores:")
print(scaled_scores2)

print("\nAttention Weights:")
print(weights2)

print("\nRow sums:")
print(weights2.sum(dim=-1))

print("\nHead 2 Output:")
print(head2)

print(
    "shape:",
    head2.shape
)


# ==================================================
# 9. 두 Head 결과 비교
# ==================================================
#
# 같은 x를 입력했지만
#
# Head 1:
# W_Q1 / W_K1 / W_V1
#
# Head 2:
# W_Q2 / W_K2 / W_V2
#
# 가 다르기 때문에
# 서로 다른 attention 결과가 나온다.
# ==================================================

print("\n======================================")
print("Compare Head Outputs")
print("======================================")

print("\nHead 1:")
print(head1)

print("\nHead 2:")
print(head2)


# ==================================================
# 10. Head 결과 이어 붙이기
# ==================================================
#
# head1:
# [3, 2]
#
# head2:
# [3, 2]
#
# concat:
#
# [3, 4]
#
#
# 토큰별로:
#
# [Head1 특징 | Head2 특징]
#
# 를 붙인다.
# ==================================================

concat = torch.cat(
    [
        head1,
        head2
    ],
    dim=-1
)


print("\n======================================")
print("Concatenated Heads")
print("======================================")

print(concat)

print(
    "shape:",
    concat.shape
)


# ==================================================
# 11. 최종 투영 W_O
# ==================================================
#
# 여러 Head의 표현을 이어 붙인 후
#
# W_O를 통해 다시 d_model 차원의
# 표현으로 다시 섞는다.
#
#
# [3, 4]
# ↓
# [3, 4]
# ==================================================

W_O = nn.Linear(
    d_model,
    d_model,
    bias=False
)


print("\n======================================")
print("W_O")
print("======================================")

print(W_O.weight)


output = W_O(
    concat
)


print("\n======================================")
print("Multi-Head Attention Output")
print("======================================")

print(output)

print(
    "shape:",
    output.shape
)