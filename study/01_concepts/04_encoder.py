import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 0. 재현성
# ==================================================

torch.manual_seed(42)


# ==================================================
# 1. Toy 입력 x
# ==================================================
#
# 이전 단계와 동일한 x
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
print("Initial Input x")
print("======================================")

print(x)
print("shape:", x.shape)


# ==================================================
# 2. 모델 설정
# ==================================================

d_model = 4
num_heads = 2
d_k = d_model // num_heads
d_v = d_model // num_heads

d_ff = 8

num_layers = 6


print("\n======================================")
print("Configuration")
print("======================================")

print("d_model   :", d_model)
print("num_heads :", num_heads)
print("d_k       :", d_k)
print("d_v       :", d_v)
print("d_ff      :", d_ff)
print("num_layers:", num_layers)


# ==================================================
# 3. 멀티헤드 셀프 어텐션
# ==================================================

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads


        self.W_Q = nn.ModuleList([
            nn.Linear(
                d_model,
                self.d_k,
                bias=False
            )
            for _ in range(num_heads)
        ])


        self.W_K = nn.ModuleList([
            nn.Linear(
                d_model,
                self.d_k,
                bias=False
            )
            for _ in range(num_heads)
        ])


        self.W_V = nn.ModuleList([
            nn.Linear(
                d_model,
                self.d_v,
                bias=False
            )
            for _ in range(num_heads)
        ])


        self.W_O = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


    def scaled_dot_product_attention(
        self,
        Q,
        K,
        V
    ):

        scores = Q @ K.T

        scaled_scores = (
            scores / math.sqrt(self.d_k)
        )

        attention_weights = F.softmax(
            scaled_scores,
            dim=-1
        )

        output = (
            attention_weights @ V
        )

        return (
            output,
            scores,
            scaled_scores,
            attention_weights
        )


    def forward(self, x):

        head_outputs = []

        head_information = []

        for i in range(self.num_heads):

            Q = self.W_Q[i](x)
            K = self.W_K[i](x)
            V = self.W_V[i](x)

            (
                head_output,
                scores,
                scaled_scores,
                attention_weights
            ) = self.scaled_dot_product_attention(
                Q,
                K,
                V
            )

            head_outputs.append(
                head_output
            )

            head_information.append({
                "Q": Q,
                "K": K,
                "V": V,
                "scores": scores,
                "scaled_scores": scaled_scores,
                "attention_weights": attention_weights,
                "output": head_output,
            })


        concat = torch.cat(
            head_outputs,
            dim=-1
        )


        output = self.W_O(
            concat
        )


        return (
            output,
            concat,
            head_information
        )


# ==================================================
# 4. Encoder Layer 구조
# ==================================================
#
# 논문의 구조:
#
# x
# ↓
# Multi-Head Self-Attention
# ↓
# x + attention_output
# ↓
# LayerNorm
# ↓
# FFN
# ↓
# + residual
# ↓
# LayerNorm
# ==================================================

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff
    ):
        super().__init__()


        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )


        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                d_ff
            ),

            nn.ReLU(),

            nn.Linear(
                d_ff,
                d_model
            )
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


    def forward(self, x):

        # ------------------------------------------
        # 1) Multi-Head Self-Attention
        # ------------------------------------------

        (
            attention_output,
            concat,
            head_information
        ) = self.attention(x)


        # ------------------------------------------
        # 2) First Residual Connection
        # ------------------------------------------
        #
        # x
        # +
        # MultiHeadAttention(x)
        #
        # shape:
        #
        # [3,4] + [3,4]
        # =
        # [3,4]
        # ------------------------------------------

        residual1 = (
            x + attention_output
        )


        # ------------------------------------------
        # 3) First LayerNorm
        # ------------------------------------------

        x1 = self.norm1(
            residual1
        )


        # ------------------------------------------
        # 4) Feed-Forward Network
        # ------------------------------------------
        #
        # 각 token에 독립적으로:
        #
        # 4
        # ↓
        # 8
        # ↓ ReLU
        # 4
        #
        # ------------------------------------------

        ffn_output = self.ffn(
            x1
        )


        # ------------------------------------------
        # 5) Second Residual Connection
        # ------------------------------------------

        residual2 = (
            x1 + ffn_output
        )


        # ------------------------------------------
        # 6) Second LayerNorm
        # ------------------------------------------

        output = self.norm2(
            residual2
        )


        return {
            "output": output,

            "attention_output": attention_output,
            "concat": concat,

            "residual1": residual1,
            "norm1_output": x1,

            "ffn_output": ffn_output,

            "residual2": residual2,
            "norm2_output": output,

            "head_information": head_information,
        }


# ==================================================
# 5. Encoder Stack
# ==================================================

class Encoder(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers
    ):
        super().__init__()


        self.layers = nn.ModuleList([

            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )

            for _ in range(num_layers)

        ])


    def forward(self, x):

        layer_results = []


        for i, layer in enumerate(
            self.layers
        ):

            print(
                f"\n======================================"
            )

            print(
                f"Encoder Layer {i + 1}"
            )

            print(
                "======================================"
            )


            result = layer(x)


            print("\nInput:")
            print(x)
            print("shape:", x.shape)


            print("\nAttention Output:")
            print(
                result["attention_output"]
            )


            print("\nResidual 1:")
            print(
                result["residual1"]
            )


            print("\nAfter LayerNorm 1:")
            print(
                result["norm1_output"]
            )


            print("\nFFN Output:")
            print(
                result["ffn_output"]
            )


            print("\nResidual 2:")
            print(
                result["residual2"]
            )


            print("\nLayer Output:")
            print(
                result["output"]
            )

            print(
                "shape:",
                result["output"].shape
            )


            layer_results.append(
                result
            )


            # 다음 Encoder Layer의 input
            x = result["output"]


        return (
            x,
            layer_results
        )


# ==================================================
# 6. Encoder 생성
# ==================================================

encoder = Encoder(

    d_model=d_model,

    num_heads=num_heads,

    d_ff=d_ff,

    num_layers=num_layers
)


# ==================================================
# 7. 첫 번째 Encoder Layer의 Weight 확인
# ==================================================
#
# 03에서 하던 것처럼
# 실제 parameter도 확인한다.
# ==================================================

first_layer = encoder.layers[0]


print("\n======================================")
print("Encoder Layer 1 - Attention Weights")
print("======================================")


for i in range(num_heads):

    print(
        f"\nHead {i + 1} W_Q:"
    )

    print(
        first_layer
        .attention
        .W_Q[i]
        .weight
    )


    print(
        f"\nHead {i + 1} W_K:"
    )

    print(
        first_layer
        .attention
        .W_K[i]
        .weight
    )


    print(
        f"\nHead {i + 1} W_V:"
    )

    print(
        first_layer
        .attention
        .W_V[i]
        .weight
    )


print("\nW_O:")

print(
    first_layer
    .attention
    .W_O
    .weight
)


# ==================================================
# 8. FFN Weight Shape 확인
# ==================================================

print("\n======================================")
print("Encoder Layer 1 - FFN")
print("======================================")


print(
    "Linear 1 weight shape:",
    first_layer.ffn[0].weight.shape
)

print(
    "Linear 2 weight shape:",
    first_layer.ffn[2].weight.shape
)


# ==================================================
# 9. Encoder 순전파
# ==================================================

(
    encoder_output,
    layer_results
) = encoder(x)


print("\n======================================")
print("Final Encoder Output")
print("======================================")

print(
    encoder_output
)

print(
    "shape:",
    encoder_output.shape
)


# ==================================================
# 10. LayerNorm 통계
# ==================================================
#
# 첫 번째 Encoder Layer만 확인
# ==================================================

first_result = layer_results[0]


before_norm = (
    first_result["residual1"]
)

after_norm = (
    first_result["norm1_output"]
)


print("\n======================================")
print("LayerNorm Statistics")
print("======================================")


print("\nBefore LayerNorm mean:")

print(
    before_norm.mean(
        dim=-1
    )
)


print("\nBefore LayerNorm std:")

print(
    before_norm.std(
        dim=-1,
        unbiased=False
    )
)


print("\nAfter LayerNorm mean:")

print(
    after_norm.mean(
        dim=-1
    )
)


print("\nAfter LayerNorm std:")

print(
    after_norm.std(
        dim=-1,
        unbiased=False
    )
)


# ==================================================
# 11. LayerNorm 시각화
# ==================================================

before = (
    before_norm
    .detach()
    .numpy()
)

after = (
    after_norm
    .detach()
    .numpy()
)


tokens = [
    "I",
    "love",
    "robotics"
]


for i, token in enumerate(tokens):

    plt.figure(
        figsize=(7, 4)
    )


    plt.plot(
        range(d_model),
        before[i],
        marker="o",
        label="Before LayerNorm"
    )


    plt.plot(
        range(d_model),
        after[i],
        marker="o",
        label="After LayerNorm"
    )


    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )


    plt.title(
        f"Encoder Layer 1 - LayerNorm: {token}"
    )


    plt.xlabel(
        "Feature dimension"
    )

    plt.ylabel(
        "Value"
    )


    plt.xticks(
        range(d_model)
    )


    plt.legend()

    plt.grid()


    plt.show()