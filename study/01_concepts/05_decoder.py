import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 0. 재현성
# ==================================================

torch.manual_seed(42)


# ==================================================
# 1. 모델 설정
# ==================================================

d_model = 4
num_heads = 2

d_k = d_model // num_heads
d_v = d_model // num_heads

d_ff = 8
num_layers = 6


print("======================================")
print("Configuration")
print("======================================")

print("d_model   :", d_model)
print("num_heads :", num_heads)
print("d_k       :", d_k)
print("d_v       :", d_v)
print("d_ff      :", d_ff)
print("num_layers:", num_layers)


# ==================================================
# 2. Toy Encoder 출력
# ==================================================
#
# 실제로는 04_encoder.py의
# 최종 encoder output이라고 생각한다.
#
# Source 문장:
#
# I / love / robotics
#
# 텐서 크기:
# [source_len, d_model]
# = [3, 4]
# ==================================================

encoder_output = torch.tensor([
    [ 0.7641, -0.5353,  1.1313, -1.3600],   # I
    [-0.9590,  0.9754, -1.0399,  1.0235],   # love
    [-1.5318,  1.2271,  0.3777, -0.0730],   # robotics
])


print("\n======================================")
print("Encoder Output")
print("======================================")

print(encoder_output)
print("shape:", encoder_output.shape)


# ==================================================
# 3. Toy Decoder 입력
# ==================================================
#
# GT:
#
# I love robotics
#
# Decoder 입력은 한 칸 이동시킨다:
#
# <SOS> I love
#
# 여기서는 embedding + PE 결과라고 가정하고
# 간단한 예제 벡터를 사용한다.
#
# row 0 = <SOS>
# row 1 = I
# row 2 = love
#
# 텐서 크기:
# [target_len, d_model]
# = [3, 4]
# ==================================================

decoder_input = torch.tensor([
    [1.0, 0.0, 0.0, 1.0],   # <SOS>
    [1.0, 0.0, 1.0, 0.0],   # I
    [0.0, 2.0, 0.0, 2.0],   # love
])


print("\n======================================")
print("Decoder Input")
print("======================================")

print(decoder_input)
print("shape:", decoder_input.shape)


# ==================================================
# 4. Causal Mask
# ==================================================
#
# 미래 토큰을 보지 못하게 한다.
#
# False = 볼 수 있음
# True  = 마스킹
#
#
#        0      1      2
#
# 0   False   True   True
# 1   False  False   True
# 2   False  False  False
#
# ==================================================

target_len = decoder_input.shape[0]

causal_mask = torch.triu(
    torch.ones(
        target_len,
        target_len,
        dtype=torch.bool
    ),
    diagonal=1
)


print("\n======================================")
print("Causal Mask")
print("======================================")

print(causal_mask)


# ==================================================
# 5. 멀티헤드 어텐션
# ==================================================
#
# Self-Attention:
#
# query_input     = decoder x
# key_value_input = decoder x
#
#
# Cross-Attention:
#
# query_input     = decoder representation
# key_value_input = encoder output
#
# ==================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
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
        V,
        mask=None
    ):

        # ------------------------------------------
        # QK^T
        # ------------------------------------------

        scores = (
            Q @ K.T
        )


        # ------------------------------------------
        # 스케일링
        # ------------------------------------------

        scaled_scores = (
            scores
            / math.sqrt(self.d_k)
        )


        # ------------------------------------------
        # 마스킹
        # ------------------------------------------
        #
        # 미래 위치의 score를 -inf로 만든다.
        #
        # exp(-inf) = 0
        #
        # 따라서 Softmax 후 가중치는 0이 된다.
        # ------------------------------------------

        if mask is not None:

            scaled_scores = (
                scaled_scores.masked_fill(
                    mask,
                    float("-inf")
                )
            )


        # ------------------------------------------
        # Softmax
        # ------------------------------------------

        attention_weights = F.softmax(
            scaled_scores,
            dim=-1
        )


        # ------------------------------------------
        # V의 가중합
        # ------------------------------------------

        output = (
            attention_weights @ V
        )


        return (
            output,
            scores,
            scaled_scores,
            attention_weights
        )


    def forward(
        self,
        query_input,
        key_value_input,
        mask=None
    ):

        head_outputs = []
        head_information = []


        for i in range(
            self.num_heads
        ):

            # --------------------------------------
            # Query 생성
            #
            # query_input에서 생성
            # --------------------------------------

            Q = self.W_Q[i](
                query_input
            )


            # --------------------------------------
            # Key / Value 생성
            #
            # key_value_input에서 생성
            # --------------------------------------

            K = self.W_K[i](
                key_value_input
            )

            V = self.W_V[i](
                key_value_input
            )


            (
                head_output,
                scores,
                scaled_scores,
                attention_weights
            ) = self.scaled_dot_product_attention(
                Q,
                K,
                V,
                mask
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


        # ------------------------------------------
        # Head 결과 이어 붙이기
        #
        # [3,2] + [3,2]
        # →
        # [3,4]
        # ------------------------------------------

        concat = torch.cat(
            head_outputs,
            dim=-1
        )


        # ------------------------------------------
        # W_O
        # ------------------------------------------

        output = self.W_O(
            concat
        )


        return (
            output,
            concat,
            head_information
        )


# ==================================================
# 6. Decoder Layer 구조
# ==================================================

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff
    ):
        super().__init__()


        # ------------------------------------------
        # 1) Masked Self-Attention
        # ------------------------------------------

        self.masked_self_attention = (
            MultiHeadAttention(
                d_model,
                num_heads
            )
        )


        self.norm1 = nn.LayerNorm(
            d_model
        )


        # ------------------------------------------
        # 2) Encoder-Decoder Cross-Attention
        # ------------------------------------------

        self.cross_attention = (
            MultiHeadAttention(
                d_model,
                num_heads
            )
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        # ------------------------------------------
        # 3) FFN
        # ------------------------------------------

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


        self.norm3 = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x,
        encoder_output,
        causal_mask
    ):

        # ==========================================
        # 1. Masked Self-Attention
        # ==========================================
        #
        # Q, K, V 모두 Decoder 입력에서 생성
        #
        # 단, 미래 토큰은 마스킹
        # ==========================================

        (
            masked_attention_output,
            masked_concat,
            masked_head_information
        ) = self.masked_self_attention(
            query_input=x,
            key_value_input=x,
            mask=causal_mask
        )


        # ------------------------------------------
        # Residual + LayerNorm
        # ------------------------------------------

        residual1 = (
            x
            +
            masked_attention_output
        )


        x1 = self.norm1(
            residual1
        )


        # ==========================================
        # 2. Cross-Attention
        # ==========================================
        #
        # Q:
        #
        # Decoder 표현 x1
        #
        #
        # K, V:
        #
        # Encoder 출력
        #
        #
        # Causal Mask는 사용하지 않음
        #
        # Decoder의 각 위치는
        # Source 전체를 볼 수 있음
        # ==========================================

        (
            cross_attention_output,
            cross_concat,
            cross_head_information
        ) = self.cross_attention(
            query_input=x1,
            key_value_input=encoder_output,
            mask=None
        )


        # ------------------------------------------
        # Residual + LayerNorm
        # ------------------------------------------

        residual2 = (
            x1
            +
            cross_attention_output
        )


        x2 = self.norm2(
            residual2
        )


        # ==========================================
        # 3. FFN
        # ==========================================

        ffn_output = self.ffn(
            x2
        )


        # ------------------------------------------
        # Residual + LayerNorm
        # ------------------------------------------

        residual3 = (
            x2
            +
            ffn_output
        )


        output = self.norm3(
            residual3
        )


        return {

            "output": output,

            "masked_attention_output":
                masked_attention_output,

            "masked_head_information":
                masked_head_information,

            "residual1":
                residual1,

            "norm1_output":
                x1,

            "cross_attention_output":
                cross_attention_output,

            "cross_head_information":
                cross_head_information,

            "residual2":
                residual2,

            "norm2_output":
                x2,

            "ffn_output":
                ffn_output,

            "residual3":
                residual3,
        }


# ==================================================
# 7. Decoder Stack
# ==================================================

class Decoder(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers
    ):
        super().__init__()


        self.layers = nn.ModuleList([

            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )

            for _ in range(
                num_layers
            )
        ])


    def forward(
        self,
        x,
        encoder_output,
        causal_mask
    ):

        layer_results = []


        for i, layer in enumerate(
            self.layers
        ):

            print(
                "\n======================================"
            )

            print(
                f"Decoder Layer {i + 1}"
            )

            print(
                "======================================"
            )


            result = layer(
                x,
                encoder_output,
                causal_mask
            )


            print("\nInput:")
            print(x)


            print(
                "\nMasked Self-Attention Output:"
            )

            print(
                result[
                    "masked_attention_output"
                ]
            )


            print("\nAfter Norm 1:")

            print(
                result[
                    "norm1_output"
                ]
            )


            print(
                "\nCross-Attention Output:"
            )

            print(
                result[
                    "cross_attention_output"
                ]
            )


            print("\nAfter Norm 2:")

            print(
                result[
                    "norm2_output"
                ]
            )


            print("\nFFN Output:")

            print(
                result[
                    "ffn_output"
                ]
            )


            print("\nDecoder Layer Output:")

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


            # 다음 Decoder Layer의 입력
            x = result["output"]


        return (
            x,
            layer_results
        )


# ==================================================
# 8. Decoder 생성
# ==================================================

decoder = Decoder(

    d_model=d_model,

    num_heads=num_heads,

    d_ff=d_ff,

    num_layers=num_layers
)


# ==================================================
# 9. Decoder Layer 1 Weight Matrix 확인
# ==================================================

first_layer = decoder.layers[0]


print("\n======================================")
print("Decoder Layer 1")
print("Masked Self-Attention Weights")
print("======================================")


for i in range(num_heads):

    print(
        f"\nHead {i + 1} W_Q:"
    )

    print(
        first_layer
        .masked_self_attention
        .W_Q[i]
        .weight
    )


    print(
        f"\nHead {i + 1} W_K:"
    )

    print(
        first_layer
        .masked_self_attention
        .W_K[i]
        .weight
    )


    print(
        f"\nHead {i + 1} W_V:"
    )

    print(
        first_layer
        .masked_self_attention
        .W_V[i]
        .weight
    )


# ==================================================
# 10. Decoder 순전파
# ==================================================

(
    decoder_output,
    layer_results
) = decoder(
    decoder_input,
    encoder_output,
    causal_mask
)


# ==================================================
# 11. Masked Self-Attention Weight 확인
# ==================================================
#
# Decoder Layer 1
# Head 1
# ==================================================

first_result = (
    layer_results[0]
)


masked_weights = (
    first_result[
        "masked_head_information"
    ][0][
        "attention_weights"
    ]
)


print("\n======================================")
print("Decoder Layer 1")
print("Head 1 Masked Attention Weights")
print("======================================")

print(
    masked_weights
)


# ==================================================
# 12. Cross-Attention Weight 확인
# ==================================================
#
# 행:
# Decoder target 위치
#
# 열:
# Encoder source 위치
# ==================================================

cross_weights = (
    first_result[
        "cross_head_information"
    ][0][
        "attention_weights"
    ]
)


print("\n======================================")
print("Decoder Layer 1")
print("Head 1 Cross-Attention Weights")
print("======================================")

print(
    cross_weights
)


# ==================================================
# 13. 최종 Decoder 출력
# ==================================================

print("\n======================================")
print("Final Decoder Output")
print("======================================")

print(
    decoder_output
)

print(
    "shape:",
    decoder_output.shape
)