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

vocab = {
    0: "<PAD>",
    1: "<SOS>",
    2: "I",
    3: "love",
    4: "robotics",
    5: "<EOS>",
}

vocab_size = len(vocab)


# ==================================================
# 2. 최종 Decoder 출력
# ==================================================
# 05_decoder.py의 최종 output을 고정해서 사용

decoder_output = torch.tensor([
    [ 0.1769, -1.0267, -0.7050,  1.5548],
    [-0.4914,  0.0975,  1.5519, -1.1579],
    [-1.2938,  0.2286, -0.3912,  1.4563],
])

print("======================================")
print("Decoder Output")
print("======================================")
print(decoder_output)
print("shape:", decoder_output.shape)


# ==================================================
# 3. 출력 선형 투영
# ==================================================

output_linear = nn.Linear(
    d_model,
    vocab_size,
    bias=False
)

print("\n======================================")
print("Output Linear Weight")
print("======================================")
print(output_linear.weight)
print("shape:", output_linear.weight.shape)


# ==================================================
# 4. Logit / 확률
# ==================================================

logits = output_linear(decoder_output)

probabilities = F.softmax(
    logits,
    dim=-1
)

print("\n======================================")
print("Logits")
print("======================================")
print(logits)
print("shape:", logits.shape)

print("\n======================================")
print("Probabilities")
print("======================================")
print(probabilities)
print("row sums:", probabilities.sum(dim=-1))


# ==================================================
# 5. 예측
# ==================================================

pred_ids = torch.argmax(
    probabilities,
    dim=-1
)

pred_tokens = [
    vocab[token_id.item()]
    for token_id in pred_ids
]

print("\n======================================")
print("Predictions")
print("======================================")
print("Predicted IDs:", pred_ids)
print("Predicted Tokens:", pred_tokens)


# ==================================================
# 6. 정답
# ==================================================

gt_ids = torch.tensor([
    2,  # I
    3,  # love
    4,  # robotics
])

gt_tokens = [
    vocab[token_id.item()]
    for token_id in gt_ids
]

print("\n======================================")
print("Ground Truth")
print("======================================")
print("GT IDs:", gt_ids)
print("GT Tokens:", gt_tokens)


# ==================================================
# 7. 정답 토큰 확률
# ==================================================

gt_probabilities = probabilities[
    torch.arange(len(gt_ids)),
    gt_ids
]

print("\n======================================")
print("GT Probabilities")
print("======================================")
print("P(I)        :", gt_probabilities[0])
print("P(love)     :", gt_probabilities[1])
print("P(robotics) :", gt_probabilities[2])


# ==================================================
# 8. Cross Entropy 직접 계산
# ==================================================

manual_losses = -torch.log(
    gt_probabilities
)

manual_loss = manual_losses.mean()

print("\n======================================")
print("Manual CE Loss")
print("======================================")
print("per position:", manual_losses)
print("mean:", manual_loss)


# ==================================================
# 9. PyTorch CrossEntropyLoss
# ==================================================

criterion = nn.CrossEntropyLoss()

loss = criterion(
    logits,
    gt_ids
)

print("\n======================================")
print("PyTorch CrossEntropy Loss")
print("======================================")
print(loss)


# ==================================================
# 10. Label Smoothing
# ==================================================

criterion_smoothing = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

loss_smoothing = criterion_smoothing(
    logits,
    gt_ids
)

print("\n======================================")
print("Label Smoothing")
print("======================================")
print("CrossEntropy:", loss)
print("Label Smoothing Loss:", loss_smoothing)
