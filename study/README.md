# Attention Is All You Need - Study

이 폴더는 논문을 읽는 순서가 아니라 **이해가 쌓이는 순서**로 구성한다.

## 공부 순서

```text
01_concepts
    ↓
02_visual_intuition
    ↓
03_training_basics
    ↓
04_full_training
    ↓
05_experiments
```

`src/`는 위 실습들이 공통으로 사용하는 실제 구현 코드다.
평소에는 메인 실습 코드를 읽고, 내부 계산이 궁금할 때만 `src/`로 들어간다.

---

## 폴더 구조

```text
study/
├── 01_concepts/
│   ├── 01_embedding_pe.py
│   ├── 02_attention.py
│   ├── 03_multihead.py
│   ├── 04_encoder.py
│   ├── 05_decoder.py
│   └── 06_output_loss.py
│
├── 02_visual_intuition/
│   ├── 01_embedding_pe_3d.py
│   ├── 02_single_head_attention_3d.py
│   ├── 03_multi_head_attention_3d.py
│   └── plot_utils.py
│
├── 03_training_basics/
│   ├── 01_tokenization_dataset.py
│   └── 02_single_training_step.py
│
├── 04_full_training/
│   ├── 01_train.py
│   ├── 02_visualize_training.py
│   ├── 03_evaluate.py
│   └── 04_analyze.py
│
├── 05_experiments/
│   ├── 01_training_trace.py
│   ├── 02_visualize_training_trace.py
│   └── 03_visualize_learned_vectors.py
│
├── src/
├── data/
├── outputs/
└── notebooks/
```

---

## 01_concepts

Transformer 계산을 작은 tensor로 직접 확인한다.

```text
Token ID
→ Embedding
→ Positional Encoding
→ Q / K / V
→ Attention
→ Multi-Head Attention
→ Encoder
→ Decoder
→ Output Linear
→ Cross Entropy Loss
```

여기서는 모델을 잘 학습시키는 것이 목적이 아니라,
**각 블록에서 어떤 계산이 일어나는지 이해하는 것**이 목적이다.

---

## 02_visual_intuition

고차원 벡터를 사람이 볼 수 있는 3차원 예제로 단순화하여 시각화한다.

주요 질문:

- Embedding과 PE를 더한다는 것은 벡터 관점에서 무엇인가?
- 같은 token이 Q, K, V에서 왜 다른 벡터가 되는가?
- Q와 K의 내적은 어떤 관계 점수를 만드는가?
- Attention weight가 V에 어떻게 적용되는가?
- 여러 head가 왜 서로 다른 관계를 볼 수 있는가?

이 폴더의 그림은 **개념 설명용 toy vector**다.
실제 학습된 모델의 내부 값과 구분한다.

---

## 03_training_basics

학습 데이터가 모델 안으로 어떻게 들어가고,
한 번의 optimizer step이 어떻게 이루어지는지 확인한다.

```text
01_tokenization_dataset.py
= Source / Decoder Input / GT 생성

02_single_training_step.py
= Forward
→ Loss
→ Backward
→ Gradient
→ Optimizer Step
→ Parameter Update
```

여기까지 이해하면 Transformer의 한 번의 학습 흐름을 설명할 수 있어야 한다.

---

## 04_full_training

작은 EN-KO 데이터셋으로 Tiny Transformer를 실제로 학습하고 평가한다.

```text
01_train.py
→ Train으로 parameter 학습
→ Validation으로 best checkpoint 선택
→ Early Stopping

02_visualize_training.py
→ Train / Validation Loss
→ Train / Validation Token Accuracy
→ Best Epoch 시각화

03_evaluate.py
→ Train / Validation / Test 평가
→ Teacher Forcing
→ Greedy Autoregressive Decoding

04_analyze.py
→ 학습 전후 GT probability 비교
→ Cross-Attention 비교
→ Embedding 이동 비교
```

실행 순서:

```bash
python study/04_full_training/01_train.py
python study/04_full_training/02_visualize_training.py
python study/04_full_training/03_evaluate.py
python study/04_full_training/04_analyze.py
```

---

## 05_experiments

학습 초기에 parameter와 attention이 어떻게 움직이는지 관찰하기 위해 만든 보조 실험이다.

메인 학습 흐름은 아니다.

처음 공부할 때는 건너뛰어도 되고,
`04_full_training`까지 이해한 뒤 내부 변화가 더 궁금할 때 다시 본다.

---

## src

실제 Tiny Transformer 구현이다.

```text
tokenization.py
data.py
positional_encoding.py
attention.py
encoder.py
decoder.py
transformer.py
factory.py
tracing.py
```

읽는 방식:

```text
메인 실습 코드 실행
↓
궁금한 함수 / 클래스 발견
↓
src 구현 확인
↓
다시 메인 실습으로 복귀
```

`src/`를 처음부터 순서대로 외우듯 읽을 필요는 없다.

---

## Data

```text
en_ko_train.csv
= parameter 학습

en_ko_val.csv
= best checkpoint 선택
= early stopping 판단

en_ko_test.csv
= 최종 성능 확인
```

Test data는 checkpoint 선택에 사용하지 않는다.

---

## 이 실습에서 최종적으로 설명할 수 있어야 하는 것

1. Embedding과 Positional Encoding이 Transformer 입력을 어떻게 만드는가
2. Q / K / V가 왜 서로 다른 projection인가
3. QK^T와 Softmax가 Attention Weight를 어떻게 만드는가
4. Weighted Sum of V가 어떤 정보를 가져오는가
5. Multi-Head Attention이 왜 여러 관계 공간을 만들 수 있는가
6. Encoder와 Decoder의 차이는 무엇인가
7. Masked Self-Attention과 Cross-Attention은 무엇이 다른가
8. Decoder output이 target vocabulary logits로 어떻게 변환되는가
9. Cross Entropy Loss가 어떻게 parameter update로 이어지는가
10. 실제 학습 후 loss, probability, attention, embedding이 어떻게 변했는가
