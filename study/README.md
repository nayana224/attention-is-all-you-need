# Study Guide

이 디렉터리는 **Attention Is All You Need**를 두 단계로 나눠 공부하도록 구성합니다.

```text
basics/
= Transformer 내부 블록을 작은 toy tensor로 분해해서 계산

visuals/
= 같은 개념을 2D / 3D vector 공간에서 시각적으로 확인

training/
= 실제 EN-KO 문자열을 넣어 forward → loss → backward → update를 확인

src/
= training 실습에서 재사용하는 구현

data/
= parallel sentence pairs

outputs/
= 학습 trace CSV와 시각화 결과

notebooks/
= 전체 흐름을 합친 통합 notebook
```

## Directory Structure

```text
study/
├── README.md
├── setup_venv.sh
│
├── basics/
│   ├── 01_embedding_pe.py
│   ├── 02_attention.py
│   ├── 03_multihead.py
│   ├── 04_encoder.py
│   ├── 05_decoder.py
│   └── 06_output_loss.py
│
├── visuals/
│   ├── README.md
│   ├── 01_embedding_pe_3d.py
│   ├── 02_single_head_attention_3d.py
│   └── 03_multi_head_attention_3d.py
│
├── training/
│   ├── 01_tokenization_dataset.py
│   ├── 02_single_training_step.py
│   ├── 03_training_trace.py
│   └── 04_visualize_training.py
│
├── src/
│   ├── tokenization.py
│   ├── data.py
│   ├── positional_encoding.py
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer.py
│   ├── factory.py
│   └── tracing.py
│
├── data/
│   ├── en_ko_train.csv
│   └── en_ko_test.csv
│
├── outputs/
│   ├── README.md
│   ├── csv/
│   ├── visuals/
│   └── training_figures/
│
├── notebooks/
│   └── Attention_Is_All_You_Need_Study.ipynb
│
└── images/
```

## Visualization Track

계산 흐름을 한 번 따라간 뒤에는 같은 Attention을 **벡터 공간의 움직임**으로 다시 봅니다.

```text
basics/
계산을 따라감

        ↓

visuals/
3D vector로 의미를 직관화

        ↓

training/
실제 EN-KO 학습에서 값이 어떻게 변하는지 추적
```

추천 순서:

```text
visuals/01_embedding_pe_3d.py
→ Embedding과 Position 정보

visuals/02_single_head_attention_3d.py
→ x → Q/K/V → Attention Weight → Weighted Sum of V

visuals/03_multi_head_attention_3d.py
→ 서로 다른 head가 서로 다른 projection 공간을 사용하는 모습

training/05_visualize_learned_vectors.py
→ 실제 TinyTransformer를 EN-KO pair로 SGD 학습한 뒤,
  실제 embedding / cross-attention 변화를 3D로 투영해서 관찰
```

실행:

```bash
python study/visuals/01_embedding_pe_3d.py
python study/visuals/02_single_head_attention_3d.py
python study/visuals/03_multi_head_attention_3d.py
python study/training/05_visualize_learned_vectors.py
```

시각화 이미지는:

```text
study/outputs/visuals/
```

에 저장됩니다.

> 3차원은 실제 Transformer의 차원이 아니라, 고차원 representation을 사람이 이해하기 위한 mental model입니다.

## Recommended Reading Order

새로운 end-to-end 학습 코드를 분석할 때는 아래 순서를 권장합니다.

### 1. 실제 데이터부터 보기

```text
data/en_ko_train.csv
data/en_ko_test.csv
```

먼저 모델 코드보다 **무엇이 Input이고 무엇이 GT인지** 확인합니다.

### 2. Tokenization / GT 생성

```text
src/tokenization.py
↓
src/data.py
↓
training/01_tokenization_dataset.py
```

여기서 다음 흐름만 확실히 이해하면 됩니다.

```text
Raw string
→ token
→ vocabulary
→ token ID

Target
→ <SOS> + target = Decoder Input
→ target + <EOS> = GT
```

실행:

```bash
python study/training/01_tokenization_dataset.py
```

### 3. Transformer 전체 forward의 큰 흐름

먼저:

```text
src/transformer.py
```

만 읽습니다.

처음에는 내부 Attention 수식을 다시 파고들기보다:

```text
src_ids
→ source embedding + PE
→ Encoder
→ encoder_output

decoder_input_ids
→ target embedding + PE
→ Decoder
→ logits
```

이 큰 흐름을 봅니다.

그다음 필요할 때 아래 파일로 내려갑니다.

```text
src/attention.py
src/encoder.py
src/decoder.py
src/positional_encoding.py
```

### 4. 역전파 한 번 보기

```text
training/02_single_training_step.py
```

실행:

```bash
python study/training/02_single_training_step.py
```

여기서는 코드 전체보다 이 다섯 줄의 연결이 핵심입니다.

```text
model(...)
→ CrossEntropyLoss
→ loss.backward()
→ gradient
→ optimizer.step()
```

### 5. 여러 step의 변화 추적

```text
src/tracing.py
↓
training/03_training_trace.py
```

실행:

```bash
python study/training/03_training_trace.py
```

생성되는 CSV:

```text
outputs/csv/
├── training_trace.csv
├── tensor_trace.csv
├── attention_trace.csv
└── parameter_trace.csv
```

### 6. 값의 변화 시각화

```text
training/04_visualize_training.py
```

실행:

```bash
python study/training/04_visualize_training.py
```

그래프는 `outputs/training_figures/`에 저장됩니다.

## Code Style for Study

이 저장소의 `study/src/`와 `study/training/` 코드는 짧게 쓰는 것보다 **한 줄씩 따라가기 쉬운 것**을 우선합니다.

가능하면 다음처럼 풀어서 작성합니다.

```text
사용
- 일반 for 문
- 일반 if 문
- 중간 변수
- append()
- 여러 줄 함수 호출

당분간 피함
- list comprehension
- dict comprehension
- lambda
- 한 줄에 여러 단계가 섞인 축약 표현
```

예를 들어:

```python
# 축약형
ids = [vocab[token] for token in tokens]

# 학습용 형태
ids = []

for token in tokens:
    token_id = vocab[token]
    ids.append(token_id)
```

Transformer 자체의 연산을 이해하는 것이 목적이므로,
Python 문법 때문에 data flow가 가려지지 않도록 합니다.

## Why src/ Is Separated

기존 `trace_utils.py`에는 tokenization, dataset, positional encoding,
Attention, Encoder, Decoder, Transformer, tracing 기능이 한 파일에 섞여 있었습니다.

현재는 **파일 하나 = 역할 하나**에 가깝게 분리했습니다.

```text
tokenization.py       문자열 → token / ID
data.py               CSV → 학습 example
positional_encoding.py 위치 정보
attention.py          Multi-Head Attention
encoder.py            Encoder Layer
decoder.py            Decoder Layer
transformer.py        전체 forward data flow
factory.py            실습용 model / vocab 생성
tracing.py            값 기록 및 CSV 저장
```

따라서 처음부터 `src/` 전체를 읽을 필요는 없습니다.
현재 공부 순서에서 필요한 파일만 따라가면 됩니다.


## Concept Visualization vs Actual Training Visualization

두 종류의 그림을 구분해서 봅니다.

```text
study/visuals/
= 개념을 이해하기 위한 설명용 toy vector
= 사람이 보기 쉽게 3D로 직접 구성

study/training/05_visualize_learned_vectors.py
= 실제 study/src/TinyTransformer 사용
= 실제 EN-KO train pair로 SGD 학습
= 실제 d_model=8 representation을 PCA로 3D에 투영
```

실제 학습 시각화에서 생성되는 대표 그림:

```text
outputs/training_figures/
├── 06_actual_training_loss.png
├── 07_actual_probe_quality.png
├── 08_actual_embedding_trajectory_pca3d.png
├── 09_actual_cross_attention_vs_step.png
└── 10_actual_final_cross_qk_pca3d.png
```

PCA 3D는 고차원 벡터를 사람이 보기 위해 투영한 것입니다.
따라서 3D 그림의 각도나 거리를 실제 원본 공간과 완전히 동일하다고 해석하지 않습니다.
Attention score 숫자는 원본 Q/K 차원에서 계산한 실제 값을 사용합니다.


## Full Training -> Evaluation -> Analysis

학습 초반의 변화만 보는 03~05 다음에는,
작은 EN-KO dataset을 충분히 학습시킨 뒤 결과를 분석합니다.

```text
training/06_train_tiny_transformer.py
= 40개 train pair를 300 epoch 학습
= Adam optimizer 사용
= train/test loss와 token accuracy 기록
= best/final checkpoint 저장

training/07_evaluate_trained_model.py
= best checkpoint 평가
= teacher-forced token accuracy
= 실제 inference 방식의 greedy autoregressive decoding
= train/test prediction CSV 저장

training/08_analyze_trained_model.py
= 같은 probe에 대해 학습 전 vs 학습 후 비교
= GT probability
= Cross-Attention heatmap
= source embedding shift(PCA 3D)
```

실행 순서:

```bash
python study/training/06_train_tiny_transformer.py
python study/training/07_evaluate_trained_model.py
python study/training/08_analyze_trained_model.py
```

생성 파일:

```text
outputs/checkpoints/
├── tiny_transformer_best.pt
└── tiny_transformer_final.pt

outputs/csv/
├── full_training_history.csv
└── trained_model_evaluation.csv

outputs/training_figures/
├── 12_trained_gt_probability_before_after.png
├── 13_trained_cross_attention_before_after.png
└── 14_trained_embedding_before_after_pca3d.png
```

06은 기존의 SGD 한 step 관찰 실습과 목적이 다릅니다.
충분히 학습된 작은 모델을 만들기 위해 Adam을 사용하고,
best checkpoint는 held-out test loss가 가장 낮은 epoch에서 저장합니다.

07에서는 teacher forcing 평가와 greedy decoding을 분리해서 봅니다.
teacher-forced accuracy가 높아도 실제 autoregressive generation은 다를 수 있습니다.
