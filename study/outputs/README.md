# Study Outputs

이 디렉터리는 **개념 시각화**와 **실제 학습 추적 결과**를 분리해서 저장합니다.

```text
outputs/
├── visuals/
│   └── 3D vector / Attention 개념 시각화
│
├── csv/
│   └── 실제 EN-KO 학습의 step별 tensor / attention / parameter 기록
│
└── training_figures/
    └── CSV를 기반으로 만든 실제 학습 변화 그래프
```

## 1. Concept Visualizations

`study/visuals/`의 코드를 실행하면 `outputs/visuals/`에 저장됩니다.

예:

- Token Embedding / Positional Encoding 3D
- Query / Key / Value 공간
- Attention Weight bar chart
- Weighted Value와 Attention Output
- Multi-Head의 서로 다른 Query 공간
- Head별 Attention Weight 비교

이 그림들은 **실제 고차원 Transformer를 3차원으로 그대로 표현한 것이 아니라**,
벡터 연산의 의미를 직관적으로 이해하기 위한 toy visualization입니다.

## 2. Training Trace CSV

`study/training/03_training_trace.py`를 실행하면
실제 EN-KO toy training 과정의 값을 `outputs/csv/`에 저장합니다.

학습은 40개의 train pair를 순회하면서 진행하고,
매 optimizer step 뒤에 train에 없는 고정 test probe와 전체 test set을 다시 평가합니다.

고정 probe:

```text
The teacher likes robots
→ 선생님은 로봇을 좋아한다
```

### training_trace.csv

step별 학습/평가 요약값:

- train loss
- probe loss
- probe accuracy
- probe GT probability
- test mean loss
- test token accuracy

### tensor_trace.csv

고정 probe의 forward tensor 값을 step마다 저장:

- source embedding
- positional encoding
- encoder input/output
- target embedding
- decoder input/output
- logits
- probabilities

### attention_trace.csv

고정 probe의 Attention Weight:

- Encoder Self-Attention
- Decoder Masked Self-Attention
- Decoder Cross-Attention

Attention Weight를 곧바로 언어학적 설명으로 단정하지 않고,
같은 query/key 위치의 weight가 학습 과정에서 어떻게 변하는지를 관찰합니다.

### parameter_trace.csv

각 optimizer step에서 learnable parameter의:

- update 전 값
- gradient
- update 후 값
- delta

를 저장합니다.

## 3. Training Figures

`study/training/04_visualize_training.py`를 실행하면
`outputs/training_figures/`에 다음 그림을 저장합니다.

- Held-out Probe / Test Loss vs Step
- Probe GT Probability / Test Token Accuracy vs Step
- Probe Cross-Attention Weight vs Step
- Selected W_Q parameter vs Step
- Selected W_Q gradient vs Step

여기서는 **개념용 3D 그림**과 **실제 학습 중 측정된 값**을 구분해서 보는 것이 핵심입니다.

```text
visuals/
= "Attention 계산은 공간적으로 어떤 의미인가?"

training_figures/
= "실제 학습하면서 값은 어떻게 변했는가?"
```
