# Visualization Track

이 디렉터리는 Transformer의 수식과 tensor 계산을 **벡터 공간의 움직임**으로 이해하기 위한 실습입니다.

기존 `study/basics/` 코드는 계산을 직접 따라가는 용도이고,
이 디렉터리의 코드는 같은 개념을 2D / 3D 그림으로 다시 확인하는 용도입니다.

## Recommended Order

```text
01_embedding_pe_3d.py
→ token embedding과 position 정보가 vector에 어떻게 더해지는지 보기

02_single_head_attention_3d.py
→ x → Q/K/V → QK^T → softmax → weighted sum of V 보기

03_multi_head_attention_3d.py
→ 같은 입력을 서로 다른 head가 서로 다른 projection 공간에서 보는 것 확인
```

## Important

3차원은 **실제 Transformer 차원**이 아니라 사람이 이해하기 위한 mental model입니다.

실제 모델에서는 `d_model`이 훨씬 크고,
한 차원 하나가 사람이 이해할 수 있는 의미 하나와 직접 대응한다고 볼 수 없습니다.

시각화 결과는:

```text
study/outputs/visuals/
```

에 저장됩니다.
