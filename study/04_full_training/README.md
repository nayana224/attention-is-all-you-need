# 04_full_training

작은 EN-KO 데이터셋으로 Tiny Transformer를 실제로 학습하는 메인 실습이다.

실행 순서:

```bash
python study/04_full_training/01_train.py
python study/04_full_training/02_visualize_training.py
python study/04_full_training/03_evaluate.py
python study/04_full_training/04_analyze.py
```

역할:

- `01_train.py`: Train / Validation / Early Stopping
- `02_visualize_training.py`: 학습 곡선
- `03_evaluate.py`: Teacher Forcing / Greedy Decoding / Test
- `04_analyze.py`: 학습 전후 내부 표현 비교
