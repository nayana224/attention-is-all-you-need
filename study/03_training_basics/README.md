# 03_training_basics

Transformer 학습의 가장 작은 단위를 이해하는 단계다.

- `01_tokenization_dataset.py`: Source / Decoder Input / GT 준비
- `02_single_training_step.py`: Forward → Loss → Backward → Update

이 단계에서는 여러 epoch보다 **한 번의 학습 step이 왜 parameter를 바꾸는지**에 집중한다.
