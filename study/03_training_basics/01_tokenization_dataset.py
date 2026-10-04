import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from src.data import load_translation_pairs
from src.tokenization import (
    build_vocab,
    encode_source,
    encode_target,
    invert_vocab,
    whitespace_tokenize,
)


TRAIN_PATH = STUDY_DIR / "data" / "en_ko_train.csv"
TEST_PATH = STUDY_DIR / "data" / "en_ko_test.csv"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_vocab(title, vocab):
    print_section(title)

    id_to_token = invert_vocab(vocab)

    for index in range(len(id_to_token)):
        print(f"{index:>2} -> {id_to_token[index]}")


# ==================================================
# 0. 병렬 번역 데이터 불러오기
# ==================================================

train_pairs = load_translation_pairs(TRAIN_PATH)
test_pairs = load_translation_pairs(TEST_PATH)

source_sentences = []
target_sentences = []

for pair in train_pairs:
    source_sentences.append(
        pair["source"]
    )

    target_sentences.append(
        pair["target"]
    )

src_vocab = build_vocab(
    source_sentences
)

tgt_vocab = build_vocab(
    target_sentences
)


print("=" * 70)
print("English -> Korean Parallel Dataset")
print("=" * 70)

print(f"Train pairs: {len(train_pairs)}")
print(f"Test pairs : {len(test_pairs)}")

print("\nTrain samples:")
for i, pair in enumerate(train_pairs[:5]):
    print(f"[{i}] {pair['source']} -> {pair['target']}")

print("\nTest samples:")
for i, pair in enumerate(test_pairs[:5]):
    print(f"[{i}] {pair['source']} -> {pair['target']}")


print_vocab("Source Vocabulary (English)", src_vocab)
print_vocab("Target Vocabulary (Korean)", tgt_vocab)


# ==================================================
# 1. 학습 문장 하나 선택
# ==================================================

sample = train_pairs[0]

source_text = sample["source"]
target_text = sample["target"]


print_section("1. Raw Parallel Sentence Pair")

print("English source:")
print(source_text)

print("\nKorean target:")
print(target_text)


# ==================================================
# 2. 예제 문장 토큰화
# ==================================================

source_tokens = whitespace_tokenize(source_text)
target_tokens = whitespace_tokenize(target_text)


print_section("2. Whitespace Tokenization")

print("English source tokens:")
print(source_tokens)

print("\nKorean target tokens:")
print(target_tokens)


# ==================================================
# 3. Source 입력 흐름
#
# English source
# -> tokenize
# -> + <EOS>
# -> vocabulary lookup
# -> Encoder Input IDs
# ==================================================

(
    encoder_input_tokens,
    encoder_input_ids,
) = encode_source(
    source_text,
    src_vocab,
)


print_section("3. Source Branch -> Encoder Input")

print("Raw English:")
print(source_text)

print("\nTokenized:")
print(source_tokens)

print("\nAdd <EOS>:")
print(encoder_input_tokens)

print("\nVocabulary lookup -> Encoder Input IDs:")
print(encoder_input_ids)

print("\nFlow:")
print(
    "English Source"
    " -> Tokenization"
    " -> + <EOS>"
    " -> Vocabulary Lookup"
    " -> Encoder Input IDs"
)


# ==================================================
# 4. Target 입력과 정답 흐름
#
# Korean target에서 두 갈래가 만들어진다.
#
# A) Decoder Input
#    <SOS> + target
#
# B) Ground Truth
#    target + <EOS>
# ==================================================

(
    decoder_input_tokens,
    decoder_input_ids,
    gt_tokens,
    gt_ids,
) = encode_target(
    target_text,
    tgt_vocab,
)


print_section("4. Target Branch -> Decoder Input / GT")

print("Raw Korean:")
print(target_text)

print("\nTokenized:")
print(target_tokens)

print("\nA) Decoder Input")
print("Add <SOS> at front:")
print(decoder_input_tokens)

print("Vocabulary lookup -> Decoder Input IDs:")
print(decoder_input_ids)

print("\nB) Ground Truth")
print("Add <EOS> at end:")
print(gt_tokens)

print("Vocabulary lookup -> GT IDs:")
print(gt_ids)


# ==================================================
# 5. Shifted target과 다음 토큰 예측
# ==================================================

print_section("5. Shifted Target -> Next-token Prediction")

print(
    "Decoder Input and GT are shifted by one token."
)

print()

for i in range(len(gt_tokens)):
    visible_prefix = decoder_input_tokens[: i + 1]

    print(
        f"position {i}: "
        f"{visible_prefix} "
        f"-> predict {gt_tokens[i]}"
    )


# ==================================================
# 6. 학습에 사용하지 않은 test 문장
# ==================================================

test_sample = test_pairs[0]

test_encoder_tokens, test_encoder_ids = encode_source(
    test_sample["source"],
    src_vocab,
)

(
    test_decoder_tokens,
    test_decoder_ids,
    test_gt_tokens,
    test_gt_ids,
) = encode_target(
    test_sample["target"],
    tgt_vocab,
)


print_section("6. Held-out Test Sentence")

print("English source:")
print(test_sample["source"])

print("\nKorean target:")
print(test_sample["target"])

print("\nEncoder Input:")
print(test_encoder_tokens)
print(test_encoder_ids)

print("\nDecoder Input:")
print(test_decoder_tokens)
print(test_decoder_ids)

print("\nGT:")
print(test_gt_tokens)
print(test_gt_ids)

print(
    "\n이 test 문장은 train에 동일한 완성 문장으로 존재하지 않는다. "
    "하지만 현재 실습에서는 구성 token이 train vocabulary 안에 있도록 만들었다."
)


# ==================================================
# 7. 전체 흐름 정리
# ==================================================

print_section("7. Final Data Flow Summary")

print(
    """
[Source branch]

English Source
"I like robots"
    ↓
Whitespace Tokenization
["I", "like", "robots"]
    ↓
+ <EOS>
["I", "like", "robots", "<EOS>"]
    ↓
Vocabulary Lookup
    ↓
Encoder Input IDs


[Target branch]

Korean Target
"나는 로봇을 좋아한다"
    ↓
Whitespace Tokenization
["나는", "로봇을", "좋아한다"]
    ↓
    ├─────────────────────────────┐
    ↓                             ↓
+ <SOS> at front              + <EOS> at end
    ↓                             ↓
Decoder Input                    Ground Truth
    ↓                             ↓
Decoder Input IDs                 GT IDs
"""
)

print("Concrete example:")

print(
    f"Encoder Input : {encoder_input_tokens}"
)
print(
    f"Decoder Input : {decoder_input_tokens}"
)
print(
    f"GT            : {gt_tokens}"
)

print(
    "\n핵심: Source는 Encoder로 들어가고, "
    "Target은 Decoder Input과 GT 두 갈래로 나뉜다."
)
