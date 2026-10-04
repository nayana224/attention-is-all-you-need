import csv
from pathlib import Path

import matplotlib.pyplot as plt


STUDY_DIR = Path(__file__).resolve().parents[1]
CSV_DIR = STUDY_DIR / "outputs" / "csv"
FIGURE_DIR = STUDY_DIR / "outputs" / "training_figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        rows = []

        for row in reader:
            rows.append(
                row
            )

        return rows


training_rows = read_csv(
    CSV_DIR / "training_trace.csv"
)

attention_rows = read_csv(
    CSV_DIR / "attention_trace.csv"
)

parameter_rows = read_csv(
    CSV_DIR / "parameter_trace.csv"
)


steps = []

for row in training_rows:
    step = int(
        row["step"]
    )

    steps.append(
        step
    )


# ==================================================
# 1. Probe / Test Loss 변화
# ==================================================

probe_losses = []
test_losses = []

for row in training_rows:
    probe_loss = float(
        row["probe_loss"]
    )

    test_loss = float(
        row["test_mean_loss"]
    )

    probe_losses.append(
        probe_loss
    )

    test_losses.append(
        test_loss
    )

plt.figure(figsize=(8, 4))

plt.plot(
    steps,
    probe_losses,
    label="Held-out probe",
)

plt.plot(
    steps,
    test_losses,
    label="Test mean",
)

plt.title("Held-out Loss vs Step")
plt.xlabel("Optimizer Step")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.grid()

loss_path = (
    FIGURE_DIR
    / "01_heldout_loss_vs_step.png"
)

plt.tight_layout()
plt.savefig(loss_path, dpi=150)
plt.close()


# ==================================================
# 2. 학습에 사용하지 않은 데이터의 정확도 / 정답 확률
# ==================================================

probe_gt_probs = []
test_accuracies = []

for row in training_rows:
    probe_gt_prob = float(
        row[
            "probe_mean_gt_probability"
        ]
    )

    test_accuracy = float(
        row["test_token_accuracy"]
    )

    probe_gt_probs.append(
        probe_gt_prob
    )

    test_accuracies.append(
        test_accuracy
    )

plt.figure(figsize=(8, 4))

plt.plot(
    steps,
    probe_gt_probs,
    label="Probe mean GT probability",
)

plt.plot(
    steps,
    test_accuracies,
    label="Test token accuracy",
)

plt.title(
    "Held-out Prediction Quality vs Step"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Value")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid()

quality_path = (
    FIGURE_DIR
    / "02_heldout_quality_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    quality_path,
    dpi=150,
)
plt.close()


# ==================================================
# 3. Decoder Cross-Attention
#    Head 1 / probe의 마지막 query가
#    English source token을 보는 weight 변화
# ==================================================

cross_rows = []

for row in attention_rows:
    is_cross_attention = (
        row["attention_type"]
        == "decoder_cross"
    )

    is_head_zero = (
        int(row["head"]) == 0
    )

    if (
        is_cross_attention
        and is_head_zero
    ):
        cross_rows.append(
            row
        )

query_indices = []

for row in cross_rows:
    query_index = int(
        row["query_index"]
    )

    query_indices.append(
        query_index
    )

last_query_index = max(
    query_indices
)

cross_last_rows = []

for row in cross_rows:
    query_index = int(
        row["query_index"]
    )

    if query_index == last_query_index:
        cross_last_rows.append(
            row
        )

by_key = {}

for row in cross_last_rows:
    key_token = row["key_token"]

    if key_token not in by_key:
        by_key[key_token] = []

    step = int(
        row["step"]
    )

    weight = float(
        row["weight"]
    )

    step_and_weight = (
        step,
        weight,
    )

    by_key[key_token].append(
        step_and_weight
    )

plt.figure(figsize=(8, 5))

for key_token, values in by_key.items():
    # 값은 step 순서대로 저장되므로
    # 복잡한 정렬 문법 없이 그대로 사용한다.
    x = []
    y = []

    for value in values:
        step = value[0]
        weight = value[1]

        x.append(
            step
        )

        y.append(
            weight
        )

    plt.plot(
        x,
        y,
        label=key_token,
    )

plt.title(
    "Probe Cross-Attention "
    "(Head 1, Last Decoder Query)"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Attention Weight")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid()

attention_path = (
    FIGURE_DIR
    / "03_probe_cross_attention_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    attention_path,
    dpi=150,
)
plt.close()


# ==================================================
# 4. 대표 W_Q scalar 변화
# ==================================================

tracked_parameter = (
    "decoder_layers.0."
    "masked_self_attention."
    "W_Q.0.weight"
)

tracked_rows = []

for row in parameter_rows:
    same_parameter = (
        row["parameter"]
        == tracked_parameter
    )

    same_index = (
        row["index"] == "0,0"
    )

    if same_parameter and same_index:
        tracked_rows.append(
            row
        )


parameter_steps = []
parameter_values = []

for row in tracked_rows:
    step = int(
        row["step"]
    )

    value = float(
        row["value_before"]
    )

    parameter_steps.append(
        step
    )

    parameter_values.append(
        value
    )

plt.figure(figsize=(8, 4))

plt.plot(
    parameter_steps,
    parameter_values,
)

plt.title(
    "Selected W_Q[0,0] vs Step"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Parameter Value")
plt.grid()

parameter_path = (
    FIGURE_DIR
    / "04_selected_parameter_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    parameter_path,
    dpi=150,
)
plt.close()


# ==================================================
# 5. 같은 parameter의 gradient 변화
# ==================================================

gradients = []

for row in tracked_rows:
    gradient = float(
        row["gradient"]
    )

    gradients.append(
        gradient
    )

plt.figure(figsize=(8, 4))

plt.plot(
    parameter_steps,
    gradients,
)

plt.axhline(
    y=0.0,
    linestyle="--",
    linewidth=1,
)

plt.title(
    "Selected W_Q[0,0] Gradient vs Step"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Gradient")
plt.grid()

gradient_path = (
    FIGURE_DIR
    / "05_selected_gradient_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    gradient_path,
    dpi=150,
)
plt.close()


print("=" * 70)
print("Saved Figures")
print("=" * 70)

for path in [
    loss_path,
    quality_path,
    attention_path,
    parameter_path,
    gradient_path,
]:
    print(path)

print(
    "\nCSV 전체 값은 study/outputs/csv/에서 "
    "직접 열어 확인할 수 있다."
)
