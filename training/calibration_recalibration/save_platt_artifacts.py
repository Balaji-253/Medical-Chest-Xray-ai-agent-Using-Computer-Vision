from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    ROOT
    / "artifacts"
    / "calibration_recalibration"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "calibration_recalibration"
)

MODELS_FILE = (
    OUTPUT_DIR
    / "platt_models.json"
)

THRESHOLDS_FILE = (
    OUTPUT_DIR
    / "platt_thresholds.json"
)


def clip_probability(
    probability,
    eps=1e-6,
):
    return np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        eps,
        1.0 - eps,
    )


def to_logit(probability):
    probability = clip_probability(
        probability
    )

    return np.log(
        probability
        / (1.0 - probability)
    )


def find_best_threshold(
    y_true,
    probability,
):
    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        prediction = (
            probability >= threshold
        ).astype(int)

        tp = np.sum(
            (y_true == 1)
            & (prediction == 1)
        )

        fp = np.sum(
            (y_true == 0)
            & (prediction == 1)
        )

        fn = np.sum(
            (y_true == 1)
            & (prediction == 0)
        )

        denominator = (
            2 * tp
            + fp
            + fn
        )

        f1 = (
            2 * tp / denominator
            if denominator > 0
            else 0.0
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(
                threshold
            )

    return best_threshold, best_f1


def main():

    print("=" * 70)
    print("Saving Platt Calibration Artifacts")
    print("=" * 70)

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation predictions not found:\n"
            f"{VALIDATION_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        VALIDATION_FILE
    )

    required_columns = {
        "label",
        "true",
        "probability",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    labels = sorted(
        df["label"].unique()
    )

    models = {}
    thresholds = {}

    print(
        f"Validation rows: {len(df)}"
    )

    print(
        f"Classes: {len(labels)}"
    )

    print("")

    for label in labels:

        subset = df[
            df["label"] == label
        ].copy()

        y = pd.to_numeric(
            subset["true"],
            errors="coerce",
        ).to_numpy()

        probability = pd.to_numeric(
            subset["probability"],
            errors="coerce",
        ).to_numpy()

        valid = (
            np.isfinite(y)
            & np.isfinite(probability)
        )

        y = y[valid].astype(int)

        probability = (
            probability[valid]
            .astype(float)
        )

        if len(y) == 0:
            print(
                f"SKIP {label}: no valid rows"
            )
            continue

        if len(np.unique(y)) < 2:
            print(
                f"SKIP {label}: "
                "only one validation class"
            )
            continue

        probability = clip_probability(
            probability
        )

        logit = to_logit(
            probability
        )

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )

        model.fit(
            logit.reshape(-1, 1),
            y,
        )

        calibrated = model.predict_proba(
            logit.reshape(-1, 1)
        )[:, 1]

        threshold, validation_f1 = (
            find_best_threshold(
                y,
                calibrated,
            )
        )

        models[label] = {
            "coef": (
                model.coef_
                .astype(float)
                .tolist()
            ),
            "intercept": (
                model.intercept_
                .astype(float)
                .tolist()
            ),
            "classes": (
                model.classes_
                .astype(int)
                .tolist()
            ),
        }

        thresholds[label] = threshold

        print(
            f"{label:22s} "
            f"threshold={threshold:.2f} "
            f"validation_F1={validation_f1:.4f}"
        )

    with open(
        MODELS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            models,
            f,
            indent=2,
        )

    with open(
        THRESHOLDS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            thresholds,
            f,
            indent=2,
        )

    print("")
    print("=" * 70)
    print("Platt calibration artifacts saved")
    print("=" * 70)
    print("")
    print(
        f"Models:    {MODELS_FILE}"
    )
    print(
        f"Thresholds:{THRESHOLDS_FILE}"
    )


if __name__ == "__main__":
    main()
