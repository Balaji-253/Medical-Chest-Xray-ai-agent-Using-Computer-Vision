from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)


ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = (
    ROOT
    / "artifacts"
    / "calibration_recalibration"
    / "validation_predictions.csv"
)

TEST_FILE = (
    ROOT
    / "artifacts"
    / "error_analysis"
    / "prediction_level_errors.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "calibration_recalibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EPS = 1e-6
N_BINS = 10


def clip_probability(p):
    return np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )


def to_logit(p):
    p = clip_probability(p)
    return np.log(p / (1.0 - p))


def calculate_ece(y_true, probability, n_bins=N_BINS):
    y_true = np.asarray(y_true).astype(int)
    probability = clip_probability(probability)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probability >= lower)
                & (probability <= upper)
            )
        else:
            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        if not np.any(mask):
            continue

        confidence = np.mean(probability[mask])
        observed = np.mean(y_true[mask])
        fraction = np.sum(mask) / len(y_true)

        ece += fraction * abs(
            confidence - observed
        )

    return float(ece)


def evaluate_probability_quality(y_true, probability):
    probability = clip_probability(probability)

    brier = brier_score_loss(
        y_true,
        probability,
    )

    ece = calculate_ece(
        y_true,
        probability,
    )

    return float(brier), float(ece)


def fit_platt(y_true, probability):
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        to_logit(probability).reshape(-1, 1),
        y_true,
    )

    return model


def fit_isotonic(y_true, probability):
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    model.fit(
        probability,
        y_true,
    )

    return model


def apply_platt(model, probability):
    return model.predict_proba(
        to_logit(probability).reshape(-1, 1)
    )[:, 1]


def apply_isotonic(model, probability):
    return clip_probability(
        model.predict(probability)
    )


def find_best_threshold(y_true, probability):
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

        score = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(
                threshold
            )

    return best_threshold, best_f1


def evaluate_classification(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    auroc = roc_auc_score(
        y_true,
        probability,
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    brier, ece = (
        evaluate_probability_quality(
            y_true,
            probability,
        )
    )

    return {
        "AUROC": float(auroc),
        "F1": float(f1),
        "Precision": float(precision),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Brier": float(brier),
        "ECE": float(ece),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "threshold": float(threshold),
    }


def main():

    print("=" * 70)
    print("Recalibrated Model Performance Comparison")
    print("=" * 70)

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {VAL_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {TEST_FILE}"
        )

    validation = pd.read_csv(
        VAL_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    labels = sorted(
        set(validation["label"])
        & set(test["label"])
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        f"Test rows:       {len(test)}"
    )

    print(
        f"Classes:         {len(labels)}"
    )

    results = []
    threshold_records = []

    methods = [
        "Original",
        "Platt",
        "Isotonic",
    ]

    for label in labels:

        val_subset = validation[
            validation["label"] == label
        ].copy()

        test_subset = test[
            test["label"] == label
        ].copy()

        val_true = pd.to_numeric(
            val_subset["true"],
            errors="coerce",
        ).to_numpy()

        val_probability = pd.to_numeric(
            val_subset["probability"],
            errors="coerce",
        ).to_numpy()

        test_true = pd.to_numeric(
            test_subset["true"],
            errors="coerce",
        ).to_numpy()

        test_probability = pd.to_numeric(
            test_subset["probability"],
            errors="coerce",
        ).to_numpy()

        val_valid = (
            np.isfinite(val_true)
            & np.isfinite(val_probability)
        )

        test_valid = (
            np.isfinite(test_true)
            & np.isfinite(test_probability)
        )

        val_true = (
            val_true[val_valid]
            .astype(int)
        )

        val_probability = (
            val_probability[val_valid]
            .astype(float)
        )

        test_true = (
            test_true[test_valid]
            .astype(int)
        )

        test_probability = (
            test_probability[test_valid]
            .astype(float)
        )

        if len(val_true) == 0:
            continue

        if len(test_true) == 0:
            continue

        if len(np.unique(val_true)) < 2:
            print(
                f"Skipping {label}: "
                "validation split contains only one class."
            )
            continue

        if len(np.unique(test_true)) < 2:
            print(
                f"Skipping {label}: "
                "test split contains only one class."
            )
            continue

        # -------------------------------------------------
        # Original model
        # -------------------------------------------------

        original_threshold, _ = (
            find_best_threshold(
                val_true,
                val_probability,
            )
        )

        original_test = (
            evaluate_classification(
                test_true,
                test_probability,
                original_threshold,
            )
        )

        original_test["label"] = label
        original_test["method"] = "Original"

        results.append(
            original_test
        )

        threshold_records.append(
            {
                "label": label,
                "method": "Original",
                "validation_threshold": original_threshold,
            }
        )

        # -------------------------------------------------
        # Platt calibration
        # -------------------------------------------------

        platt_model = fit_platt(
            val_true,
            val_probability,
        )

        val_platt = apply_platt(
            platt_model,
            val_probability,
        )

        test_platt = apply_platt(
            platt_model,
            test_probability,
        )

        platt_threshold, _ = (
            find_best_threshold(
                val_true,
                val_platt,
            )
        )

        platt_test = (
            evaluate_classification(
                test_true,
                test_platt,
                platt_threshold,
            )
        )

        platt_test["label"] = label
        platt_test["method"] = "Platt"

        results.append(
            platt_test
        )

        threshold_records.append(
            {
                "label": label,
                "method": "Platt",
                "validation_threshold": platt_threshold,
            }
        )

        # -------------------------------------------------
        # Isotonic calibration
        # -------------------------------------------------

        isotonic_model = fit_isotonic(
            val_true,
            val_probability,
        )

        val_isotonic = apply_isotonic(
            isotonic_model,
            val_probability,
        )

        test_isotonic = apply_isotonic(
            isotonic_model,
            test_probability,
        )

        isotonic_threshold, _ = (
            find_best_threshold(
                val_true,
                val_isotonic,
            )
        )

        isotonic_test = (
            evaluate_classification(
                test_true,
                test_isotonic,
                isotonic_threshold,
            )
        )

        isotonic_test["label"] = label
        isotonic_test["method"] = "Isotonic"

        results.append(
            isotonic_test
        )

        threshold_records.append(
            {
                "label": label,
                "method": "Isotonic",
                "validation_threshold": isotonic_threshold,
            }
        )

        print(
            f"{label:22s} "
            f"Original F1={original_test['F1']:.4f} "
            f"Platt F1={platt_test['F1']:.4f} "
            f"Isotonic F1={isotonic_test['F1']:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    threshold_df = pd.DataFrame(
        threshold_records
    )

    results_file = (
        OUTPUT_DIR
        / "recalibrated_performance.csv"
    )

    thresholds_file = (
        OUTPUT_DIR
        / "recalibrated_thresholds.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    threshold_df.to_csv(
        thresholds_file,
        index=False,
    )

    aggregate_rows = []

    for method in methods:

        subset = results_df[
            results_df["method"] == method
        ]

        aggregate_rows.append(
            {
                "method": method,
                "mean_AUROC": subset[
                    "AUROC"
                ].mean(),
                "mean_F1": subset[
                    "F1"
                ].mean(),
                "mean_Precision": subset[
                    "Precision"
                ].mean(),
                "mean_Sensitivity": subset[
                    "Sensitivity"
                ].mean(),
                "mean_Specificity": subset[
                    "Specificity"
                ].mean(),
                "mean_Brier": subset[
                    "Brier"
                ].mean(),
                "mean_ECE": subset[
                    "ECE"
                ].mean(),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    aggregate_file = (
        OUTPUT_DIR
        / "recalibrated_aggregate_metrics.csv"
    )

    aggregate_df.to_csv(
        aggregate_file,
        index=False,
    )

    print("")
    print("Aggregate results:")
    print("")
    print(
        aggregate_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary = {
        "model": (
            "ResNet-18 weighted baseline"
        ),
        "calibration_fit_split": "validation",
        "threshold_selection_split": "validation",
        "final_evaluation_split": "test",
        "methods": methods,
        "aggregate_metrics": (
            aggregate_df
            .round(6)
            .to_dict(
                orient="records"
            )
        ),
        "note": (
            "Calibration and decision thresholds "
            "were fitted/selected on validation data. "
            "Final metrics were computed on the "
            "untouched test prediction set."
        ),
        "clinical_warning": (
            "These are research evaluation results "
            "and do not establish clinical validity."
        ),
    }

    summary_file = (
        OUTPUT_DIR
        / "recalibrated_performance_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    report_file = (
        OUTPUT_DIR
        / "recalibrated_performance_report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Recalibrated Model Performance Comparison\n\n"
        )

        f.write(
            "## Methodology\n\n"
        )

        f.write(
            "Calibration models and class-specific "
            "decision thresholds were fitted using "
            "the validation split. Final metrics were "
            "then calculated on the untouched test "
            "prediction set.\n\n"
        )

        f.write(
            "## Aggregate Test Results\n\n"
        )

        f.write(
            "| Method | AUROC | F1 | Precision | "
            "Sensitivity | Specificity | Brier | ECE |\n"
        )

        f.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for _, row in aggregate_df.iterrows():

            f.write(
                f"| {row['method']} "
                f"| {row['mean_AUROC']:.4f} "
                f"| {row['mean_F1']:.4f} "
                f"| {row['mean_Precision']:.4f} "
                f"| {row['mean_Sensitivity']:.4f} "
                f"| {row['mean_Specificity']:.4f} "
                f"| {row['mean_Brier']:.4f} "
                f"| {row['mean_ECE']:.4f} |\n"
            )

        f.write(
            "\n## Per-Class Test Results\n\n"
        )

        f.write(
            "| Class | Method | AUROC | F1 | "
            "Precision | Sensitivity | Specificity | "
            "Brier | ECE | Threshold |\n"
        )

        f.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for _, row in results_df.iterrows():

            f.write(
                f"| {row['label']} "
                f"| {row['method']} "
                f"| {row['AUROC']:.4f} "
                f"| {row['F1']:.4f} "
                f"| {row['Precision']:.4f} "
                f"| {row['Sensitivity']:.4f} "
                f"| {row['Specificity']:.4f} "
                f"| {row['Brier']:.4f} "
                f"| {row['ECE']:.4f} "
                f"| {row['threshold']:.2f} |\n"
            )

        f.write(
            "\n## Interpretation\n\n"
        )

        f.write(
            "The comparison separates probability "
            "calibration from final test-set evaluation. "
            "Lower Brier score and ECE indicate improved "
            "probability calibration under these metrics. "
            "AUROC measures ranking performance and is "
            "generally unchanged by strictly monotonic "
            "probability transformations.\n\n"
        )

        f.write(
            "The calibrated methods are not assumed to "
            "improve disease classification performance "
            "merely because their probabilities are better "
            "calibrated. Classification metrics are "
            "reported separately.\n\n"
        )

        f.write(
            "These results are research-only and do not "
            "establish clinical diagnostic performance."
        )

    print("")
    print("=" * 70)
    print("Recalibrated performance comparison completed.")
    print("=" * 70)
    print("")
    print(
        f"Per-class results: {results_file}"
    )
    print(
        f"Thresholds:        {thresholds_file}"
    )
    print(
        f"Aggregate results: {aggregate_file}"
    )
    print(
        f"Summary:           {summary_file}"
    )
    print(
        f"Report:            {report_file}"
    )


if __name__ == "__main__":
    main()
