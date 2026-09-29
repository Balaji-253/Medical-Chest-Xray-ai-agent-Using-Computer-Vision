from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    ROOT
    / "artifacts"
    / "error_analysis"
    / "prediction_level_errors.csv"
)

METRICS_FILE = (
    ROOT
    / "artifacts"
    / "evaluation"
    / "weighted_model_test_metrics.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "statistical_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAPS = 1000


def bootstrap_ci(values, statistic_fn, rng, n_bootstraps=N_BOOTSTRAPS):
    """
    Calculate a percentile bootstrap 95% confidence interval.
    """
    values = np.asarray(values)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    estimates = []

    for _ in range(n_bootstraps):
        sample_indices = rng.integers(
            0,
            len(values),
            size=len(values),
        )

        sample = values[sample_indices]

        try:
            value = statistic_fn(sample)

            if np.isfinite(value):
                estimates.append(float(value))
        except Exception:
            continue

    if not estimates:
        return np.nan, np.nan, np.nan

    estimates = np.asarray(estimates)

    estimate = float(statistic_fn(values))
    lower = float(np.percentile(estimates, 2.5))
    upper = float(np.percentile(estimates, 97.5))

    return estimate, lower, upper


def bootstrap_binary_metric(y_true, y_score, metric_name, rng):
    """
    Bootstrap AUROC or thresholded F1.

    y_true:
        Binary ground-truth labels.

    y_score:
        Model probabilities/scores.

    metric_name:
        'AUROC' or 'F1'.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    if metric_name == "AUROC":

        def statistic(sample_indices):
            yt = y_true[sample_indices]
            ys = y_score[sample_indices]

            # AUROC is undefined if bootstrap sample
            # contains only one class.
            if len(np.unique(yt)) < 2:
                return np.nan

            return roc_auc_score(yt, ys)

        estimates = []

        for _ in range(N_BOOTSTRAPS):
            indices = rng.integers(
                0,
                len(y_true),
                size=len(y_true),
            )

            value = statistic(indices)

            if np.isfinite(value):
                estimates.append(value)

        if not estimates:
            return np.nan, np.nan, np.nan

        point_estimate = roc_auc_score(y_true, y_score)
        lower = np.percentile(estimates, 2.5)
        upper = np.percentile(estimates, 97.5)

        return (
            float(point_estimate),
            float(lower),
            float(upper),
        )

    if metric_name == "F1":

        threshold = 0.5

        y_pred = (y_score >= threshold).astype(int)

        point_estimate = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        bootstrap_values = []

        for _ in range(N_BOOTSTRAPS):
            indices = rng.integers(
                0,
                len(y_true),
                size=len(y_true),
            )

            yt = y_true[indices]
            yp = y_pred[indices]

            value = f1_score(
                yt,
                yp,
                zero_division=0,
            )

            bootstrap_values.append(value)

        lower = np.percentile(bootstrap_values, 2.5)
        upper = np.percentile(bootstrap_values, 97.5)

        return (
            float(point_estimate),
            float(lower),
            float(upper),
        )

    raise ValueError(f"Unsupported metric: {metric_name}")


def find_columns(df):
    """
    Try to identify the expected prediction-level columns.

    The existing project may use slightly different names,
    so this function searches common alternatives.
    """

    label_candidates = [
        "label",
        "class",
        "disease",
        "target",
    ]

    true_candidates = [
        "y_true",
        "true",
        "actual",
        "ground_truth",
        "target_value",
    ]

    score_candidates = [
        "probability",
        "prob",
        "score",
        "prediction_probability",
        "y_score",
    ]

    threshold_candidates = [
        "threshold",
        "optimal_threshold",
    ]

    label_col = next(
        (c for c in label_candidates if c in df.columns),
        None,
    )

    true_col = next(
        (c for c in true_candidates if c in df.columns),
        None,
    )

    score_col = next(
        (c for c in score_candidates if c in df.columns),
        None,
    )

    threshold_col = next(
        (c for c in threshold_candidates if c in df.columns),
        None,
    )

    return (
        label_col,
        true_col,
        score_col,
        threshold_col,
    )


def main():

    print("=" * 70)
    print("Bootstrap Statistical Evaluation")
    print("=" * 70)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}"
        )

    df = pd.read_csv(PREDICTION_FILE)
    metrics = pd.read_csv(METRICS_FILE)

    print(f"Prediction rows: {len(df)}")
    print(f"Metric rows:     {len(metrics)}")

    print("")
    print("Prediction file columns:")
    print(list(df.columns))

    (
        label_col,
        true_col,
        score_col,
        threshold_col,
    ) = find_columns(df)

    if label_col is None:
        raise ValueError(
            "Could not identify the class/label column in "
            "prediction_level_errors.csv"
        )

    if true_col is None:
        raise ValueError(
            "Could not identify the ground-truth column in "
            "prediction_level_errors.csv"
        )

    if score_col is None:
        raise ValueError(
            "Could not identify the probability/score column in "
            "prediction_level_errors.csv"
        )

    print("")
    print("Detected columns:")
    print(f"  Label:     {label_col}")
    print(f"  True:      {true_col}")
    print(f"  Score:     {score_col}")
    print(f"  Threshold: {threshold_col}")

    thresholds = {}

    for _, row in metrics.iterrows():
        thresholds[str(row["label"])] = float(row["threshold"])

    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    labels = sorted(df[label_col].dropna().unique())

    print("")
    print(f"Evaluating {len(labels)} classes...")
    print("")

    for label in labels:

        subset = df[df[label_col] == label].copy()

        y_true = pd.to_numeric(
            subset[true_col],
            errors="coerce",
        ).to_numpy()

        y_score = pd.to_numeric(
            subset[score_col],
            errors="coerce",
        ).to_numpy()

        valid = np.isfinite(y_true) & np.isfinite(y_score)

        y_true = y_true[valid].astype(int)
        y_score = y_score[valid].astype(float)

        if len(y_true) == 0:
            continue

        threshold = thresholds.get(str(label), 0.5)

        # AUROC bootstrap
        auroc_point, auroc_lower, auroc_upper = (
            bootstrap_binary_metric(
                y_true,
                y_score,
                "AUROC",
                rng,
            )
        )

        # F1 using the class-specific threshold
        y_pred = (
            y_score >= threshold
        ).astype(int)

        f1_point = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1_values = []

        for _ in range(N_BOOTSTRAPS):

            indices = rng.integers(
                0,
                len(y_true),
                size=len(y_true),
            )

            f1_values.append(
                f1_score(
                    y_true[indices],
                    y_pred[indices],
                    zero_division=0,
                )
            )

        f1_lower = np.percentile(
            f1_values,
            2.5,
        )

        f1_upper = np.percentile(
            f1_values,
            97.5,
        )

        results.append(
            {
                "label": label,
                "samples": len(y_true),
                "positive_samples": int(y_true.sum()),
                "threshold": threshold,
                "AUROC": auroc_point,
                "AUROC_CI_lower": auroc_lower,
                "AUROC_CI_upper": auroc_upper,
                "F1": f1_point,
                "F1_CI_lower": float(f1_lower),
                "F1_CI_upper": float(f1_upper),
            }
        )

        print(
            f"{label:22s} "
            f"AUROC={auroc_point:.4f} "
            f"95%CI=[{auroc_lower:.4f}, {auroc_upper:.4f}] "
            f"F1={f1_point:.4f} "
            f"95%CI=[{f1_lower:.4f}, {f1_upper:.4f}]"
        )

    results_df = pd.DataFrame(results)

    output_csv = (
        OUTPUT_DIR
        / "bootstrap_confidence_intervals.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    summary = {
        "model": "ResNet-18 weighted baseline",
        "random_seed": RANDOM_SEED,
        "bootstrap_iterations": N_BOOTSTRAPS,
        "confidence_level": 0.95,
        "classes_evaluated": int(len(results_df)),
        "mean_AUROC": float(results_df["AUROC"].mean()),
        "mean_F1": float(results_df["F1"].mean()),
        "note": (
            "Bootstrap confidence intervals are statistical "
            "uncertainty estimates for the current local test subset. "
            "They do not constitute clinical validation."
        ),
    }

    summary_file = (
        OUTPUT_DIR
        / "bootstrap_summary.json"
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
        / "bootstrap_report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Bootstrap Statistical Evaluation\n\n"
        )

        f.write(
            "## Experimental Configuration\n\n"
        )

        f.write(
            f"- Model: **ResNet-18 weighted baseline**\n"
            f"- Bootstrap iterations: **{N_BOOTSTRAPS}**\n"
            f"- Confidence level: **95%**\n"
            f"- Random seed: **{RANDOM_SEED}**\n\n"
        )

        f.write(
            "## Results\n\n"
        )

        f.write(
            "| Class | AUROC | 95% CI | F1 | 95% CI |\n"
        )

        f.write(
            "|---|---:|---|---:|---|\n"
        )

        for _, row in results_df.iterrows():

            f.write(
                f"| {row['label']} "
                f"| {row['AUROC']:.4f} "
                f"| [{row['AUROC_CI_lower']:.4f}, "
                f"{row['AUROC_CI_upper']:.4f}] "
                f"| {row['F1']:.4f} "
                f"| [{row['F1_CI_lower']:.4f}, "
                f"{row['F1_CI_upper']:.4f}] |\n"
            )

        f.write("\n")

        f.write(
            "## Interpretation\n\n"
        )

        f.write(
            "The confidence intervals quantify sampling uncertainty "
            "within the current local test subset. Wider intervals "
            "indicate greater uncertainty in the estimated metric. "
            "Classes with fewer positive samples can have substantially "
            "wider intervals.\n\n"
        )

        f.write(
            "These results are research evaluation results and should "
            "not be interpreted as clinical validation or evidence of "
            "diagnostic performance in clinical practice.\n"
        )

    print("")
    print("=" * 70)
    print("Bootstrap evaluation completed successfully.")
    print("=" * 70)
    print("")
    print(f"CSV:     {output_csv}")
    print(f"Summary: {summary_file}")
    print(f"Report:  {report_file}")
    print("")
    print(
        f"Mean AUROC: {results_df['AUROC'].mean():.4f}"
    )
    print(
        f"Mean F1:    {results_df['F1'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
