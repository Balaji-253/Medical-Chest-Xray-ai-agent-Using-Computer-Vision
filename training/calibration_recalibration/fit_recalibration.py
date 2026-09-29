from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


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


def safe_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=float),
        EPS,
        1.0 - EPS,
    )


def probability_to_logit(probability):
    p = safe_clip(probability)
    return np.log(p / (1.0 - p))


def calculate_ece(
    y_true,
    probability,
    n_bins=N_BINS,
):
    y_true = np.asarray(y_true).astype(int)
    probability = safe_clip(probability)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

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

        confidence = np.mean(
            probability[mask]
        )

        observed = np.mean(
            y_true[mask]
        )

        fraction = (
            np.sum(mask)
            / len(y_true)
        )

        ece += fraction * abs(
            confidence - observed
        )

    return float(ece)


def evaluate_calibration(
    y_true,
    probability,
):
    probability = safe_clip(
        probability
    )

    brier = brier_score_loss(
        y_true,
        probability,
    )

    ece = calculate_ece(
        y_true,
        probability,
    )

    return float(brier), float(ece)


def fit_platt(
    y_true,
    probability,
):
    """
    Fit logistic calibration using
    model logits as the single feature.

    The calibrator is fitted only on
    validation data.
    """

    logits = probability_to_logit(
        probability
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        logits,
        y_true,
    )

    return calibrator


def fit_isotonic(
    y_true,
    probability,
):
    """
    Fit isotonic regression using
    validation probabilities.
    """

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probability,
        y_true,
    )

    return calibrator


def main():

    print("=" * 70)
    print("Probability Recalibration Experiment")
    print("=" * 70)

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VAL_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test prediction file not found: {TEST_FILE}"
        )

    validation = pd.read_csv(
        VAL_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    required_val = {
        "label",
        "true",
        "probability",
    }

    required_test = {
        "label",
        "true",
        "probability",
    }

    missing_val = (
        required_val
        - set(validation.columns)
    )

    missing_test = (
        required_test
        - set(test.columns)
    )

    if missing_val:
        raise ValueError(
            "Validation file is missing: "
            + ", ".join(
                sorted(missing_val)
            )
        )

    if missing_test:
        raise ValueError(
            "Test file is missing: "
            + ", ".join(
                sorted(missing_test)
            )
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
    calibrated_rows = []

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

        val_prob = pd.to_numeric(
            val_subset["probability"],
            errors="coerce",
        ).to_numpy()

        test_true = pd.to_numeric(
            test_subset["true"],
            errors="coerce",
        ).to_numpy()

        test_prob = pd.to_numeric(
            test_subset["probability"],
            errors="coerce",
        ).to_numpy()

        val_valid = (
            np.isfinite(val_true)
            & np.isfinite(val_prob)
        )

        test_valid = (
            np.isfinite(test_true)
            & np.isfinite(test_prob)
        )

        val_true = (
            val_true[val_valid]
            .astype(int)
        )

        val_prob = (
            val_prob[val_valid]
            .astype(float)
        )

        test_true = (
            test_true[test_valid]
            .astype(int)
        )

        test_prob = (
            test_prob[test_valid]
            .astype(float)
        )

        if len(val_true) == 0:
            continue

        if len(test_true) == 0:
            continue

        # -------------------------------------------------
        # Original test probabilities
        # -------------------------------------------------

        original_brier, original_ece = (
            evaluate_calibration(
                test_true,
                test_prob,
            )
        )

        # -------------------------------------------------
        # Platt calibration
        # -------------------------------------------------

        platt = fit_platt(
            val_true,
            val_prob,
        )

        test_logits = probability_to_logit(
            test_prob
        ).reshape(-1, 1)

        platt_probability = platt.predict_proba(
            test_logits
        )[:, 1]

        platt_brier, platt_ece = (
            evaluate_calibration(
                test_true,
                platt_probability,
            )
        )

        # -------------------------------------------------
        # Isotonic calibration
        # -------------------------------------------------

        isotonic = fit_isotonic(
            val_true,
            val_prob,
        )

        isotonic_probability = (
            isotonic.predict(
                test_prob
            )
        )

        isotonic_probability = safe_clip(
            isotonic_probability
        )

        isotonic_brier, isotonic_ece = (
            evaluate_calibration(
                test_true,
                isotonic_probability,
            )
        )

        results.append(
            {
                "label": label,
                "test_samples": len(test_true),
                "positive_samples": int(
                    test_true.sum()
                ),
                "original_Brier": original_brier,
                "platt_Brier": platt_brier,
                "isotonic_Brier": isotonic_brier,
                "original_ECE": original_ece,
                "platt_ECE": platt_ece,
                "isotonic_ECE": isotonic_ece,
                "platt_Brier_change": (
                    platt_brier
                    - original_brier
                ),
                "isotonic_Brier_change": (
                    isotonic_brier
                    - original_brier
                ),
                "platt_ECE_change": (
                    platt_ece
                    - original_ece
                ),
                "isotonic_ECE_change": (
                    isotonic_ece
                    - original_ece
                ),
            }
        )

        for index in range(
            len(test_true)
        ):

            calibrated_rows.append(
                {
                    "image_name": test_subset.iloc[
                        np.where(test_valid)[0][index]
                    ]["image_name"],
                    "patient_id": test_subset.iloc[
                        np.where(test_valid)[0][index]
                    ]["patient_id"],
                    "label": label,
                    "true": int(
                        test_true[index]
                    ),
                    "original_probability": float(
                        test_prob[index]
                    ),
                    "platt_probability": float(
                        platt_probability[index]
                    ),
                    "isotonic_probability": float(
                        isotonic_probability[index]
                    ),
                }
            )

        print(
            f"{label:22s} "
            f"Original Brier={original_brier:.4f} "
            f"Platt={platt_brier:.4f} "
            f"Isotonic={isotonic_brier:.4f} "
            f"| Original ECE={original_ece:.4f} "
            f"Platt={platt_ece:.4f} "
            f"Isotonic={isotonic_ece:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    calibrated_df = pd.DataFrame(
        calibrated_rows
    )

    results_file = (
        OUTPUT_DIR
        / "recalibration_comparison.csv"
    )

    calibrated_file = (
        OUTPUT_DIR
        / "calibrated_test_predictions.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    calibrated_df.to_csv(
        calibrated_file,
        index=False,
    )

    summary = {
        "model": (
            "ResNet-18 weighted baseline"
        ),
        "validation_rows": int(
            len(validation)
        ),
        "test_rows": int(
            len(test)
        ),
        "classes": int(
            len(results_df)
        ),
        "fitting_split": "validation",
        "evaluation_split": "test",
        "original_mean_Brier": float(
            results_df[
                "original_Brier"
            ].mean()
        ),
        "platt_mean_Brier": float(
            results_df[
                "platt_Brier"
            ].mean()
        ),
        "isotonic_mean_Brier": float(
            results_df[
                "isotonic_Brier"
            ].mean()
        ),
        "original_mean_ECE": float(
            results_df[
                "original_ECE"
            ].mean()
        ),
        "platt_mean_ECE": float(
            results_df[
                "platt_ECE"
            ].mean()
        ),
        "isotonic_mean_ECE": float(
            results_df[
                "isotonic_ECE"
            ].mean()
        ),
        "methodology": (
            "Platt and isotonic calibration "
            "were fitted on validation predictions "
            "and evaluated on the untouched test "
            "prediction set."
        ),
        "clinical_warning": (
            "Calibration results are research "
            "measurements and do not establish "
            "clinical validity."
        ),
    }

    with open(
        OUTPUT_DIR
        / "recalibration_summary.json",
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
        / "recalibration_report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Probability Recalibration Experiment\n\n"
        )

        f.write(
            "## Methodology\n\n"
        )

        f.write(
            "Calibration methods were fitted using the "
            "**validation split only** and evaluated on "
            "the untouched test prediction set.\n\n"
        )

        f.write(
            "Methods compared:\n\n"
        )

        f.write(
            "1. Original model probabilities\n"
            "2. Platt/logistic calibration\n"
            "3. Isotonic regression calibration\n\n"
        )

        f.write(
            "## Aggregate Results\n\n"
        )

        f.write(
            "| Method | Mean Brier | Mean ECE |\n"
        )

        f.write(
            "|---|---:|---:|\n"
        )

        f.write(
            f"| Original | "
            f"{summary['original_mean_Brier']:.4f} | "
            f"{summary['original_mean_ECE']:.4f} |\n"
        )

        f.write(
            f"| Platt | "
            f"{summary['platt_mean_Brier']:.4f} | "
            f"{summary['platt_mean_ECE']:.4f} |\n"
        )

        f.write(
            f"| Isotonic | "
            f"{summary['isotonic_mean_Brier']:.4f} | "
            f"{summary['isotonic_mean_ECE']:.4f} |\n"
        )

        f.write(
            "\n## Per-Class Results\n\n"
        )

        f.write(
            "| Class | Original Brier | "
            "Platt Brier | Isotonic Brier | "
            "Original ECE | Platt ECE | "
            "Isotonic ECE |\n"
        )

        f.write(
            "|---|---:|---:|---:|---:|---:|---:|\n"
        )

        for _, row in results_df.iterrows():

            f.write(
                f"| {row['label']} "
                f"| {row['original_Brier']:.4f} "
                f"| {row['platt_Brier']:.4f} "
                f"| {row['isotonic_Brier']:.4f} "
                f"| {row['original_ECE']:.4f} "
                f"| {row['platt_ECE']:.4f} "
                f"| {row['isotonic_ECE']:.4f} |\n"
            )

        f.write(
            "\n## Interpretation\n\n"
        )

        f.write(
            "A lower Brier score indicates lower "
            "probabilistic prediction error. A lower "
            "ECE indicates closer agreement between "
            "predicted probabilities and observed "
            "frequencies under the selected binning "
            "procedure.\n\n"
        )

        f.write(
            "Calibration method selection should be "
            "based on validation methodology and "
            "independent test evaluation rather than "
            "optimizing directly on the test set.\n\n"
        )

        f.write(
            "These results are research-only and do "
            "not establish clinical reliability."
        )

    print("")
    print("=" * 70)
    print("Probability recalibration completed successfully.")
    print("=" * 70)
    print("")
    print(f"Comparison: {results_file}")
    print(f"Predictions: {calibrated_file}")
    print(
        f"Summary: "
        f"{OUTPUT_DIR / 'recalibration_summary.json'}"
    )
    print(
        f"Report: "
        f"{report_file}"
    )
    print("")
    print(
        f"Original mean Brier: "
        f"{summary['original_mean_Brier']:.4f}"
    )
    print(
        f"Platt mean Brier:    "
        f"{summary['platt_mean_Brier']:.4f}"
    )
    print(
        f"Isotonic mean Brier:  "
        f"{summary['isotonic_mean_Brier']:.4f}"
    )
    print("")
    print(
        f"Original mean ECE: "
        f"{summary['original_mean_ECE']:.4f}"
    )
    print(
        f"Platt mean ECE:    "
        f"{summary['platt_mean_ECE']:.4f}"
    )
    print(
        f"Isotonic mean ECE: "
        f"{summary['isotonic_mean_ECE']:.4f}"
    )


if __name__ == "__main__":
    main()
