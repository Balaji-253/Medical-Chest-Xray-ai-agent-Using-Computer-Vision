from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    ROOT
    / "artifacts"
    / "error_analysis"
    / "prediction_level_errors.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "calibration"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 10


def expected_calibration_error(
    y_true,
    probability,
    n_bins=N_BINS,
):
    """
    Calculate Expected Calibration Error (ECE).

    ECE is the weighted average difference between:
        mean predicted probability
    and
        observed positive frequency
    across probability bins.
    """

    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    rows = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

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

        bin_probability = probability[mask]
        bin_true = y_true[mask]

        confidence = float(
            np.mean(bin_probability)
        )

        observed_frequency = float(
            np.mean(bin_true)
        )

        count = int(np.sum(mask))

        gap = abs(
            confidence - observed_frequency
        )

        ece += (
            count / len(y_true)
        ) * gap

        rows.append(
            {
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": confidence,
                "observed_frequency": observed_frequency,
                "absolute_gap": gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def create_calibration_plot(
    calibration_df,
    label,
):
    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        calibration_df["mean_probability"],
        calibration_df["observed_frequency"],
        marker="o",
        label="Model",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive frequency")
    plt.title(
        f"Calibration Curve - {label}"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(alpha=0.25)
    plt.legend()

    safe_label = (
        str(label)
        .replace("/", "_")
        .replace(" ", "_")
    )

    output_file = (
        PLOT_DIR
        / f"calibration_{safe_label}.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    return output_file


def create_ece_plot(results):

    data = results.sort_values(
        "ECE",
        ascending=True,
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        data["label"],
        data["ECE"],
    )

    plt.xlabel(
        "Expected Calibration Error"
    )

    plt.ylabel("Class")

    plt.title(
        "Expected Calibration Error by Class"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR
        / "ece_by_class.png"
    )

    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    return output_file


def create_brier_plot(results):

    data = results.sort_values(
        "Brier_score",
        ascending=True,
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        data["label"],
        data["Brier_score"],
    )

    plt.xlabel(
        "Brier Score"
    )

    plt.ylabel("Class")

    plt.title(
        "Brier Score by Class"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR
        / "brier_score_by_class.png"
    )

    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    return output_file


def main():

    print("=" * 70)
    print("Medical AI Probability Calibration Evaluation")
    print("=" * 70)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    df = pd.read_csv(
        PREDICTION_FILE
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
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Prediction rows: {len(df)}"
    )

    labels = sorted(
        df["label"]
        .dropna()
        .unique()
    )

    results = []

    for label in labels:

        subset = df[
            df["label"] == label
        ].copy()

        y_true = pd.to_numeric(
            subset["true"],
            errors="coerce",
        ).to_numpy()

        probability = pd.to_numeric(
            subset["probability"],
            errors="coerce",
        ).to_numpy()

        valid = (
            np.isfinite(y_true)
            & np.isfinite(probability)
        )

        y_true = (
            y_true[valid]
            .astype(int)
        )

        probability = (
            probability[valid]
            .astype(float)
        )

        if len(y_true) == 0:
            continue

        brier = brier_score_loss(
            y_true,
            probability,
        )

        ece, calibration_df = (
            expected_calibration_error(
                y_true,
                probability,
                N_BINS,
            )
        )

        plot_file = (
            create_calibration_plot(
                calibration_df,
                label,
            )
        )

        results.append(
            {
                "label": label,
                "samples": len(y_true),
                "positive_samples": int(
                    y_true.sum()
                ),
                "Brier_score": float(
                    brier
                ),
                "ECE": float(
                    ece
                ),
            }
        )

        calibration_df.to_csv(
            OUTPUT_DIR
            / (
                f"calibration_bins_"
                f"{label}.csv"
            ),
            index=False,
        )

        print(
            f"{label:22s} "
            f"Brier={brier:.4f} "
            f"ECE={ece:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_csv = (
        OUTPUT_DIR
        / "calibration_metrics.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False,
    )

    ece_plot = create_ece_plot(
        results_df
    )

    brier_plot = create_brier_plot(
        results_df
    )

    summary = {
        "model": (
            "ResNet-18 weighted baseline"
        ),
        "classes_evaluated": int(
            len(results_df)
        ),
        "mean_Brier_score": float(
            results_df[
                "Brier_score"
            ].mean()
        ),
        "mean_ECE": float(
            results_df["ECE"].mean()
        ),
        "calibration_bins": N_BINS,
        "note": (
            "Calibration results are based "
            "on the current local test subset "
            "and should not be interpreted as "
            "clinical validation."
        ),
    }

    summary_file = (
        OUTPUT_DIR
        / "calibration_summary.json"
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
        / "calibration_report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Probability Calibration Evaluation\n\n"
        )

        f.write(
            "## Model\n\n"
        )

        f.write(
            "**ResNet-18 weighted baseline**\n\n"
        )

        f.write(
            "## Aggregate Calibration Metrics\n\n"
        )

        f.write(
            f"- Mean Brier Score: "
            f"**{summary['mean_Brier_score']:.4f}**\n"
        )

        f.write(
            f"- Mean Expected Calibration Error: "
            f"**{summary['mean_ECE']:.4f}**\n"
        )

        f.write(
            f"- Calibration bins: "
            f"**{N_BINS}**\n\n"
        )

        f.write(
            "## Per-Class Results\n\n"
        )

        f.write(
            "| Class | Samples | Positive | "
            "Brier Score | ECE |\n"
        )

        f.write(
            "|---|---:|---:|---:|---:|\n"
        )

        for _, row in results_df.iterrows():

            f.write(
                f"| {row['label']} "
                f"| {int(row['samples'])} "
                f"| {int(row['positive_samples'])} "
                f"| {row['Brier_score']:.4f} "
                f"| {row['ECE']:.4f} |\n"
            )

        f.write(
            "\n## Interpretation\n\n"
        )

        f.write(
            "The Brier score measures the mean squared "
            "difference between predicted probability "
            "and the binary outcome. Lower values "
            "indicate smaller probabilistic error.\n\n"
        )

        f.write(
            "Expected Calibration Error (ECE) summarizes "
            "the difference between predicted probability "
            "and observed frequency across probability "
            "bins. Lower values indicate closer agreement "
            "between predicted confidence and observed "
            "frequency under this evaluation procedure.\n\n"
        )

        f.write(
            "Calibration metrics are descriptive research "
            "measurements for the current test subset. "
            "They do not establish clinical reliability "
            "or diagnostic validity.\n"
        )

    print("")
    print("=" * 70)
    print("Calibration evaluation completed successfully.")
    print("=" * 70)
    print("")
    print(f"CSV:     {results_csv}")
    print(f"Summary: {summary_file}")
    print(f"Report:  {report_file}")
    print(f"ECE plot:   {ece_plot}")
    print(f"Brier plot: {brier_plot}")
    print("")
    print(
        f"Mean Brier Score: "
        f"{summary['mean_Brier_score']:.4f}"
    )
    print(
        f"Mean ECE: "
        f"{summary['mean_ECE']:.4f}"
    )


if __name__ == "__main__":
    main()
