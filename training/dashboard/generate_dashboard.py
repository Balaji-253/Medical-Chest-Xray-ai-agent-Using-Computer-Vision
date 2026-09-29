from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = ROOT / "artifacts" / "evaluation" / "weighted_model_test_metrics.csv"
ERROR_FILE = ROOT / "artifacts" / "error_analysis" / "class_error_analysis.csv"

OUT_DIR = ROOT / "artifacts" / "evaluation_dashboard"
PLOT_DIR = OUT_DIR / "plots"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Missing evaluation file: {EVAL_FILE}")

    if not ERROR_FILE.exists():
        raise FileNotFoundError(f"Missing error-analysis file: {ERROR_FILE}")

    metrics = pd.read_csv(EVAL_FILE)
    errors = pd.read_csv(ERROR_FILE)

    return metrics, errors


def save_bar_chart(df, column, title, ylabel, filename, ylim=None):
    data = df.sort_values(column, ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(data["label"], data[column])
    plt.xlabel(ylabel)
    plt.ylabel("Class")
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)

    if ylim is not None:
        plt.xlim(ylim)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def save_sensitivity_specificity(df):
    data = df.sort_values("Sensitivity", ascending=True)

    y = np.arange(len(data))
    height = 0.38

    plt.figure(figsize=(12, 8))
    plt.barh(y - height / 2, data["Sensitivity"], height, label="Sensitivity")
    plt.barh(y + height / 2, data["Specificity"], height, label="Specificity")

    plt.yticks(y, data["label"])
    plt.xlabel("Score")
    plt.ylabel("Class")
    plt.title("Sensitivity vs Specificity by Class")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "sensitivity_vs_specificity.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def save_precision_sensitivity(df):
    data = df.sort_values("Precision", ascending=True)

    y = np.arange(len(data))
    height = 0.38

    plt.figure(figsize=(12, 8))
    plt.barh(y - height / 2, data["Precision"], height, label="Precision")
    plt.barh(y + height / 2, data["Sensitivity"], height, label="Sensitivity")

    plt.yticks(y, data["label"])
    plt.xlabel("Score")
    plt.ylabel("Class")
    plt.title("Precision vs Sensitivity by Class")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "precision_vs_sensitivity.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def save_confusion_components(df):
    data = df.copy()

    x = np.arange(len(data))
    width = 0.18

    plt.figure(figsize=(14, 8))

    plt.bar(x - 1.5 * width, data["TP"], width, label="TP")
    plt.bar(x - 0.5 * width, data["FP"], width, label="FP")
    plt.bar(x + 0.5 * width, data["TN"], width, label="TN")
    plt.bar(x + 1.5 * width, data["FN"], width, label="FN")

    plt.xticks(x, data["label"], rotation=55, ha="right")
    plt.ylabel("Number of samples")
    plt.xlabel("Class")
    plt.title("Confusion-Matrix Components by Class")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "confusion_matrix_components.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def save_positive_distribution(df):
    data = df.sort_values("positive_samples", ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(data["label"], data["positive_samples"])
    plt.xlabel("Positive samples")
    plt.ylabel("Class")
    plt.title("Positive Sample Distribution in Test Set")
    plt.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "positive_sample_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def save_threshold_distribution(df):
    data = df.sort_values("threshold", ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(data["label"], data["threshold"])
    plt.xlabel("Selected threshold")
    plt.ylabel("Class")
    plt.title("Per-Class Decision Thresholds")
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "threshold_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def create_summary(metrics, errors):
    mean_metrics = {
        "mean_AUROC": float(metrics["AUROC"].mean()),
        "mean_F1": float(metrics["F1"].mean()),
        "mean_Precision": float(metrics["Precision"].mean()),
        "mean_Sensitivity": float(metrics["Sensitivity"].mean()),
        "mean_Specificity": float(metrics["Specificity"].mean()),
    }

    best_auroc = metrics.loc[metrics["AUROC"].idxmax()]
    worst_auroc = metrics.loc[metrics["AUROC"].idxmin()]
    best_f1 = metrics.loc[metrics["F1"].idxmax()]
    highest_sensitivity = metrics.loc[metrics["Sensitivity"].idxmax()]
    highest_specificity = metrics.loc[metrics["Specificity"].idxmax()]

    summary = {
        "model": "ResNet-18 weighted baseline",
        "evaluation_file": str(EVAL_FILE.relative_to(ROOT)),
        "error_analysis_file": str(ERROR_FILE.relative_to(ROOT)),
        "test_classes": int(len(metrics)),
        "total_test_positive_labels": int(metrics["positive_samples"].sum()),
        "mean_metrics": mean_metrics,
        "highest_AUROC_class": {
            "label": str(best_auroc["label"]),
            "AUROC": float(best_auroc["AUROC"]),
        },
        "lowest_AUROC_class": {
            "label": str(worst_auroc["label"]),
            "AUROC": float(worst_auroc["AUROC"]),
        },
        "highest_F1_class": {
            "label": str(best_f1["label"]),
            "F1": float(best_f1["F1"]),
        },
        "highest_sensitivity_class": {
            "label": str(highest_sensitivity["label"]),
            "Sensitivity": float(highest_sensitivity["Sensitivity"]),
        },
        "highest_specificity_class": {
            "label": str(highest_specificity["label"]),
            "Specificity": float(highest_specificity["Specificity"]),
        },
        "research_warning": (
            "These results are research evaluation results from the current "
            "local NIH ChestX-ray14 subset. They are not clinical validation "
            "or evidence of diagnostic performance in clinical practice."
        ),
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def create_markdown_report(metrics, errors, summary):
    lines = []

    lines.append("# Medical AI Research Evaluation Report")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append("**ResNet-18 weighted baseline**")
    lines.append("")
    lines.append(
        "The evaluation uses the currently generated test-set metrics "
        "and class-level error analysis artifacts."
    )
    lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Mean |")
    lines.append("|---|---:|")

    for key, value in summary["mean_metrics"].items():
        display_name = key.replace("_", " ").title()
        lines.append(f"| {display_name} | {value:.4f} |")

    lines.append("")

    lines.append("## Per-Class Results")
    lines.append("")
    lines.append(
        "| Class | Positive | AUROC | F1 | Precision | Sensitivity | Specificity | Threshold |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['label']} "
            f"| {int(row['positive_samples'])} "
            f"| {row['AUROC']:.4f} "
            f"| {row['F1']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['Sensitivity']:.4f} "
            f"| {row['Specificity']:.4f} "
            f"| {row['threshold']:.2f} |"
        )

    lines.append("")

    lines.append("## Confusion-Matrix Error Analysis")
    lines.append("")
    lines.append("| Class | TP | FP | TN | FN |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, row in errors.iterrows():
        lines.append(
            f"| {row['label']} "
            f"| {int(row['TP'])} "
            f"| {int(row['FP'])} "
            f"| {int(row['TN'])} "
            f"| {int(row['FN'])} |"
        )

    lines.append("")

    lines.append("## Research Interpretation")
    lines.append("")

    lines.append(
        f"- Mean AUROC across the 14 classes is "
        f"**{summary['mean_metrics']['mean_AUROC']:.4f}**."
    )

    lines.append(
        f"- Mean F1 is **{summary['mean_metrics']['mean_F1']:.4f}**, "
        f"indicating substantial variation between ranking performance "
        f"and thresholded classification performance."
    )

    lines.append(
        f"- The highest AUROC in this evaluation is for "
        f"**{summary['highest_AUROC_class']['label']}** "
        f"({summary['highest_AUROC_class']['AUROC']:.4f})."
    )

    lines.append(
        f"- The lowest AUROC in this evaluation is for "
        f"**{summary['lowest_AUROC_class']['label']}** "
        f"({summary['lowest_AUROC_class']['AUROC']:.4f})."
    )

    lines.append(
        f"- The highest F1 in this evaluation is for "
        f"**{summary['highest_F1_class']['label']}** "
        f"({summary['highest_F1_class']['F1']:.4f})."
    )

    lines.append("")

    lines.append("## Important Limitations")
    lines.append("")
    lines.append(
        "- The current evaluation is based on the project's local test subset."
    )
    lines.append(
        "- The results should not be interpreted as clinical validation."
    )
    lines.append(
        "- Performance varies considerably across disease classes."
    )
    lines.append(
        "- Several classes have relatively few positive test samples."
    )
    lines.append(
        "- Threshold optimization was performed using the validation split."
    )
    lines.append(
        "- Further robustness, calibration, confidence-interval, and "
        "external-validation analysis should be performed before making "
        "stronger research claims."
    )

    report_path = OUT_DIR / "research_evaluation_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main():
    print("=" * 70)
    print("Medical AI Research Evaluation Dashboard")
    print("=" * 70)

    metrics, errors = load_data()

    print(f"Loaded evaluation rows: {len(metrics)}")
    print(f"Loaded error-analysis rows: {len(errors)}")

    save_bar_chart(
        metrics,
        "AUROC",
        "AUROC by Disease Class",
        "AUROC",
        "auroc_by_class.png",
        ylim=(0, 1),
    )

    save_bar_chart(
        metrics,
        "F1",
        "F1 Score by Disease Class",
        "F1",
        "f1_by_class.png",
        ylim=(0, 1),
    )

    save_sensitivity_specificity(metrics)
    save_precision_sensitivity(metrics)
    save_confusion_components(errors)
    save_positive_distribution(metrics)
    save_threshold_distribution(metrics)

    summary = create_summary(metrics, errors)
    report_path = create_markdown_report(metrics, errors, summary)

    metrics.to_csv(
        OUT_DIR / "metrics_snapshot.csv",
        index=False,
    )

    errors.to_csv(
        OUT_DIR / "error_analysis_snapshot.csv",
        index=False,
    )

    print("")
    print("Dashboard generated successfully.")
    print("")
    print(f"Output directory: {OUT_DIR}")
    print(f"Plots directory:  {PLOT_DIR}")
    print(f"Research report:  {report_path}")
    print("")
    print("Mean metrics:")
    for key, value in summary["mean_metrics"].items():
        print(f"  {key}: {value:.4f}")

    print("")
    print("Highest AUROC:")
    print(
        f"  {summary['highest_AUROC_class']['label']}: "
        f"{summary['highest_AUROC_class']['AUROC']:.4f}"
    )

    print("Lowest AUROC:")
    print(
        f"  {summary['lowest_AUROC_class']['label']}: "
        f"{summary['lowest_AUROC_class']['AUROC']:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
