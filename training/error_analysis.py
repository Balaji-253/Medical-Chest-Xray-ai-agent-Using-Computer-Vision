from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from training.dataset import ChestXrayDataset
from training.evaluate_weighted import (
    LABELS,
    ROOT,
    CHECKPOINT,
    DEVICE,
    BATCH_SIZE,
    build_model,
)


OUTPUT_DIR = Path("artifacts/error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    model.load_state_dict(cleaned)
    model.to(DEVICE)
    model.eval()

    return model


@torch.no_grad()
def collect_predictions(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    targets = []
    probabilities = []
    image_names = []
    patient_ids = []

    for batch in loader:
        images = batch["image"].to(DEVICE)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        targets.append(batch["target"].numpy())
        probabilities.append(probs)

        image_names.extend(batch["image_name"])
        patient_ids.extend(batch["patient_id"])

    return (
        np.concatenate(targets),
        np.concatenate(probabilities),
        image_names,
        patient_ids,
    )


def main():
    print("=" * 70)
    print("CHEST X-RAY ERROR ANALYSIS")
    print("=" * 70)

    threshold_file = Path(
        "artifacts/evaluation/optimal_thresholds.json"
    )

    thresholds = pd.read_json(threshold_file, typ="series")

    thresholds = np.array(
        [float(thresholds[label]) for label in LABELS]
    )

    dataset = ChestXrayDataset(ROOT / "test")
    model = load_model()

    print(f"Test samples: {len(dataset)}")
    print(f"Device: {DEVICE}")

    (
        y_true,
        y_prob,
        image_names,
        patient_ids,
    ) = collect_predictions(model, dataset)

    rows = []

    for i, label in enumerate(LABELS):
        true = y_true[:, i].astype(int)
        prob = y_prob[:, i]

        threshold = thresholds[i]
        pred = (prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            true,
            pred,
            labels=[0, 1],
        ).ravel()

        auroc = roc_auc_score(true, prob)

        rows.append(
            {
                "label": label,
                "positive_samples": int(true.sum()),
                "negative_samples": int((true == 0).sum()),
                "threshold": threshold,
                "AUROC": auroc,
                "F1": f1_score(
                    true,
                    pred,
                    zero_division=0,
                ),
                "Precision": precision_score(
                    true,
                    pred,
                    zero_division=0,
                ),
                "Sensitivity": recall_score(
                    true,
                    pred,
                    zero_division=0,
                ),
                "Specificity": (
                    tn / (tn + fp)
                    if (tn + fp) > 0
                    else 0
                ),
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            }
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "AUROC",
        ascending=False,
    )

    output = OUTPUT_DIR / "class_error_analysis.csv"
    results.to_csv(output, index=False)

    print("\nClass-level results:")
    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Save prediction-level errors.
    prediction_rows = []

    for row_index in range(len(y_true)):
        for class_index, label in enumerate(LABELS):
            true = int(y_true[row_index, class_index])
            prob = float(y_prob[row_index, class_index])
            threshold = float(thresholds[class_index])
            pred = int(prob >= threshold)

            error_type = "CORRECT"

            if true == 1 and pred == 0:
                error_type = "FALSE_NEGATIVE"
            elif true == 0 and pred == 1:
                error_type = "FALSE_POSITIVE"

            prediction_rows.append(
                {
                    "image_name": image_names[row_index],
                    "patient_id": patient_ids[row_index],
                    "label": label,
                    "true": true,
                    "probability": prob,
                    "threshold": threshold,
                    "prediction": pred,
                    "error_type": error_type,
                }
            )

    prediction_df = pd.DataFrame(prediction_rows)

    prediction_output = (
        OUTPUT_DIR / "prediction_level_errors.csv"
    )

    prediction_df.to_csv(
        prediction_output,
        index=False,
    )

    print("\nSaved:")
    print(output)
    print(prediction_output)

    print("\nBest AUROC classes:")
    print(
        results[
            ["label", "AUROC", "F1", "Sensitivity"]
        ].head(5).to_string(index=False)
    )

    print("\nClasses needing improvement:")
    print(
        results[
            ["label", "AUROC", "F1", "Sensitivity"]
        ].tail(5).to_string(index=False)
    )

    print("\nError analysis complete.")


if __name__ == "__main__":
    main()
