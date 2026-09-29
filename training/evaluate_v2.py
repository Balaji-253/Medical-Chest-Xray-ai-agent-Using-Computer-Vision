import json
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


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

ROOT = Path("data/processed/NIH_ChestXray14")
CHECKPOINT = Path("artifacts/chest_xray_resnet18_v2.pt")
OUTPUT_DIR = Path("artifacts/evaluation_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def build_model():
    from torchvision.models import resnet18

    model = resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.35),
        torch.nn.Linear(in_features, len(LABELS)),
    )

    return model


def load_model():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

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

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


@torch.no_grad()
def predict(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_targets = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        targets = batch["target"]

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_prob


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(y_true, y_prob, thresholds):
    rows = []

    for i, label in enumerate(LABELS):
        true = y_true[:, i].astype(int)
        prob = y_prob[:, i]
        threshold = thresholds[i]

        pred = (prob >= threshold).astype(int)

        positive_samples = int(true.sum())

        if len(np.unique(true)) >= 2:
            try:
                auroc = float(roc_auc_score(true, prob))
            except Exception:
                auroc = float("nan")
        else:
            auroc = float("nan")

        f1 = float(
            f1_score(
                true,
                pred,
                zero_division=0,
            )
        )

        precision = float(
            precision_score(
                true,
                pred,
                zero_division=0,
            )
        )

        sensitivity = float(
            recall_score(
                true,
                pred,
                zero_division=0,
            )
        )

        tn, fp, fn, tp = confusion_matrix(
            true,
            pred,
            labels=[0, 1],
        ).ravel()

        specificity = (
            float(tn / (tn + fp))
            if (tn + fp) > 0
            else float("nan")
        )

        rows.append(
            {
                "label": label,
                "positive_samples": positive_samples,
                "threshold": threshold,
                "AUROC": auroc,
                "F1": f1,
                "Precision": precision,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("WEIGHTED CHEST X-RAY MODEL EVALUATION")
    print("=" * 70)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    train_csv = ROOT / "train" / "labels.csv"
    val_csv = ROOT / "val" / "labels.csv"
    test_csv = ROOT / "test" / "labels.csv"

    print("\nLoading datasets...")

    val_dataset = ChestXrayDataset(ROOT / "val")
    test_dataset = ChestXrayDataset(ROOT / "test")

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = load_model()

    print("\nRunning validation predictions...")
    val_true, val_prob = predict(model, val_dataset)

    print("\nFinding per-class thresholds...")

    thresholds = []

    for i, label in enumerate(LABELS):
        threshold, best_f1 = find_best_threshold(
            val_true[:, i],
            val_prob[:, i],
        )

        thresholds.append(threshold)

        print(
            f"{label:20s} "
            f"threshold={threshold:.2f} "
            f"validation_F1={best_f1:.4f}"
        )

    thresholds = np.array(thresholds)

    threshold_file = OUTPUT_DIR / "optimal_thresholds.json"

    threshold_data = {
        label: float(thresholds[i])
        for i, label in enumerate(LABELS)
    }

    with open(threshold_file, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"\nSaved thresholds: {threshold_file}")

    print("\nRunning TEST predictions...")
    test_true, test_prob = predict(model, test_dataset)

    results = calculate_metrics(
        test_true,
        test_prob,
        thresholds,
    )

    results_file = OUTPUT_DIR / "weighted_model_test_metrics.csv"

    results.to_csv(
        results_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    valid_auroc = results["AUROC"].dropna()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Mean AUROC:       {valid_auroc.mean():.4f}"
    )

    print(
        f"Mean F1:           {results['F1'].mean():.4f}"
    )

    print(
        f"Mean Precision:    {results['Precision'].mean():.4f}"
    )

    print(
        f"Mean Sensitivity:  {results['Sensitivity'].mean():.4f}"
    )

    print(
        f"Mean Specificity:  {results['Specificity'].mean():.4f}"
    )

    print(f"\nSaved metrics: {results_file}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
