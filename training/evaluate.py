from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataset import ChestXrayDataset, LABELS


ROOT = Path("data/processed/NIH_ChestXray14")
MODEL_PATH = Path("artifacts/chest_xray_resnet18.pt")

DEVICE = torch.device("cpu")
BATCH_SIZE = 16


def build_model():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(LABELS),
    )
    return model


def main():
    print("Loading test dataset...")

    dataset = ChestXrayDataset(ROOT / "test")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_targets = []
    all_probs = []

    print("Running inference...")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_targets.append(
                batch["target"].numpy()
            )

            all_probs.append(
                probs.cpu().numpy()
            )

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    y_pred = (y_prob >= 0.5).astype(int)

    results = []

    print("\nPer-label metrics")
    print("=" * 70)

    for i, label in enumerate(LABELS):

        true = y_true[:, i]
        prob = y_prob[:, i]
        pred = y_pred[:, i]

        positives = int(true.sum())
        negatives = int(len(true) - positives)

        # AUROC requires both positive and negative examples.
        if positives > 0 and negatives > 0:
            auc = roc_auc_score(true, prob)
        else:
            auc = float("nan")

        f1 = f1_score(
            true,
            pred,
            zero_division=0,
        )

        recall = recall_score(
            true,
            pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            true,
            pred,
            labels=[0, 1],
        ).ravel()

        specificity = tn / max(tn + fp, 1)

        results.append({
            "label": label,
            "positive_samples": positives,
            "AUROC": auc,
            "F1": f1,
            "Sensitivity": recall,
            "Specificity": specificity,
        })

        print(
            f"{label:20s} "
            f"AUC={auc:.4f} "
            f"F1={f1:.4f} "
            f"Sens={recall:.4f} "
            f"Spec={specificity:.4f}"
        )

    results_df = pd.DataFrame(results)

    macro_auc = results_df["AUROC"].mean()

    try:
        micro_auc = roc_auc_score(
            y_true.ravel(),
            y_prob.ravel(),
        )
    except ValueError:
        micro_auc = float("nan")

    print("\n" + "=" * 70)
    print(f"Macro AUROC: {macro_auc:.4f}")
    print(f"Micro AUROC: {micro_auc:.4f}")

    output = Path("artifacts/evaluation_metrics.csv")
    results_df.to_csv(output, index=False)

    print(f"\nSaved metrics to: {output}")


if __name__ == "__main__":
    main()
