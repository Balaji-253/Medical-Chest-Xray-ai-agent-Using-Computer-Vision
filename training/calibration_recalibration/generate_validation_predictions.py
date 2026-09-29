from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.dataset import ChestXrayDataset


CHECKPOINT = ROOT / "artifacts" / "chest_xray_resnet18_weighted.pt"
VAL_DIR = ROOT / "data" / "processed" / "NIH_ChestXray14" / "val"

OUTPUT_DIR = ROOT / "artifacts" / "calibration_recalibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "validation_predictions.csv"

BATCH_SIZE = 16
IMAGE_SIZE = 224

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


def build_model(device):
    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(LABELS),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def main():

    print("=" * 70)
    print("Generate Validation Predictions")
    print("=" * 70)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {VAL_DIR}"
        )

    device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Validation directory: {VAL_DIR}")

    dataset = ChestXrayDataset(
        VAL_DIR,
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Validation samples: {len(dataset)}"
    )

    model = build_model(device)

    rows = []

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            images = batch["image"].to(device)

            targets = batch["target"]

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            targets = targets.cpu().numpy()

            image_names = batch["image_name"]

            patient_ids = batch["patient_id"]

            batch_size = len(image_names)

            for i in range(batch_size):

                image_name = image_names[i]

                patient_id = patient_ids[i]

                for class_index, label in enumerate(LABELS):

                    rows.append(
                        {
                            "image_name": image_name,
                            "patient_id": patient_id,
                            "label": label,
                            "true": int(
                                targets[
                                    i,
                                    class_index
                                ]
                            ),
                            "probability": float(
                                probabilities[
                                    i,
                                    class_index
                                ]
                            ),
                        }
                    )

            if (
                batch_index + 1
            ) % 10 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    metadata = {
        "model": "ResNet-18 weighted baseline",
        "checkpoint": str(
            CHECKPOINT.relative_to(ROOT)
        ),
        "split": "validation",
        "samples": int(len(dataset)),
        "prediction_rows": int(len(result)),
        "classes": LABELS,
        "device": "cpu",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "purpose": (
            "Validation probabilities for fitting "
            "probability recalibration without using "
            "the final test set."
        ),
    }

    with open(
        OUTPUT_DIR / "validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("")
    print("=" * 70)
    print("Validation prediction generation completed.")
    print("=" * 70)
    print("")
    print(f"Output: {OUTPUT_FILE}")
    print(
        f"Rows:   {len(result)}"
    )
    print(
        f"Images: {len(dataset)}"
    )
    print("")
    print(
        "The validation predictions can now be used "
        "to fit recalibration methods."
    )


if __name__ == "__main__":
    main()
