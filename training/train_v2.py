from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd


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
OUTPUT = Path("artifacts/chest_xray_resnet18_v2.pt")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
EPOCHS = 8
PATIENCE = 2

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayDatasetV2(torch.utils.data.Dataset):

    def __init__(self, split_dir, train=False):

        self.split_dir = Path(split_dir)
        self.train = train

        self.df = pd.read_csv(
            self.split_dir / "labels.csv"
        )

        if train:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=5
                ),

                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD
                ),
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD
                ),
            ])

    def __len__(self):
        return len(self.df)

    def encode_labels(self, label_string):

        target = torch.zeros(
            len(LABELS),
            dtype=torch.float32
        )

        for label in str(label_string).split("|"):

            label = label.strip()

            if label in LABELS:

                target[
                    LABELS.index(label)
                ] = 1.0

        return target

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = (
            self.split_dir /
            row["Image Index"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        target = self.encode_labels(
            row["Finding Labels"]
        )

        return {
            "image": image,
            "target": target,
            "image_name": row["Image Index"],
            "patient_id": str(
                row["Patient ID"]
            ),
        }


def calculate_pos_weights():

    df = pd.read_csv(
        ROOT / "train" / "labels.csv"
    )

    weights = []

    print("\nClass statistics:")

    for label in LABELS:

        positive = (
            df["Finding Labels"]
            .fillna("")
            .apply(
                lambda x:
                label in str(x).split("|")
            )
            .sum()
        )

        negative = len(df) - positive

        weight = (
            negative / max(positive, 1)
        )

        # Less aggressive than previous model.
        weight = min(max(weight, 1.0), 8.0)

        weights.append(weight)

        print(
            f"{label:20s} "
            f"positive={positive:4d} "
            f"weight={weight:.2f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def build_model():

    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # Freeze early feature extraction layers.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune only layer4.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Replace classifier.
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(
            in_features,
            len(LABELS)
        )
    )

    return model


def evaluate_loss(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                DEVICE
            )

            targets = batch["target"].to(
                DEVICE
            )

            logits = model(images)

            loss = criterion(
                logits,
                targets
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return (
        total_loss /
        max(total_samples, 1)
    )


def main():

    print("=" * 70)
    print("RESNET-18 V2 TRAINING")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    train_dataset = ChestXrayDatasetV2(
        ROOT / "train",
        train=True
    )

    val_dataset = ChestXrayDatasetV2(
        ROOT / "val",
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = build_model().to(DEVICE)

    pos_weight = (
        calculate_pos_weights()
        .to(DEVICE)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith("fc"):

            classifier_parameters.append(
                parameter
            )

        else:

            backbone_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": 1e-5
            },
            {
                "params": classifier_parameters,
                "lr": 5e-5
            }
        ],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )

    best_val_loss = float("inf")
    patience_counter = 0

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:

            images = batch["image"].to(
                DEVICE
            )

            targets = batch["target"].to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0
            )

            optimizer.step()

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

        train_loss = (
            running_loss /
            max(sample_count, 1)
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_loss
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "labels": LABELS,

                    "best_val_loss":
                        best_val_loss,

                    "epoch":
                        epoch
                },
                OUTPUT
            )

            print(
                "Saved improved V2 model."
            )

        else:

            patience_counter += 1

            print(
                f"No improvement. "
                f"Patience "
                f"{patience_counter}/{PATIENCE}"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping."
                )

                break

    print("\n" + "=" * 70)
    print("V2 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Model: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
