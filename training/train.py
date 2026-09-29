from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from dataset import ChestXrayDataset, LABELS


ROOT = Path("data/processed/NIH_ChestXray14")
ARTIFACTS = Path("artifacts")

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

DEVICE = torch.device("cpu")


def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze the early feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final residual block.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    model.fc = nn.Linear(
        model.fc.in_features,
        len(LABELS),
    )

    return model


def compute_pos_weights(dataset):
    df = dataset.df

    positive_counts = np.zeros(len(LABELS), dtype=np.float64)

    for labels in df["Finding Labels"].astype(str):
        for label in labels.split("|"):
            if label in LABELS:
                positive_counts[LABELS.index(label)] += 1

    total = len(df)

    negative_counts = total - positive_counts

    # Balanced BCE:
    # weight = negatives / positives
    weights = negative_counts / np.maximum(positive_counts, 1)

    # Avoid extreme weights for extremely rare classes.
    weights = np.clip(weights, 1.0, 15.0)

    print("\nPositive counts:")
    for label, count, weight in zip(
        LABELS,
        positive_counts,
        weights,
    ):
        print(
            f"{label:20s}: "
            f"{int(count):4d} positives | "
            f"weight={weight:.2f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_items = 0

    progress = tqdm(loader, leave=False)

    for batch in progress:

        images = batch["image"].to(DEVICE)
        targets = batch["target"].to(DEVICE)

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()

                # Prevent unstable updates.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_items += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / max(total_items, 1)


def main():

    print("Device:", DEVICE)
    print("Labels:", len(LABELS))

    train_dataset = ChestXrayDataset(
        ROOT / "train"
    )

    val_dataset = ChestXrayDataset(
        ROOT / "val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model().to(DEVICE)

    pos_weights = compute_pos_weights(
        train_dataset
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        with torch.no_grad():

            val_loss = run_epoch(
                model,
                val_loader,
                criterion,
            )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "labels": LABELS,
                },
                ARTIFACTS /
                "chest_xray_resnet18_weighted.pt",
            )

            print(
                "Saved improved model."
            )

    print(
        "\nImproved training complete."
    )

    print(
        "Model:",
        ARTIFACTS /
        "chest_xray_resnet18_weighted.pt",
    )


if __name__ == "__main__":
    main()
