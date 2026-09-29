from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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

LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


class ChestXrayDataset(Dataset):
    def __init__(self, split_dir, image_size=224):
        self.split_dir = Path(split_dir)
        self.df = pd.read_csv(self.split_dir / "labels.csv")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def _encode_labels(self, label_string):
        target = torch.zeros(len(LABELS), dtype=torch.float32)

        for label in str(label_string).split("|"):
            label = label.strip()

            if label in LABEL_TO_INDEX:
                target[LABEL_TO_INDEX[label]] = 1.0

        return target

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self.split_dir / row["Image Index"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        target = self._encode_labels(row["Finding Labels"])

        return {
            "image": image,
            "target": target,
            "image_name": row["Image Index"],
            "patient_id": str(row["Patient ID"]),
        }
