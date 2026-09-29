from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT
    / "artifacts"
    / "chest_xray_resnet18_weighted.pt"
)

THRESHOLDS_FILE = (
    ROOT
    / "artifacts"
    / "evaluation"
    / "optimal_thresholds.json"
)

PLATT_DIR = (
    ROOT
    / "artifacts"
    / "calibration_recalibration"
)

PLATT_MODELS_FILE = (
    PLATT_DIR
    / "platt_models.json"
)

PLATT_THRESHOLDS_FILE = (
    PLATT_DIR
    / "platt_thresholds.json"
)


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


class ChestXrayInference:
    """
    Research-only Chest X-ray inference engine.

    The underlying ResNet-18 classifier is unchanged.

    Two probability views are exposed:

    1. raw_probability
       Original model output.

    2. calibrated_probability
       Platt-calibrated probability learned from the
       validation split.

    Classification remains based on the original model
    thresholds so that the selected baseline behavior
    remains reproducible.

    This system is NOT a clinical diagnostic device.
    """

    def __init__(self):

        self.device = torch.device("cpu")

        self.labels = LABELS

        self.model = self._load_model()

        self.original_thresholds = (
            self._load_original_thresholds()
        )

        self.platt_models = (
            self._load_platt_models()
        )

        self.platt_thresholds = (
            self._load_platt_thresholds()
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

    def _load_model(self):

        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: "
                f"{CHECKPOINT}"
            )

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            len(self.labels),
        )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(
            checkpoint,
            dict,
        ) and "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:
            state_dict = checkpoint

        model.load_state_dict(
            state_dict
        )

        model.to(
            self.device
        )

        model.eval()

        return model

    def _load_original_thresholds(self):

        if not THRESHOLDS_FILE.exists():

            return {
                label: 0.5
                for label in self.labels
            }

        with open(
            THRESHOLDS_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        return {
            label: float(
                data.get(
                    label,
                    0.5,
                )
            )
            for label in self.labels
        }

    def _load_platt_models(self):

        if not PLATT_MODELS_FILE.exists():
            return {}

        with open(
            PLATT_MODELS_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    def _load_platt_thresholds(self):

        if not PLATT_THRESHOLDS_FILE.exists():

            return {
                label: 0.5
                for label in self.labels
            }

        with open(
            PLATT_THRESHOLDS_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        return {
            label: float(
                data.get(
                    label,
                    0.5,
                )
            )
            for label in self.labels
        }

    @staticmethod
    def _clip_probability(
        probability,
        eps=1e-6,
    ):

        return np.clip(
            float(probability),
            eps,
            1.0 - eps,
        )

    @classmethod
    def _to_logit(
        cls,
        probability,
    ):

        probability = (
            cls._clip_probability(
                probability
            )
        )

        return float(
            np.log(
                probability
                / (
                    1.0
                    - probability
                )
            )
        )

    def _platt_calibrate(
        self,
        label,
        probability,
    ):

        model_data = (
            self.platt_models.get(
                label
            )
        )

        if model_data is None:

            return float(
                probability
            )

        logit = self._to_logit(
            probability
        )

        coef = np.asarray(
            model_data["coef"],
            dtype=float,
        )

        intercept = np.asarray(
            model_data["intercept"],
            dtype=float,
        )

        value = (
            float(coef[0][0])
            * logit
            + float(intercept[0])
        )

        calibrated = (
            1.0
            / (
                1.0
                + np.exp(-value)
            )
        )

        return float(
            calibrated
        )

    @staticmethod
    def _uncertainty(
        max_probability
    ):

        if max_probability >= 0.75:
            return "LOW"

        if max_probability >= 0.50:
            return "MODERATE"

        return "HIGH"

    def predict(
        self,
        image_path: str | Path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = self.transform(
            image
        ).unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            raw_probabilities = (
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()[0]
            )

        predictions = []

        positive_predictions = []

        calibrated_values = []

        for index, label in enumerate(
            self.labels
        ):

            raw_probability = float(
                raw_probabilities[
                    index
                ]
            )

            calibrated_probability = (
                self._platt_calibrate(
                    label,
                    raw_probability,
                )
            )

            original_threshold = (
                self.original_thresholds[
                    label
                ]
            )

            platt_threshold = (
                self.platt_thresholds[
                    label
                ]
            )

            # The original classifier threshold
            # remains the primary research signal.
            predicted = (
                raw_probability
                >= original_threshold
            )

            predictions.append(
                {
                    "label": label,
                    "raw_probability": (
                        raw_probability
                    ),
                    "calibrated_probability": (
                        calibrated_probability
                    ),
                    "original_threshold": (
                        original_threshold
                    ),
                    "platt_threshold": (
                        platt_threshold
                    ),
                    "predicted": bool(
                        predicted
                    ),
                }
            )

            calibrated_values.append(
                calibrated_probability
            )

            if predicted:

                positive_predictions.append(
                    {
                        "label": label,
                        "raw_probability": (
                            raw_probability
                        ),
                        "calibrated_probability": (
                            calibrated_probability
                        ),
                        "threshold": (
                            original_threshold
                        ),
                    }
                )

        max_raw_probability = float(
            np.max(
                raw_probabilities
            )
        )

        max_calibrated_probability = float(
            np.max(
                calibrated_values
            )
        )

        uncertainty = self._uncertainty(
            max_raw_probability
        )

        human_review_required = (
            uncertainty != "LOW"
            or len(
                positive_predictions
            ) > 0
        )

        return {
            "model": "ResNet-18",
            "device": "cpu",
            "image": str(
                image_path
            ),
            "predictions": predictions,
            "positive_predictions": (
                positive_predictions
            ),
            "max_probability": (
                max_raw_probability
            ),
            "max_calibrated_probability": (
                max_calibrated_probability
            ),
            "uncertainty": uncertainty,
            "human_review_required": (
                human_review_required
            ),
            "calibration": {
                "method": "Platt",
                "fitted_on": (
                    "validation split"
                ),
                "classification_thresholds": (
                    "original validation-selected "
                    "thresholds"
                ),
                "probability_output": (
                    "calibrated probability is "
                    "reported alongside raw probability"
                ),
            },
            "research_only": True,
            "notice": (
                "This output is for research and "
                "software evaluation only. It is not "
                "a medical diagnosis and must not be "
                "used as a substitute for professional "
                "clinical assessment."
            ),
        }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Research-only Chest X-ray inference"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to chest X-ray image",
    )

    args = parser.parse_args()

    engine = ChestXrayInference()

    result = engine.predict(
        args.image
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
