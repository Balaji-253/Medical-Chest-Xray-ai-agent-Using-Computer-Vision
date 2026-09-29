from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """
    Research-only Platt probability calibration.

    The calibration parameters are learned from the validation
    split and then applied to unseen predictions.

    This does NOT change the underlying classifier.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

        self.models = {}
        self.thresholds = {}

        self.models_file = (
            self.model_dir / "platt_models.json"
        )

        self.thresholds_file = (
            self.model_dir / "platt_thresholds.json"
        )

        self.labels = []

        self._load()

    @staticmethod
    def _clip_probability(
        probability,
        eps=1e-6,
    ):
        probability = np.asarray(
            probability,
            dtype=float,
        )

        return np.clip(
            probability,
            eps,
            1.0 - eps,
        )

    @staticmethod
    def _to_logit(probability):
        probability = PlattCalibrator._clip_probability(
            probability
        )

        return np.log(
            probability
            / (1.0 - probability)
        )

    def _load(self):

        if not self.models_file.exists():
            raise FileNotFoundError(
                f"Platt calibration file not found: "
                f"{self.models_file}"
            )

        if not self.thresholds_file.exists():
            raise FileNotFoundError(
                f"Platt threshold file not found: "
                f"{self.thresholds_file}"
            )

        with open(
            self.models_file,
            "r",
            encoding="utf-8",
        ) as f:
            model_data = json.load(f)

        with open(
            self.thresholds_file,
            "r",
            encoding="utf-8",
        ) as f:
            self.thresholds = json.load(f)

        self.labels = sorted(
            model_data.keys()
        )

        for label in self.labels:

            data = model_data[label]

            model = LogisticRegression()

            model.coef_ = np.asarray(
                data["coef"],
                dtype=float,
            )

            model.intercept_ = np.asarray(
                data["intercept"],
                dtype=float,
            )

            model.classes_ = np.asarray(
                data["classes"],
                dtype=int,
            )

            model.n_features_in_ = 1

            self.models[label] = model

    def calibrate(
        self,
        label: str,
        probability: float,
    ) -> float:

        if label not in self.models:
            raise KeyError(
                f"No Platt calibration model "
                f"available for label: {label}"
            )

        probability = float(
            self._clip_probability(
                [probability]
            )[0]
        )

        logit = self._to_logit(
            [probability]
        )[0]

        calibrated = self.models[
            label
        ].predict_proba(
            np.asarray(
                [[logit]],
                dtype=float,
            )
        )[0, 1]

        return float(calibrated)

    def threshold(
        self,
        label: str,
    ) -> float:

        if label not in self.thresholds:
            return 0.5

        return float(
            self.thresholds[label]
        )

    def predict(
        self,
        label: str,
        probability: float,
    ) -> dict:

        calibrated_probability = (
            self.calibrate(
                label,
                probability,
            )
        )

        threshold = self.threshold(
            label
        )

        predicted = (
            calibrated_probability
            >= threshold
        )

        return {
            "raw_probability": float(
                probability
            ),
            "calibrated_probability": float(
                calibrated_probability
            ),
            "threshold": float(
                threshold
            ),
            "predicted": bool(
                predicted
            ),
            "calibration": "platt",
        }
