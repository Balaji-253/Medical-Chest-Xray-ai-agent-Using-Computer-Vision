import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_DIR = (
    ROOT
    / "artifacts"
    / "calibration_recalibration"
)

MODELS_FILE = (
    CALIBRATION_DIR
    / "platt_models.json"
)

THRESHOLDS_FILE = (
    CALIBRATION_DIR
    / "platt_thresholds.json"
)


EXPECTED_LABELS = {
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
}


def test_platt_artifacts_exist():
    """Both calibration artifacts must exist."""

    assert MODELS_FILE.exists()
    assert THRESHOLDS_FILE.exists()


def test_platt_contains_all_labels():
    """Every one of the 14 X-ray labels must have calibration data."""

    with open(
        MODELS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        models = json.load(f)

    with open(
        THRESHOLDS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        thresholds = json.load(f)

    assert set(models.keys()) == EXPECTED_LABELS
    assert set(thresholds.keys()) == EXPECTED_LABELS


def test_platt_parameters_are_finite():
    """Calibration coefficients must contain valid finite numbers."""

    with open(
        MODELS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        models = json.load(f)

    for label, data in models.items():

        assert "coef" in data
        assert "intercept" in data
        assert "classes" in data

        coefficient = np.asarray(
            data["coef"],
            dtype=float,
        )

        intercept = np.asarray(
            data["intercept"],
            dtype=float,
        )

        assert coefficient.shape == (1, 1)
        assert intercept.shape == (1,)

        assert np.all(
            np.isfinite(coefficient)
        )

        assert np.all(
            np.isfinite(intercept)
        )


def test_platt_thresholds_are_valid():
    """Validation-selected thresholds must be in [0, 1]."""

    with open(
        THRESHOLDS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        thresholds = json.load(f)

    for label, threshold in thresholds.items():

        value = float(threshold)

        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_inference_exposes_calibrated_probability():
    """Inference must expose raw and calibrated probabilities."""

    from app.ml.chest_xray_inference import (
        ChestXrayInference,
    )

    image_path = (
        ROOT
        / "data"
        / "processed"
        / "NIH_ChestXray14"
        / "test"
        / "00001170_003.png"
    )

    if not image_path.exists():
        pytest.skip(
            "Reference test image is not available."
        )

    engine = ChestXrayInference()

    result = engine.predict(
        image_path
    )

    assert result["research_only"] is True

    assert (
        result["calibration"]["method"]
        == "Platt"
    )

    assert len(
        result["predictions"]
    ) == 14

    for prediction in result[
        "predictions"
    ]:

        assert (
            "raw_probability"
            in prediction
        )

        assert (
            "calibrated_probability"
            in prediction
        )

        assert (
            "original_threshold"
            in prediction
        )

        assert (
            "platt_threshold"
            in prediction
        )

        raw = float(
            prediction[
                "raw_probability"
            ]
        )

        calibrated = float(
            prediction[
                "calibrated_probability"
            ]
        )

        assert np.isfinite(raw)
        assert np.isfinite(calibrated)

        assert 0.0 <= raw <= 1.0
        assert 0.0 <= calibrated <= 1.0
