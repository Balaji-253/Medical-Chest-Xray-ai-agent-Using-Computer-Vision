from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


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


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "artifacts/chest_xray_resnet18_weighted.pt"
)

IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        class_index
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ].detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

        probability = torch.sigmoid(
            score
        ).item()

        return cam, probability

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(LABELS)
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = image.copy()

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN,
            STD
        )
    ])

    tensor = transform(
        image
    ).unsqueeze(0)

    return original, tensor


def create_overlay(
    original,
    cam,
    output_path
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    original = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.asarray(
        original
    ) / 255.0

    fig = plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image_array
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1
    )

    plt.axis("off")

    plt.tight_layout(
        pad=0
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)


def generate_gradcam(
    image_path,
    class_name,
    output_path
):

    if class_name not in LABELS:

        raise ValueError(
            f"Unknown class: {class_name}"
        )

    model = load_model()

    original, tensor = preprocess_image(
        image_path
    )

    tensor = tensor.to(
        DEVICE
    )

    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    class_index = LABELS.index(
        class_name
    )

    cam, probability = (
        gradcam.generate(
            tensor,
            class_index
        )
    )

    gradcam.close()

    create_overlay(
        original,
        cam,
        output_path
    )

    return probability


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--class-name",
        required=True,
        choices=LABELS
    )

    parser.add_argument(
        "--output",
        default="artifacts/gradcam_result.png"
    )

    args = parser.parse_args()

    Path(
        args.output
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    probability = generate_gradcam(
        args.image,
        args.class_name,
        args.output
    )

    print(
        f"Class: {args.class_name}"
    )

    print(
        f"Model probability: "
        f"{probability:.4f}"
    )

    print(
        f"Grad-CAM saved to: "
        f"{args.output}"
    )
