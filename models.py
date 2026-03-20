"""
models.py — Model definitions for Hessian-Guided Gradient Unlearning
Provides ResNet-18 and VGG-16 adapted for various input sizes / class counts.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─── ResNet-18 ────────────────────────────────────────────────────────────────

def build_resnet18(num_classes: int = 10, input_channels: int = 3,
                   pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 with an adjustable first conv for small images (32×32).
    For CIFAR-style data we replace the 7×7 stem with 3×3 and remove maxpool.
    """
    model = tv_models.resnet18(weights=None)

    # Adapt stem for small images
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                             stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Adapt final FC
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ─── VGG-16 ───────────────────────────────────────────────────────────────────

def build_vgg16(num_classes: int = 10, input_channels: int = 3,
                pretrained: bool = False) -> nn.Module:
    """
    VGG-16 with an adjustable classifier head.
    For 32×32 inputs the adaptive pool ensures the FC layers still work.
    """
    model = tv_models.vgg16(weights=None)

    # Handle non-RGB inputs
    if input_channels != 3:
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

    model.avgpool = nn.AdaptiveAvgPool2d((4, 4))

    # Rebuild classifier for potentially small spatial maps
    model.classifier = nn.Sequential(
        nn.Linear(512 * 4 * 4, 1024),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_model(name: str, num_classes: int = 10,
              input_channels: int = 3) -> nn.Module:
    """Return the requested model architecture."""
    name = name.lower()
    if name == "resnet18":
        return build_resnet18(num_classes, input_channels)
    elif name in ("vgg16", "vgg"):
        return build_vgg16(num_classes, input_channels)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'resnet18' or 'vgg16'.")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, path: str,
                    extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str,
                    device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    print(f"  ✓ Checkpoint loaded ← {path}")
    return payload
