from pathlib import Path

import timm
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_PATH = MODELS_DIR / "resnet18_mnist_baseline.pt"
IMAGE_SIZE = 28


def build_model() -> nn.Module:
    # Must mirror Deployment_2/backend/main.py exactly: a timm resnet18 with
    # its first conv layer swapped for single-channel (grayscale) input.
    net = timm.create_model("resnet18", pretrained=False, num_classes=10)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    net.eval()
    return net
