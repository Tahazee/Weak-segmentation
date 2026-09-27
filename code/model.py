import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 2,
    device: torch.device = None,
) -> nn.Module:
    """
    Constructs a U-Net architecture with a specified encoder backbone.

    Args:
        encoder_name (str): Backbone encoder name (e.g. 'resnet34', 'resnet18').
        encoder_weights (str): Pretrained weights source ('imagenet' or None).
        in_channels (int): Input image color channels (3 for RGB).
        classes (int): Number of target output segmentation classes.
        device (torch.device): Device placement (CPU or CUDA).

    Returns:
        nn.Module: Instantiated PyTorch U-Net model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    ).to(device)

    return model
