import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import build_unet_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net Road Segmentation Inference on Image")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input aerial image")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to trained model weights (.pth)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save output prediction figure")
    parser.add_argument("--image-size", type=int, default=256, help="Target height and width for image resizing")
    return parser.parse_args()


def predict(image_path: str, weights_path: str, save_path: str = None, image_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and transform image
    raw_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Build model & load weights
    model = build_unet_model(encoder_name="resnet34", device=device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights path '{weights_path}' not found. Using initialized model weights.")

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.array(raw_image.resize((image_size, image_size))))
    axes[0].set_title("Input Aerial Image")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Predicted Road Mask")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction saved to: {save_path}")
    plt.show()

    return pred


if __name__ == "__main__":
    args = parse_args()

    if args.weights_path is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        weights_path = os.path.join(root_dir, "weights", "unet_resnet34_pts10.pth")
    else:
        weights_path = args.weights_path

    predict(
        image_path=args.image_path,
        weights_path=weights_path,
        save_path=args.save_path,
        image_size=args.image_size,
    )
