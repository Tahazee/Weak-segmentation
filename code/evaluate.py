import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import download_massachusetts_dataset, get_dataloaders
from model import build_unet_model
from metrics import evaluate_model, plot_predictions, plot_checkpoint_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Weakly Supervised U-Net Checkpoints")
    parser.add_argument("--image-folder", type=str, default=None, help="Path to aerial images folder")
    parser.add_argument("--mask-folder", type=str, default=None, help="Path to ground truth masks folder")
    parser.add_argument("--weights-dir", type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--assets-dir", type=str, default=None, help="Directory to output result figures")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for validation loader")
    parser.add_argument("--image-size", type=int, default=256, help="Target height and width for image resizing")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Determine directories
    weights_dir = args.weights_dir if args.weights_dir else os.path.join(root_dir, "weights")
    assets_dir = args.assets_dir if args.assets_dir else os.path.join(root_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    if args.image_folder is None or args.mask_folder is None:
        image_folder, mask_folder = download_massachusetts_dataset()
    else:
        image_folder, mask_folder = args.image_folder, args.mask_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating checkpoints using device: {device}")

    # Build validation dataloader
    _, val_loader = get_dataloaders(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    model = build_unet_model(encoder_name="resnet34", device=device)

    # Map available checkpoints
    checkpoints = {
        "5 Points": os.path.join(weights_dir, "unet_resnet34_pts5.pth"),
        "10 Points": os.path.join(weights_dir, "unet_resnet34_pts10.pth"),
        "50 Points": os.path.join(weights_dir, "unet_resnet34_pts50.pth"),
    }

    # Evaluate IoU across checkpoints
    bar_chart_path = os.path.join(assets_dir, "checkpoint_ious.png")
    results = plot_checkpoint_comparison(checkpoints, model, val_loader, device, save_path=bar_chart_path)

    # Qualitative comparison for 10-point model (or default)
    default_ckpt = checkpoints["10 Points"]
    if os.path.exists(default_ckpt):
        model.load_state_dict(torch.load(default_ckpt, map_location=device))
        model.eval()

        val_iter = iter(val_loader)
        images, targets = next(val_iter)
        images_dev = images.to(device)

        with torch.no_grad():
            outputs = model(images_dev)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        comparison_path = os.path.join(assets_dir, "road_segmentation_comparison10pts.png")
        plot_predictions(images, targets, preds, save_path=comparison_path, num_samples=3)

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
