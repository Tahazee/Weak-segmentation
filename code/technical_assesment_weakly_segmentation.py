# -*- coding: utf-8 -*-
"""
TECHNICAL ASSESSMENT: WEAKLY SUPERVISED ROAD SEGMENTATION

Converted and modularized pipeline for weakly supervised aerial road segmentation
using U-Net architecture with a ResNet34 backbone.
"""

import os
import sys

# Ensure local module directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import modular components
from dataset import download_massachusetts_dataset, get_dataloaders
from model import build_unet_model
from metrics import compute_iou, evaluate_model, plot_predictions, plot_checkpoint_comparison


def main():
    # Set directories
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(root_dir, "weights")
    assets_dir = os.path.join(root_dir, "assets")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Weakly Supervised Segmentation Pipeline on device: {device}")

    # 1. Download / acquire dataset
    image_folder, mask_folder = download_massachusetts_dataset()

    # 2. Setup DataLoaders (10 points weak supervision)
    train_loader, val_loader = get_dataloaders(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=8,
        num_points=10,
        image_size=256,
        num_workers=2,
    )

    print(f"Train Dataset Batches: {len(train_loader)} | Val Dataset Batches: {len(val_loader)}")

    # 3. Instantiate U-Net model with ResNet-34 encoder
    model = build_unet_model(encoder_name="resnet34", device=device)

    # 4. Training loop configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    epochs = 10

    print(f"\n--- Starting Model Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, targets in train_bar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 5. Evaluate final validation IoU metric
    mean_val_iou = evaluate_model(model, val_loader, device)
    print(f"\nFinal Validation Mean Road IoU: {mean_val_iou:.4f}")

    # 6. Save trained model checkpoint
    checkpoint_path = os.path.join(weights_dir, "unet_resnet34_pts10.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # 7. Generate qualitative prediction plot
    val_iter = iter(val_loader)
    images, targets = next(val_iter)
    images_dev = images.to(device)
    with torch.no_grad():
        outputs = model(images_dev)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    comparison_fig_path = os.path.join(assets_dir, "road_segmentation_comparison10pts.png")
    plot_predictions(images, targets, preds, save_path=comparison_fig_path, num_samples=3)

    # 8. Evaluate all checkpoints if available
    checkpoints = {
        "5 Points": os.path.join(weights_dir, "unet_resnet34_pts5.pth"),
        "10 Points": os.path.join(weights_dir, "unet_resnet34_pts10.pth"),
        "50 Points": os.path.join(weights_dir, "unet_resnet34_pts50.pth"),
    }
    bar_chart_path = os.path.join(assets_dir, "checkpoint_ious.png")
    plot_checkpoint_comparison(checkpoints, model, val_loader, device, save_path=bar_chart_path)


if __name__ == "__main__":
    main()