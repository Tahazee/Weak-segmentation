import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import download_massachusetts_dataset, get_dataloaders
from model import build_unet_model
from metrics import evaluate_model, plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train Weakly Supervised Road Segmentation U-Net")
    parser.add_argument("--image-folder", type=str, default=None, help="Path to aerial input images folder")
    parser.add_argument("--mask-folder", type=str, default=None, help="Path to ground truth masks folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--num-points", type=int, default=10, help="Number of sparse point annotations per class")
    parser.add_argument("--image-size", type=int, default=256, help="Target height and width for image resizing")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder network")
    parser.add_argument("--save-path", type=str, default=None, help="Filepath to save trained model weights")
    return parser.parse_args()


def train():
    args = parse_args()

    # Determine dataset directories
    if args.image_folder is None or args.mask_folder is None:
        image_folder, mask_folder = download_massachusetts_dataset()
    else:
        image_folder, mask_folder = args.image_folder, args.mask_folder

    # Set save path
    if args.save_path is None:
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
        os.makedirs(weights_dir, exist_ok=True)
        save_path = os.path.join(weights_dir, f"unet_{args.encoder}_pts{args.num_points}.pth")
    else:
        save_path = args.save_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training parameters: epochs={args.epochs}, num_points={args.num_points}, batch_size={args.batch_size}, lr={args.lr}")

    # Build DataLoaders
    train_loader, val_loader = get_dataloaders(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=args.batch_size,
        num_points=args.num_points,
        image_size=args.image_size,
    )

    # Build Model
    model = build_unet_model(encoder_name=args.encoder, device=device)

    # Setup Optimizer & Loss (ignores -1 unannotated pixels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, targets in train_bar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, targets in val_bar:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item()
                val_bar.set_postfix(val_loss=f"{loss.item():.4f}")

        train_epoch_loss = running_train_loss / len(train_loader)
        val_epoch_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    # Evaluate final validation IoU
    mean_val_iou = evaluate_model(model, val_loader, device)
    print(f"\nFinal Validation Mean Road IoU: {mean_val_iou:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint successfully saved to: {save_path}")


if __name__ == "__main__":
    train()
