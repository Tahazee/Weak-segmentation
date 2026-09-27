import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, class_id: int = 1, smooth: float = 1e-6) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index) for a specific binary target class.

    Args:
        preds (torch.Tensor): Predicted class label tensor (same shape as targets).
        targets (torch.Tensor): Target ground truth mask tensor.
        class_id (int): Target class identifier (default: 1 for road).
        smooth (float): Smoothing denominator constant to avoid zero division.

    Returns:
        float: Computed IoU score.
    """
    pred_mask = (preds == class_id)
    true_mask = (targets == class_id)

    intersection = (pred_mask & true_mask).float().sum()
    union = (pred_mask | true_mask).float().sum()

    iou = ((intersection + smooth) / (union + smooth)).item()
    return iou


def evaluate_model(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on the full validation DataLoader and returns mean IoU.
    """
    model.eval()
    total_iou = 0.0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            total_iou += compute_iou(preds, targets, class_id=1)

    mean_iou = total_iou / len(val_loader)
    return mean_iou


def plot_sample(image: torch.Tensor, mask: torch.Tensor, save_path: Optional[str] = None):
    """
    Plots an input image and its corresponding mask side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask (Ground Truth / Sparse)")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    preds: np.ndarray,
    save_path: Optional[str] = None,
    num_samples: int = 3,
):
    """
    Plots a side-by-side grid of Aerial Input, True Ground Truth, and Model Predictions.
    """
    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    titles = ["Aerial Input", "True Ground Truth", "Model Prediction"]

    for i in range(num_samples):
        axes[i, 0].imshow(np.clip(images[i], 0, 1))
        axes[i, 1].imshow(targets[i], cmap="gray")
        axes[i, 2].imshow(preds[i], cmap="gray")

        for j in range(3):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visual comparison plot saved to: {save_path}")
    plt.show()


def plot_checkpoint_comparison(
    checkpoints: Dict[str, str],
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluates multiple saved checkpoints and plots a bar chart comparing IoU metrics.
    """
    results = {}

    for name, path in checkpoints.items():
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            mean_iou = evaluate_model(model, val_loader, device)
            results[name] = mean_iou
            print(f"Checkpoint '{name}' ({path}) -> Mean IoU: {mean_iou:.4f}")
        except Exception as e:
            print(f"Could not load checkpoint '{name}' from {path}: {e}")

    if not results:
        return {}

    labels = list(results.keys())
    ious = list(results.values())

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, ious, color=["#d9534f", "#f0ad4e", "#2e6da4"][: len(labels)])
    plt.ylabel("Jaccard Index (IoU)")
    plt.title("Weak Supervision Checkpoint Evaluation")
    plt.ylim(0, 1.0)

    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.02, f"{y:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Checkpoint comparison bar chart saved to: {save_path}")
    plt.show()

    return results
