"""
Weakly Supervised Road Segmentation Package
"""

from .dataset import AerialDataset, make_sparse_mask, download_massachusetts_dataset, get_dataloaders
from .model import build_unet_model
from .metrics import compute_iou, evaluate_model, plot_sample, plot_predictions, plot_checkpoint_comparison

__all__ = [
    "AerialDataset",
    "make_sparse_mask",
    "download_massachusetts_dataset",
    "get_dataloaders",
    "build_unet_model",
    "compute_iou",
    "evaluate_model",
    "plot_sample",
    "plot_predictions",
    "plot_checkpoint_comparison",
]
