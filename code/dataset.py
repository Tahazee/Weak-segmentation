import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def download_massachusetts_dataset(dataset_handle="insaff/massachusetts-roads-dataset"):
    """
    Downloads the Massachusetts Roads Dataset using kagglehub and returns
    the paths to the image and mask training folders.
    """
    import kagglehub

    print(f"Downloading dataset '{dataset_handle}' via kagglehub...")
    path = kagglehub.dataset_download(dataset_handle)

    base_train_path = os.path.join(path, "road_segmentation_ideal", "training")
    image_folder = os.path.join(base_train_path, "input")
    mask_folder = os.path.join(base_train_path, "output")

    return image_folder, mask_folder


def make_sparse_mask(mask: torch.Tensor, num_points_per_class: int = 5) -> torch.Tensor:
    """
    Converts a dense 2D target mask into a sparse point-supervised mask.
    Unannotated pixels are set to -1 (to be ignored during loss computation).

    Args:
        mask (torch.Tensor): 2D tensor ground truth mask (0 for background, 1 for road).
        num_points_per_class (int): Number of point labels to randomly sample per class.

    Returns:
        torch.Tensor: Sparse target mask with values in {-1, 0, 1}.
    """
    sparse_mask = torch.full_like(mask, fill_value=-1)

    for class_id in [0, 1]:
        coords = (mask == class_id).nonzero(as_tuple=False)

        if len(coords) > 0:
            k = min(len(coords), num_points_per_class)
            rand_indices = torch.randperm(len(coords))[:k]
            chosen_coords = coords[rand_indices]

            sparse_mask[chosen_coords[:, 0], chosen_coords[:, 1]] = class_id

    return sparse_mask


class AerialDataset(Dataset):
    """
    PyTorch Dataset for Aerial Road Segmentation with optional weak (point-level) supervision.
    """

    def __init__(
        self,
        image_folder: str,
        mask_folder: str,
        split: str = "train",
        num_points: int = 5,
        image_size: int = 256,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            image_folder (str): Directory path containing input aerial images.
            mask_folder (str): Directory path containing corresponding mask images.
            split (str): 'train' or 'val'.
            num_points (int): Number of point labels per class for weak supervision (train set only).
            image_size (int): Target width and height for resizing.
            val_split (float): Ratio of total images to use for validation split.
            seed (int): Random seed for reproducible splitting.
        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.split = split
        self.num_points = num_points
        self.image_size = image_size

        # Match shared image and mask files
        input_files = set(os.listdir(image_folder))
        output_files = set(os.listdir(mask_folder))
        common_files = sorted(list(input_files.intersection(output_files)))

        # Split train / val deterministically
        train_samples = int((1.0 - val_split) * len(common_files))

        if self.split == "train":
            self.images = common_files[:train_samples]
        else:
            self.images = common_files[train_samples:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        image_path = os.path.join(self.image_folder, filename)
        mask_path = os.path.join(self.mask_folder, filename)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        mask_transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
        ])

        image = image_transform(image)
        mask = mask_transform(mask)

        import numpy as np
        dense_mask = (torch.from_numpy(np.array(mask)) > 0).long()

        # Apply weak point-supervision sparsification during training; keep dense for evaluation
        if self.split == "train":
            target_mask = make_sparse_mask(dense_mask, num_points_per_class=self.num_points)
        else:
            target_mask = dense_mask

        return image, target_mask


def get_dataloaders(
    image_folder: str,
    mask_folder: str,
    batch_size: int = 8,
    num_points: int = 5,
    image_size: int = 256,
    num_workers: int = 2,
    val_split: float = 0.2,
):
    """
    Creates and returns train and validation DataLoaders.
    """
    train_dataset = AerialDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        split="train",
        num_points=num_points,
        image_size=image_size,
        val_split=val_split,
    )

    val_dataset = AerialDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        split="val",
        num_points=num_points,
        image_size=image_size,
        val_split=val_split,
    )

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader
