"""
Loads CIFAR-10 (32×32) and produces:
  - 32×32 grayscale dataset
  - 64×64 grayscale dataset (upsampled)
  - 128×128 grayscale dataset (upsampled)

All outputs saved as NumPy arrays:
    data/processed/cifar_gray_32.npy
    data/processed/cifar_gray_64.npy
    data/processed/cifar_gray_128.npy

Run:
    python prepare_data.py
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

OUT_DIR = "./data/processed"
NUM_IMAGES = 200   # number of images to process


def save_dataset(arr, filename):
    """Save NumPy array and print shape."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    np.save(path, arr)
    print(f"Saved {filename} | shape = {arr.shape}")


def load_cifar10_gray(num_images):
    """Load first num_images from CIFAR-10 converted to grayscale."""
    transform = T.Compose([
        T.ToTensor(),      # (3,32,32)
        T.Grayscale(),     # -> (1,32,32)
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data/raw",
        train=False,
        download=True,
        transform=transform
    )

    imgs = []
    for i in range(num_images):
        img, _ = dataset[i]
        imgs.append(img.numpy())   # (1,32,32)

    return np.stack(imgs)          # (N,1,32,32)


def upsample_images(images, size):
    """Upsample a batch of images to (size, size) using PyTorch bilinear interpolation."""
    imgs_t = torch.tensor(images)  # (N,1,H,W)
    imgs_resized = torch.nn.functional.interpolate(
        imgs_t,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )
    return imgs_resized.numpy()    # (N,1,size,size)


def main():
    print("Loading CIFAR-10 and generating multiple resolutions...\n")

    # -------------------------------------------------------------
    # 1. Load original 32×32 grayscale
    # -------------------------------------------------------------
    cifar_32 = load_cifar10_gray(NUM_IMAGES)
    save_dataset(cifar_32, "cifar_gray_32.npy")

    # -------------------------------------------------------------
    # 2. Generate 64×64 upsampled version
    # -------------------------------------------------------------
    cifar_64 = upsample_images(cifar_32, 64)
    save_dataset(cifar_64, "cifar_gray_64.npy")

    # -------------------------------------------------------------
    # 3. Generate 128×128 upsampled version
    # -------------------------------------------------------------
    cifar_128 = upsample_images(cifar_32, 128)
    save_dataset(cifar_128, "cifar_gray_128.npy")

    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()
