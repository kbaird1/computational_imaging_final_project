"""
utils.py — updated

Now fully resolution-agnostic and robust for:
    32×32, 64×64, 128×128 images.

Includes:
  - Image normalization
  - PSNR / SSIM metrics
  - Safe shape handling (H×W, 1×H×W, H×W×1)
  - Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# ============================================================
#  Image helpers
# ============================================================

def ensure_2d(img):
    """
    Convert any of:
      (H, W)
      (1, H, W)
      (H, W, 1)
    into:
      (H, W)
    """
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 3:
        # (1,H,W) → (H,W)
        if img.shape[0] == 1:
            return img[0]
        # (H,W,1) → (H,W)
        if img.shape[2] == 1:
            return img[:, :, 0]

    if img.ndim == 2:
        return img

    raise ValueError(f"ensure_2d expected 2D or 3D image, got shape {img.shape}")


def normalize_image(img):
    """
    Normalize to [0,1] regardless of resolution or shape.
    """
    img = ensure_2d(img)

    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


# ============================================================
#  PSNR and SSIM
# ============================================================

def compute_psnr(gt, recon):
    gt = normalize_image(gt)
    recon = normalize_image(recon)
    return peak_signal_noise_ratio(gt, recon, data_range=1.0)


def compute_ssim(gt, recon):
    gt = normalize_image(gt)
    recon = normalize_image(recon)

    # Gaussian-weighted SSIM is more stable for larger images
    return structural_similarity(
        gt,
        recon,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )


def compute_metrics(gt, recon):
    """
    Convenience function returning dict {psnr, ssim}.
    """
    return {
        "psnr": compute_psnr(gt, recon),
        "ssim": compute_ssim(gt, recon)
    }


# ============================================================
#  Plotting
# ============================================================

def plot_reconstruction(gt, recon, title="URNet Reconstruction", figsize=(10,4)):
    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.imshow(normalize_image(gt), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(normalize_image(recon), cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history, log_scale=True):
    plt.figure(figsize=(7,4))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("L1 Loss")

    if log_scale:
        plt.yscale("log")

    plt.title("URNet Optimization Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def view_pattern(pattern, figsize=(3,3)):
    pattern = ensure_2d(pattern)
    plt.figure(figsize=figsize)
    plt.imshow(pattern, cmap="gray")
    plt.title("SPI Pattern")
    plt.axis("off")
    plt.show()


def plot_measurements(s, figsize=(7,3)):
    plt.figure(figsize=figsize)
    plt.plot(np.asarray(s), linewidth=1)
    plt.title("SPI Measurements")
    plt.xlabel("Pattern Index")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
