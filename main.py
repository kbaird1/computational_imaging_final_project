"""
main.py

End-to-end debugging framework for the SPI → Reconstruction pipeline,
now supporting multiple image sizes:

    - 32×32   (cifar_gray_32.npy)
    - 64×64   (cifar_gray_64.npy)
    - 128×128 (cifar_gray_128.npy)

Switch modes at the bottom of the file.
"""

import os
import numpy as np

from src.forward_model import generate_hadamard_patterns
from src.reconstruction import (
    reconstruct_image_direct,
    reconstruct_image,
    linear_reconstruct_image,
)
from src.utils import (
    compute_psnr,
    compute_ssim,
    plot_reconstruction,
    plot_loss_curve,
)

# ============================================================
# Dataset Loader with Resolution Selection
# ============================================================

def load_test_image(size=32, idx=0):
    """
    Load an image of specified resolution: 32, 64, or 128.
    Files must be created by prepare_data.py.
    """
    name_map = {
        32: "cifar_gray_32.npy",
        64: "cifar_gray_64.npy",
        128: "cifar_gray_128.npy",
    }

    if size not in name_map:
        raise ValueError(f"Invalid size={size}. Choose from 32, 64, 128.")

    file_path = os.path.join("./data/processed/", name_map[size])

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"{file_path} not found. Run prepare_data.py first."
        )

    images = np.load(file_path)
    print(f"Loaded dataset: {file_path}, shape={images.shape}")

    return images[idx, 0]   # returns (H,W)


# ============================================================
# MODE 1 — Direct Pixel Optimization
# ============================================================

def run_direct_optimization_test(img):
    print("\n==========================")
    print(" MODE 1: DIRECT OPTIMIZATION RECONSTRUCTION")
    print("==========================\n")

    sampling_ratio = 0.5
    snr_db = 100.0

    recon_img, loss_history = reconstruct_image_direct(
        img,
        sampling_ratio=sampling_ratio,
        snr_db=snr_db,
        num_iters=2000,
        lr=1e-2,
        device="cpu",
        verbose=True,
    )

    psnr = compute_psnr(img, recon_img)
    ssim = compute_ssim(img, recon_img)

    print(f"\nDIRECT Reconstruction Metrics:")
    print(f"  PSNR = {psnr:.2f} dB")
    print(f"  SSIM = {ssim:.4f}")

    plot_reconstruction(img, recon_img, title="Direct Optimization Reconstruction")
    plot_loss_curve(loss_history)

    return recon_img


# ============================================================
# MODE 3 — URNet Reconstruction
# ============================================================

def run_urnet_reconstruction(img):
    print("\n==========================")
    print(" MODE 3: URNet RECONSTRUCTION")
    print("==========================\n")

    sampling_ratio = 0.50
    snr_db = 100.0

    recon_img, loss_history = reconstruct_image(
        img,
        sampling_ratio=sampling_ratio,
        snr_db=snr_db,
        num_iters=5000,
        lr=0.01,
        device="cpu",
        verbose=True,
    )

    psnr = compute_psnr(img, recon_img)
    ssim = compute_ssim(img, recon_img)

    print(f"\nURNet Reconstruction Metrics:")
    print(f"  PSNR = {psnr:.2f} dB")
    print(f"  SSIM = {ssim:.4f}")

    plot_reconstruction(img, recon_img, title="URNet Reconstruction")
    plot_loss_curve(loss_history)

    return recon_img


# ============================================================
# MODE 2 — Linear LS Reconstruction
# ============================================================

def run_linear_ls_test(img):
    print("\n==========================")
    print(" MODE 2: LINEAR LS RECONSTRUCTION")
    print("==========================\n")

    sampling_ratio = 0.50
    snr_db = 40.0
    
    recon_img = linear_reconstruct_image(
        img,
        sampling_ratio=sampling_ratio,
        snr_db=snr_db,
        l2_reg=0.0,
        verbose=True,
    )

    psnr = compute_psnr(img, recon_img)
    ssim = compute_ssim(img, recon_img)

    print(f"\n[LS] Reconstruction Metrics:")
    print(f"  PSNR = {psnr:.2f} dB")
    print(f"  SSIM = {ssim:.4f}")

    plot_reconstruction(img, recon_img, title="Linear LS Reconstruction")
    return recon_img


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------
    # Choose image resolution
    # --------------------------------------------
    IMAGE_SIZE = 32     # choose: 32, 64, or 128

    # --------------------------------------------
    # Choose which tests to run
    # --------------------------------------------
    RUN_DIRECT_OPTIMIZATION = False
    RUN_LINEAR_LS          = False
    RUN_URNET              = True

    # Load selected resolution
    img = load_test_image(size=IMAGE_SIZE, idx=0)

    if RUN_DIRECT_OPTIMIZATION:
        run_direct_optimization_test(img)

    if RUN_LINEAR_LS:
        run_linear_ls_test(img)

    if RUN_URNET:
        run_urnet_reconstruction(img)
