"""
URNet experiments for 32x32 SPI reconstruction.

Modes:
  --mode hyperparam
      Run hyperparameter search on 5 images.
      Output directory: results/hyperparam_test_<timestamp>/

  --mode analysis --config path/to/best_config.json
      Run sampling-ratio and noise-robustness sweeps
      on the SAME 5 random images.
      Output directory: results/analysis_<timestamp>/
      For each of 5 images:
          image_00_sampling.png
          image_00_noise.png
"""

import os
import json
import argparse
import itertools
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import torch

from src.reconstruction import reconstruct_image


# ============================================================
# Global constants
# ============================================================

DATA_PATH = "./data/processed/cifar_gray_32.npy"
ROOT_RESULTS_DIR = "./results"
os.makedirs(ROOT_RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_dataset(num_images=5, seed=0):
    arr = np.load(DATA_PATH)
    if arr.ndim == 4:
        arr = arr[:, 0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=num_images, replace=False)
    imgs = arr[idx].astype(np.float32)

    if imgs.max() > 1.0:
        imgs /= 255.0

    return imgs


def compute_metrics(gt, rec):
    gt = np.clip(gt, 0, 1)
    rec = np.clip(rec, 0, 1)
    psnr = psnr_metric(gt, rec, data_range=1.0)
    ssim = ssim_metric(gt, rec, data_range=1.0)
    return float(psnr), float(ssim)


# ============================================================
# Hyperparameter search
# ============================================================

def generate_hparam_grid():
    """
    Define the hyperparameter search grid.
    FULL GRID SEARCH.
    """
    grid = {
        "base_channels": [128, 256, 512],
        "min_channels": [32, 64],
        "num_refine_convs": [2, 4],
        "lr": [0.01 ,0.05],
        "num_iters": [3000, 6000],
    }
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run_single_config(imgs, cfg, sampling_ratio=0.5, snr_db=100.0):
    """
    Evaluate one hyperparameter config.
    Includes tqdm for the 5-image sweep.
    """
    psnrs, ssims = [], []

    for img in tqdm(imgs, desc="Images", leave=False):
        rec, _ = reconstruct_image(
            img,
            sampling_ratio=sampling_ratio,
            snr_db=snr_db,
            num_iters=cfg["num_iters"],
            lr=cfg["lr"],
            lr_decay_every=int(0.4 * cfg["num_iters"]),
            lr_decay_factor=0.5,
            base_channels=cfg["base_channels"],
            min_channels=cfg["min_channels"],
            num_refine_convs=cfg["num_refine_convs"],
            device=DEVICE,
            verbose=False,
        )
        p, s = compute_metrics(img, rec)
        psnrs.append(p)
        ssims.append(s)

    return (
        float(np.mean(psnrs)),
        float(np.std(psnrs)),
        float(np.mean(ssims)),
        float(np.std(ssims)),
    )


def hyperparam_search():
    ts = timestamp_str()
    out_dir = os.path.join(ROOT_RESULTS_DIR, f"hyperparam_test_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Hyperparameter Search ===")
    print(f"Saving results to: {out_dir}")

    imgs = load_dataset(num_images=5, seed=0)

    sampling_ratio = 0.5
    snr_db = 100.0

    results = []
    hparam_list = list(generate_hparam_grid())

    print(f"\nTotal configs: {len(hparam_list)}\n")

    for cfg in tqdm(hparam_list, desc="Hyperparameter configs"):
        avg_psnr, std_psnr, avg_ssim, std_ssim = run_single_config(
            imgs, cfg, sampling_ratio, snr_db
        )

        results.append({
            "config": cfg,
            "avg_psnr": avg_psnr,
            "std_psnr": std_psnr,
            "avg_ssim": avg_ssim,
            "std_ssim": std_ssim,
        })

    # Sort by best SSIM
    results_sorted = sorted(results, key=lambda x: x["avg_ssim"], reverse=True)
    best_cfg = results_sorted[0]["config"]

    # Save JSON
    with open(os.path.join(out_dir, "hyperparam_results.json"), "w") as f:
        json.dump(results_sorted, f, indent=2)

    # Save best config
    with open(os.path.join(out_dir, "best_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    # CSV export
    csv_path = os.path.join(out_dir, "hyperparam_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "base_channels", "min_channels", "num_refine_convs",
            "lr", "num_iters",
            "avg_psnr", "std_psnr", "avg_ssim", "std_ssim"
        ])
        for e in results_sorted:
            c = e["config"]
            writer.writerow([
                c["base_channels"], c["min_channels"], c["num_refine_convs"],
                c["lr"], c["num_iters"],
                e["avg_psnr"], e["std_psnr"], e["avg_ssim"], e["std_ssim"],
            ])

    # LaTeX table export
    tex_path = os.path.join(out_dir, "hyperparam_results.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{URNet Hyperparameter Search Results (Sorted by SSIM)}\n")
        f.write("\\begin{tabular}{ccccccccc}\n")
        f.write("\\hline\n")
        f.write("base & min & refine & lr & iters & PSNR & $\\sigma$PSNR & SSIM & $\\sigma$SSIM \\\\ \\hline\n")
        for e in results_sorted:
            c = e["config"]
            f.write(
                f"{c['base_channels']} & {c['min_channels']} & {c['num_refine_convs']} & "
                f"{c['lr']} & {c['num_iters']} & "
                f"{e['avg_psnr']:.2f} & {e['std_psnr']:.2f} & "
                f"{e['avg_ssim']:.3f} & {e['std_ssim']:.3f} \\\\ \n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("\n=== Hyperparameter Search Complete ===")
    print(f"Best config: {best_cfg}")
    print(f"CSV saved to: {csv_path}")
    print(f"TEX table saved to: {tex_path}")

    return best_cfg, out_dir


# ============================================================
# Plotting helpers for analysis mode
# ============================================================

def plot_sampling_panel_for_image(gt_img, recon_list, sampling_ratios, metric_list, save_path):
    K = len(sampling_ratios)

    fig, axes = plt.subplots(
        3,                # 3 rows: image, PSNR row, SSIM row
        K + 1,            # Number of columns (GT + K reconstructions)
        figsize=(3.2 * (K + 1), 5.0),
        gridspec_kw={"height_ratios": [5, 0.8, 0.8]}
    )

    # ----- Row 0: images -----
    axes[0, 0].imshow(gt_img, cmap="gray")
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")

    for j in range(K):
        ax = axes[0, j + 1]
        ax.imshow(recon_list[j], cmap="gray")
        r = sampling_ratios[j]
        ax.set_title(f"{int(100*r)}%")
        ax.axis("off")

    # ----- Row 1: PSNR scores -----
    axes[1, 0].text(0.5, 0.5, "", ha="center", va="center")
    axes[1, 0].axis("off")

    for j in range(K):
        psnr, _ = metric_list[j]
        axes[1, j + 1].text(
            0.5, 0.5, f"PSNR = {psnr:.2f}",
            ha="center", va="center", fontsize=10
        )
        axes[1, j + 1].axis("off")

    # ----- Row 2: SSIM scores -----
    axes[2, 0].text(0.5, 0.5, "", ha="center", va="center")
    axes[2, 0].axis("off")

    for j in range(K):
        _, ssim = metric_list[j]
        axes[2, j + 1].text(
            0.5, 0.5, f"SSIM = {ssim:.3f}",
            ha="center", va="center", fontsize=10
        )
        axes[2, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_noise_panel_for_image(gt_img, recon_list, snr_list, metric_list, save_path):
    K = len(snr_list)

    fig, axes = plt.subplots(
        3,            # 3 rows: image, PSNR row, SSIM row
        K + 1,
        figsize=(3.2 * (K + 1), 5.0),
        gridspec_kw={"height_ratios": [5, 0.8, 0.8]}
    )

    # ----- Row 0: images -----
    axes[0, 0].imshow(gt_img, cmap="gray")
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")

    for j in range(K):
        ax = axes[0, j + 1]
        ax.imshow(recon_list[j], cmap="gray")
        ax.set_title(f"{snr_list[j]} dB")
        ax.axis("off")

    # ----- Row 1: PSNR scores -----
    axes[1, 0].text(0.5, 0.5, "", ha="center", va="center")
    axes[1, 0].axis("off")

    for j in range(K):
        psnr, _ = metric_list[j]
        axes[1, j + 1].text(
            0.5, 0.5, f"PSNR = {psnr:.2f}",
            ha="center", va="center", fontsize=10
        )
        axes[1, j + 1].axis("off")

    # ----- Row 2: SSIM scores -----
    axes[2, 0].text(0.5, 0.5, "", ha="center", va="center")
    axes[2, 0].axis("off")

    for j in range(K):
        _, ssim = metric_list[j]
        axes[2, j + 1].text(
            0.5, 0.5, f"SSIM = {ssim:.3f}",
            ha="center", va="center", fontsize=10
        )
        axes[2, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



# ============================================================
# Analysis mode
# ============================================================

def analysis_mode(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ts = timestamp_str()
    out_dir = os.path.join(ROOT_RESULTS_DIR, f"analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n=== Analysis Mode ===")
    print(f"Using config: {cfg}")
    print(f"Results will be saved to: {out_dir}")

    imgs = load_dataset(num_images=5, seed=1)

    sampling_ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    snr_list = [5, 10, 20, 40, 60, 80, 100]
    fixed_sampling_for_noise = 0.75

    # tqdm for image loop
    for idx, img in enumerate(tqdm(imgs, desc="Images")):

        # --- Sampling sweep ---
        sampling_recons = []
        sampling_metrics = []

        for r in tqdm(sampling_ratios, desc=f"Sampling sweep (img {idx})", leave=False):
            rec, _ = reconstruct_image(
                img,
                sampling_ratio=r,
                snr_db=100.0,
                num_iters=cfg["num_iters"],
                lr=cfg["lr"],
                lr_decay_every=int(0.4 * cfg["num_iters"]),
                lr_decay_factor=0.5,
                base_channels=cfg["base_channels"],
                min_channels=cfg["min_channels"],
                num_refine_convs=cfg["num_refine_convs"],
                device=DEVICE,
                verbose=False,
            )
            p, s = compute_metrics(img, rec)
            sampling_recons.append(rec)
            sampling_metrics.append((p, s))

        samp_path = os.path.join(out_dir, f"image_{idx:02d}_sampling.png")
        plot_sampling_panel_for_image(
            img, sampling_recons, sampling_ratios, sampling_metrics, samp_path
        )

        # --- Noise sweep ---
        noise_recons = []
        noise_metrics = []

        for snr in tqdm(snr_list, desc=f"Noise sweep (img {idx})", leave=False):
            rec, _ = reconstruct_image(
                img,
                sampling_ratio=fixed_sampling_for_noise,
                snr_db=snr,
                num_iters=cfg["num_iters"],
                lr=cfg["lr"],
                lr_decay_every=int(0.4 * cfg["num_iters"]),
                lr_decay_factor=0.5,
                base_channels=cfg["base_channels"],
                min_channels=cfg["min_channels"],
                num_refine_convs=cfg["num_refine_convs"],
                device=DEVICE,
                verbose=False,
            )
            p, s = compute_metrics(img, rec)
            noise_recons.append(rec)
            noise_metrics.append((p, s))

        noise_path = os.path.join(out_dir, f"image_{idx:02d}_noise.png")
        plot_noise_panel_for_image(
            img, noise_recons, snr_list, noise_metrics, noise_path
        )

    print("\n=== Analysis Complete ===")
    print(f"All results saved to: {out_dir}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["hyperparam", "analysis"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "hyperparam":
        hyperparam_search()

    elif args.mode == "analysis":
        if args.config is None:
            raise ValueError("--config required for analysis mode")
        analysis_mode(args.config)
