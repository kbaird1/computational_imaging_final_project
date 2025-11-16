"""
Supports URNet reconstruction at:
    - 32×32
    - 64×64
    - 128×128

Implements the optimization described in:
  Li et al., "URNet: High-quality single-pixel imaging with untrained
  reconstruction network" (Optics & Laser Engineering, 2023)

Training configuration:
  - Loss: L1( s_pred, s_true )
  - Optimizer: Adam(lr = 0.08)
  - LR decay: multiply by 0.8 every 3000 iters
  - Initialization: Xavier uniform (handled in URNet class)

Pattern selection:
  - Hadamard (Walsh-ordered) basis of size N×N
  - Sampling ratio r → K = floor(r * M)
  - Use FIRST K patterns (lowest K frequencies)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .forward_model import (
    generate_hadamard_patterns,
    measure_image,
    add_gaussian_noise,
)

from .urnet_model import URNet


# -----------------------------------------------------------
# Deterministic subsampling helper
# -----------------------------------------------------------

def deterministic_subsample_measurements(patterns, measurements, sampling_ratio: float):
    """
    Deterministically subsample by taking FIRST K Hadamard patterns.
    (Assumes generate_hadamard_patterns gives low-frequency-first order.)
    """
    M = patterns.shape[0]
    K = int(np.floor(M * sampling_ratio))
    K = max(1, min(K, M))

    indices = np.arange(K, dtype=np.int64)
    return patterns[indices], measurements[indices], indices


# -----------------------------------------------------------
# Torch measurement operator
# -----------------------------------------------------------

def torch_measurement_operator(img, patterns):
    """
    img: (B,1,N,N)
    patterns: (K,N,N)
    Returns s_pred : (B,K)
    """
    p = patterns.unsqueeze(1)     # (K,1,N,N)
    meas = (p * img).sum(dim=[2, 3])  # (K,1)
    return meas.permute(1, 0)    # (B,K)


# -----------------------------------------------------------
# URNet reconstruction
# -----------------------------------------------------------

def reconstruct_image(
    image: np.ndarray,
    sampling_ratio: float = 0.1,
    snr_db: float = 40.0,
    num_iters: int = 5000,
    lr: float = 0.08,
    lr_decay_every: int = 2000,
    lr_decay_factor: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
    #URNet model params
    base_channels: int = 512,
    min_channels: int = 128,
    num_refine_convs: int = 4,
    
):

    # -------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------
    N = image.shape[0]
    assert N in {32, 64, 128}, f"URNet supports 32/64/128 (got {N})"

    patterns = generate_hadamard_patterns(N)    # (M,N,N)
    M = patterns.shape[0]

    s_clean = measure_image(image, patterns)    # (M,)
    s_noisy = add_gaussian_noise(s_clean, snr_db)

    # Deterministic first-K sampling
    patterns_sub, s_sub, _ = deterministic_subsample_measurements(
        patterns, s_noisy, sampling_ratio
    )

    K = patterns_sub.shape[0]
    if verbose:
        print(f"Using first {K}/{M} patterns ({sampling_ratio*100:.1f}%)")
        print(f"SNR = {snr_db} dB")

    device = torch.device(device)
    patterns_t = torch.tensor(patterns_sub, dtype=torch.float32, device=device)
    s_true_t  = torch.tensor(s_sub, dtype=torch.float32, device=device).unsqueeze(0)  # (1,K)

    # -------------------------------------------------------
    # 2. Construct URNet
    # -------------------------------------------------------
    model = URNet(
        measurement_dim=K,
        image_size=N,
        base_channels=base_channels,
        min_channels=min_channels,
        num_refine_convs=num_refine_convs,
    ).to(device)

    # -------------------------------------------------------
    # 3. Optimization
    # -------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    loss_history = []

    for it in range(1, num_iters + 1):

        optimizer.zero_grad()

        # Forward through URNet
        x_pred = model(s_true_t)                     # (1,1,N,N)
        s_pred = torch_measurement_operator(x_pred, patterns_t)  # (1,K)

        # L1 loss (no normalization)
        loss = loss_fn(s_pred, s_true_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # LR decay
        if it == lr_decay_every:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay_factor
            if verbose:
                print(f"[iter {it}] LR decayed to {g['lr']:.5f}")

        if verbose and (it % 500 == 0 or it == 1):
            print(f"Iter {it:5d} | Loss = {loss.item():.6f}")

    # -------------------------------------------------------
    # 4. Final output
    # -------------------------------------------------------
    final_img = x_pred.detach().cpu().numpy()[0, 0]
    final_img = np.clip(final_img, 0, 1)
    return final_img, loss_history


# -----------------------------------------------------------
# Direct pixel optimization (baseline)
# -----------------------------------------------------------

def reconstruct_image_direct(
    image: np.ndarray,
    sampling_ratio: float = 0.1,
    snr_db: float = 40.0,
    num_iters: int = 2000,
    lr: float = 1e-2,
    device: str = "cpu",
    verbose: bool = True,
):

    N = image.shape[0]
    device = torch.device(device)

    patterns = generate_hadamard_patterns(N)
    s_clean = measure_image(image, patterns)
    s_noisy = add_gaussian_noise(s_clean, snr_db)

    patterns_sub, s_sub, _ = deterministic_subsample_measurements(
        patterns, s_noisy, sampling_ratio
    )
    K = patterns_sub.shape[0]

    if verbose:
        print(f"[DIRECT] Using {K}/{N*N} patterns, SNR={snr_db} dB")

    patterns_t = torch.tensor(patterns_sub, dtype=torch.float32, device=device)
    s_true_t   = torch.tensor(s_sub, dtype=torch.float32, device=device).unsqueeze(0)

    # Start with random reconstruction
    recon = torch.rand(1, 1, N, N, device=device, requires_grad=True)

    opt = torch.optim.Adam([recon], lr=lr)
    loss_fn = nn.L1Loss()
    losses = []

    for it in range(1, num_iters + 1):

        opt.zero_grad()
        s_pred = torch_measurement_operator(recon, patterns_t)
        loss = loss_fn(s_pred, s_true_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if verbose and (it % 500 == 0 or it == 1):
            print(f"[DIRECT] iter {it:4d} | loss = {loss.item():.6f}")

    final_img = recon.detach().cpu().numpy()[0, 0]
    final_img = np.clip(final_img, 0, 1)
    return final_img, losses


# -----------------------------------------------------------
# Least squares reconstruction (baseline)
# -----------------------------------------------------------

def linear_reconstruct_image(
    image: np.ndarray,
    sampling_ratio: float = 1.0,
    snr_db: float = 40.0,
    l2_reg: float = 0.0,
    verbose: bool = True,
):

    N = image.shape[0]
    M = N * N

    patterns = generate_hadamard_patterns(N)
    s_clean = measure_image(image, patterns)
    s_noisy = add_gaussian_noise(s_clean, snr_db)

    patterns_sub, s_sub, _ = deterministic_subsample_measurements(
        patterns, s_noisy, sampling_ratio
    )
    K = patterns_sub.shape[0]

    if verbose:
        print(f"[LS] Using first {K}/{M} patterns ({100*sampling_ratio:.1f}%), SNR={snr_db}")

    A = patterns_sub.reshape(K, -1).astype(np.float32)
    b = s_sub.astype(np.float32)

    AtA = A.T @ A
    Atb = A.T @ b

    if l2_reg > 0:
        AtA += l2_reg * np.eye(M, dtype=np.float32)

    x_hat, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)

    final = x_hat.reshape(N, N)
    return np.clip(final, 0, 1)
