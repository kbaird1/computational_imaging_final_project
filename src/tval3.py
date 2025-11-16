"""
tval3.py

TVAL3-style baseline for compressed sensing SPI reconstruction.

This is a Python implementation of a TV-regularized least-squares
inverse problem, inspired by the TVAL3 algorithm:

    min_x  0.5 * || A x - b ||_2^2  +  lambda_tv * || D x ||_1

where:
  - A is the sensing matrix (flattened SPI patterns)
  - b is the measurement vector
  - D is a discrete gradient operator (finite differences)

We solve this using ADMM. This is NOT the official TVAL3 code,
but a small, self-contained TV-CS solver you can use as a baseline
against URNet in your experiments.
"""

import numpy as np

from .forward_model import (
    generate_hadamard_patterns,
    measure_image,
    add_gaussian_noise,
    subsample_measurements,
)


# ============================================================
#  Gradient Operator D (finite differences)
# ============================================================

def build_gradient_matrix(N: int) -> np.ndarray:
    """
    Build a discrete gradient operator D for an N x N image.

    D maps x (flattened image, length P = N^2)
    to stacked [Dx_horiz; Dx_vert] of length 2P.

    Using forward differences with Neumann-like boundaries
    (last row/column effectively has zero gradient).

    Returns
    -------
    D : np.ndarray
        Shape (2P, P), where P = N*N.
    """
    P = N * N
    D = np.zeros((2 * P, P), dtype=np.float32)

    row = 0
    for i in range(N):
        for j in range(N):
            k = i * N + j  # index of (i,j) in flattened x

            # Horizontal gradient: x[i, j+1] - x[i, j]
            if j < N - 1:
                k_right = i * N + (j + 1)
                D[row, k_right] = 1.0
                D[row, k]      = -1.0
            # else gradient is zero (row stays all zeros)
            row += 1

            # Vertical gradient: x[i+1, j] - x[i, j]
            if i < N - 1:
                k_down = (i + 1) * N + j
                D[row, k_down] = 1.0
                D[row, k]      = -1.0
            # else gradient is zero
            row += 1

    return D


# ============================================================
#  Core ADMM solver for TV-regularized CS
# ============================================================

def tv_cs_admm(
    A: np.ndarray,
    b: np.ndarray,
    N: int,
    lambda_tv: float = 0.1,
    rho: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
):
    """
    Solve: 0.5 * ||A x - b||_2^2 + lambda_tv * ||D x||_1

    using ADMM splitting with z = D x.

    Parameters
    ----------
    A : np.ndarray
        Sensing matrix, shape (M, P) with P = N*N.
    b : np.ndarray
        Measurements, shape (M,).
    N : int
        Image size (N x N).
    lambda_tv : float
        TV regularization weight.
    rho : float
        ADMM penalty parameter.
    max_iters : int
        Maximum number of ADMM iterations.
    tol : float
        Convergence tolerance (on primal residual).
    verbose : bool
        Print progress.

    Returns
    -------
    x_img : np.ndarray
        Reconstructed image, shape (N, N).
    """
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    M, P = A.shape
    assert P == N * N, "A shape mismatch with N"

    # Build gradient operator D
    D = build_gradient_matrix(N)          # (2P, P)
    Dt = D.T                              # (P, 2P)

    # Precompute matrices for x-update
    AtA = A.T @ A                         # (P, P)
    DtD = Dt @ D                          # (P, P)

    # System matrix: H = A^T A + rho D^T D
    H = AtA + rho * DtD + 1e-6 * np.eye(P, dtype=np.float32)  # small reg

    # Factorize H once (Cholesky)
    L = np.linalg.cholesky(H)

    def solve_H(rhs):
        # Solve H x = rhs using the Cholesky factorization
        y = np.linalg.solve(L, rhs)
        return np.linalg.solve(L.T, y)

    # Initialize variables
    x = np.zeros(P, dtype=np.float32)
    z = np.zeros(2 * P, dtype=np.float32)
    u = np.zeros(2 * P, dtype=np.float32)

    Atb = A.T @ b

    for it in range(1, max_iters + 1):
        # x-update: solve (AtA + rho DtD) x = Atb + rho Dt(z - u)
        rhs = Atb + rho * Dt @ (z - u)
        x = solve_H(rhs)

        # z-update: soft-thresholding on Dx + u
        Dx = D @ x
        v = Dx + u

        # Soft-thresholding with threshold = lambda_tv / rho
        thresh = lambda_tv / rho
        z = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)

        # u-update (dual variable)
        u = u + Dx - z

        # Convergence check (primal residual)
        r_norm = np.linalg.norm(Dx - z)
        if verbose and it % 50 == 0:
            print(f"[TV-CS ADMM] iter {it:4d} | r_norm = {r_norm:.4e}")

        if r_norm < tol:
            if verbose:
                print(f"[TV-CS ADMM] Converged at iter {it}, r_norm = {r_norm:.4e}")
            break

    x_img = x.reshape(N, N)
    return x_img


# ============================================================
#  High-level function: TV baseline for SPI (like TVAL3)
# ============================================================

def tval3_reconstruct_image(
    image: np.ndarray,          # (N, N)
    sampling_ratio: float = 0.1,
    snr_db: float = 40.0,
    lambda_tv: float = 0.1,
    rho: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
):
    """
    TVAL3-style baseline reconstruction for SPI.

    This mirrors the setup of your URNet reconstruction, but uses
    a convex TV-regularized solver instead of a neural network.

    Steps:
      1. Generate Hadamard patterns.
      2. Measure the ground-truth image.
      3. Add Gaussian noise with given SNR.
      4. Subsample patterns & measurements according to sampling_ratio.
      5. Build sensing matrix A from patterns.
      6. Run TV-regularized CS solver (ADMM).
    
    Parameters
    ----------
    image : np.ndarray
        Ground-truth image, shape (N, N).
    sampling_ratio : float
        Fraction of patterns used (e.g. 0.1 = 10%).
    snr_db : float
        Measurement SNR in dB.
    lambda_tv : float
        TV regularization weight.
    rho : float
        ADMM penalty parameter.
    max_iters : int
        Maximum ADMM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.

    Returns
    -------
    recon_img : np.ndarray
        Reconstructed image, shape (N, N).
    """
    N = image.shape[0]
    M = N * N

    # 1) Generate patterns and measurements
    patterns = generate_hadamard_patterns(N)      # (M, N, N)
    s_clean = measure_image(image, patterns)      # (M,)
    s_noisy = add_gaussian_noise(s_clean, snr_db)

    # 2) Subsample patterns and measurements
    patterns_sub, s_sub, idx = subsample_measurements(
        patterns, s_noisy, compression_ratio=sampling_ratio
    )
    K = patterns_sub.shape[0]

    if verbose:
        print(f"[TVAL3 baseline] Using {K} / {M} patterns "
              f"({sampling_ratio*100:.1f}% sampling), SNR={snr_db} dB")

    # 3) Build sensing matrix A from subsampled patterns
    #    patterns_sub: (K, N, N) -> A: (K, N*N)
    A = patterns_sub.reshape(K, -1)

    # 4) Run TV-regularized CS solver
    recon_img = tv_cs_admm(
        A,
        s_sub,
        N=N,
        lambda_tv=lambda_tv,
        rho=rho,
        max_iters=max_iters,
        tol=tol,
        verbose=verbose,
    )

    return recon_img


# ============================================================
#  Self-test
# ============================================================

if __name__ == "__main__":
    # Simple test: reconstruct a gradient image
    N = 32
    x = np.linspace(0, 1, N, dtype=np.float32)
    test_img = np.outer(x, x)

    recon = tval3_reconstruct_image(
        test_img,
        sampling_ratio=0.1,
        snr_db=20.0,
        lambda_tv=0.1,
        rho=1.0,
        max_iters=200,
        verbose=True,
    )

    print("TVAL3-style reconstruction complete. "
          f"Recon shape: {recon.shape}")
