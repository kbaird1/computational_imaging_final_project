"""
forward_model.py

Forward single-pixel imaging (SPI) model for URNet-style experiments.

Provides:
  - True Walsh–Hadamard 1D sequency ordering (via Gray-code bit reversal)
  - Correct 2D Walsh basis: outer products of Walsh1D[i] × Walsh1D[j]
  - Low-frequency-first ordering suitable for deterministic subsampling
  - SPI forward operator and AWGN noise model

NOTE:
  N must be a power of 2 (32, 64, 128, ...).
"""

import numpy as np
from scipy.linalg import hadamard


# ============================================================
# Validate N for Hadamard
# ============================================================

def _validate_hadamard_size(N: int):
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    if (N & (N - 1)) != 0:
        raise ValueError(
            f"N={N} is not a power of 2. "
            f"Supported sizes include 32, 64, 128."
        )


# ============================================================
# True Walsh–Hadamard sequency ordering via Gray-code rule
# ============================================================

def _gray_code(n: int) -> int:
    """Return Gray code of integer n: g = n XOR (n >> 1)."""
    return n ^ (n >> 1)


def _walsh_sequency_order(H: np.ndarray) -> np.ndarray:
    """
    Convert a Hadamard matrix H into **Walsh sequency order**.

    This uses the canonical Gray-code rule:
        index i maps to gray_code(i)

    This produces a basis ordered by increasing "sequency"
    (number of zero-crossings), matching standard Walsh functions.
    """
    N = H.shape[0]
    order = np.array([_gray_code(i) for i in range(N)], dtype=int)
    return H[order]


# ============================================================
# Generate 2D Walsh–Hadamard Patterns (low-frequency-first)
# ============================================================

def generate_hadamard_patterns(N: int) -> np.ndarray:
    """
    Generate 2D Walsh–Hadamard patterns in true sequency order.

    Steps:
      1. Build 1D Hadamard (±1)
      2. Reorder rows by Gray-code (Walsh ordering)
      3. Construct 2D basis: outer(W[i], W[j])
      4. Lexicographic (i, j) ensures low → high frequency order

    Returns:
        patterns : (N*N, N, N) float32 in {+1, -1}
    """
    _validate_hadamard_size(N)

    # 1D Hadamard ±1 matrix
    H = hadamard(N).astype(np.float32)

    # Convert to Walsh sequency order
    W = _walsh_sequency_order(H)

    # Build full 2D Walsh basis
    M = N * N
    patterns = np.zeros((M, N, N), dtype=np.float32)

    idx = 0
    for i in range(N):
        for j in range(N):
            patterns[idx] = np.outer(W[i], W[j])
            idx += 1

    return patterns


# ============================================================
# Forward Measurement Operator
# ============================================================

def measure_image(image: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    """
    Compute SPI measurements using 2D Walsh–Hadamard patterns.

    image    : (N, N)
    patterns : (M, N, N)

    Returns:
        s : (M,)
    """
    return np.sum(patterns * image[None, :, :], axis=(1, 2))


# ============================================================
# Additive White Gaussian Noise (AWGN)
# ============================================================

def add_gaussian_noise(
    s: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Add AWGN for target SNR (dB):
        SNR_dB = 10 * log10(P_signal / P_noise)
    """
    if rng is None:
        rng = np.random.default_rng()

    s = s.astype(np.float32)
    signal_power = np.mean(s ** 2)

    if signal_power == 0:
        return s

    noise_power = signal_power / (10 ** (snr_db / 10))
    sigma = np.sqrt(noise_power)

    noise = rng.normal(0.0, sigma, size=s.shape).astype(np.float32)
    return s + noise


# ============================================================
# Sanity Test
# ============================================================

if __name__ == "__main__":
    print("\n=== Sanity Test for 2D Walsh–Hadamard Patterns ===\n")

    for N in [32, 64]:
        print(f"\nTesting N={N}...")
        patterns = generate_hadamard_patterns(N)
        print("patterns shape:", patterns.shape)

        # Create a test image
        x = np.linspace(0, 1, N, dtype=np.float32)
        test_img = np.outer(x, x)

        s = measure_image(test_img, patterns)
        print("measurements shape:", s.shape)

        s_noisy = add_gaussian_noise(s, 20)
        print("noisy first 5:", s_noisy[:5])
