"""
Generalized URNet implementation supporting image sizes:
    - 32 × 32
    - 64 × 64
    - 128 × 128

Follows Li et al. (2023), "URNet: High-quality single-pixel imaging
with untrained reconstruction network".

Architecture summary:

    s ∈ R^M
        ↓ FC projection
    (B, 512*4*4)
        ↓ reshape
    (B, 512, 4, 4)
        ↓ deconv blocks (kernel=4, stride=2)
    ... upsample to (B, 64, H, W)
        ↓ 4 refinement conv blocks
    (B, 64, H, W)
        ↓ final conv
    (B, 1, H, W)
"""

import math
import torch
import torch.nn as nn


class URNet(nn.Module):
    def __init__(
        self,
        measurement_dim: int,
        image_size: int = 128,
        base_channels: int = 512,
        min_channels: int = 64,
        num_refine_convs: int = 4,
    ):
        """
        Parameters
        ----------
        measurement_dim : int
            Dimension M of the SPI measurement vector.
        image_size : int
            One of {32, 64, 128}. Must satisfy image_size / 4 = 2^k.
        base_channels : int
            Number of channels in the initial 4×4 latent map.
        min_channels : int
            Minimum number of channels used in the upsampling path.
        num_refine_convs : int
            Number of refinement conv layers (paper uses 4).
        """
        super().__init__()

        assert image_size in {32, 64, 128}, \
            f"image_size must be 32, 64, or 128 (got {image_size})"

        self.measurement_dim = measurement_dim
        self.image_size = image_size

        # -----------------------------
        # 1) Determine number of upsampling steps
        # -----------------------------
        upscale_factor = image_size // 4    # 4 → 8 → 16 → ...
        num_upsamples = int(math.log2(upscale_factor))   # 2^k = upscale_factor

        # -----------------------------
        # 2) Channel reduction schedule (paper: 512 → 256 → 128 → 64 → 64 → 64)
        # -----------------------------
        # Channels for each upsampling step
        channel_schedule = [base_channels]
        for _ in range(num_upsamples):
            next_ch = max(min_channels, channel_schedule[-1] // 2)
            channel_schedule.append(next_ch)

        # -----------------------------
        # 3) FC projection → 4×4×base_channels
        # -----------------------------
        self.fc = nn.Linear(measurement_dim, base_channels * 4 * 4)

        # -----------------------------
        # 4) Deconvolution blocks
        # -----------------------------
        deconvs = []
        for in_ch, out_ch in zip(channel_schedule[:-1], channel_schedule[1:]):
            deconvs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.deconv_blocks = nn.ModuleList(deconvs)

        # -----------------------------
        # 5) Refinement conv layers (paper uses 4)
        # -----------------------------
        refine_blocks = []
        feat_ch = channel_schedule[-1]  # final deconv output channels

        for _ in range(num_refine_convs - 1):
            refine_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        feat_ch,
                        feat_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(feat_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.refine_blocks = nn.ModuleList(refine_blocks)

        # Final conv to 1 channel (NO activation)
        self.final_conv = nn.Conv2d(
            feat_ch,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._init_weights()

    # -----------------------------
    # Weight initialization
    # -----------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, s):
        """
        s: (B, M)
        returns: (B, 1, H, W)
        """
        B = s.size(0)

        # FC projection → 4×4×base_channels
        x = self.fc(s)
        x = x.view(B, -1, 4, 4)

        # Upsample
        for block in self.deconv_blocks:
            x = block(x)

        # Refinement
        for block in self.refine_blocks:
            x = block(x)

        # Output
        x = self.final_conv(x)
        return x


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    for N in [32, 64, 128]:
        M = (N * N) // 4   # e.g., 25% sampling
        print(f"\nTesting URNet with image_size={N}, M={M}")

        model = URNet(measurement_dim=M, image_size=N)
        dummy_s = torch.randn(1, M)
        out = model(dummy_s)

        print("Output:", out.shape)   # (1, 1, N, N)

