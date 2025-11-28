import math
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer for FNO.
    x: [B, C_in, H, W]
    We keep only the lowest `modes1` x `modes2` Fourier modes.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # complex weights: [C_in, C_out, modes1, modes2]
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input, weights):
        # input: [B, C_in, M1, M2]
        # weight: [C_in, C_out, M1, M2]
        # output: [B, C_out, M1, M2]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        x: [B, C_in, H, W]
        """
        B, C, H, W = x.shape
        # FFT: [B, C_in, H, W//2+1]
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        # NO spectral normalization here now; weights can be learned freely.
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2],
        )

        # inverse FFT: back to real space
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNO2D(nn.Module):
    """
    2D FNO block for one-step prediction:
    (S_t, params) -> S_{t+Δt}
    """
    def __init__(
        self,
        in_channels=1,
        width=96,
        depth=4,
        modes1=48,
        modes2=48,
        param_channels=0,
    ):
        super().__init__()
        self.width = width

        self.in_proj = nn.Conv2d(in_channels + param_channels, width, 1)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)]
        )
        self.w_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(depth)]
        )

        self.act = nn.GELU()
        self.out_proj = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, x, params=None):
        """
        x: [B, 1, H, W]
        params: [B, P] or None
        """
        if params is not None:
            B, _, H, W = x.shape
            p = params.view(B, -1, 1, 1).expand(B, -1, H, W)
            x = torch.cat([x, p], dim=1)  # [B, 1+P, H, W]

        x = self.in_proj(x)  # [B, width, H, W]

        for spec, w in zip(self.spectral_layers, self.w_layers):
            x_spec = spec(x)
            x_lin = w(x)
            x = self.act(x_spec + x_lin)

        out = self.out_proj(x)  # [B, 1, H, W]
        return out
