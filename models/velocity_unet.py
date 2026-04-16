
import math
import torch
from torch import nn
import torch.nn.functional as F

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding.
    t: (B,) or (B,1) in [0,1]
    returns: (B, dim)
    """
    if t.dim() == 2:
        t = t.squeeze(1)
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.time_mlp(t_emb)  # (B, 2*out_ch)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class VelocityUNet(nn.Module):
    """
    Small time-conditioned U-Net for velocity v_theta(z, t).
    Input/Output: (B, C, H, W)
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_mults=(1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down path: store skips after each ResBlock only (standard)
        self.downs = nn.ModuleList()
        ch = base_channels
        skip_chs = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch, time_dim=time_dim, dropout=dropout))
                ch = out_ch
                skip_chs.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(ch))

        # Mid
        self.mid1 = ResBlock(ch, ch, time_dim=time_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, time_dim=time_dim, dropout=dropout)

        # Up path: mirror
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_chs.pop()
                self.ups.append(ResBlock(ch + skip_ch, out_ch, time_dim=time_dim, dropout=dropout))
                ch = out_ch
            if i != len(channel_mults) - 1:
                self.ups.append(Upsample(ch))

        assert len(skip_chs) == 0, "skip channel bookkeeping error"

        self.out_norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch, eps=1e-6)
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(1)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.in_conv(x)

        skips = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:
                h = layer(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                s = skips.pop()
                if s.shape[-2:] != h.shape[-2:]:
                    s = F.interpolate(s, size=h.shape[-2:], mode='nearest')
                h = torch.cat([h, s], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        out = self.out_conv(F.silu(self.out_norm(h)))
        return out
