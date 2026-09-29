"""Real-ESRNet and Real-ESRGAN (x4plus).

Both use the same RRDBNet generator and the same synthetic degradations for
training; Real-ESRNet is trained with an L1 loss only, Real-ESRGAN is then
fine-tuned from it with L1 + perceptual + GAN losses. Comparing the two isolates
the effect of generative (adversarial) training.

The RRDBNet definition follows BasicSR (Apache-2.0, X. Wang et al.) so that the
official checkpoints load without depending on the ``basicsr`` package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import SRModel, to_tensor, to_uint8


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """x4 RRDBNet (ESRGAN generator)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in state:
            return state[key]
    return state


class _RealESRBase(SRModel):
    weight_file: str = ""

    def __init__(self, device: torch.device, weights_dir: Path, half_on_cuda: bool = True):
        super().__init__(device)
        net = RRDBNet()
        net.load_state_dict(load_state(Path(weights_dir) / self.weight_file), strict=True)
        # Official inference runs in fp16 on CUDA; fp32 elsewhere.
        self.half = half_on_cuda and device.type == "cuda"
        self.net = (net.half() if self.half else net).eval().to(device)

    @torch.inference_mode()
    def upscale(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        x = to_tensor(lr, self.device)
        if self.half:
            x = x.half()
        return to_uint8(self.net(x))


class RealESRNet(_RealESRBase):
    name = "realesrnet"
    weight_file = "RealESRNet_x4plus.pth"


class RealESRGAN(_RealESRBase):
    name = "realesrgan"
    weight_file = "RealESRGAN_x4plus.pth"
