"""Fallback diffusion model: Stable Diffusion x4 Upscaler (diffusers).

Only used if ResShift cannot be run (``--methods sd_upscaler``). Needs the
optional dependencies: ``pip install -e .[fallback]``.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .base import SRModel


class SDUpscaler(SRModel):
    name = "sd_upscaler"
    stochastic = True

    def __init__(self, device: torch.device, repo: str, steps: int = 20, noise_level: int = 20,
                 guidance_scale: float = 0.0, prompt: str = "", **_):
        super().__init__(device)
        from diffusers import StableDiffusionUpscalePipeline

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(repo, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.kw = dict(num_inference_steps=steps, noise_level=noise_level,
                       guidance_scale=guidance_scale, prompt=prompt)

    @torch.inference_mode()
    def upscale(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        gen = torch.Generator("cpu").manual_seed(0 if seed is None else seed)
        out = self.pipe(image=Image.fromarray(lr), generator=gen, **self.kw).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.uint8)
