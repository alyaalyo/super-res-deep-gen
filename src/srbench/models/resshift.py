"""ResShift (Yue et al., NeurIPS 2023 / TPAMI 2024), journal version v3: 4 steps.

We reuse the official model code in ``third_party/ResShift`` but not its
``ResShiftSampler``, which hard-codes CUDA. This wrapper builds the same
diffusion, Swin-UNet and VQGAN from the official config and calls the same
``p_sample_loop``, so it runs on CUDA, MPS or CPU.

ResShift is stochastic: the chain starts from the upsampled LR latent plus
Gaussian noise. We reseed per image (``image_seed``), so a result does not
depend on the order in which images are processed.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ROOT
from .base import SRModel, to_tensor, to_uint8

REPO = ROOT / "third_party" / "ResShift"
CONFIGS = {"v3": "realsr_swinunet_realesrgan256_journal.yaml"}


def _import_resshift():
    if not REPO.exists():
        raise FileNotFoundError(
            f"{REPO} missing. Run: git clone https://github.com/zsyOAOA/ResShift.git {REPO}"
        )
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf
    from utils import util_common, util_net
    return OmegaConf, util_common, util_net


def _load(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("state_dict", state)


class ResShift(SRModel):
    name = "resshift"
    stochastic = True

    def __init__(self, device: torch.device, weights_dir: Path, version: str = "v3", amp: bool = True,
                 seed: int = 0, **_):
        super().__init__(device)
        OmegaConf, util_common, util_net = _import_resshift()
        cfg = OmegaConf.load(REPO / "configs" / CONFIGS[version])
        cfg.diffusion.params.sf = self.scale

        self.diffusion = util_common.instantiate_from_config(cfg.diffusion)
        if device.type == "mps":
            # MPS has no float64; the schedule arrays are float64 numpy arrays.
            for k, v in vars(self.diffusion).items():
                if isinstance(v, np.ndarray) and v.dtype == np.float64:
                    setattr(self.diffusion, k, v.astype(np.float32))
        model = util_common.instantiate_from_config(cfg.model)
        util_net.reload_model(model, _load(Path(weights_dir) / f"resshift_realsrx4_s4_{version}.pth"))
        self.model = model.eval().to(device)

        vq = util_common.get_obj_from_str(cfg.autoencoder.target)(**cfg.autoencoder.params)
        util_net.reload_model(vq, _load(Path(weights_dir) / "autoencoder_vq_f4.pth"))
        self.autoencoder = vq.eval().to(device)
        for p in list(self.model.parameters()) + list(self.autoencoder.parameters()):
            p.requires_grad_(False)

        self.pad_to = cfg.model.params.get("lq_size", 64)
        self.seed = seed
        # The official sampler runs everything under fp16 autocast on CUDA.
        self.amp = amp and device.type in ("cuda", "mps")

    def _autocast(self):
        if not self.amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    @torch.inference_mode()
    def upscale(self, lr: np.ndarray, seed: int | None = None) -> np.ndarray:
        torch.manual_seed(self.seed if seed is None else seed)
        y = to_tensor(lr, self.device) * 2 - 1                       # [-1, 1]
        h, w = y.shape[2:]
        ph = math.ceil(h / self.pad_to) * self.pad_to - h
        pw = math.ceil(w / self.pad_to) * self.pad_to - w
        if ph or pw:
            y = F.pad(y, (0, pw, 0, ph), mode="reflect")
        with self._autocast():
            out = self.diffusion.p_sample_loop(
                y=y, model=self.model, first_stage_model=self.autoencoder, noise=None,
                noise_repeat=False, clip_denoised=False, denoised_fn=None,
                model_kwargs={"lq": y}, progress=False,
            )
        out = out[:, :, : h * self.scale, : w * self.scale].float().clamp(-1, 1)
        return to_uint8(out * 0.5 + 0.5)
