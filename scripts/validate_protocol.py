"""Sanity check of the degradation + fidelity protocol on Set5 x4.

    python scripts/validate_protocol.py

Downloads the Set5 HR images and checks that our end-to-end protocol (MATLAB-
style bicubic x1/4, bicubic x4 back, PSNR/SSIM on Y with a 4-px crop)
reproduces the published MATLAB bicubic numbers: 28.42 dB / 0.8104
(Kim et al. 2016; Lim et al. 2017). PIL's bicubic would not: it uses a
different antialiasing filter.
"""

import argparse
import tarfile

import numpy as np

from srbench.config import Paths, load_config
from srbench.data.download import download_file
from srbench.imageio import imread
from srbench.imresize import imresize_uint8
from srbench.metrics.fidelity import psnr_y, ssim_y

URL = "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/{}.tar.gz"
PUBLISHED = (28.42, 0.8104)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Paths(load_config()).raw / "set5"
    archive = download_file(URL.format("Set5_HR"), root / "Set5_HR.tar.gz")
    if not (root / "Set5_HR").exists():
        with tarfile.open(archive) as t:
            t.extractall(root, filter="data")

    psnrs, ssims = [], []
    for hr_path in sorted((root / "Set5_HR").glob("*.png")):
        hr = imread(hr_path)
        h, w = hr.shape[:2]
        hr = hr[: h - h % 4, : w - w % 4]
        sr = imresize_uint8(imresize_uint8(hr, scale=0.25), scale=4)
        psnrs.append(psnr_y(sr, hr, 4))
        ssims.append(ssim_y(sr, hr, 4))
        print(f"{hr_path.stem:10s} PSNR {psnrs[-1]:.2f}  SSIM {ssims[-1]:.4f}")

    print(f"Set5 x4 bicubic: PSNR {np.mean(psnrs):.2f} dB, SSIM {np.mean(ssims):.4f} "
          f"(published {PUBLISHED[0]} / {PUBLISHED[1]})")
    ok = abs(np.mean(psnrs) - PUBLISHED[0]) < 0.05 and abs(np.mean(ssims) - PUBLISHED[1]) < 0.002
    print("OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
