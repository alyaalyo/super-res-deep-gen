import numpy as np
import pandas as pd
import pytest
import torch

from srbench.config import Paths, load_yaml, DEFAULT_CONFIG, jpeg_tag, parse_jpeg_tag
from srbench.data import textgen
from srbench.imageio import jpeg_roundtrip
from srbench.imresize import imresize, imresize_uint8
from srbench.metrics.face import threshold_at_fmr
from srbench.metrics.fidelity import psnr_y, rgb2y, ssim_y
from srbench.metrics.ocr import cer, normalize, wer
from srbench.models.base import image_seed
from srbench.models.rrdbnet import RRDBNet
from srbench.pipeline import TestLock
from srbench.stats import bootstrap_ci, paired_difference


def test_imresize_constant_and_shape():
    img = np.full((128, 96, 3), 77, np.uint8)
    down = imresize_uint8(img, scale=0.25)
    assert down.shape == (32, 24, 3) and (down == 77).all()
    up = imresize(img[:32, :24], scale=4)
    assert up.shape == (128, 96, 3) and np.allclose(up, 77)


def test_imresize_reproduces_linear_ramp_in_interior():
    ramp = np.tile(np.arange(64, dtype=np.float64), (64, 1))
    up = imresize(ramp, scale=2)
    # cubic convolution with a = -0.5 is exact for linear functions away from the border
    expected = (np.arange(128) + 0.5) / 2 - 0.5
    assert np.allclose(up[10, 8:-8], expected[8:-8], atol=1e-9)


def test_fidelity_metrics():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert psnr_y(a, a) == float("inf")
    assert ssim_y(a, a) == pytest.approx(1.0)
    b = np.clip(a.astype(int) + rng.integers(-10, 11, a.shape), 0, 255).astype(np.uint8)
    assert 20 < psnr_y(b, a) < 40
    assert ssim_y(b, a) < 1
    assert rgb2y(np.zeros((1, 1, 3), np.uint8))[0, 0] == pytest.approx(16)
    assert rgb2y(np.full((1, 1, 3), 255, np.uint8))[0, 0] == pytest.approx(235)


def test_cer_wer():
    assert normalize(" a\n b  c ") == "a b c"
    assert cer("hello world", "hello\nworld") == 0
    assert cer("hallo world", "hello world") == pytest.approx(1 / 11)
    assert wer("hallo world", "hello world") == pytest.approx(0.5)
    assert cer("", "abc") == 1


def test_bootstrap_and_paired_difference():
    x = np.arange(100, dtype=float)
    m, lo, hi = bootstrap_ci(x, n_boot=2000)
    assert m == pytest.approx(49.5) and lo < m < hi
    assert bootstrap_ci(x, n_boot=2000) == bootstrap_ci(x, n_boot=2000)   # seeded
    df = pd.DataFrame({"id": list(range(10)) * 2, "content": "objects", "jpeg": "none",
                       "method": ["a"] * 10 + ["b"] * 10, "psnr": list(np.arange(10.0) + 1) + list(np.arange(10.0))})
    d = paired_difference(df, "a", "b", ["content", "jpeg"], ["psnr"], n_boot=500)
    assert d["mean"].iloc[0] == pytest.approx(1) and d.lo.iloc[0] == pytest.approx(1) and d.significant.iloc[0]


def test_textgen_deterministic_and_fits():
    a = textgen.render("DejaVuSans.ttf", 24, "codes", np.random.default_rng(1))
    b = textgen.render("DejaVuSans.ttf", 24, "codes", np.random.default_rng(1))
    assert a.text == b.text and (a.image == b.image).all()
    assert a.image.shape == (512, 512, 3) and len(a.lines) > 5
    assert all(c in textgen.CODE_ALPHABET + " " for c in "".join(a.lines))
    w = textgen.render("DejaVuSerif.ttf", 16, "words", np.random.default_rng(2))
    assert all(t in textgen.WORDS for t in w.text.split())


def test_jpeg_and_tags():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = jpeg_roundtrip(img, 30)
    assert out.shape == img.shape and out.dtype == np.uint8 and not (out == img).all()
    assert jpeg_tag(None) == "none" and jpeg_tag(30) == "q30"
    assert parse_jpeg_tag("q30") == 30 and parse_jpeg_tag("none") is None


def test_threshold_at_fmr():
    scores = np.linspace(0, 1, 1001)
    t = threshold_at_fmr(scores, 0.01)
    assert (scores >= t).mean() <= 0.01
    assert (scores >= t - 0.002).mean() > 0.01


def test_image_seed_is_order_independent():
    assert image_seed(0, "face_00001") == image_seed(0, "face_00001")
    assert image_seed(0, "face_00001") != image_seed(1, "face_00001")


def test_rrdbnet_shape():
    net = RRDBNet(num_block=1)
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 8, 12)).shape == (1, 3, 32, 48)


def test_test_lock(tmp_path):
    cfg = load_yaml(DEFAULT_CONFIG)
    cfg["paths"]["outputs"] = str(tmp_path)
    paths = Paths(cfg)
    lock = TestLock(paths, cfg)
    assert lock.check("bicubic", force=False)
    lock.mark("bicubic", "started", "cpu")
    assert TestLock(paths, cfg).check("bicubic", force=False)          # interrupted -> resume
    lock.mark("bicubic", "finished", "cpu")
    assert not TestLock(paths, cfg).check("bicubic", force=False)      # done -> skip
    cfg2 = load_yaml(DEFAULT_CONFIG)
    cfg2["paths"]["outputs"] = str(tmp_path)
    cfg2["seed"] = 1
    with pytest.raises(SystemExit):
        TestLock(paths, cfg2).check("bicubic", force=False)             # settings changed -> refuse
