"""Identity preservation with ArcFace (InsightFace buffalo_l: SCRFD detector +
ResNet-50 ArcFace trained on WebFace600K).

Landmarks are detected once on the HR face and reused for every SR output of
that image (HR and SR are pixel-aligned). A detector failure on a degraded
output therefore cannot be confused with an identity change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 5-point landmarks (eyes, nose, mouth corners) of the FFHQ alignment at
# 512x512, as used by facexlib/GFPGAN. Every FFHQ image is aligned to this
# template, so it is a valid fallback when the detector misses a face.
FFHQ_512_TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                              [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


class ArcFace:
    def __init__(self, root: Path, det_size: int = 512):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", root=str(root), allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self.rec = self.app.models["recognition"]

    def landmarks(self, hr: np.ndarray) -> tuple[np.ndarray, str]:
        """5-point landmarks of the largest face in an RGB image, and their source."""
        det = self.app.models["detection"]
        default = det.det_thresh
        try:
            for thresh in (default, 0.3):
                det.det_thresh = thresh
                faces = self.app.get(hr[..., ::-1].copy())      # InsightFace expects BGR
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    return face.kps.astype(np.float32), f"detector@{thresh}"
        finally:
            det.det_thresh = default
        scale = hr.shape[0] / 512
        return FFHQ_512_TEMPLATE * scale, "ffhq_template"

    def embed(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align

        crop = face_align.norm_crop(img[..., ::-1].copy(), landmark=kps, image_size=112)
        e = self.rec.get_feat(crop).ravel()
        return e / np.linalg.norm(e)


def threshold_at_fmr(impostor_scores: np.ndarray, fmr: float) -> float:
    """Smallest threshold whose false-match rate on impostor pairs is <= fmr."""
    s = np.sort(np.asarray(impostor_scores))[::-1]
    k = int(np.floor(fmr * len(s)))          # number of impostors allowed above threshold
    return float(np.nextafter(s[k], np.inf)) if k < len(s) else float(s[-1])
