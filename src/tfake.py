"""T-FAKE thermal landmarker (pip: thermal-face-alignment) adapted to low-res Lepton frames.

The package runs a TFW YOLO5Face detector and then a MobileNetV2 landmark
regressor on a 224x224 crop. On 160x120 frames the detector often finds
nothing, so two face-box sources are supported:
    face_box given  -> skip the detector, crop around that box
    face_box None   -> run the package detector on the frame upscaled by `upscale`

Requires Python 3.10-3.12 (use .venv312). Weights download on first use to ~/.tfan.
Paper: Flotho et al., "T-FAKE: Synthesizing Thermal Images for Facial Landmarking", CVPR 2025.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

# 70-point layout = iBUG 68 + 2 pupils; 478-point layout = MediaPipe face mesh
NOSE_TIP = {70: 30, 478: 1}
NOSE_LOWER = {70: list(range(30, 36)), 478: [1, 2, 98, 327]}  # tip + nostrils

# 70-point layout only. Jawline from ear to ear (iBUG 0-16), used to outline the face.
JAWLINE = list(range(0, 17))
# Nose tip, nostrils and upper lip (polygon, in order): from the left nostril wing (31) up to
# the lower nose bridge (29) and down to the right wing (35), on to the right mouth corner
# (54), back along the inner edge of the upper lip (64-60) and up from the left mouth corner
# (48). Point 29 rather than the tip landmark (30) is the top, because 30 sits almost level
# with the nostrils when the head is tilted down. This is the region shown as analysed.
NOSE_LIP_ROI = [31, 29, 35, 54, 64, 63, 62, 61, 60, 48]


class TFakeLandmarker:
    def __init__(self, n_landmarks: int = 70, upscale: int = 4):
        from tfan import ThermalLandmarks
        from tfan.core import models as tfan_models

        warnings.filterwarnings("ignore", category=RuntimeWarning, module="tfan")
        # tfan builds its MobileNetV2 backbone with timm pretrained=True, which fetches ImageNet
        # weights from the Hugging Face hub. The T-FAKE checkpoint loaded right after replaces
        # every one of them (all 314 keys, none missing), so build the backbone empty instead
        # and avoid the network call.
        timm_create = tfan_models.timm.create_model
        tfan_models.timm.create_model = lambda *a, **kw: timm_create(*a, **{**kw, "pretrained": False})
        try:
            self.model = ThermalLandmarks(device="cpu", n_landmarks=n_landmarks)
        finally:
            tfan_models.timm.create_model = timm_create
        self.n_landmarks = n_landmarks
        self.upscale = upscale

    def predict(self, gray: np.ndarray, face_box: tuple[int, int, int, int] | None = None):
        """Landmarks (n x 2, frame pixel coords) and confidences, or (None, None) if no face."""
        img3 = np.repeat(gray.astype(np.float32)[..., None], 3, axis=2)
        if face_box is not None:
            x0, y0, x1, y1 = face_box
            # _refine_landmarks crops a square of 1.25x the box's larger side around its centre
            return self.model._refine_landmarks(img3, np.array([[x0, y0], [x1, y1]], float))

        s = self.upscale
        big = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        lms, confs = self.model.process(big, mode="pixel")
        lm = np.asarray(lms[0])
        if lm.ndim != 2 or (lm < 0).all():
            return None, None
        return lm / s, np.asarray(confs[0])
