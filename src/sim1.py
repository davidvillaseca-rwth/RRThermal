"""Read SIM1 raw thermal recordings and convert them to Lepton-like frames.

SIM1 is the distracted-driving dataset of Taamneh et al. (Scientific Data, 2017),
https://osf.io/c42cn. Each drive has two files in "Raw Thermal Data":

    <ID>-<DRIVE>.inf   three text lines: last frame index, width, height (e.g. 4404 / 640 / 512)
    <ID>-<DRIVE>.dat   headerless little-endian uint16 frames, one after another

Decoded on T001-PD (not documented by the dataset itself):
    - pixel values look like thousandths of a degree Celsius (background ~26 C, face ~34 C)
    - rows 480-511 are constant padding, so the real image is 640x480
    - 640x480 is exactly 4x the FLIR Lepton 3.5 (160x120)
    - frame rate is ~7.5 fps from the .pp timestamps, but 4405 frames over the
      504 s chest-strap record would mean ~8.7 fps; unresolved
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FPS = 7.5                 # from .pp timestamps; see module docstring
IMAGE_H, IMAGE_W = 480, 640
LEPTON_W, LEPTON_H = 160, 120


def read_inf(path: Path) -> tuple[int, int, int]:
    """(frame count, width, height) from a SIM1 .inf file."""
    last, w, h = (int(v) for v in Path(path).read_text().split())
    return last + 1, w, h


def open_thermal(dat: Path) -> np.ndarray:
    """Memory-mapped raw frames, shape (n, 480, 640), uint16, padding rows dropped."""
    dat = Path(dat)
    _, w, h = read_inf(dat.with_suffix(".inf"))
    frames = np.memmap(dat, dtype="<u2", mode="r").reshape(-1, h, w)
    return frames[:, :IMAGE_H, :]


def to_celsius(raw: np.ndarray) -> np.ndarray:
    """Raw SIM1 values to degrees Celsius (float32)."""
    return raw.astype(np.float32) / 1000.0


def to_lepton(frame_c: np.ndarray) -> np.ndarray:
    """Downsample a 640x480 frame to Lepton 3.5 resolution (160x120) by area averaging."""
    return cv2.resize(frame_c, (LEPTON_W, LEPTON_H), interpolation=cv2.INTER_AREA)


def to_uint8(frame_c: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a fixed temperature range [lo, hi] C to 0-255 grayscale."""
    return np.clip((frame_c - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
