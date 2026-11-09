"""Synthetic breathing for testing the pipeline before real recordings exist.

Adds a known breathing-like temperature change under the nose of real thermal frames, so the
signal-processing steps can be checked against a ground-truth rate. This is NOT a physical
model of exhaled air: it is a smooth warm/cool pulse with a chosen rate and amplitude.

Spatial pattern: two small elliptical Gaussian "bulbs", one just below each nostril
(landmarks 32 and 34, moved a third of the way down towards the upper lip, 50 and 52),
where exhaled air meets the skin. Each is about a fifth of the nose width wide, so the two
stay separate; the weight is the larger of the two (peak 1 in each bulb).

Temporal pattern: a periodic waveform whose rate follows a schedule (e.g. 12, then 18, then 24
breaths/min), with optional slow breath-to-breath variation. Positive = warmer (exhaling).
"""

from __future__ import annotations

import numpy as np


def breathing_waveform(n: int, fps: float, rates_bpm: list[float], irregularity: float = 0.05,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Waveform in [-1, 1] and the true instantaneous rate (breaths/min), both length n.

    The clip is split into len(rates_bpm) equal parts, one rate each. `irregularity` is the
    standard deviation of a slow random variation of the rate, as a fraction of the rate.
    """
    rng = np.random.default_rng(seed)
    part = np.minimum(np.arange(n) * len(rates_bpm) // n, len(rates_bpm) - 1)
    rate = np.asarray(rates_bpm, float)[part]
    if irregularity > 0:
        # slow variation: white noise smoothed over ~10 s, scaled to the requested std
        k = max(int(10 * fps), 1)
        wobble = np.convolve(rng.standard_normal(n + k), np.ones(k) / k, mode="same")[:n]
        rate = rate * (1 + irregularity * wobble / (wobble.std() + 1e-12))
    phase = 2 * np.pi * np.cumsum(rate / 60.0) / fps
    # exhalation a little shorter and sharper than inhalation: add a small second harmonic
    wave = np.sin(phase) + 0.25 * np.sin(2 * phase + np.pi / 4)
    return wave / np.abs(wave).max(), rate


def weight_map(shape: tuple[int, int], lm: np.ndarray) -> np.ndarray:
    """Spatial weight in [0, 1] (image shape) for one frame's 70 landmarks (pixel coords):
    one bulb below each nostril."""
    nose_w = np.linalg.norm(lm[35] - lm[31])             # nostril wing to wing
    gap = np.linalg.norm(lm[51] - lm[33])                # nostril base to top of upper lip
    sx, sy = 0.2 * nose_w, 0.3 * gap
    # rotate with the face: x axis along the nostril line
    ax = (lm[35] - lm[31]) / (nose_w + 1e-9)
    ay = np.array([-ax[1], ax[0]])
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    w = np.zeros(shape, np.float32)
    for nostril, lip in ((32, 50), (34, 52)):
        centre = lm[nostril] + (lm[lip] - lm[nostril]) / 3
        d = np.stack([xx - centre[0], yy - centre[1]], -1)
        u, v = d @ ax, d @ ay
        w = np.maximum(w, np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)))
    return w.astype(np.float32)


def inject(frame_c: np.ndarray, lm: np.ndarray, delta_c: float) -> np.ndarray:
    """Frame (deg C) with delta_c * weight_map added under the nose. Unchanged if lm has NaN."""
    if np.isnan(lm).any() or delta_c == 0:
        return frame_c
    return frame_c + delta_c * weight_map(frame_c.shape, lm)
