"""Breathing signal and respiratory rate from the nose/upper-lip region (pipeline steps 5-7).

1. Align. Each frame's region is warped onto one fixed canvas with a similarity transform
   fitted to the nose and upper-lip landmarks, so a given canvas pixel is the same patch of
   skin in every frame. The region mask (tfake.NOSE_LIP_ROI) is drawn once on that canvas.
   Result: a pixel x time matrix X.
2. Per sliding window (default 30 s, 5 s step), two ways to get one breathing waveform:
   - "mean": average over all region pixels (the plan's step 5).
   - "svd":  remove each pixel's linear trend (static pattern + slow drift), take the SVD of
             the pixel x time matrix and keep the component whose time course looks most
             like breathing (most power in the breathing band, concentrated in one peak).
3. Rate: peak of the spectrum inside the breathing band 0.1-0.85 Hz (6-51 breaths/min).
   The spectrum is a Hann-windowed periodogram of the whole window (Welch with one segment,
   zero-padded), which keeps the best frequency resolution for a 30 s window.
   Searching only inside the band is equivalent to band-passing first.
4. Quality (SQI): share of the in-band power that lies in the peak's main lobe (+-2/T Hz).
   About 1 for a clean periodic signal, about 0.2 for noise. Windows below `sqi_min`
   are marked invalid.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import signal

from tfake import NOSE_LIP_ROI

BAND = (0.1, 0.85)                                   # Hz, 6-51 breaths/min
ALIGN_POINTS = list(range(27, 36)) + list(range(48, 55))   # nose bridge, nose, outer upper lip


# ----------------------------------------------------------------------------- alignment
def warp_rois(frames_c: np.ndarray, lms: np.ndarray, roi: list[int] = NOSE_LIP_ROI, margin: int = 2):
    """Warp each frame's region onto a fixed canvas.

    frames_c: (n, H, W) deg C. lms: (n, 70, 2) landmarks in the same pixel coordinates, NaN = no face.
    Returns X (pixels, n) with NaN columns for frames without a face, the canvas mask (h, w)
    bool, the valid-frame flags (n,), and the warped canvases (n, h, w) for display.
    """
    ref = np.nanmedian(lms, axis=0)                          # typical face position in the clip
    pts = ref[roi]
    x0, y0 = np.floor(pts.min(0)).astype(int) - margin
    x1, y1 = np.ceil(pts.max(0)).astype(int) + margin
    w, h = x1 - x0 + 1, y1 - y0 + 1
    offset = np.array([x0, y0], float)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round((pts - offset) * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    mask = mask.astype(bool)

    dst = (ref[ALIGN_POINTS] - offset).astype(np.float32)
    n = len(frames_c)
    canv = np.full((n, h, w), np.nan, np.float32)
    valid = ~np.isnan(lms).any(axis=(1, 2))
    for k in np.flatnonzero(valid):
        M, _ = cv2.estimateAffinePartial2D(lms[k][ALIGN_POINTS].astype(np.float32), dst, method=cv2.LMEDS)
        if M is None:
            valid[k] = False
            continue
        canv[k] = cv2.warpAffine(frames_c[k].astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return canv[:, mask].T, mask, valid, canv


# ----------------------------------------------------------------------------- signal
def bandpass(x: np.ndarray, fs: float, band=BAND, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = signal.butter(order, band, btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def spectrum(x: np.ndarray, fs: float, nfft: int = 4096):
    """Hann-windowed one-segment Welch spectrum (linear trend removed), zero-padded to nfft."""
    return signal.welch(x, fs, window="hann", nperseg=len(x), noverlap=0, nfft=max(nfft, len(x)),
                        detrend="linear")


def spectral_rate(x: np.ndarray, fs: float, band=BAND):
    """(rate breaths/min, SQI, in-band power fraction) of one waveform."""
    f, p = spectrum(x, fs)
    inb = (f >= band[0]) & (f <= band[1])
    if p[inb].sum() <= 0:
        return np.nan, 0.0, 0.0
    f_pk = f[inb][np.argmax(p[inb])]
    lobe = np.abs(f - f_pk) <= 2 * fs / len(x)               # Hann main lobe = +-2/T
    sqi = p[lobe & inb].sum() / p[inb].sum()
    band_frac = p[inb].sum() / p[f > 0].sum()
    return 60 * f_pk, float(sqi), float(band_frac)


def svd_breathing(Xw: np.ndarray, fs: float, n_comp: int = 5):
    """Breathing component of one window's pixel x time matrix.

    Returns time course (frames,), spatial map (pixels,), component index and its share of
    the variance. Each pixel's linear trend is removed first, so the static pattern and slow
    drift do not enter the decomposition. The component sign is chosen so the map is mostly
    positive (sign does not affect the rate).
    """
    Xc = signal.detrend(Xw, axis=1, type="linear")
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    best, best_score = 0, -1.0
    for k in range(min(n_comp, len(S))):
        _, sqi, band_frac = spectral_rate(Vt[k], fs)
        if sqi * band_frac > best_score:
            best, best_score = k, sqi * band_frac
    u, v = U[:, best], Vt[best] * S[best] / np.sqrt(len(U))
    if u.sum() < 0:
        u, v = -u, -v
    return v, u, best, float(S[best] ** 2 / (S ** 2).sum())


def analyse(X: np.ndarray, valid: np.ndarray, fs: float, win_s: float = 30, step_s: float = 5,
            sqi_min: float = 0.5) -> list[dict]:
    """Rate per sliding window, by region mean and by SVD. Windows with a missing frame are skipped."""
    W, step = int(round(win_s * fs)), int(round(step_s * fs))
    rows = []
    for s in range(0, X.shape[1] - W + 1, step):
        row = {"start": s, "stop": s + W, "t_start": s / fs, "t_end": (s + W) / fs,
               "complete": bool(valid[s:s + W].all())}
        if row["complete"]:
            Xw = X[:, s:s + W].astype(np.float64)
            rr, sqi, _ = spectral_rate(Xw.mean(0), fs)
            row.update(rr_mean=rr, sqi_mean=sqi, valid_mean=sqi >= sqi_min)
            v, _, k, var = svd_breathing(Xw, fs)
            rr, sqi, _ = spectral_rate(v, fs)
            row.update(rr_svd=rr, sqi_svd=sqi, valid_svd=sqi >= sqi_min, svd_comp=k, svd_var=var)
        rows.append(row)
    return rows
