"""Side-by-side figure: original, real (processed) and synthetic (processed) breathing data.

Columns
    Original   the region's mean temperature as recorded: no injection, no filtering
    Real       the same frames after processing (detrend + band-pass, region mean and SVD)
    Synthetic  the same frames with synthetic breathing added (synth.py), same processing
Rows
    1  signal over the whole clip (processed columns share one y-axis)
    2  spectrum over the whole clip, breathing band shaded
    3  Original: one Lepton-like frame with the region; Real / Synthetic: rate per 30 s window

Example (from the project root):
    .venv312/Scripts/python.exe src/plot_compare.py data/SIM1/T001/raw_thermal/T001-PD.dat \
        outputs/face_detection/T001-PD_f2090-2989_landmarks.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import breathing
import sim1
import synth
from tfake import NOSE_LIP_ROI

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dat", type=Path)
    ap.add_argument("landmarks", type=Path, help="*_landmarks.npz from detect_faces.py --res both")
    ap.add_argument("--rates", type=float, nargs="+", default=[12, 18, 24])
    ap.add_argument("--amplitude", type=float, default=0.3)
    ap.add_argument("--irregularity", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sqi-min", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "breathing")
    args = ap.parse_args()

    fs = sim1.FPS
    npz = np.load(args.landmarks)
    idx, lm_lep, lm_nat = npz["frames"], npz["lepton"], npz["native"]
    n = len(idx)
    wave, rate = synth.breathing_waveform(n, fs, args.rates, args.irregularity, args.seed)

    frames = sim1.open_thermal(args.dat)
    real = np.empty((n, sim1.LEPTON_H, sim1.LEPTON_W), np.float32)
    simu = np.empty_like(real)
    for k, i in enumerate(idx):
        c = sim1.to_celsius(frames[i])
        real[k] = sim1.to_lepton(c)
        simu[k] = sim1.to_lepton(synth.inject(c, lm_nat[k], args.amplitude * wave[k]))

    t = np.arange(n) / fs
    data = {}
    for key, stack in (("real", real), ("synthetic", simu)):
        X, mask, valid, _ = breathing.warp_rois(stack, lm_lep)
        Xf = np.where(valid, X, np.nanmean(X, axis=1, keepdims=True)).astype(np.float64)
        mean = breathing.bandpass(Xf.mean(0) - Xf.mean(), fs)
        v, _, comp, _ = breathing.svd_breathing(Xf, fs)          # whole clip, for display
        svd = breathing.bandpass(v, fs)
        svd *= np.std(mean) / np.std(svd)                         # same scale as the mean, for display
        rows = breathing.analyse(X, valid, fs, sqi_min=args.sqi_min)
        data[key] = dict(raw=Xf.mean(0), mean=mean, svd=svd, comp=comp, rows=rows)

    fig, ax = plt.subplots(3, 3, figsize=(16, 10.5))
    fig.suptitle(f"{args.landmarks.stem.removesuffix('_landmarks')}: original vs real vs synthetic "
                 f"(Lepton-like 160x120, {fs} fps; synthetic {args.amplitude} C at "
                 f"{'/'.join(f'{r:g}' for r in args.rates)} breaths/min)")

    # column 1: original
    a = ax[0, 0]
    a.plot(t, data["real"]["raw"], color="0.3", lw=1)
    a.set(title="Original: region mean temperature, unprocessed", xlabel="time (s)", ylabel="deg C")
    a = ax[1, 0]
    plot_spectrum(a, data["real"]["raw"] - data["real"]["raw"].mean(), fs, "original", "0.3", detrend=False)
    a.set_title("Original: spectrum (static level removed only)")
    a = ax[2, 0]
    k = n // 2
    a.imshow(real[k], cmap="inferno", vmin=24, vmax=34)
    a.fill(*lm_lep[k][NOSE_LIP_ROI].T, fc="cyan", ec="cyan", alpha=0.35)
    a.set(title=f"Original frame {idx[k]} with the analysed region", xticks=[], yticks=[])

    # columns 2-3: processed real and synthetic, shared y-axes per row
    lim = 1.1 * max(np.abs(d[s]).max() for d in data.values() for s in ("mean", "svd"))
    for col, key in ((1, "real"), (2, "synthetic")):
        d = data[key]
        a = ax[0, col]
        a.plot(t, d["mean"], lw=1, color="C0", label="region mean")
        a.plot(t, d["svd"], lw=1, color="C3", alpha=0.8, label=f"SVD component {d['comp'] + 1}")
        if key == "synthetic":
            gain = float(d["mean"] @ wave / (wave @ wave))
            a.plot(t, gain * wave, lw=1, color="k", alpha=0.5, label="injected (scaled)")
        a.set(title=f"{key.capitalize()}: detrended + band-passed 0.1-0.85 Hz", xlabel="time (s)",
              ylabel="deg C", ylim=(-lim, lim))
        a.legend(loc="lower right", fontsize=8, ncol=3)

        a = ax[1, col]
        plot_spectrum(a, d["mean"], fs, "region mean", "C0")
        plot_spectrum(a, d["svd"], fs, "SVD", "C3")
        if key == "synthetic":
            for r in args.rates:
                a.axvline(r, color="k", ls="--", lw=0.8)
        a.set_title(f"{key.capitalize()}: spectrum" + (" (dashed = injected rates)" if key == "synthetic" else ""))
        a.legend(fontsize=8)

        a = ax[2, col]
        if key == "synthetic":
            a.plot(t, rate, color="k", lw=1, label="true")
        tc = np.array([(r["t_start"] + r["t_end"]) / 2 for r in d["rows"]])
        for m, c in (("mean", "C0"), ("svd", "C3")):
            rr = np.array([r.get(f"rr_{m}", np.nan) for r in d["rows"]])
            ok = np.array([bool(r.get(f"valid_{m}", False)) for r in d["rows"]])
            a.plot(tc[ok], rr[ok], "o", color=c, label=f"{m}, valid")
            a.plot(tc[~ok], rr[~ok], "o", mfc="none", color=c, label=f"{m}, low SQI")
        a.set(title=f"{key.capitalize()}: rate per 30 s window", xlabel="window centre (s)",
              ylabel="breaths/min", ylim=(0, 55), xlim=(0, t[-1]))
        a.legend(loc="upper left", fontsize=8, ncol=3)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.landmarks.stem.removesuffix('_landmarks')}_compare.png"
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


def plot_spectrum(a, x, fs, label, color, detrend=True):
    """Whole-clip spectrum, normalised to its own maximum, breathing band shaded."""
    if detrend:
        f, p = breathing.spectrum(x, fs)
    else:
        from scipy import signal
        f, p = signal.welch(x, fs, window="hann", nperseg=len(x), noverlap=0, nfft=8192, detrend=False)
    a.plot(60 * f, p / p.max(), color=color, lw=1, label=label)
    a.axvspan(60 * breathing.BAND[0], 60 * breathing.BAND[1], color="0.92", zorder=-1)
    a.set(xlabel="breaths/min", ylabel="normalised power", xlim=(0, 60))


if __name__ == "__main__":
    main()
