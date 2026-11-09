"""Prototype respiratory-rate estimation on a SIM1 clip, optionally with synthetic breathing.

Needs the landmarks of the clip from detect_faces.py (run with --res both when using
--simulate, because the synthetic signal is added at native resolution before downsampling).

Per frame: raw -> deg C -> [add synthetic breathing under the nose] -> Lepton-like 160x120.
Then breathing.py: align the nose/upper-lip region, sliding windows, region mean vs SVD,
spectral peak -> breaths/min, SQI.

Outputs (in --out), named <clip>_<real|sim>_*:
    _windows.csv   one row per window: true rate (sim), rate + SQI by mean and by SVD
    _summary.png   waveform, rate over time, example spectrum, SVD spatial map
    _roi.mp4       (with --video) Lepton frame + the aligned region minus its 30 s mean

Examples (from the project root):
    .venv312/Scripts/python.exe src/rr_prototype.py data/SIM1/T001/raw_thermal/T001-PD.dat \
        outputs/face_detection/T001-PD_f2090-2989_landmarks.npz --simulate --video
    .venv312/Scripts/python.exe src/rr_prototype.py data/SIM1/T001/raw_thermal/T001-PD.dat \
        outputs/face_detection/T001-PD_f2090-2989_landmarks.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
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
    ap.add_argument("dat", type=Path, help="SIM1 .dat file")
    ap.add_argument("landmarks", type=Path, help="*_landmarks.npz from detect_faces.py for this clip")
    ap.add_argument("--simulate", action="store_true", help="add synthetic breathing under the nose")
    ap.add_argument("--rates", type=float, nargs="+", default=[12, 18, 24],
                    help="synthetic rates in breaths/min, one per equal part of the clip (default 12 18 24)")
    ap.add_argument("--amplitude", type=float, default=0.3, help="synthetic peak change in deg C (default 0.3)")
    ap.add_argument("--irregularity", type=float, default=0.05, help="slow rate variation, fraction (default 0.05)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, default=30, help="window length in s (default 30)")
    ap.add_argument("--step", type=float, default=5, help="window step in s (default 5)")
    ap.add_argument("--sqi-min", type=float, default=0.5, help="SQI threshold for a valid window (default 0.5)")
    ap.add_argument("--video", action="store_true", help="also write the region video")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "breathing")
    args = ap.parse_args()

    fs = sim1.FPS
    npz = np.load(args.landmarks)
    idx = npz["frames"]
    lm_lep = npz["lepton"]
    if args.simulate and "native" not in npz:
        ap.error("--simulate needs native landmarks: run detect_faces.py with --res both")
    n = len(idx)

    wave, rate = synth.breathing_waveform(n, fs, args.rates, args.irregularity, args.seed)
    frames = sim1.open_thermal(args.dat)
    lep = np.empty((n, sim1.LEPTON_H, sim1.LEPTON_W), np.float32)
    for k, i in enumerate(idx):
        c = sim1.to_celsius(frames[i])
        if args.simulate:
            c = synth.inject(c, npz["native"][k], args.amplitude * wave[k])
        lep[k] = sim1.to_lepton(c)

    X, mask, valid, canv = breathing.warp_rois(lep, lm_lep)
    rows = breathing.analyse(X, valid, fs, args.window, args.step, args.sqi_min)
    for r in rows:
        r["rr_true"] = float(rate[r["start"]:r["stop"]].mean()) if args.simulate else np.nan

    tag = "sim" if args.simulate else "real"
    name = f"{args.landmarks.stem.removesuffix('_landmarks')}_{tag}"
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["t_start", "t_end", "complete", "rr_true", "rr_mean", "sqi_mean", "valid_mean",
              "rr_svd", "sqi_svd", "valid_svd", "svd_comp", "svd_var"]
    with open(args.out / f"{name}_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{n} frames ({n / fs:.0f} s at {fs} fps), region {mask.sum()} pixels on a "
          f"{mask.shape[1]}x{mask.shape[0]} canvas, {len(rows)} windows of {args.window:.0f} s")
    if args.simulate:
        print(f"synthetic breathing: rates {args.rates} breaths/min, amplitude {args.amplitude} C")
    for m in ("mean", "svd"):
        ok = [r for r in rows if r.get(f"valid_{m}")]
        line = f"{m:>4}: valid windows {len(ok)}/{len(rows)}"
        if args.simulate and ok:
            err = np.array([r[f"rr_{m}"] - r["rr_true"] for r in ok])
            line += f", MAE {np.abs(err).mean():.2f}, bias {err.mean():+.2f}, max error {np.abs(err).max():.2f} breaths/min"
        elif ok:
            line += ", rates " + " ".join(f"{r[f'rr_{m}']:.1f}" for r in ok)
        print(line)

    plot_summary(args.out / f"{name}_summary.png", X, valid, mask, rows, fs, rate if args.simulate else None,
                 wave * args.amplitude if args.simulate else None, args.window, name)
    if args.video:
        write_video(args.out / f"{name}_roi.mp4", lep, lm_lep, canv, mask, idx, fs, args.window)
    print(f"saved to {args.out}")


def plot_summary(path, X, valid, mask, rows, fs, rate, wave, win_s, title):
    n = X.shape[1]
    t = np.arange(n) / fs
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title)

    # (a) whole-clip waveform: region mean, band-passed (display only; frames without face filled)
    x = X.mean(0)
    x = np.where(valid, x, np.nanmean(x))
    a = ax[0, 0]
    bp = breathing.bandpass(x - x.mean(), fs)
    a.plot(t, bp, lw=1, label="region mean, band-passed")
    if wave is not None:
        # the injected pulse covers only part of the region, so scale it to the region mean for display
        gain = float(bp @ wave / (wave @ wave))
        a.plot(t, gain * wave, lw=1, alpha=0.7, label=f"injected waveform x {gain:.2f}")
    a.set(xlabel="time (s)", ylabel="deg C", title="Breathing waveform")
    a.legend(loc="upper right", fontsize=8)

    # (b) rate per window
    a = ax[0, 1]
    tc = np.array([(r["t_start"] + r["t_end"]) / 2 for r in rows])
    if rate is not None:
        a.plot(t, rate, color="k", lw=1, label="true (synthetic)")
    for m, c in (("mean", "C0"), ("svd", "C3")):
        rr = np.array([r.get(f"rr_{m}", np.nan) for r in rows])
        ok = np.array([bool(r.get(f"valid_{m}", False)) for r in rows])
        a.plot(tc[ok], rr[ok], "o", color=c, label=f"{m} (valid)")
        a.plot(tc[~ok], rr[~ok], "o", mfc="none", color=c, label=f"{m} (low SQI)")
    a.set(xlabel="window centre (s)", ylabel="breaths/min", title=f"Rate per {win_s:.0f} s window",
          ylim=(0, 55))
    a.legend(loc="upper left", fontsize=8, ncol=2)

    # (c) spectrum of the middle complete window, (d) its SVD spatial map
    mid = [r for r in rows if r["complete"]]
    r = mid[len(mid) // 2]
    Xw = X[:, r["start"]:r["stop"]].astype(np.float64)
    v, u, k, var = breathing.svd_breathing(Xw, fs)
    a = ax[1, 0]
    for sig, lab, c in ((Xw.mean(0), "region mean", "C0"), (v, f"SVD component {k + 1} ({var:.0%} of variance)", "C3")):
        f, p = breathing.spectrum(sig, fs)
        a.plot(60 * f, p / p.max(), color=c, label=lab)
    a.axvspan(60 * breathing.BAND[0], 60 * breathing.BAND[1], color="0.9", zorder=-1)
    if rate is not None:
        a.axvline(r["rr_true"], color="k", ls="--", lw=1, label="true rate")
    a.set(xlabel="breaths/min", ylabel="normalised power", xlim=(0, 60),
          title=f"Spectrum, window {r['t_start']:.0f}-{r['t_end']:.0f} s")
    a.legend(fontsize=8)

    a = ax[1, 1]
    img = np.full(mask.shape, np.nan)
    img[mask] = u
    lim = np.nanmax(np.abs(img))
    im = a.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set(title=f"SVD component {k + 1}: where it is strong (same window)", xticks=[], yticks=[])
    fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_video(path, lep, lms, canv, mask, idx, fs, win_s):
    """Left: Lepton frame with the region outline. Right: aligned region minus its moving 30 s mean."""
    n = len(lep)
    half = int(win_s * fs / 2)
    c = np.nan_to_num(canv, nan=np.nanmean(canv))
    cs = np.concatenate([np.zeros((1,) + c.shape[1:]), np.cumsum(c, 0)])
    lut = (plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256))[:, :3][:, ::-1] * 255).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fs, (1280, 480))
    rng = 0.5
    for k in range(n):
        a, b = max(0, k - half), min(n, k + half)
        diff = c[k] - (cs[b] - cs[a]) / (b - a)
        g = np.clip((diff / rng + 1) * 127.5, 0, 255).astype(np.uint8)
        right = lut[g]
        right[~mask] = (right[~mask] * 0.35).astype(np.uint8)
        scale = min(600 / mask.shape[1], 440 / mask.shape[0])
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((480, 640, 3), np.uint8)
        y0, x0 = (480 - right.shape[0]) // 2, (640 - right.shape[1]) // 2
        panel[y0:y0 + right.shape[0], x0:x0 + right.shape[1]] = right
        cv2.putText(panel, f"aligned region minus {win_s:.0f} s mean, +-{rng} C", (10, 25), 0, 0.6, (255, 255, 255), 2)

        left = cv2.applyColorMap(sim1.to_uint8(lep[k], 24, 34), cv2.COLORMAP_INFERNO)
        left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_NEAREST)
        if not np.isnan(lms[k]).any():
            cv2.polylines(left, [np.round(lms[k][NOSE_LIP_ROI] * 4).astype(np.int32)], True, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(left, "Lepton-like 160x120", (10, 25), 0, 0.65, (255, 255, 255), 2)
        cv2.putText(left, f"frame {idx[k]}  t={k / fs:.1f}s", (10, 465), 0, 0.6, (255, 255, 255), 2)
        writer.write(np.hstack([left, panel]))
    writer.release()


if __name__ == "__main__":
    main()
