"""Presentation figure of the processing pipeline, run on SYNTHETIC breathing, as SVG (+ PNG preview).

Eight panels, one per step, each with a small graph made from the actual data passing
through that step:
    1 Load recording      2 Detect face + region   3 Add synthetic breathing   4 Align region
    5 Raw waveform        6 Isolate breathing      7 Rate + quality (SQI)      8 Compare with truth

The breathing signal is synthetic (see synth.py): it is added to real SIM1 T001 thermal
frames, and the figure footer says so. Steps 2 "select closest person" and 4 "clean"
(FFC/outlier masking) of the planned pipeline are not implemented and are not shown.

Example (from the project root):
    .venv312/Scripts/python.exe src/pipeline_figure.py data/SIM1/T001/raw_thermal/T001-PD.dat \
        outputs/face_detection/T001-PD_f2090-2989_landmarks.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import breathing
import sim1
import synth
from tfake import JAWLINE, NOSE_LIP_ROI

ROOT = Path(__file__).resolve().parent.parent
RED, BLUE, GREY, CYAN = "#c0392b", "#2c7fb8", "#7f7f7f", "#17becf"

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dat", type=Path)
    ap.add_argument("landmarks", type=Path, help="*_landmarks.npz from detect_faces.py --res both")
    ap.add_argument("--rates", type=float, nargs="+", default=[12, 18, 24])
    ap.add_argument("--amplitude", type=float, default=0.3)
    ap.add_argument("--irregularity", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sqi-min", type=float, default=0.5)
    ap.add_argument("--example", type=float, default=60, help="centre (s) of the example window (default 60)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "pipeline")
    ap.add_argument("--hero", action="store_true",
                    help="minimal-text single-row version (pipeline_hero.svg) instead of the annotated one")
    args = ap.parse_args()

    # ---------------------------------------------------------------- run the pipeline
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

    X, mask, valid, canv = breathing.warp_rois(simu, lm_lep)
    rows = breathing.analyse(X, valid, fs, sqi_min=args.sqi_min)
    for r in rows:
        r["rr_true"] = float(rate[r["start"]:r["stop"]].mean())
    ex = min(rows, key=lambda r: abs((r["t_start"] + r["t_end"]) / 2 - args.example))
    s0, s1 = ex["start"], ex["stop"]
    Xw = X[:, s0:s1].astype(np.float64)
    v, _, comp, var = breathing.svd_breathing(Xw, fs)
    v_bp = breathing.bandpass(v, fs)
    f, p = breathing.spectrum(v, fs)
    rr, sqi, _ = breathing.spectral_rate(v, fs)
    ok = [r for r in rows if r.get("valid_svd")]
    err = np.array([r["rr_svd"] - r["rr_true"] for r in ok])
    t = np.arange(n) / fs
    k = int(round(args.example * fs))                       # example frame inside the example window
    box = face_box(lm_lep[k])

    if args.hero:
        hero(args, fs, real, lm_lep, k, box, wave, rate, canv, mask, X, s0, s1, v_bp, f, p, rr, rows, err, t)
        return

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, left=0.035, right=0.975, top=0.86, bottom=0.2, wspace=0.32, hspace=0.6)
    ax = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    fig.text(0.035, 0.955, "Respiratory rate from thermal video: processing pipeline", fontsize=17, weight="bold")
    fig.text(0.035, 0.918, "Demonstrated on SYNTHETIC breathing added to a real thermal recording "
             "(see footer). Lepton-like 160×120 frames, 7.5 fps, 120 s clip.", fontsize=11, color="#444")

    # 1 load
    a = ax[0]
    a.imshow(real[k], cmap="inferno", vmin=24, vmax=34)
    img_axes(a, "1  Load recording")
    caption(fig, a, "Raw camera file → temperature\nin °C per pixel, downsampled to\n160×120 (FLIR Lepton 3.5 size).")

    # 2 detect face + region
    a = ax[1]
    a.imshow(real[k], cmap="gray", vmin=24, vmax=34)
    a.plot(*lm_lep[k][JAWLINE].T, color="#2ecc71", lw=1.5)
    a.plot(*lm_lep[k].T, ".", color="#2ecc71", ms=2)
    a.fill(*lm_lep[k][NOSE_LIP_ROI].T, fc=CYAN, ec=CYAN, alpha=0.55, lw=1)
    a.set(xlim=box[:2], ylim=box[2:][::-1])
    img_axes(a, "2  Detect face + region")
    caption(fig, a, "Thermal face detector (T-FAKE)\nfinds 70 landmarks per frame; they\ndefine the nose / upper-lip region.")

    # 3 add synthetic breathing
    a = ax[2]
    a.imshow(real[k], cmap="gray", vmin=24, vmax=34)
    wmap = synth.weight_map(real[k].shape, lm_lep[k])
    a.imshow(np.ma.masked_less(wmap, 0.05), cmap="Reds", vmin=0, vmax=1, alpha=0.85)
    a.set(xlim=box[:2], ylim=box[2:][::-1])
    img_axes(a, "3  Add synthetic breathing", color=RED)
    ins = a.inset_axes([0.05, 0.04, 0.9, 0.22])
    ins.plot(t[:int(20 * fs)], args.amplitude * wave[:int(20 * fs)], color=RED, lw=1)
    ins.axis("off")
    caption(fig, a, f"SIMULATION ONLY: ±{args.amplitude:g} °C pulse, one bulb\nunder each nostril, at {'/'.join(f'{r:g}' for r in args.rates)} "
            "breaths/min,\nadded to every frame (inset: first 20 s).", color=RED)

    # 4 align region
    a = ax[3]
    m = np.nanmean(canv[s0:s1], axis=0)
    shown = np.where(mask, m, np.nan)
    a.imshow(shown, cmap="inferno", interpolation="nearest")
    a.contour(mask, levels=[0.5], colors=CYAN, linewidths=1)
    img_axes(a, "4  Align region")
    caption(fig, a, f"Region warped to a fixed grid in\nevery frame: {mask.sum()} pixels × {s1 - s0} frames\nper 30 s window.")

    # 5 raw waveform
    a = ax[4]
    raw = X.mean(0)
    a.plot(t, raw, color=GREY, lw=0.8)
    a.axvspan(s0 / fs, s1 / fs, color="#f1c40f", alpha=0.25, lw=0)
    a.text(s0 / fs + 1, raw.max(), "example\nwindow", fontsize=8, va="top")
    a.set(xlabel="time (s)", ylabel="°C")
    step_title(a, "5  Raw waveform")
    caption(fig, a, "Mean region temperature: breathing\nis hidden under the static level,\nslow drift and head movement.")

    # 6 isolate breathing
    a = ax[5]
    tw = t[s0:s1]
    truth = args.amplitude * wave[s0:s1]
    a.plot(tw, truth / np.abs(truth).max(), color="k", lw=1, ls="--", label="injected (truth)")
    a.plot(tw, v_bp / np.abs(v_bp).max(), color=RED, lw=1.4, label="recovered")
    a.set(xlabel="time (s)", ylabel="normalised", ylim=(-1.6, 1.6))
    a.legend(loc="upper right", fontsize=7.5, ncol=2, frameon=False)
    step_title(a, "6  Isolate breathing")
    caption(fig, a, "Per pixel: remove level + trend. SVD\nsplits the pixels into patterns; keep\nthe breathing-like one, band-pass 6–51/min.")

    # 7 rate + SQI
    a = ax[6]
    a.plot(60 * f, p / p.max(), color=RED, lw=1.4)
    a.axvspan(60 * breathing.BAND[0], 60 * breathing.BAND[1], color="0.93", lw=0, zorder=-2)
    lobe = 2 * fs / (s1 - s0) * 60
    a.axvspan(rr - lobe, rr + lobe, color=RED, alpha=0.15, lw=0, zorder=-1)
    a.annotate(f"{rr:.1f} /min\nSQI {sqi:.2f}", (rr, 1.0), (rr + 8, 0.75), fontsize=9,
               arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
    a.set(xlabel="breaths/min", ylabel="power", xlim=(0, 55), ylim=(0, 1.1), yticks=[])
    step_title(a, "7  Rate + quality (SQI)")
    caption(fig, a, "Spectrum peak = rate. SQI = share of\nbreathing-band power in the peak;\nwindows below 0.5 are rejected.")

    # 8 compare
    a = ax[7]
    a.plot(t, rate, color="k", lw=1, label="injected (truth)")
    tc = np.array([(r["t_start"] + r["t_end"]) / 2 for r in rows])
    rr_s = np.array([r["rr_svd"] for r in rows])
    vs = np.array([r["valid_svd"] for r in rows])
    a.plot(tc[vs], rr_s[vs], "o", color=RED, ms=4, label="estimate")
    a.plot(tc[~vs], rr_s[~vs], "o", mfc="none", color=RED, ms=4, label="rejected")
    a.set(xlabel="time (s)", ylabel="breaths/min", ylim=(5, 30))
    a.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3)
    a.text(0.98, 0.05, f"MAE {np.abs(err).mean():.1f} /min\n{len(ok)}/{len(rows)} windows valid",
           transform=a.transAxes, ha="right", fontsize=9)
    step_title(a, "8  Compare with truth")
    caption(fig, a, "Rate every 5 s (30 s windows) vs the\ninjected rate. With real data: vs the\nchest-belt reference.")

    arrows(fig, ax)
    fig.text(0.035, 0.035,
             "Synthetic data disclosure: frames 2090–2989 of SIM1 participant T001, practice drive (Taamneh et al., Sci Data 2017; "
             "FLIR Tau 640, downsampled to 160×120). No breathing signal could be found in these frames, so a smooth\n"
             f"temperature pulse (max {args.amplitude:g} °C, rates {'/'.join(f'{r:g}' for r in args.rates)} breaths/min with "
             f"±{args.irregularity:.0%} variation) was added just below each nostril in software. It is a test signal, not a model of real "
             "breathing; results show the pipeline works, not that breathing can be measured.\n"
             "Not yet implemented: selecting the closest person (single person in view) and masking camera recalibration (FFC) and "
             "motion outliers. Frame rate 7.5 fps assumed (unresolved for SIM1).",
             fontsize=8, color="#444", va="bottom")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "pipeline_synthetic"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=110)
    print(f"example window {ex['t_start']:.0f}-{ex['t_end']:.0f} s: SVD component {comp + 1} ({var:.0%}), "
          f"rate {rr:.1f} (true {ex['rr_true']:.1f}), SQI {sqi:.2f}")
    print(f"all windows: {len(ok)}/{len(rows)} valid, MAE {np.abs(err).mean():.2f} breaths/min")
    print(f"saved {stem.with_suffix('.svg')} and .png")


HERO_STEPS = ["Thermal frame", "Find the face", "Add simulated\nbreathing", "Align the\nnose region",
              "Raw temperature", "Isolate\nbreathing", "Read the rate", "Compare with\nthe true rate"]
HERO_DISCLOSURE = ("The breathing is simulated: a small warm pulse (up to {amp:g} °C) was added below each nostril "
                   "of a real thermal recording (SIM1 dataset, participant T001). These results test the processing; "
                   "they are not a measurement of anyone's breathing.")


def hero(args, fs, real, lm_lep, k, box, wave, rate, canv, mask, X, s0, s1, v_bp, f, p, rr, rows, err, t):
    """Single row of eight square panels, step names only, one disclosure line."""
    fig = plt.figure(figsize=(20, 5.4))
    fig.text(0.02, 0.89, "Respiratory rate from thermal video", fontsize=26, weight="bold")
    fig.text(0.02, 0.81, "How a thermal video becomes a breathing rate, shown on simulated breathing.",
             fontsize=14, color="#555")
    w, gap, y0, h = 0.105, 0.0186, 0.19, 0.105 * 20 / 5.4    # square panels
    ax = [fig.add_axes([0.02 + i * (w + gap), y0, w, h]) for i in range(8)]

    sq = square(box)
    gray = dict(cmap="gray", vmin=24, vmax=34)
    ax[0].imshow(real[k], cmap="inferno", vmin=24, vmax=34)
    ax[0].set(xlim=square((20, 140, 0, 120))[:2], ylim=square((20, 140, 0, 120))[2:][::-1])

    ax[1].imshow(real[k], **gray)
    ax[1].plot(*lm_lep[k][JAWLINE].T, color="#2ecc71", lw=1.5)
    ax[1].plot(*lm_lep[k].T, ".", color="#2ecc71", ms=2)
    ax[1].fill(*lm_lep[k][NOSE_LIP_ROI].T, fc=CYAN, ec=CYAN, alpha=0.55, lw=1)

    ax[2].imshow(real[k], **gray)
    ax[2].imshow(np.ma.masked_less(synth.weight_map(real[k].shape, lm_lep[k]), 0.05), cmap="Reds",
                 vmin=0, vmax=1, alpha=0.9)
    for a in ax[1:3]:
        a.set(xlim=sq[:2], ylim=sq[2:][::-1])

    ax[3].imshow(np.where(mask, np.nanmean(canv[s0:s1], axis=0), np.nan), cmap="inferno", interpolation="nearest")
    ax[3].contour(mask, levels=[0.5], colors=CYAN, linewidths=1)

    a = ax[4]
    a.plot(t, X.mean(0), color=GREY, lw=0.9)
    a.axvspan(s0 / fs, s1 / fs, color="#f1c40f", alpha=0.3, lw=0)

    a = ax[5]
    tw, truth = t[s0:s1], wave[s0:s1]
    a.plot(tw, truth / np.abs(truth).max(), color="k", lw=1, ls="--")
    a.plot(tw, v_bp / np.abs(v_bp).max(), color=RED, lw=1.6)
    a.set_ylim(-1.5, 1.5)

    a = ax[6]
    a.fill_between(60 * f, p / p.max(), color=RED, alpha=0.15, lw=0)
    a.plot(60 * f, p / p.max(), color=RED, lw=1.6)
    a.set(xlim=(0, 55), ylim=(0, 1.25))
    a.text(rr, 1.08, f"{rr:.1f} /min", ha="center", fontsize=12, weight="bold")

    a = ax[7]
    a.plot(t, rate, color="k", lw=1)
    tc = np.array([(r["t_start"] + r["t_end"]) / 2 for r in rows])
    vs = np.array([r["valid_svd"] for r in rows])
    a.plot(tc[vs], np.array([r["rr_svd"] for r in rows])[vs], "o", color=RED, ms=4)
    a.set_ylim(5, 30)
    a.text(0.95, 0.06, f"error {np.abs(err).mean():.1f} /min", transform=a.transAxes, ha="right",
           fontsize=12, weight="bold")

    for i, a in enumerate(ax):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        if i >= 4:
            a.set_facecolor("#f4f4f4")
        pos = drawn(a)
        color = RED if i == 2 else "k"
        fig.text(pos.x0, y0 + h + 0.05, f"{i + 1}", fontsize=22, weight="bold", color=RED if i == 2 else "#bbb", va="bottom")
        fig.text(pos.x0 + 0.017, y0 + h + 0.05, HERO_STEPS[i], fontsize=13, weight="bold", color=color, va="bottom",
                 linespacing=1.15)
        if i < 7:
            q = drawn(ax[i + 1])
            fig.add_artist(FancyArrowPatch((pos.x1 + 0.003, y0 + h / 2), (q.x0 - 0.003, y0 + h / 2),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                           color="#888", lw=1.3))

    fig.text(0.02, 0.05, HERO_DISCLOSURE.format(amp=args.amplitude), fontsize=10.5, color="#666")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "pipeline_hero"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=110)
    print(f"saved {stem.with_suffix('.svg')} and .png")


def square(box):
    """Square (x0, x1, y0, y1) limits around the centre of a box, side = its larger extent."""
    x0, x1, y0, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
    return cx - s, cx + s, cy - s, cy + s


def face_box(lm, pad=0.25):
    (x0, y0), (x1, y1) = lm.min(0), lm.max(0)
    w, h = x1 - x0, y1 - y0
    return x0 - pad * w, x1 + pad * w, y0 - 0.6 * h, y1 + pad * h


def img_axes(a, title, color="k"):
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    step_title(a, title, color)


def step_title(a, title, color="k"):
    a.set_title(title, loc="left", fontsize=12, weight="bold", color=color, pad=8)


def drawn(a):
    """Axes position as drawn (image panels shrink to keep pixels square)."""
    a.apply_aspect()
    return a.get_position()


def caption(fig, a, text, color="#333"):
    pos = drawn(a)
    gap = 0.02 if a.get_xlabel() == "" else 0.055               # leave room for an x-axis label
    fig.text(pos.x0, pos.y0 - gap, text, fontsize=9, va="top", color=color, linespacing=1.35)


def arrows(fig, ax):
    """Arrows between panels: along each row, and a bent arrow from the end of row 1 to row 2."""
    kw = dict(transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color="#555", lw=1.4)
    for i in (0, 1, 2, 4, 5, 6):
        p, q = drawn(ax[i]), drawn(ax[i + 1])
        y = (p.y0 + p.y1) / 2
        end = q.x0 - (0.008 if i < 3 else 0.035)                  # row 2: stop before the y-axis label
        fig.add_artist(FancyArrowPatch((p.x1 + 0.006, y), (end, y), **kw))
    p, q = drawn(ax[3]), drawn(ax[4])
    yb = q.y1 + 0.06                                              # just above the row-2 titles
    xr = p.x1 + 0.008
    fig.add_artist(Line2D([xr, xr, q.x0 - 0.012], [(p.y0 + p.y1) / 2, yb, yb], transform=fig.transFigure,
                          color="#555", lw=1.4))
    fig.add_artist(FancyArrowPatch((q.x0 - 0.012, yb), (q.x0 - 0.012, q.y1 + 0.03), **kw))


if __name__ == "__main__":
    main()
