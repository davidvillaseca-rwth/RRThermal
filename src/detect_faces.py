"""Run T-FAKE face detection + 70 landmarks on a SIM1 recording and save the results.

For each frame the 640x480 thermal image is downsampled to Lepton-like 160x120
(and optionally also kept at native resolution), mapped to 8-bit with a fixed
temperature range, and passed to the T-FAKE detector and landmarker.

Outputs (in --out):
    <name>_faces.csv    one row per frame and resolution: found, face box, nose tip
    <name>_landmarks.npz  landmarks per resolution, shape (frames, 70, 2), NaN = no face
    <name>_faces.mp4    overlay video (only with --video)

Examples (from the project root, with .venv312):
    .venv312/Scripts/python.exe src/detect_faces.py data/SIM1/T001/raw_thermal/T001-PD.dat --start 2090 --seconds 30 --video
    .venv312/Scripts/python.exe src/detect_faces.py data/SIM1/T001/raw_thermal/T001-PD.dat --res lepton
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import sim1
from tfake import JAWLINE, NOSE_LIP_ROI, NOSE_LOWER, NOSE_TIP, TFakeLandmarker

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dat", type=Path, help="SIM1 .dat file (its .inf must sit next to it)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--seconds", type=float, help="length to process (default: whole recording)")
    ap.add_argument("--res", choices=["lepton", "native", "both"], default="both")
    ap.add_argument("--range", type=float, nargs=2, default=(24.0, 34.0), metavar=("LO", "HI"),
                    help="temperature range in C mapped to black..white (default 24 34)")
    ap.add_argument("--video", action="store_true", help="also write an overlay video")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "face_detection")
    args = ap.parse_args()

    frames = sim1.open_thermal(args.dat)
    n = len(frames) - args.start if args.seconds is None else int(round(args.seconds * sim1.FPS))
    idx = range(args.start, min(args.start + n, len(frames)))
    if not idx:
        ap.error(f"no frames to process: --start {args.start} with {len(frames)} frames "
                 f"(valid start 0-{len(frames) - 1}, --seconds must be > 0)")
    resolutions = ["native", "lepton"] if args.res == "both" else [args.res]
    lo, hi = args.range

    partial = args.start > 0 or idx[-1] < len(frames) - 1
    name = args.dat.stem + (f"_f{args.start}-{idx[-1]}" if partial else "")
    args.out.mkdir(parents=True, exist_ok=True)
    model = TFakeLandmarker(n_landmarks=70)
    landmarks = {r: np.full((len(idx), 70, 2), np.nan, np.float32) for r in resolutions}
    writer = None
    if args.video:
        writer = cv2.VideoWriter(str(args.out / f"{name}_faces.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                 sim1.FPS, (640 * len(resolutions), 480))

    rows, t0 = [], time.time()
    for k, i in enumerate(idx):
        frame_c = sim1.to_celsius(frames[i])
        panels = []
        for r in resolutions:
            img = frame_c if r == "native" else sim1.to_lepton(frame_c)
            gray = sim1.to_uint8(img, lo, hi)
            model.upscale = 1 if r == "native" else 4   # detector needs ~640 px input
            lm, _ = model.predict(gray, None)
            row = {"frame": i, "time_s": round(i / sim1.FPS, 3), "res": r, "found": lm is not None}
            if lm is not None:
                landmarks[r][k] = lm
                (x0, y0), (x1, y1) = lm.min(0), lm.max(0)
                row.update(box_x0=round(x0, 1), box_y0=round(y0, 1), box_x1=round(x1, 1), box_y1=round(y1, 1),
                           nose_x=round(lm[NOSE_TIP[70], 0], 1), nose_y=round(lm[NOSE_TIP[70], 1], 1))
            rows.append(row)
            if writer is not None:
                panels.append(draw(gray, lm, r, i))
        if writer is not None:
            writer.write(np.hstack(panels))
        if k % 250 == 0:
            print(f"frame {i} ({k + 1}/{len(idx)}), {time.time() - t0:.0f} s", flush=True)

    if writer is not None:
        writer.release()
    fields = ["frame", "time_s", "res", "found", "box_x0", "box_y0", "box_x1", "box_y1", "nose_x", "nose_y"]
    with open(args.out / f"{name}_faces.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    np.savez(args.out / f"{name}_landmarks.npz", frames=np.array(list(idx)), fps=sim1.FPS, **landmarks)

    for r in resolutions:
        found = sum(row["found"] for row in rows if row["res"] == r)
        print(f"{r}: face found in {found}/{len(idx)} frames")
    print(f"saved to {args.out}")


def draw(gray: np.ndarray, lm: np.ndarray | None, res: str, frame: int) -> np.ndarray:
    """Colour-mapped 640x480 panel: jawline outline, landmarks (nose points in cyan) and the
    analysed nose/upper-lip region filled semi-transparent."""
    vis = cv2.resize(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), (640, 480),
                     interpolation=cv2.INTER_NEAREST)
    label = "native 640x480" if res == "native" else "Lepton-like 160x120"
    if lm is None:
        label += ": NO FACE"
    else:
        p = lm * (640 / gray.shape[1])
        roi = np.round(p[NOSE_LIP_ROI]).astype(np.int32)
        fill = vis.copy()
        cv2.fillPoly(fill, [roi], (255, 255, 0), cv2.LINE_AA)
        vis = cv2.addWeighted(fill, 0.45, vis, 0.55, 0)
        cv2.polylines(vis, [roi], True, (255, 255, 0), 1, cv2.LINE_AA)
        jaw = np.round(p[JAWLINE]).astype(np.int32)
        cv2.polylines(vis, [jaw], False, (0, 255, 0), 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(p):
            nose = j in NOSE_LOWER[70]
            cv2.circle(vis, (int(x), int(y)), 3 if nose else 2, (255, 255, 0) if nose else (0, 255, 0), -1)
    cv2.putText(vis, label, (10, 25), 0, 0.65, (255, 255, 255), 2)
    cv2.putText(vis, f"frame {frame}  t={frame / sim1.FPS:.1f}s", (10, 465), 0, 0.6, (255, 255, 255), 2)
    return vis


if __name__ == "__main__":
    main()
