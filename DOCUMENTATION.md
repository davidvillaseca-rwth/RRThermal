# RRThermal technical documentation

Current scope: **face detection on thermal video**, plus a **respiratory-rate prototype
tested on synthetic breathing** added to real SIM1 frames. Real respiratory-rate results
wait for recordings with a synchronized breathing reference (the lab's Lepton 3.5 with a
chest belt). The project goal and research questions are in [README.md](README.md); the
planned pipeline is slide 6 of `project_proposal_presentation_2026-10-25.pptx`.

## Folder layout

```
RRThermal/
├── README.md                  project description (from the brief)
├── DOCUMENTATION.md           this file
├── CLAUDE.md                  instructions for Claude Code
├── requirements.txt           Python packages (installed in .venv312)
├── .gitignore                 keeps .venv312, data/ and output videos out of git
├── contactless_vital_sign_detection_project_brief.pdf
├── project_proposal_presentation_2026-10-25.pptx
├── Thermal_RR_Project_Plan.xlsx
├── src/
│   ├── sim1.py                read SIM1 raw thermal files, convert to °C and to 160×120
│   ├── tfake.py               T-FAKE face detector + landmarker wrapper, region definitions
│   ├── detect_faces.py        command-line face detection on a recording
│   ├── synth.py               synthetic breathing added below the nostrils (for testing)
│   ├── breathing.py           region alignment, mean / SVD waveform, spectral rate, SQI
│   ├── rr_prototype.py        command-line respiratory-rate prototype
│   ├── plot_compare.py        figure: original vs real vs synthetic, side by side
│   └── pipeline_figure.py     presentation figure of the pipeline on synthetic data (SVG)
├── data/
│   └── SIM1/
│       ├── Dataset-Table-Index.xlsx
│       └── T001/
│           ├── raw_thermal/   T001-PD.dat + .inf (thermal video, practice drive)
│           └── structured/    per-drive signals and ordinary videos (see below)
└── outputs/
    ├── face_detection/        results of detect_faces.py
    ├── breathing/             results of rr_prototype.py and plot_compare.py
    └── pipeline/              pipeline_synthetic.svg/.png from pipeline_figure.py
```

## Setup

Everything runs in the Python 3.12 environment `.venv312` (T-FAKE needs Python 3.10–3.12).
The existing one was made with Python 3.12.13 from `uv`. To recreate it:

```bash
uv venv --python 3.12 .venv312
uv pip install --python .venv312 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

The extra index is needed because `requirements.txt` pins the CPU builds of torch and
torchvision (`+cpu`), which are not on PyPI. `--index-strategy unsafe-best-match` is needed
because the PyTorch index also carries old copies of ordinary packages (such as `requests`),
and without it `uv` refuses to look on PyPI for a newer one. All pinned versions match what
is installed in `.venv312`.

If `.venv312/Scripts/python.exe` answers "No Python at ...", the base interpreter the
environment was created from is no longer at that path (this happens when it was created
from inside the Claude desktop app, which keeps `uv`'s Python in its own storage). Delete
`.venv312` and recreate it with the two commands above from an ordinary terminal.

The T-FAKE weights download to `~/.tfan/models` on first use; after that, face detection
runs without a network connection. The package itself would also fetch ImageNet weights for
its MobileNetV2 backbone from the Hugging Face hub every time it loads. `src/tfake.py`
skips that download, because the T-FAKE checkpoint replaces all of those weights anyway
(checked: all 314 keys present, none missing).

Run scripts from the project root. To use the modules in your own code, put `src/` on the
path first, for example `sys.path.insert(0, "src")`.

## Data: SIM1, participant T001

**Source.** Taamneh et al., *A multimodal dataset for various forms of distracted
driving*, Scientific Data 2017. Public at [osf.io/c42cn](https://osf.io/c42cn). 68 participants
drove a simulator under different conditions while a FLIR Tau 640 filmed their face.

**Drive codes.** `BL` baseline, `PD` practice drive, `RD` relaxing drive, `ND` normal
drive, `CD` cognitive distraction, `ED` emotional distraction, `MD` motoric (texting)
distraction, `FDN`/`FDL` failure drive (startle event). These meanings come from the SIM1
paper as remembered, not from a file in the dataset; `Dataset-Table-Index.xlsx` lists the
codes without defining them, and the `N`/`L` suffix is unknown. Only `T001-PD` is downloaded
as raw thermal video. The other T001 `.dat` files (1.2–4.5 GB each) are on OSF under
*Raw Thermal Data/T001*.

### Raw thermal files (`data/SIM1/T001/raw_thermal/`)

| File | Format |
|---|---|
| `T001-PD.inf` | 3 text lines: last frame index (4404), width (640), height (512) |
| `T001-PD.dat` | Headerless little-endian `uint16`, frames back to back: 4,405 frames × 640×512 |

These facts were worked out from the files, not from dataset documentation:

- Values are **thousandths of a degree Celsius** (background about 26 °C, face about 34 °C, steps of 0.04 °C).
- Rows 480–511 are constant padding, so the **real image is 640×480**. That is exactly 4× the FLIR Lepton 3.5 (160×120).
- **Frame rate is uncertain, probably 7.5 fps.** The `.pp` timestamps step by 0.13343 s (7.49 fps). 4,405 frames against the 504 s chest-strap record would mean about 8.7 fps, but that assumes both cover the same time span, which nothing confirms; at 7.5 fps the `.dat` would simply be about 85 s longer (587 s). `sim1.FPS` is set to 7.5 until this is resolved.

```python
import sim1
frames = sim1.open_thermal("data/SIM1/T001/raw_thermal/T001-PD.dat")   # (4405, 480, 640) uint16
frame_c = sim1.to_celsius(frames[2090])                                  # °C, 640×480
lepton  = sim1.to_lepton(frame_c)                                        # °C, 160×120
```

### Structured files (`data/SIM1/T001/structured/<n> <drive>/`)

Checked on the PD drive (`2 PD`); the other drives have the same file types.

| Extension | Contents |
|---|---|
| `.avi1/.avi2/.avi3.avi` | Ordinary colour video, DivX: 640×480 at 27 fps, 640×480 at 25 fps, 1280×720 at 10 fps. **Not thermal** |
| `.pp` | Excel `.xlsx` despite the name. A title row "Perinasal Perspiration", then columns `Frame#, Time, Perinasal EDA, NR Perinasal EDA`, one row per thermal frame (0.133 s steps). Extracted from the thermal video by the dataset authors |
| `.BR`, `.HR` | Old-format Excel `.xls`. Columns `Frame, Time, Breathing Rate` (or heart rate), one value per second from the chest strap |
| `.peda` | Palm skin conductance (from the dataset description; not opened) |
| `.FACS` | Facial action codes (from the file name; not opened) |
| `.res`, `.stm` | Not opened; meaning unknown |

`.BR` is the breathing reference for later work. The baseline drive (`1 BL`) has no `.BR`
or `.HR`. Nothing in `.venv312` reads old `.xls` files, so reading `.BR` needs a package
such as `xlrd`.

For PD, the `.dat` has 4,405 frames, while `.pp` has 3,763 frames (502 s) and `.BR` 504 s.
How the thermal frames line up with them in time is not yet known.

## Face detection

**Method.** T-FAKE (Flotho et al., CVPR 2025; pip package `thermal-face-alignment`): a YOLO5Face
detector trained on thermal faces (TFW), followed by a 70-point landmark regressor.
`src/tfake.py` wraps it for low-resolution input: 160×120 frames are enlarged 4× before
detection, because the detector misses faces at native Lepton size.

**Per frame.**
1. Convert the raw frame to °C and drop the padding rows (640×480).
2. Downsample to 160×120 by area averaging (Lepton-like). Optionally also keep the native 640×480.
3. Map a fixed range (default 24–34 °C) to 8-bit grayscale.
4. Run T-FAKE. The face box is the bounding box of the 70 landmarks.

**Run it** (from the project root):

```bash
.venv312/Scripts/python.exe src/detect_faces.py data/SIM1/T001/raw_thermal/T001-PD.dat --start 2090 --seconds 30 --video
```

| Option | Meaning |
|---|---|
| `--start N` | First frame (default 0) |
| `--seconds S` | Length to process (default: whole recording). The whole T001-PD at one resolution took about 5 min on CPU, so `both` takes longer |
| `--res lepton\|native\|both` | Which resolution(s) to run (default both) |
| `--range LO HI` | Temperature range mapped to black..white (default 24 34) |
| `--video` | Also write an overlay video |
| `--out DIR` | Output folder (default `outputs/face_detection`) |

**Outputs** (in `outputs/face_detection/`). Files are named `<recording>_f<first>-<last>_*`
for part of a recording, and `<recording>_*` for a whole one.

| File | Contents |
|---|---|
| `*_faces.csv` | One row per frame and resolution: `frame, time_s, res, found, box_x0, box_y0, box_x1, box_y1, nose_x, nose_y`. Coordinates are in that resolution's pixels; empty when no face |
| `*_landmarks.npz` | Arrays `native` / `lepton` of shape (frames, 70, 2), NaN when no face. Also `frames` and `fps` |
| `*_faces.mp4` | Overlay video (with `--video`). Jawline outline and landmarks in green, nose and nostril points in cyan, and the analysed region under the nose including the upper lip (`tfake.NOSE_LIP_ROI`) filled semi-transparent cyan. The face box is still written to the CSV but not drawn. H.264, written at `sim1.FPS` |

Current contents:

| File | Made by |
|---|---|
| `T001-PD_f2090-2314_faces.csv`, `_landmarks.npz`, `_faces.mp4` | The example command above |
| `T001-PD_f2090-2989_faces.csv`, `_landmarks.npz` | Same with `--seconds 120`, no video; input for the breathing prototype |
| `T001-PD_f2090-2314_raw.mp4` | An earlier one-off script that is not in `src/`: the same 30 s at native and 160×120 side by side, colour range 24.1–33.9 °C, no detection |

Landmark layout: iBUG 68 points plus 2 pupils. Index 30 is the nose tip, 31–35 the nostril base.

## Respiratory-rate prototype (synthetic breathing)

Covers pipeline steps 5–7 of the proposal (waveform, band-pass, RR + SQI) so they can be
built and tested before the camera arrives. Step 4 (clean: FFC and outlier masking) is
not implemented yet.

**Synthetic breathing** (`src/synth.py`). Not a physical model, only a known test signal.
A smooth warm/cool pulse is added to each native 640×480 frame before downsampling, as two
small elliptical Gaussian bulbs, one just below each nostril, where exhaled air meets the
skin. Each bulb is centred on a nostril landmark (32, 34) moved a third of the way towards
the upper lip below it (50, 52), is about a fifth of the nose width wide, and turns with
the face; the two stay separate (peak weight 1 in each). Its time course
is a periodic waveform (sine plus a small second harmonic) whose rate follows a schedule,
by default 12, 18 and 24 breaths/min over equal thirds of the clip, with a slow ±5 % rate
variation. Default peak amplitude 0.3 °C. The face positions come from detection on the
unmodified frames.

**Processing** (`src/breathing.py`), on the Lepton-like 160×120 frames:
1. **Align.** Each frame's region (`tfake.NOSE_LIP_ROI`) is warped onto one fixed canvas
   with a similarity transform fitted to the nose and outer upper-lip landmarks (27–35,
   48–54), so a canvas pixel is the same patch of skin in every frame. On T001 this gives
   290 pixels on a 28×24 canvas. Result: a pixel × time matrix.
2. **Waveform per window** (30 s, 5 s step), two methods:
   - `mean`: average of all region pixels.
   - `svd`: remove each pixel's linear trend (static pattern and slow drift), take the SVD
     of the pixel × time matrix, and keep the one of the first 5 components whose time
     course has the most breathing-band power concentrated in one peak.
3. **Rate**: peak of the spectrum inside 0.1–0.85 Hz (6–51 breaths/min). Hann-windowed
   periodogram of the whole window (Welch with one segment), zero-padded.
4. **SQI**: share of in-band power in the peak's main lobe (±2/T Hz); about 1 for a clean
   rhythm, about 0.2 for noise. Default threshold 0.5.

**Run it** (needs the landmarks of the clip from `detect_faces.py --res both`):

```bash
.venv312/Scripts/python.exe src/rr_prototype.py data/SIM1/T001/raw_thermal/T001-PD.dat outputs/face_detection/T001-PD_f2090-2989_landmarks.npz --simulate --video
```

Without `--simulate` it runs on the real frames. Other options: `--rates`, `--amplitude`,
`--irregularity`, `--seed`, `--window`, `--step`, `--sqi-min`, `--out`
(default `outputs/breathing`).

**Outputs** (in `outputs/breathing/`, named `<clip>_<sim|real>_*`):

| File | Contents |
|---|---|
| `*_windows.csv` | One row per window: `t_start, t_end, complete, rr_true` (synthetic only), `rr_mean, sqi_mean, valid_mean, rr_svd, sqi_svd, valid_svd, svd_comp, svd_var` |
| `*_summary.png` | Band-passed region mean (with the injected waveform), rate per window, spectrum of the middle window, spatial map of the chosen SVD component |
| `*_roi.mp4` | With `--video`: Lepton-like frame with the region outline, next to the aligned region minus its moving 30 s mean (±0.5 °C, red = warmer) |
| `<clip>_compare.png` | From `src/plot_compare.py` (same two arguments as above, plus `--rates`, `--amplitude`, `--irregularity`, `--seed`, `--sqi-min`, `--out`). Three columns: original (region mean as recorded, unfiltered), real and synthetic (both detrended and band-passed, region mean and SVD, shared y-axis). Rows: whole-clip signal, whole-clip spectrum, rate per window (original column: one frame with the region). In the processed columns the SVD trace is rescaled to the mean's spread for display |

**Results on T001-PD frames 2090–2989** (120 s, 18 windows, `sim1.FPS` = 7.5):

| Data | Method | Valid windows | MAE | Bias | Max error (breaths/min) |
|---|---|---|---|---|---|
| Synthetic, 0.3 °C, 12/18/24 | mean | 6/18 | 6.62 | −6.39 | 13.85 |
| Synthetic, 0.3 °C, 12/18/24 | svd | 14/18 | 1.45 | −0.71 | 9.81 |
| Real (no injection) | mean | 6/18 | – | – | – |
| Real (no injection) | svd | 5/18 | – | – | – |

- With the two small nostril bulbs the injected signal covers only a small part of the
  region, so averaging all 290 pixels dilutes it below the drift and motion: the mean
  mostly fails (rates stuck near 6 breaths/min). The SVD still finds it in most windows.
  An earlier version with one larger pulse under the nose gave MAE 1.17 (mean) and 0.72
  (SVD).
- The SVD's largest error is the 71–101 s window (12.0 against a true 21.8 breaths/min),
  which spans a change of rate and still passed the SQI threshold (0.52). The neighbouring
  windows were rejected. The threshold needs tuning.
- The mean is pulled off by slow drift and by two sharp drops.
- Those drops (about 14 s and 33 s) are in the real data too. The nose landmarks jump
  7–11 px in one frame there, against a typical 0.3 px, so they are fast head movements or
  detection glitches; the whole frame changes by under 0.08 °C, so they are not FFC events.
  This is what the planned cleaning step must flag.
- On the real frames neither method finds a believable rate: estimates cluster at the
  6 breaths/min band edge, and the SVD map sits on the lip line (likely lip or jaw
  movement). One SVD window passes the SQI threshold at 45 breaths/min, so the threshold
  needs tuning on real data.
- Synthetic results depend only on `sim1.FPS` being used consistently; they say nothing
  about the 7.5 vs 8.7 fps question.

## Results so far

| Recording | Frames | Native 640×480 | Lepton-like 160×120 |
|---|---|---|---|
| T001-PD, frames 2090–2314 (30 s) | 225 | 225 found | 225 found |
| T001-PD, whole drive (earlier run, Lepton-like only, same settings; its output files were deleted) | 4,405 | not run | 4,401 found |

Checked by eye only: the eye, nose, mouth and nostril points sit in the right places at
both resolutions, and the jawline stops slightly above the chin. There are no
hand-labelled positions to measure accuracy against.

## Pipeline figure (synthetic data)

`outputs/pipeline/pipeline_synthetic.svg` is a one-page figure of the pipeline for
presentations, made by `src/pipeline_figure.py`. Each of its eight panels shows the actual
data at that step. A PNG copy (`pipeline_synthetic.png`) sits next to it for quick viewing.
In the SVG, text stays as editable text (font Arial).

```bash
.venv312/Scripts/python.exe src/pipeline_figure.py data/SIM1/T001/raw_thermal/T001-PD.dat outputs/face_detection/T001-PD_f2090-2989_landmarks.npz
```

Options: `--rates`, `--amplitude`, `--irregularity`, `--seed`, `--sqi-min` (as for
`rr_prototype.py`), `--example` (centre of the window shown in panels 4, 6 and 7, default
60 s) and `--out` (default `outputs/pipeline`).

**Hero version.** With `--hero` the same script writes `pipeline_hero.svg` (and `.png`)
instead: the same eight steps in one row of square panels, with only a title, a subtitle,
the step names, two numbers (the example window's rate and the mean error) and a one-line
disclosure. No axes, captions or tick labels. It is meant as the opening image of a
presentation; the annotated figure above is the one to explain the steps with. The text
was edited for plain wording:

- Title: "Respiratory rate from thermal video"
- Subtitle: "How a thermal video becomes a breathing rate, shown on simulated breathing."
- Steps: Thermal frame · Find the face · Add simulated breathing · Align the nose region ·
  Raw temperature · Isolate breathing · Read the rate · Compare with the true rate
- Disclosure: "The breathing is simulated: a small warm pulse (up to 0.3 °C) was added below
  each nostril of a real thermal recording (SIM1 dataset, participant T001). These results
  test the processing; they are not a measurement of anyone's breathing."

Step 3 is in red in both versions, so the simulated part stands out.

### Disclosure: the breathing in this figure is synthetic

Wherever this figure (or any result from `synth.py`) is shown, it must be stated that the
breathing signal is simulated. The figure carries this in its subtitle, in red on step 3,
and in its footer. Suggested wording for slides or reports:

> The pipeline is demonstrated on synthetic data. Frames 2090–2989 (120 s) of participant
> T001, practice drive, from the public SIM1 distracted-driving dataset (Taamneh et al.,
> Scientific Data 2017; FLIR Tau 640, downsampled to the 160×120 resolution of a FLIR
> Lepton 3.5) were used as background. No breathing signal could be found in these frames,
> so a smooth temperature pulse of at most 0.3 °C, at 12, 18 and 24 breaths/min with ±5 %
> slow variation, was added just below each nostril in software. This is a test signal, not a model
> of real breathing. The results show that the processing works as intended; they do not
> show that breathing can be measured from thermal video. Steps not yet implemented:
> selecting the closest person and masking camera recalibration (FFC) and motion outliers.
> A frame rate of 7.5 fps was assumed.

What is real and what is not:

| Part | Real or synthetic |
|---|---|
| Thermal frames, face, head movement, noise, drift | Real (SIM1 recording) |
| Face landmarks | Real detector output (on the frames without the pulse) |
| Breathing pulse below the nostrils, its rates and the "truth" in step 8 | Synthetic |
| Rate estimates, SQI, MAE | Computed by the pipeline, but on synthetic breathing |

### The steps in the figure

1. **Load recording.** The camera file is read and converted to a temperature in °C for
   every pixel. SIM1 frames (640×480) are downsampled to 160×120 by area averaging, the
   resolution of the Lepton 3.5 the lab will use.
2. **Detect face + region.** The thermal face detector T-FAKE finds the face and 70
   landmarks in every frame. Ten of them outline the analysed region: nose tip, nostrils
   and upper lip (`tfake.NOSE_LIP_ROI`). The green line is the jawline.
3. **Add synthetic breathing (simulation only).** A smooth warm/cool pulse is added to the
   temperatures of every frame as two small bulbs, one just below each nostril, moving with
   the face. Its rate is known
   (12, then 18, then 24 breaths/min), so the result can be checked. With real recordings
   this step is left out.
4. **Align region.** The region is warped onto the same fixed grid in every frame using the
   nose and lip landmarks, so each grid pixel is the same patch of skin over time. The
   panel shows the average aligned region over the example window (290 pixels; cold nose
   at the top). Each 30 s window gives a table of 290 pixels × 225 frames.
5. **Raw waveform.** The average temperature of the region over time. The breathing is
   hard to see: it is small compared with the static temperature level, slow drift and
   the sharp drops caused by fast head movements. The yellow band is the example window.
6. **Isolate breathing.** For each window, each pixel's average and linear trend are
   removed. An SVD then splits the pixel changes into independent patterns (where × when),
   and the one most like breathing is kept: most power in the breathing band, in a single
   peak. It is band-pass filtered to 6–51 breaths/min. The panel compares it with the
   injected waveform (both scaled to the same height).
7. **Rate + quality (SQI).** The peak of the spectrum gives the rate. The SQI is the share
   of the breathing-band power that lies in the peak (shaded); about 1 for a clean rhythm,
   about 0.2 for noise. Windows with SQI below 0.5 are rejected. Example window 46–76 s:
   18.5 breaths/min against a true 18.3, SQI 0.66.
8. **Compare with truth.** A rate every 5 s from 30 s windows, against the injected rate.
   Here: 14 of 18 windows valid, mean absolute error 1.4 breaths/min. The rejected and
   least accurate windows span a change of rate; one of them (71–101 s, 12.0 against
   21.8) still passed the SQI threshold. With real recordings, the comparison is against
   the chest-belt reference.

## Known limitations and open points

- One participant, one drive, mostly facing the camera. Drives with head movement (`MD`, `FD*`) and talking (`CD`, `ED`) are untested.
- The frame rate (7.5 or 8.7 fps) and the time offset between `.dat` and `.BR` must be settled before any breathing-rate comparison.
- No breathing signal has been found in the real T001-PD frames: neither a first nostril test nor the prototype above (mean and SVD) finds one, and none was visible by eye in a nose close-up.
- `tfake.py` keeps only the first face the detector returns, so the planned "closest person" selection needs it to return all faces first.
- The SIM1 images come from a FLIR Tau 640, not a Lepton. Downsampling matches the resolution but not the Lepton's noise, lens or field of view.
