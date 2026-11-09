# RRThermal

Contactless respiratory-rate estimation from thermal video. See `README.md` for the project
itself and `DOCUMENTATION.md` for code, data layout and how to run things.

Current scope is face detection on SIM1 participant T001, plus a respiratory-rate prototype
tested on synthetic breathing (`src/synth.py`, `src/breathing.py`, `src/rr_prototype.py`).
Real respiratory-rate results wait for recordings with a breathing reference.

## Python

Use `.venv312/Scripts/python.exe` (Python 3.12, packages in `requirements.txt`). Plain
`python3` resolves to the Microsoft Store stub on this machine.
