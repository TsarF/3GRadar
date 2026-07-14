# 3GRadar

Antenna design workspace for a **3.1–3.4 GHz** (9% band) radar front end, targeting
manufacture on **JLCPCB PTFE** (ZYF300CA-C, Dk 2.94, Df 0.0016, 1.52 mm). Electromagnetic
design is done with **openEMS** (FDTD) driven from Python, plus some MATLAB Antenna Toolbox
prototyping and a KiCad project for the RF board.

The current focus is a high-gain **Fabry–Perot cavity (FPC) antenna** — a fed patch under a
partially-reflecting surface (PRS) and a resonant complementary metasurface (RCM), inspired
by Ding et al. (*Results in Engineering* 27, 2025), retuned for the 3.25 GHz band.

> Note: the reference paper PDF and all multi-GB openEMS run outputs are intentionally
> excluded from version control (see `.gitignore`).

## Layout

**Fabry–Perot cavity antenna (active work)**
- `fpc3_build.py` — 3-layer model: inset-fed patch → air cavity → wire-grid PRS → air cavity → subdivided RCM. Env/flag-configurable (`FEED_ONLY`, `RCM_ON`, `MATCH_ON`, …).
- `fpc3_optimize_de_gain.py` — differential-evolution optimizer maximizing **worst-in-band realized gain**, co-tuning feed (y0, L, W) with cavity (h_cav, h3) and RCM sub-spacing (rcm_gap). Full-state checkpointing (resumes after interruption); parallelism is env-configurable.
- `fpc3_validate.py` — high-fidelity full-cavity validation (S11, input Z, directivity, realized gain).
- `fpc3_characterize.py`, `fpc3_impedance.py`, `fpc3_feed_optimize.py`, `fpc3_sweep.py` — diagnostics and feed studies.
- `fpc2_*.py`, `fpc_*.py` — earlier single-PRS and first-cut FPC iterations.
- `fpc_bempp_*.py` — MoM/BEM cross-checks via `bempp-cl`.

**Guides**
- `fpc3_THREADRIPPER.md` — running the optimizer on a many-core box (worker/thread sizing).
- `fpc3_EC2.md` — running it on EC2 M8g **spot** with checkpoint durability (S3 sync / persistent EBS).

**Other antenna studies**
- `inset_slot_*.py`, `inset_patch_build.py`, `series_patch_build.py` — patch feed studies.
- `pcb_*.py`, `array2x2_build.py` — 2×2 PCB array build/optimize.
- `*.m` — MATLAB horn, patch, and matching-network prototyping.

**Board**
- `3GRadar.kicad_*`, `TX.kicad_sch`, `RX.kicad_sch`, `Parts.pretty/` — KiCad RF project.

## Requirements
Python with `openEMS` + `CSXCAD` bindings, `numpy`, `matplotlib` (and `bempp-cl` for the MoM
cross-checks). openEMS must be built for your platform (no ARM64 wheels — see `fpc3_EC2.md`).

## Optimizer quick start
```bash
python fpc3_optimize_de_gain.py                 # local defaults (2 workers x 4 threads)
FPC_WORKERS=16 FPC_THREADS=4 FPC_TAG=_run python fpc3_optimize_de_gain.py   # bigger box
```
It checkpoints the best design to `fpc3_gain_de_opt<TAG>/optimized_params.json` and the full
DE state to `de_state.pkl`; relaunching resumes mid-search.
