# Antenna Design — Autonomous Handoff Manifest

You are a Claude instance taking over an antenna-design project to **iterate on it
autonomously**. Read this whole file first. It is your single source of truth.

---

## 1. Mission

Design a **high-gain Fabry–Perot cavity (FPC) antenna** for **3.1–3.4 GHz** (9 %
band) on the JLCPCB **PTFE ZYF300CA-C** stack (Dk 2.94, Df 0.0016, 1.52 mm boards),
board side **≤ ~200 mm**. **Objective: maximize the worst-in-band realized boresight
gain** `G = D_broadside · η_rad · (1 − |S11|²)` over 3.10–3.40 GHz.

Practical ceiling at 194 mm (2.1 λ) aperture is **~15 dBi** (the source paper hit 14.9
at this electrical size). 17 dBi is not physical here without a bigger board.

---

## 2. Current state (READ `python fpc3_status.py` FIRST)

- **Best design so far:** the 3-layer antenna `fpc3_build.py` (feed patch + dual-layer
  PRS + RCM superstrate). Hand-swept best: **D = 14.5 dBi, peak realized gain 12.7 dBi**
  at `h_cav=45, rcm_s=17, h3=45, y0=9.5`.
- **A DE optimizer is ALREADY RUNNING** (`fpc3_optimize_de_gain.py`, detached, PID in
  `fpc3_gain_de_opt/driver.pid`) under a **watchdog** (`fpc3_watchdog.ps1`, PID in
  `watchdog.pid`) that relaunches it if it dies. **Do NOT start a second optimizer** —
  check `fpc3_status.py` and the watchdog first.
- It checkpoints the best to `fpc3_gain_de_opt/optimized_params.json` on every
  improvement and logs every eval to `optimization_log.csv`. Warm-starts from the JSON,
  so nothing is lost across restarts.

---

## 3. Environment & solvers

- **Windows, Python 3.13, NVIDIA RTX 4070.** PowerShell is primary; a bash tool exists.
- **openEMS (FDTD), Python bindings** — the accurate solver (has the dielectric). SLOW
  for this high-Q cavity: ~20–30 min/eval at coarse mesh (`mesh_res /20`, ~4 M cells),
  ~1 hr at fine mesh. **This is the source of truth for final numbers.**
- **bempp-cl (MoM, installed as `bempp_cl`)** — fast (~3 min/solve) but **PEC-in-air**
  (ignores the PTFE), so it **undershoots absolute gain by ~3 dB and its optimum spacings
  are ~0.83× the FDTD ones**. CPU only (GPU gives NO speedup — measured). **Use bempp
  ONLY to screen trends/direction cheaply, then validate in FDTD.**
- openEMS output is fd-redirected to per-run logs so it doesn't spam stdout.

---

## 4. File inventory

Active FPC design (use these):
- `fpc3_build.py` — the 3-layer antenna. Knobs: `h_cav, rcm_s, h3, y0` (+ fixed
  `L=26.5, W=29.4, sb=16, r1=6.5, pb=20.7, P=21.5, N_PRS=9`). `RCM_ON` toggles the RCM.
- `fpc3_optimize_de_gain.py` — the running DE (objective = worst-in-band realized gain).
- `fpc3_characterize.py` — full validation: S11 + gain-vs-freq + directivity + E/H
  patterns. `--replot` reuses saved data. ~1 hr (fine mesh).
- `fpc3_sweep.py` — lean N-point FDTD sweep (edit `POINTS`), coarse mesh.
- `fpc3_watchdog.ps1` — keeps the optimizer alive.
- `fpc3_status.py` — one-shot status.
- `fpc_bempp_directivity.py` — bempp MoM directivity screen (edit the sweep in `main()`).

Baselines / building blocks:
- `fpc2_build.py`, `fpc2_characterize.py` — single-PRS FPC (~11 dBi, matched). `FEED_ONLY`
  flag builds just the bare patch.
- `fpc2_prs_unitcell.py` — dual-layer PRS reflection extractor (|Γ|, phase, Trentini
  cavity height) via a TEM waveguide unit cell.
- `fpc2_feed_optimize.py` — Stage-A bare-patch match optimizer (fast, feed-only).
- Older experiments (context only): `fpc_build.py`, `series_patch_build.py`,
  `inset_slot_*`, `inset_patch_*`, `pcb_*`, `part2_simulate.py`.
- `fpc_metasurface paper.pdf` — the source paper (Ding et al., Results in Eng. 27
  (2025) 106647). 3-layer metasurface FPC at 7 GHz; we scaled ×2.15 to 3.25 GHz and
  dropped the CBM (RCS) layer.

---

## 5. Hard-won facts & gotchas (do not relearn these)

1. **Aperture caps gain:** uniform 9×9 (194 mm) → 17.4 dBi ideal, ~15 realized. Bigger
   board needed for more (11×11 ≈ 236 mm → ~17), but user capped ~200 mm.
2. **openEMS drops zero-height cylinders** as "Unused primitive" → build flat discs with
   `AddPolygon` (see `_disc` in the builds). Never use a degenerate `AddCylinder`.
3. **A bare patch can't match across 9 %** (patch BW ~3 %). Match it at BAND CENTER
   (3.25 GHz); the cavity provides band coverage. (Objective for the *feed alone* is S11
   at f0, NOT worst-in-band.)
4. **Adding a superstrate layer detunes the feed** → must re-tune `y0` (and `h_cav`)
   jointly. The RCM's first naive placement killed the match (S11 ≈ 0).
5. **RCM works:** a uniform square-patch RCM adds **+3–5 dB** broadside at `rcm_s≈17,
   h3≈45` — but it's a **single resonance → narrow band** (gain peaks tall, sags at band
   edges). The paper uses THREE varied square sizes to get bandwidth; we haven't.
6. **High-Q ⇒ slow FDTD** (long ringdown). Use `WORKERS=2` (4 thrashes RAM), coarse mesh
   for optimization, fine mesh only for final validation.
7. **bempp PEC-in-air is qualitative only** — screens direction, not magnitude. Its
   optimum ≠ FDTD's. Confirmed it undershoots (single-PRS: bempp 8 vs FDTD 11 dBi).
8. **DE loop bug (fixed):** the loop must terminate on stagnation + a generation cap,
   else it spins forever once the population caches out. Preserve this in any new DE.

---

## 6. Your objective & how to iterate

**Goal:** push worst-in-band realized gain as high as possible (target ~15 dBi) within
≤200 mm, keeping a usable in-band gain floor (not just a tall narrow peak).

**Loop:**
1. `python fpc3_status.py` — see best-so-far, eval count, process health.
2. Verify the optimizer + watchdog are alive (`Get-Process -Id <pid>`). If dead and not
   converged, relaunch the optimizer detached (Start-Process, WorkingDirectory the repo,
   `python fpc3_optimize_de_gain.py`, redirect to `fpc3_gain_de_opt/driver.out`/`.err`,
   hidden, PassThru; write new PID to `driver.pid`). It warm-starts — no loss.
3. When the DE converges (driver.out shows "Converged"/"BEST 3-layer"/"generation cap",
   or a `WATCHDOG_DONE` file appears), run `python fpc3_characterize.py` to VALIDATE the
   winner at fine mesh, and record the real S11 + gain-vs-freq.
4. Then pursue the next improvement (below), using **bempp to screen cheaply first**,
   then **1–2 FDTD validation runs** at the bempp-scaled point. Never blind-grid FDTD.
5. Append everything you do/find/decide to `AGENT_LOG.md` (create it) with timestamps.

**Improvement ideas (in rough priority):**
- **Bandwidth (biggest gap):** the RCM peak is narrow. Try (a) a *weaker* RCM (smaller
  `rcm_s` / bigger `sb`) trading peak for a flatter band, and/or (b) a **multi-size RCM**
  (two or three square sizes per super-cell, like the paper) to add a second resonance
  and flatten gain across 3.1–3.4. This likely needs a new `rcm` build variant.
- **Joint feed re-match** with the RCM present (`y0`, maybe `L`) so realized gain isn't
  killed by mismatch at band edges.
- **PRS reflectivity** (`sb`, `r1`) re-tune for the compound two-layer cavity.
- If the user later relaxes the size cap: 11×11 (236 mm) → ~17 dBi ceiling.

---

## 7. Guardrails (respect these)

- **Never run `WORKERS > 2`** and **never run two heavy FDTD jobs at once** (RAM). One
  optimizer + at most one manual FDTD run.
- **Never delete** `fpc3_gain_de_opt/optimized_params.json` or `optimization_log.csv`.
- **bempp = screening only.** Any gain number you report to the user must come from FDTD.
- Keep the board ≤ ~200 mm and the stack PTFE (Dk 2.94) unless the user says otherwise.
- Checkpoint/log continuously. Prefer small, reversible changes; keep `fpc3_build.py`
  buildable at all times (syntax-check + a 3×3 smoke build before any long run).
- Don't fabricate progress. If a run fails or is inconclusive, say so in `AGENT_LOG.md`.

---

## 8. Quick commands

```
python fpc3_status.py                      # status + best-so-far
python fpc3_characterize.py                # full FDTD validation (~1 hr)
python fpc3_characterize.py --replot       # replot last validation
python fpc_bempp_directivity.py            # bempp trend screen (~3 min/point, edit main)
python -c "import ast; ast.parse(open('fpc3_build.py').read())"   # syntax check
```

Best design lives in `fpc3_gain_de_opt/optimized_params.json`. When in doubt, prefer
watching the running DE to converge over starting new work.
