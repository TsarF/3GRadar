# AGENT_LOG — Autonomous FPC antenna iteration

Agent: Claude (Opus 4.8). Objective: push worst-in-band realized boresight gain
toward ~15 dBi over 3.1–3.4 GHz within a ≤200 mm board. All gain numbers reported
to the user MUST come from FDTD; bempp is trend-screening only.

---

## 2026-07-07 ~16:40 PDT — Takeover / initial status

**Read HANDOFF.md in full.** Understood mission, solvers, gotchas, guardrails.

**Process health (both alive, verified with Get-Process):**
- DE optimizer `fpc3_optimize_de_gain.py` — PID 8928 (python, started 16:22).
- Watchdog `fpc3_watchdog.ps1` — PID 37168 (powershell, started 16:26).
  Heartbeat `watchdog.hb` current (16:36).
- Did NOT start a second optimizer (per guardrail).

**DE state:** freshly started. `optimized_params.json` does NOT exist yet — still on
the initial population (16 members, NP=16), 0 evals logged in `optimization_log.csv`.
Two FDTD workers running (worker_42740 @~17 min, worker_45436 @~17 min), both ringing
down normally (high-Q cavity; energy decaying past -8 dB). Coarse opt mesh (~4M cells),
WORKERS=2 × 4 threads. Objective = worst-in-band realized gain over 3.10–3.40 GHz;
knobs h_cav[42–50], rcm_s[13–20], h3[36–50], y0[5–13]. Converges on stagnation
(3 gens no improvement) or GEN_CAP=100 / MAX_EVALS=600.

**Timeline estimate:** initial population of 16 at ~20–30 min/eval, 2 at a time
≈ 8 batches ≈ 3–4 h just for gen 0, then DE generations. This is a multi-hour/overnight
run. Plan: poll status + logs periodically, do NOT block.

**Machine:** AMD Ryzen 7 5800X3D, 8 cores / 16 logical, 32 GB RAM (10.6 GB free with
the 2 FDTD workers using ~21 GB). Headroom exists for a thread-limited bempp screen.

**Baseline (pre-DE, from HANDOFF):** hand-swept best D=14.5 dBi, realized-gain peak
12.7 dBi at h_cav=45, rcm_s=17, h3=45, y0=9.5. Known weakness: RCM peak is narrow
(single resonance) — gain sags at band edges → worst-in-band is the limiter.

**Plan:**
1. Let the DE run to convergence; validate winner with `fpc3_characterize.py` (fine mesh).
2. In parallel (thread-limited so as not to starve FDTD): bempp-screen the multi-size /
   weaker RCM idea to see if a second resonance flattens the band. bempp = direction only.
3. Only commit FDTD time to bempp-validated directions.

## 2026-07-07 ~16:55 PDT — bempp import OK; multi-size RCM screen launched

- Confirmed bempp_cl 0.4.2 imports (9 s, 2 threads). DE workers healthy: worker_45436
  at -11.5 dB ringdown (TS 34.6k/70k), worker_42740 at -5 dB (TS 30k). Ringdown is slow
  → revised eval estimate ~35–45 min each, so gen 0 (16 evals ÷ 2) ≈ ~5 h.
- Wrote `fpc_bempp_rcm_screen.py` (new): screens the TOP improvement idea from HANDOFF
  §6 — does a MULTI-SIZE RCM (3 interleaved square sizes {14.5,17,19.5} by (i+j)%3,
  PEC-in-air) sag LESS at band edges than a UNIFORM RCM? Runs one process per frequency
  (3.10/3.25/3.40 GHz) so numba compiles k fresh each time. Syntax-checked; fpc3_build
  also syntax-checked (still buildable).
- Launched it in the background at 2 threads (OMP/MKL/NUMBA/OPENBLAS=2) so it does NOT
  starve the priority-1 DE (8-core CPU already saturated by 2×4-thread FDTD). Output →
  `scratch_bempp_rcm_screen.out`. This is SCREENING ONLY (direction, not magnitude).
  Will monitor DE MC/s; if it collapses I kill the screen.

## 2026-07-07 ~17:15 PDT — Restructured bempp screen to ONE process; DE healthy

- Coordinator guidance: bempp's first solve per PROCESS pays a one-time numba/OpenCL
  compile; run the whole variant sweep in ONE process to amortize it. My first version
  was one-process-per-frequency (3 compiles) and its first solve was very slow under
  2-thread contention (only the 3.10 GHz header printed after ~15 min).
- Killed the old screen SURGICALLY (only the python whose cmdline matched
  `fpc_bempp_rcm_screen`; PID 42228) and stopped the driver loop. Verified DE optimizer
  PID 8928 + its two workers (45436, 42740) and watchdog 37168 all still alive.
- DE health throughout: workers held ~98-103 MC/s (no starvation from 2-thread bempp).
  worker_45436 reached -17 dB ringdown (TS 49k/70k) ~28 min in — first evals ~30-40 min.
  Still 0 rows in optimization_log.csv (gen 0 not finished).
- Rewrote `fpc_bempp_rcm_screen.py`: single process, FREQS(3.10/3.25/3.40) outer x
  variants(no-RCM / uniform s=17 / multi{14.5,17,19.5} interleaved) inner. Per-freq RHS
  via `make_rhs(kval)` closure factory (numba freezes k at compile, so a fresh fn per
  freq is needed). Prints a broadside-D-vs-freq trend table + sag(max-min). Syntax OK.
  Relaunched at 2 threads -> `scratch_bempp_rcm_screen.out` (bg task bgsenqhfi).
- Set up a persistent monitor on driver.out/err + watchdog.log for DE milestones
  (task bvfzcac6x): fires on NEW BEST / gen-done / Converged / generation cap /
  BEST 3-layer / Traceback / watchdog restart / WATCHDOG_DONE.

## 2026-07-07 ~17:20 PDT — Validation procedure prepared (read characterize + watchdog)

Read fpc3_characterize.py, fpc3_watchdog.ps1, fpc3_status.py. Key facts for when the
DE converges:
- `fpc3_characterize.py` imports `fpc3_build as p1` and uses its MODULE-LEVEL knobs
  (h_cav, rcm_s, h3, y0 via `_recompute`). It does NOT read optimized_params.json.
  It uses the FINE mesh (fpc3_build mesh_res = .../24) and NrTS=120000, EndCriteria
  1e-4, NFREQ=601, NGAIN=21, THREADS=8. Output -> fpc3_char/openems.log, plots +
  char_data.npz. ~1 hr. `--replot` reuses char_data.npz.
- Watchdog checks the "done" tail (BEST 3-layer DESIGN | Converged (population |
  generation cap) BEFORE the alive-check each 300 s loop, so on clean convergence it
  writes WATCHDOG_DONE and breaks with NO spurious relaunch. => after convergence the
  optimizer process has exited and nothing heavy is running -> safe to run characterize
  (respects "never two heavy FDTD at once").

VALIDATION PLAN (execute on the convergence milestone):
  1. Confirm optimizer PID 8928 exited + WATCHDOG_DONE present (or driver.out shows
     BEST 3-layer). `python fpc3_status.py` to read winner params.
  2. Set winner h_cav/rcm_s/h3/y0 into fpc3_build.py defaults (small, reversible edit;
     harmless to any watchdog relaunch since the DE warm-starts from the JSON, not the
     module defaults). Syntax-check + 3x3 smoke build first.
  3. Launch `python fpc3_characterize.py` detached; poll fpc3_char/openems.log.
  4. Record real worst-in-band S11 + realized gain-vs-freq (this is the FDTD number to
     report to the user).

Now yielding to armed notifications (bempp screen bgsenqhfi completion + DE milestones
bvfzcac6x). Not polling blindly per coordinator guidance.

## 2026-07-08 ~09:26 PDT — Re-engaged (overnight); DE MILESTONE + bempp relaunch

Was dormant overnight — the persistent monitor did NOT reliably wake me (its events
arrived only when I re-engaged). LESSON: treat resumption as self-paced, not
event-driven. Will note a next-action time before each yield.

**DE MILESTONE (verified via fpc3_status.py):** 50 evals, gen 3.
  BEST worst-in-band realized gain = **11.84 dBi**
  params: h_cav=45.5, rcm_s=17.5, h3=49.5, y0=6.6  (fixed L=26.5,W=29.4,sb=16,N=9,P=21.5)
  across band [3.10..3.40] = [12.47, 13.34, 11.84, 11.97, 11.88] dBi  (found at eval 48)
  NEW BEST progression: e1 +6.41 -> e4 +6.56 -> e5 +10.80 -> e35 +11.55 -> e47 +11.84.
  This is a FLAT band (spread ~1.5 dB, worst 11.84) vs the narrow-peak hand baseline
  (D=14.5 / realized peak 12.7 but sagging hard at edges -> low worst-in-band). The DE
  is optimizing exactly the right thing (worst-in-band) and has traded peak for
  flatness. h3=49.5 sits near the top of its [36,50] range and rcm_s=17.5 mid — worth
  noting h3 may want to exceed 50 (bound may be limiting; revisit after convergence).
  DE STILL RUNNING (2 FDTD workers, ~95 MC/s) -> NOT running characterize yet (guardrail).

**bempp screen had DIED** (bg bash task got killed when I went dormant) after ONE data
  point: no-RCM @3.10 GHz -> broadside 6.69 dBi, peak 8.27@38deg, DOF 5743, and that
  single solve took **837 s** (~14 min) under 2-thread contention. Full 3-freq x 3-var
  screen ~= 2 h. RELAUNCHED robustly as a detached hidden OS process via PowerShell
  Start-Process (PID 46968, pidfile bempp_screen.pid, env OMP/MKL/NUMBA/OPENBLAS=2),
  stdout->scratch_bempp_rcm_screen.out, stderr->.err. Verified alive; it is computing
  the 3.10 GHz no-RCM solve now. DE workers still ~95 MC/s alongside it (not starved).

NEXT ACTION (self-paced): check back ~10:15 PDT to confirm the bempp screen passed the
first point (no-RCM@3.10) and ideally has uniform-vs-multi numbers; and re-poll DE
status. Waiting on: (a) bempp screen verdict on multi-size RCM, (b) DE convergence to
trigger FDTD validation of the flat-band winner.

## 2026-07-08 — COORDINATOR NOTE (main): widened h3 bound + restarted DE
Acted on your finding that h3 was pinned at its upper bound (49.5/50) in the 11.84 dBi
best -> the bound was throttling the DE. Edited fpc3_optimize_de_gain.py PARAMS: h3
upper 50 -> 58. Killed old DE (PID 8928 + workers), relaunched cleanly (NEW PID 1440,
driver.pid updated), warm-started from optimized_params.json so the 11.84 flat-band best
is preserved as the seed. New DE now explores h3 up to 58. Watchdog unaffected (reads
driver.pid=1440). On your next self-resume: fpc3_status.py reflects the new PID; keep
monitoring for a new best above 11.84 that uses h3>50, and proceed with your validation
plan on convergence.

## 2026-07-08 — COORDINATOR NOTE (main): match is the bottleneck; added L,W knobs
Read the best design's S11 (live_s11.png): dips only to ~-4.8 dB @3.13 GHz, worst-in-band
~-1.9 dB, NEVER below -10 dB. Yet realized gain is ~12 dBi flat => directivity is
~16-17 dBi (aperture ceiling) and the POOR MATCH is throwing away ~4 dB. So directivity
is maxed; the FEED MATCH is now the bottleneck. Added L[24.5-28.5], W[26.0-33.0] to the
DE knobs (was only y0 for match) so it can center/deepen the S11 dip. Restarted DE:
NEW PID 52208, 6 knobs [h_cav,rcm_s,h3,y0,L,W], warm-started from optimized_params.json
(the 11.84 best). driver.pid updated. Watch for a new best where S11 goes <-10 and
realized gain climbs toward ~15-16. When you validate at convergence, report the S11
too (it's the story).

## 2026-07-09 ~14:45 PDT — NEW FOCUSED TASK: matched-feed design (DE PAUSED)

Coordinator paused DE + watchdog (DE_PAUSED marker 2026-07-09T14:42) so I have the full
8 cores for FDTD. Do NOT relaunch the DE. Best (11.84 dBi) checkpointed & safe. On
re-engage all processes were already stopped. Stopped the stale milestone monitor
bvfzcac6x. Dropped the multi-size bempp screen (moot now; bempp can't model feed Z).

Best design to match (optimized_params.json): h_cav=45.5, rcm_s=17.5, h3=49.5, y0=6.6,
L=26.5, W=29.4. Directivity ~16-17 dBi (aperture ceiling) but S11 only ~-5 dB (never
< -10) -> ~4 dB of realized gain lost. Cavity/RCM maxed; pure feed-impedance problem.

Feed geometry (fpc3_build): inset-fed patch, 50-ohm microstrip feed line Wf=3.89
(≈50 ohm on 1.52mm PTFE eps_r=2.94) from port x_port=-23.25 to x_in=-6.65 (inset depth
y0=6.6, gap g=1.0, Lf=10). Port = z-lumped 50 ohm at outer feed-line end. DE swept y0
(and L,W) but couldn't beat -5 dB => an impedance offset the inset can't null (cavity
detunes feed, gotcha #4). Plan: diagnose Z(f) then add a quarter-wave transformer /
reference-plane shift / stub in the FEED LINE ONLY.

STEP 1: launched `python fpc3_impedance.py` (moderate mesh /22, NRTS 90k, 8 threads)
detached -> stdout fpc3_impedance/run_stdout.log, FDTD fpc3_impedance/openems.log.
Polling. ~30-45 min.

Impedance FDTD verified running cleanly at ~190 MC/s (8 threads, full machine; DE paused).
While it runs, prepared a matching-network calculator (scratchpad/match_design.py):
reads impedance.csv, diagnoses Z at 3.10/3.25/3.40 + min-S11 + X=0 crossings, and
synthesizes microstrip dims on PTFE (eps_r=2.94, h=1.52). Validated: 50 ohm -> W=3.906 mm
(design Wf=3.89 = 50.14 ohm, confirms feed line is 50 ohm); lg/4 ~= 15 mm @3.25 GHz.
Quarter-wave transformer ready: Zt=sqrt(50*R), e.g. R=100->Zt=70.7ohm W~2.2mm,
R=25->Zt=35ohm W~6mm. NEXT: read Z when the run finishes, diagnose, design, implement.

## 2026-07-09 ~15:40 PDT — IMPEDANCE DIAGNOSIS + narrowband match designed & implemented

Baseline impedance run done (2570 s). Full picture (moderate mesh /22):
  f(GHz)  D(dBi)   Z = R + jX (ohm)      S11
  3.10    11.42    4.4  -5.1            -1.52   <- series resonance (X=0), R tiny
  3.25    15.92    38.2 +79.6           -3.38   <- band center, R near 50 but big +X
  3.30    16.39    81.5 +142.8
  3.35    16.45    264.6 +177.4          (peak D)
  3.395   ~16.2    ~365 +0              anti-resonance (X=0), R huge
  3.40    16.02    365.4 -18.4
  Directivity rises MONOTONICALLY toward the cavity anti-resonance (~3.40). In-band R
  swings 4.4 -> 365 ohm (~80x), X up to +191. |Gamma|~0.68 even at the best in-band point
  => broadband match is Bode-Fano IMPOSSIBLE (confirms coordinator). Baseline files saved
  as fpc3_impedance/impedance_baseline.{csv,png}, imp_baseline.npz.

DESIGN FREQ = 3.25 GHz (band center). Chosen over the higher-D points because R=38 is
already closest to 50 (only reactance to null) and D=15.92 is just 0.5 dB below the peak
(16.45), with more usable bandwidth than the extreme anti-resonance (3.395, R=365).

MATCH (offset line + quarter-wave transformer, all-microstrip, no stub/T-junction):
  Load at old port @3.25: Z_L=38+j80, |Gamma|=0.677 @ +56.4 deg.
  1. Extend the 50-ohm feed line OUTWARD by Loff=4.68 mm (reference-plane shift; rotates
     Gamma to the +real axis -> Z=260 ohm real).
  2. Quarter-wave transformer Zt=sqrt(50*260)=114 ohm -> W=0.75 mm, length Lt=15.68 mm
     (lambda_g/4 @3.25). Transforms 260 -> 50 ohm. New port at outer end (x=-43.6 mm).
  Prepended OUTWARD from the old port so the patch/inset/cavity/RCM are untouched and the
  inward load (38+j80) is preserved automatically. Dims from scratchpad/match_design.py.

IMPLEMENTED in fpc3_build.py behind MATCH_ON toggle (default True):
  - new params Loff=4.68, Wt=0.75, Lt=15.68; _recompute now computes x_port (extended
    feed-line end) and x_tr_out (new port); feed_x=x_tr_out when matching.
  - build_antenna adds the transformer box (x_tr_out..x_port, width Wt) + mesh lines at
    x_port/x_tr_out and y=+/-Wt/2. Syntax OK. 3x3 smoke build meshes cleanly. New port
    x=-43.6 is well inside the 9x9 board (+/-96.75 mm).
  - MATCH_ON=False reverts to the original single-port feed (reversible).

VALIDATION: launched fpc3_impedance.py with the matched feed (bg bu30a64tw), ~185 MC/s,
~40 min. Baseline outputs preserved; new run overwrites impedance.{csv,png}/imp_data.npz
= the MATCHED result. Will read S11(3.25), the -10 dB bandwidth, realized gain, and
confirm directivity held. Next self-resume ~16:25 PDT (or on task completion).