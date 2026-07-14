"""
Part 3 - Optimize the inset-fed parasitic patch for S11 < -10 dB across 3.1-3.4 GHz.

Tweaks a handful of geometry parameters and re-runs the FDTD engine until |S11|
stays below -10 dB across the whole target band. It reuses the inset_patch_build
geometry builder unchanged - the design values are simply overwritten on that
module before each build - so the hand-written build script never needs editing.

What gets tuned (see PARAMS below):
    L      driven patch length   -> centre frequency
    W      driven patch width
    y0     inset depth           -> the match (feed-point resistance)
    h_air  air gap               -> coupling / bandwidth
    Lp,Wp  parasitic patch size  -> the second resonance

Objective: minimise the WORST |S11| (in dB) over [3.1, 3.4] GHz. Once that worst
point drops below the target margin the search stops early. Only S11 is computed
during the search (no NF2FF / pattern) so each evaluation is a fraction of a full
Part 2 run, but it is still a real 3-D FDTD solve - expect each evaluation to take
roughly as long as Part 2's S11 portion.

Run:  python part3_optimize.py
Out:  inset_patch_opt/optimized_params.json   (best parameters found)
      inset_patch_opt/optimization_log.csv    (every evaluation)
      inset_patch_opt/optimized_s11.png       (S11 of the best design)
Then paste the printed DESIGN PARAMETERS block into inset_patch_build.py and run
Part 2 for the full pattern / gain characterisation.
"""

import os
import csv
import json
import time
import shutil
from collections import deque
import numpy as np
import matplotlib
from scipy.optimize import minimize

# Live view: overlay the most recent few S11 curves in a window that refreshes
# every iteration. Needs a GUI backend; if none is available we fall back to a
# refreshing PNG (cp_patch_opt/live_s11.png). Final plots save either way.
LIVE_PLOT  = True
LIVE_LAST  = 3                    # how many of the most recent evals to overlay

_gui_ok = False
if LIVE_PLOT:
    for _bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(_bk, force=True)
            _gui_ok = True
            break
        except Exception:
            continue
if not _gui_ok:
    matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import inset_patch_build as p1          # `p1` is just the geometry-module handle

# ============================ optimiser config ============================
# Target impedance band: |S11| must be below TARGET_DB across [BAND_LO, BAND_HI].
BAND_LO, BAND_HI = 3.10e9, 3.40e9
TARGET_DB        = -15.0
STOP_MARGIN_DB   = -16.0          # stop early once worst-in-band is below this
                                  # (a little headroom for the finer Part 2 mesh)

# Parameters to tune: name -> (initial, lower, upper, step) in mm.
# Initial values are the inset_patch_build defaults. Bounds keep the geometry
# valid. `step` is the quantization grid: every candidate is snapped to a multiple
# of `step` before building, so the optimizer never asks for sub-grid (micron-
# level) dimensions that force openEMS to create degenerate cells. It also caps
# how finely each dimension can move per iteration. Coarsen step if the mesher
# complains; refine it for a finer (slower) search.
# To lock the parasitic equal to the driven patch (fewer dims, faster), delete
# the Lp and Wp lines - set_params then leaves them at the module defaults.
# Bounds widened for the PTFE (Dk 2.94) / 0.762 mm driven stack: lower Dk + thinner
# substrate -> patches must grow ~20% vs the FR-4 design to resonate at 3.25 GHz.
PARAMS = {
    'L':     (p1.L,     22.0, 32.0, 0.1),
    'W':     (p1.W,     26.0, 42.0, 0.2),
    'y0':    (p1.y0,     2.0, 14.0, 0.1),
    'h_air': (p1.h_air,  2.0, 12.0, 0.1),
    'Lp':    (p1.Lp,    24.0, 36.0, 0.1),
    'Wp':    (p1.Wp,    28.0, 48.0, 0.2),
}

# Warm start: if a previous run's results are present, begin the search from
# those parameters instead of the Part 1 defaults. Lets you refine an earlier
# result (e.g. after adding new knobs like W1/W2) without losing progress.
WARM_START_FROM = os.path.join(os.getcwd(), 'inset_ptfe_opt', 'optimized_params.json')

MAX_EVALS = 500                    # hard cap on FDTD evaluations

# Per-evaluation FDTD cost. Looser than Part 2 (which also resolves far fields):
# S11 settles fast, so a relaxed energy criterion keeps each solve cheap while
# still giving a trustworthy match estimate.
EVAL_NRTS         = 80000
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121           # S11 samples across [BAND_LO-pad, BAND_HI+pad]
EVAL_FPAD         = 0.20e9        # look a bit outside the band to see drift

# After the search, run ONE full Part 2-style solve on the best design (with the
# NF2FF box) to produce the same 2x2 figure: S11, Zin, axial ratio, CP pattern.
# This is a full-accuracy run, so it costs about as much as a Part 2 run.
FINAL_CHARACTERIZE = True
FINAL_NRTS         = 60000        # match Part 2 fidelity for the keeper plot
FINAL_ENDCRITERIA  = 1e-4

sim_path = os.path.join(os.getcwd(), 'inset_ptfe_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    """Overwrite the tuned design values on the geometry module and refresh every
    derived global that build_antenna() reads, so a plain build picks them up.
    Mirrors the derived-geometry block of inset_patch_build."""
    for name, val in values.items():
        setattr(p1, name, float(val))

    # feed / inset geometry (Wf, Lf, margin are fixed module constants)
    p1.hf     = p1.Wf / 2.0
    p1.x_in   = -p1.L/2 + p1.y0
    p1.x_port = -p1.L/2 - p1.Lf
    p1.feed_x, p1.feed_y = p1.x_port, 0.0

    # ground / substrate footprint (snug around the metal extent)
    p1.gnd_x0 = p1.x_port - p1.margin
    p1.gnd_x1 =  p1.L/2   + p1.margin
    p1.gnd_y0 = -p1.W/2   - p1.margin
    p1.gnd_y1 =  p1.W/2   + p1.margin

    # z-levels depend on h_air (air gap) and h_sub
    p1.z_gnd       = 0.0
    p1.z_drv_patch = p1.h_sub
    p1.z_stk_bot   = p1.h_sub + p1.h_air
    p1.z_stk_patch = p1.h_sub + p1.h_air + p1.h_sub


def _rmtree_retry(path, tries=5):
    """Best-effort recursive delete. On Windows rmtree can race with the OS
    still releasing handles (antivirus/indexing), so retry briefly."""
    for _ in range(tries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def simulate_s11(values, run_dir):
    """Build + run an S11-only FDTD solve for the given parameters in its own
    fresh directory. Returns (f_Hz, s11_dB).

    Each evaluation uses a brand-new directory and Run(cleanup=False): openEMS's
    cleanup=True does rmtree()+mkdir() of the same path, which races on Windows
    (WinError 183). Creating the dir ourselves once sidesteps that entirely.
    """
    set_params(values)
    os.makedirs(run_dir, exist_ok=True)

    FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
    FDTD.SetGaussExcite(p1.f0, p1.fc)
    FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)

    p1.build_antenna(CSX, FDTD)

    port = FDTD.AddLumpedPort(1, p1.feed_R,
                              [p1.feed_x, p1.feed_y, 0],
                              [p1.feed_x, p1.feed_y, p1.h_sub],
                              'z', 1.0, priority=5, edges2grid='xy')

    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
    s11_dB = 20 * np.log10(np.abs(port.uf_ref / port.uf_inc))

    _rmtree_retry(run_dir)                     # results are in memory now
    return f, s11_dB


def worst_in_band(f, s11_dB):
    """Worst (highest) |S11| in dB across the target band - the thing to minimise."""
    band = (f >= BAND_LO) & (f <= BAND_HI)
    return float(np.max(s11_dB[band]))


# rolling window of the most recent evaluations, for the live overlay
_recent = deque(maxlen=LIVE_LAST)
_live   = {'fig': None, 'ax': None}


def update_live_plot(eval_no, f, s11_dB, obj, is_best):
    """Redraw the live S11 view with the last LIVE_LAST iterations overlaid.
    Draws to a window if a GUI backend is up; always refreshes live_s11.png.
    Fully guarded: a plotting failure disables the view, never the search."""
    global LIVE_PLOT
    if not LIVE_PLOT:
        return
    _recent.append((eval_no, f, s11_dB, obj, is_best))
    try:
        if _live['fig'] is None:
            if _gui_ok:
                plt.ion()
            _live['fig'], _live['ax'] = plt.subplots(figsize=(8, 5))

        ax = _live['ax']
        ax.clear()
        for i, (n, ff, ss, ob, bestflag) in enumerate(_recent):
            newest = (i == len(_recent) - 1)
            ax.plot(ff/1e9, ss,
                    lw=2.4 if newest else 1.3,
                    alpha=1.0 if newest else 0.5,
                    label='eval %d: %+.2f dB%s' % (n, ob, '  (best)' if bestflag else ''))
        ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        ax.set(title='S11 - last %d of %d evals' % (len(_recent), eval_no),
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True)
        _live['fig'].tight_layout()

        if _gui_ok:
            _live['fig'].canvas.draw_idle()
            _live['fig'].canvas.flush_events()
            plt.pause(0.001)
        _live['fig'].savefig(os.path.join(sim_path, 'live_s11.png'), dpi=110)
    except Exception as e:
        print('Live plot disabled (%s) - optimization continues.' % e)
        LIVE_PLOT = False


# ----------------------------- search driver -----------------------------
names   = list(PARAMS.keys())
bounds  = [(PARAMS[n][1], PARAMS[n][2]) for n in names]
lo      = np.array([b[0] for b in bounds])
hi      = np.array([b[1] for b in bounds])
steps   = np.array([PARAMS[n][3] for n in names])


def snap(x):
    """Clip to bounds, then quantize each parameter to its step grid. Keeps the
    geometry on a coarse grid so mesh lines coincide exactly instead of landing
    microns apart (which would force degenerate openEMS cells)."""
    x = np.clip(x, lo, hi)
    x = np.round(x / steps) * steps
    return np.clip(x, lo, hi)                  # snapping can nudge past a bound


def initial_values():
    """Starting point for each parameter: a previous run's result if present
    (so newly added knobs start from the last optimum), else the Part 1 default.
    Earlier runs stored tuned knobs under 'params_mm' and fixed ones under
    'fixed_mm', so check both."""
    init = {n: PARAMS[n][0] for n in names}
    if os.path.exists(WARM_START_FROM):
        try:
            with open(WARM_START_FROM) as fh:
                data = json.load(fh)
            for section in ('params_mm', 'fixed_mm'):
                for k, v in data.get(section, {}).items():
                    if k in init:
                        init[k] = float(v)
            print('Warm-starting from', WARM_START_FROM)
        except Exception as e:
            print('Could not warm-start (%s) - using Part 1 defaults.' % e)
    return init


_init = initial_values()
x0 = snap(np.array([_init[n] for n in names], dtype=float))

history = []                                  # (eval#, params dict, worst_dB)
best    = {'obj': np.inf, 'x': x0.copy(), 'f': None, 's11': None}
_cache  = {}                                  # snapped-params tuple -> obj


class _TargetReached(Exception):
    pass


def objective(x):
    x = snap(x)                                # quantize to the mesh grid
    key = tuple(np.round(x, 6))

    # Many continuous Nelder-Mead probes collapse onto the same snapped geometry;
    # reuse the result instead of re-running an identical (expensive) FDTD solve.
    if key in _cache:
        return _cache[key]

    if len(history) >= MAX_EVALS:
        raise _TargetReached()                # spend budget, keep best so far

    values = dict(zip(names, x))
    run_dir = os.path.join(sim_path, 'eval_%03d' % (len(history) + 1))
    f, s11_dB = simulate_s11(values, run_dir)
    obj = worst_in_band(f, s11_dB)

    history.append((len(history) + 1, values, obj))
    _cache[key] = obj
    if obj < best['obj']:
        best.update(obj=obj, x=x.copy(), f=f, s11=s11_dB)

    is_best = obj == best['obj']
    tag = '  <-- best' if is_best else ''
    print('[eval %2d] %s -> worst S11 in band = %+6.2f dB%s'
          % (len(history),
             '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             obj, tag))

    update_live_plot(len(history), f, s11_dB, obj, is_best)

    if obj <= STOP_MARGIN_DB:
        print('Target met (worst %.2f dB <= %.2f dB) - stopping.'
              % (obj, STOP_MARGIN_DB))
        raise _TargetReached()
    return obj


def run_optimization():
    print('=================== OPTIMIZING S11 < %.0f dB over %.2f-%.2f GHz '
          '===================' % (TARGET_DB, BAND_LO/1e9, BAND_HI/1e9))
    print('Tuning:', ', '.join(names))
    print('Start: ', '  '.join('%s=%.3f' % (n, v) for n, v in zip(names, x0)), '\n')
    try:
        minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                 options={'xatol': float(steps.min()), 'fatol': 0.2,
                          'maxfev': 10 * MAX_EVALS, 'maxiter': 10 * MAX_EVALS})
    except _TargetReached:
        pass


def report():
    bx = dict(zip(names, best['x']))
    set_params(bx)                            # leave module on the winning design

    print('\n=========================== BEST DESIGN FOUND ===========================')
    print('Worst |S11| in %.2f-%.2f GHz band: %+.2f dB  (%s)'
          % (BAND_LO/1e9, BAND_HI/1e9, best['obj'],
             'PASS' if best['obj'] < TARGET_DB else 'FAIL - widen bounds / band'))
    for n in names:
        print('    %-7s = %7.3f mm   (was %.3f)' % (n, bx[n], PARAMS[n][0]))

    # paste-ready block for the build script (bx.get falls back to the current
    # module value for any parameter not in the tuned set, so this stays correct
    # as PARAMS changes)
    print('\n--- paste into inset_patch_build.py DESIGN PARAMETERS ---')
    print('L  = %.2f' % bx.get('L',  p1.L))
    print('W  = %.2f' % bx.get('W',  p1.W))
    print('y0 = %.2f' % bx.get('y0', p1.y0))
    print('h_air = %.2f' % bx.get('h_air', p1.h_air))
    print('Lp = %.2f' % bx.get('Lp', p1.Lp))
    print('Wp = %.2f' % bx.get('Wp', p1.Wp))
    print('---------------------------------------------------------\n')

    # save artefacts (only genuinely-fixed dimensions go in 'fixed_mm')
    with open(os.path.join(sim_path, 'optimized_params.json'), 'w') as fh:
        json.dump({'objective_worst_S11_dB': best['obj'],
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'params_mm': bx,
                   'fixed_mm': {'h_sub': p1.h_sub, 'Wf': p1.Wf, 'g': p1.g,
                                'Lf': p1.Lf, 'margin': p1.margin}}, fh, indent=2)

    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['worst_S11_dB'])
        for i, vals, obj in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names] + ['%.3f' % obj])

    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8)
        ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12,
                   label='target band')
        ax.set(title='Optimized |S11|  (worst in band %+.2f dB)' % best['obj'],
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved: optimized_params.json, optimization_log.csv, optimized_s11.png')


def characterize_best():
    """Full Part 2-style solve on the best design -> 2x2 figure
    (S11, input impedance, broadside directivity, co-pol principal-plane pattern)."""
    set_params(dict(zip(names, best['x'])))
    run_dir = os.path.join(sim_path, 'best_full')
    os.makedirs(run_dir, exist_ok=True)

    FDTD = openEMS(NrTS=FINAL_NRTS, EndCriteria=FINAL_ENDCRITERIA)
    FDTD.SetGaussExcite(p1.f0, p1.fc)
    FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    p1.build_antenna(CSX, FDTD)

    port = FDTD.AddLumpedPort(1, p1.feed_R,
                              [p1.feed_x, p1.feed_y, 0],
                              [p1.feed_x, p1.feed_y, p1.h_sub],
                              'z', 1.0, priority=5, edges2grid='xy')
    nf2ff = FDTD.CreateNF2FFBox()

    print('\nRunning full characterization of the best design...')
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    # ---- S11 / Zin over the excitation band ----
    f = np.linspace(p1.f0 - p1.fc, p1.f0 + p1.fc, 601)
    port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))
    Zin    = port.uf_tot / port.if_tot

    # ---- broadside directivity across the band ----
    center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
    f_pat  = np.linspace(BAND_LO, BAND_HI, 7)
    d_res  = nf2ff.CalcNF2FF(run_dir, f_pat, np.array([0.0]),
                             np.array([0.0]), center=center)
    Dbroad = 10 * np.log10(np.array([d_res.Dmax[i] for i in range(len(f_pat))]))

    # ---- principal-plane co-pol patterns at band centre ----
    f_mid = 0.5 * (BAND_LO + BAND_HI)
    theta = np.arange(-180, 180.5, 1.0)                            # DEGREES for CalcNF2FF
    e_cut = nf2ff.CalcNF2FF(run_dir, f_mid, theta,
                            np.array([0.0]), center=center)        # E-plane (phi=0)
    h_cut = nf2ff.CalcNF2FF(run_dir, f_mid, theta,
                            np.array([90.0]), center=center)       # H-plane (phi=90)

    def _norm_db(res):
        E = np.sqrt(np.abs(res.E_theta[0][:, 0])**2 + np.abs(res.E_phi[0][:, 0])**2)
        return 20 * np.log10(E / np.max(E) + 1e-12)
    Eplane, Hplane = _norm_db(e_cut), _norm_db(h_cut)

    # ---- 2x2 figure (LP: S11, Zin, directivity, co-pol pattern) ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].plot(f/1e9, s11_dB)
    ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target')
    ax[0, 0].set(title='Reflection coefficient (optimized)', xlabel='Frequency (GHz)',
                 ylabel='|S11| (dB)'); ax[0, 0].legend(); ax[0, 0].grid(True)

    ax[0, 1].plot(f/1e9, np.real(Zin), label='Re')
    ax[0, 1].plot(f/1e9, np.imag(Zin), label='Im')
    ax[0, 1].axhline(50, color='k', ls=':', lw=0.8)
    ax[0, 1].set(title='Input impedance', xlabel='Frequency (GHz)',
                 ylabel='Z (ohm)'); ax[0, 1].legend(); ax[0, 1].grid(True)

    ax[1, 0].plot(f_pat/1e9, Dbroad, 'o-')
    ax[1, 0].set(title='Broadside directivity', xlabel='Frequency (GHz)',
                 ylabel='D (dBi)'); ax[1, 0].grid(True)

    ax[1, 1].plot(theta, Eplane, label='E-plane (phi=0)')
    ax[1, 1].plot(theta, Hplane, '--', label='H-plane (phi=90)')
    ax[1, 1].set(title='Co-pol pattern @ %.2f GHz' % (f_mid/1e9), xlabel='Theta (deg)',
                 ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
    ax[1, 1].legend(); ax[1, 1].grid(True)

    fig.suptitle('Optimized inset-fed parasitic patch  (worst in-band |S11| = %+.2f dB)'
                 % worst_in_band(f, s11_dB), y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'optimized_results.png')
    fig.savefig(out, dpi=130)
    print('Saved full Part 2-style plot to', out)

    _rmtree_retry(run_dir)


if __name__ == '__main__':
    run_optimization()
    report()
    if FINAL_CHARACTERIZE and best['f'] is not None:
        characterize_best()
    if LIVE_PLOT and _gui_ok:
        plt.ioff()
        plt.show()                # keep the live + result windows open until closed
