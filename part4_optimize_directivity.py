"""
Part 4 - Maximize broadside directivity of the inset-fed parasitic patch, subject
to a hard S11 < -10 dB constraint across the whole 3.1-3.4 GHz band.

Constrained optimization: maximize directivity, but OUTRIGHT REJECT any design
whose worst in-band |S11| exceeds -10 dB. The rejection is a barrier - an
infeasible design always scores worse than every feasible one - so no amount of
directivity can buy back a bad match.

Per evaluation it runs a real 3-D FDTD solve, computes |S11| over the band, and
only if the design is feasible (S11 <= limit everywhere) does it run the (more
expensive) NF2FF far-field post-process for the directivity. Infeasible designs
skip the far-field entirely, so they're cheap to discard.

Starts from the S11-optimized design (inset_patch_opt/optimized_params.json) so
the search begins feasible. Reuses inset_patch_build's geometry unchanged.

Run:  python part4_optimize_directivity.py
Out:  inset_patch_dir_opt/optimized_params.json   (best feasible, most-directive)
      inset_patch_dir_opt/optimization_log.csv
      inset_patch_dir_opt/optimized_s11.png
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

# Live view (same as Part 3): overlay recent S11 curves; falls back to a PNG.
LIVE_PLOT = True
LIVE_LAST = 3

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
BAND_LO, BAND_HI = 3.10e9, 3.40e9
S11_LIMIT        = -10.0          # feasibility gate: worst in-band |S11| must be <= this
DIR_FREQ         = p1.f0          # frequency at which directivity is maximized (band centre)
DIRECTIVITY_TARGET = None         # optional early stop, e.g. 9.0 (dBi); None = use full budget
PENALTY_BASE     = 1.0e3          # infeasible score floor (>> any feasible -directivity)

# Same 6 geometry knobs as Part 3: (initial, lower, upper, step) in mm.
PARAMS = {
    'L':     (p1.L,     18.0, 26.0, 0.1),
    'W':     (p1.W,     22.0, 36.0, 0.2),
    'y0':    (p1.y0,     3.0, 11.0, 0.1),
    'h_air': (p1.h_air,  2.0,  9.0, 0.1),
    'Lp':    (p1.Lp,    18.0, 26.0, 0.1),
    'Wp':    (p1.Wp,    22.0, 36.0, 0.2),
}

# Warm-start from the S11-optimized design so we begin feasible (Part 3's output).
WARM_START_FROM = os.path.join(os.getcwd(), 'inset_patch_opt', 'optimized_params.json')

MAX_EVALS = 120                   # far-field evals are slower than S11-only, so fewer

EVAL_NRTS         = 40000
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121
EVAL_FPAD         = 0.20e9

FINAL_CHARACTERIZE = True
FINAL_NRTS         = 60000
FINAL_ENDCRITERIA  = 1e-4

sim_path = os.path.join(os.getcwd(), 'inset_patch_dir_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    """Overwrite tuned values on the geometry module + refresh derived globals
    (mirrors inset_patch_build's derived-geometry block)."""
    for name, val in values.items():
        setattr(p1, name, float(val))
    p1.hf     = p1.Wf / 2.0
    p1.x_in   = -p1.L/2 + p1.y0
    p1.x_port = -p1.L/2 - p1.Lf
    p1.feed_x, p1.feed_y = p1.x_port, 0.0
    p1.gnd_x0 = p1.x_port - p1.margin
    p1.gnd_x1 =  p1.L/2   + p1.margin
    p1.gnd_y0 = -p1.W/2   - p1.margin
    p1.gnd_y1 =  p1.W/2   + p1.margin
    p1.z_gnd       = 0.0
    p1.z_drv_patch = p1.h_sub
    p1.z_stk_bot   = p1.h_sub + p1.h_air
    p1.z_stk_patch = p1.h_sub + p1.h_air + p1.h_sub


def _rmtree_retry(path, tries=5):
    for _ in range(tries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def worst_in_band(f, s11_dB):
    band = (f >= BAND_LO) & (f <= BAND_HI)
    return float(np.max(s11_dB[band]))


def simulate(values, run_dir):
    """One FDTD solve. Returns (f, s11_dB, worst_s11, feasible, D_dBi).
    Directivity (the costly NF2FF post-process) is only computed when feasible."""
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
    nf2ff = FDTD.CreateNF2FFBox()        # recorded during the run; cheap to set up

    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
    s11_dB = 20 * np.log10(np.abs(port.uf_ref / port.uf_inc))
    worst  = worst_in_band(f, s11_dB)
    feasible = worst <= S11_LIMIT

    D_dBi = None
    if feasible:                          # only pay for the far field if it passed the gate
        center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
        nf = nf2ff.CalcNF2FF(run_dir, np.array([DIR_FREQ]), np.array([0.0]),
                             np.array([0.0]), center=center)
        D_dBi = float(10 * np.log10(nf.Dmax[0]))

    _rmtree_retry(run_dir)
    return f, s11_dB, worst, feasible, D_dBi


# rolling window of recent evals for the live overlay
_recent = deque(maxlen=LIVE_LAST)
_live   = {'fig': None, 'ax': None}


def update_live_plot(eval_no, f, s11_dB, label, is_best):
    global LIVE_PLOT
    if not LIVE_PLOT:
        return
    _recent.append((eval_no, f, s11_dB, label, is_best))
    try:
        if _live['fig'] is None:
            if _gui_ok:
                plt.ion()
            _live['fig'], _live['ax'] = plt.subplots(figsize=(8, 5))
        ax = _live['ax']
        ax.clear()
        for i, (n, ff, ss, lab, bestflag) in enumerate(_recent):
            newest = (i == len(_recent) - 1)
            ax.plot(ff/1e9, ss, lw=2.4 if newest else 1.3, alpha=1.0 if newest else 0.5,
                    label='eval %d: %s%s' % (n, lab, '  (best)' if bestflag else ''))
        ax.axhline(S11_LIMIT, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        best_txt = ('best D = %.2f dBi' % best['D']) if np.isfinite(best['D']) else 'no feasible design yet'
        ax.set(title='Max-directivity search  (%s)  -  eval %d' % (best_txt, eval_no),
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
names  = list(PARAMS.keys())
bounds = [(PARAMS[n][1], PARAMS[n][2]) for n in names]
lo     = np.array([b[0] for b in bounds])
hi     = np.array([b[1] for b in bounds])
steps  = np.array([PARAMS[n][3] for n in names])


def snap(x):
    x = np.clip(x, lo, hi)
    x = np.round(x / steps) * steps
    return np.clip(x, lo, hi)


def initial_values():
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
            print('Could not warm-start (%s) - using build defaults.' % e)
    return init


_init = initial_values()
x0 = snap(np.array([_init[n] for n in names], dtype=float))

history = []                          # (eval#, params, D_dBi, worst_s11, feasible)
best    = {'D': -np.inf, 'x': x0.copy(), 'worst': None, 'f': None, 's11': None}
_cache  = {}                          # snapped tuple -> objective scalar


class _Stop(Exception):
    pass


def objective(x):
    x = snap(x)
    key = tuple(np.round(x, 6))
    if key in _cache:
        return _cache[key]
    if len(history) >= MAX_EVALS:
        raise _Stop()

    values = dict(zip(names, x))
    run_dir = os.path.join(sim_path, 'eval_%03d' % (len(history) + 1))
    f, s11_dB, worst, feasible, D_dBi = simulate(values, run_dir)

    # barrier objective (we MINIMIZE): feasible -> -directivity; infeasible -> big penalty
    if feasible:
        obj = -D_dBi
    else:
        obj = PENALTY_BASE + (worst - S11_LIMIT)      # +ve violation, ranks infeasibles
    _cache[key] = obj
    history.append((len(history) + 1, values, D_dBi, worst, feasible))

    if feasible and D_dBi > best['D']:
        best.update(D=D_dBi, x=x.copy(), worst=worst, f=f, s11=s11_dB)

    is_best = feasible and D_dBi == best['D']
    if feasible:
        label = 'D=%.2f dBi (S11 max %.1f)' % (D_dBi, worst)
    else:
        label = 'REJECTED (S11 max %.1f > %.0f)' % (worst, S11_LIMIT)
    print('[eval %2d] %s -> %s%s'
          % (len(history),
             '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             label, '  <-- best' if is_best else ''))

    update_live_plot(len(history), f, s11_dB, label, is_best)

    if DIRECTIVITY_TARGET is not None and feasible and D_dBi >= DIRECTIVITY_TARGET:
        print('Directivity target met (%.2f >= %.2f dBi) - stopping.'
              % (D_dBi, DIRECTIVITY_TARGET))
        raise _Stop()
    return obj


def run_optimization():
    print('========= MAXIMIZING DIRECTIVITY @ %.3f GHz  s.t. S11 <= %.0f dB over '
          '%.2f-%.2f GHz =========' % (DIR_FREQ/1e9, S11_LIMIT, BAND_LO/1e9, BAND_HI/1e9))
    print('Tuning:', ', '.join(names))
    print('Start: ', '  '.join('%s=%.3f' % (n, v) for n, v in zip(names, x0)), '\n')
    try:
        minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                 options={'xatol': float(steps.min()), 'fatol': 0.1,
                          'maxfev': 10 * MAX_EVALS, 'maxiter': 10 * MAX_EVALS})
    except _Stop:
        pass


def report():
    if not np.isfinite(best['D']):
        print('\nNo feasible design found (none met S11 <= %.0f dB across the band).'
              % S11_LIMIT)
        print('Try widening bounds, relaxing S11_LIMIT, or warm-starting from a '
              'known-good S11 design.')
        return

    bx = dict(zip(names, best['x']))
    set_params(bx)

    print('\n=========================== BEST FEASIBLE DESIGN ===========================')
    print('Directivity @ %.3f GHz: %.2f dBi   |   worst in-band |S11|: %.2f dB'
          % (DIR_FREQ/1e9, best['D'], best['worst']))
    for n in names:
        print('    %-7s = %7.3f mm   (was %.3f)' % (n, bx[n], PARAMS[n][0]))

    print('\n--- paste into inset_patch_build.py DESIGN PARAMETERS ---')
    print('L  = %.2f' % bx.get('L',  p1.L))
    print('W  = %.2f' % bx.get('W',  p1.W))
    print('y0 = %.2f' % bx.get('y0', p1.y0))
    print('h_air = %.2f' % bx.get('h_air', p1.h_air))
    print('Lp = %.2f' % bx.get('Lp', p1.Lp))
    print('Wp = %.2f' % bx.get('Wp', p1.Wp))
    print('---------------------------------------------------------\n')

    with open(os.path.join(sim_path, 'optimized_params.json'), 'w') as fh:
        json.dump({'objective': 'max directivity s.t. S11<=%.0fdB' % S11_LIMIT,
                   'directivity_dBi': best['D'],
                   'worst_S11_dB': best['worst'],
                   'dir_freq_GHz': DIR_FREQ/1e9,
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'params_mm': bx,
                   'fixed_mm': {'h_sub': p1.h_sub, 'Wf': p1.Wf, 'g': p1.g,
                                'Lf': p1.Lf, 'margin': p1.margin}}, fh, indent=2)

    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['directivity_dBi', 'worst_S11_dB', 'feasible'])
        for i, vals, D, worst, feas in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names]
                       + ['' if D is None else '%.3f' % D, '%.3f' % worst, int(feas)])

    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8)
        ax.axhline(S11_LIMIT, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target band')
        ax.set(title='Most directive feasible design  (D=%.2f dBi, worst S11 %.2f dB)'
               % (best['D'], best['worst']),
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved: optimized_params.json, optimization_log.csv, optimized_s11.png')


def characterize_best():
    """Full Part 2-style solve on the best feasible design -> 2x2 figure
    (S11, input impedance, broadside directivity, co-pol principal-plane pattern)."""
    if not np.isfinite(best['D']):
        return
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

    f = np.linspace(p1.f0 - p1.fc, p1.f0 + p1.fc, 601)
    port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))
    Zin    = port.uf_tot / port.if_tot

    center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
    f_pat  = np.linspace(BAND_LO, BAND_HI, 7)
    d_res  = nf2ff.CalcNF2FF(run_dir, f_pat, np.array([0.0]), np.array([0.0]), center=center)
    Dbroad = 10 * np.log10(np.array([d_res.Dmax[i] for i in range(len(f_pat))]))

    f_mid = 0.5 * (BAND_LO + BAND_HI)
    theta = np.arange(-180, 180.5, 1.0)                      # DEGREES for CalcNF2FF
    e_cut = nf2ff.CalcNF2FF(run_dir, f_mid, theta, np.array([0.0]), center=center)
    h_cut = nf2ff.CalcNF2FF(run_dir, f_mid, theta, np.array([90.0]), center=center)

    def _norm_db(res):
        E = np.sqrt(np.abs(res.E_theta[0][:, 0])**2 + np.abs(res.E_phi[0][:, 0])**2)
        return 20 * np.log10(E / np.max(E) + 1e-12)
    Eplane, Hplane = _norm_db(e_cut), _norm_db(h_cut)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(f/1e9, s11_dB)
    ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target')
    ax[0, 0].set(title='Reflection coefficient', xlabel='Frequency (GHz)',
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

    fig.suptitle('Max-directivity inset-fed parasitic patch  (D=%.2f dBi @ %.2f GHz, '
                 'worst S11 %.2f dB)' % (best['D'], DIR_FREQ/1e9, best['worst']), y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'optimized_results.png')
    fig.savefig(out, dpi=130)
    print('Saved full Part 2-style plot to', out)
    _rmtree_retry(run_dir)


if __name__ == '__main__':
    run_optimization()
    report()
    if FINAL_CHARACTERIZE and np.isfinite(best['D']):
        characterize_best()
    if LIVE_PLOT and _gui_ok:
        plt.ioff()
        plt.show()
