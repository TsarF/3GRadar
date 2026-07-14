"""
Optimizer for the stacked via-fed patch on the JLCPCB 4-layer stackup
(pcb_patch_build): minimize the worst |S11| across 3.1-3.4 GHz.

Tunes all knobs of the new design:
    L, W    driven patch (L2)          -> centre frequency / impedance
    Lp, Wp  parasitic patch (L1)       -> second resonance / bandwidth
    fv      via offset from centre     -> the match
    antipad ground clearance (L3)      -> via reactance / fine match

Starts from the previously found optimum (carried in pcb_patch_build's defaults).
Same machinery as part3/part5 (snap-to-grid + dedup cache + live S11 overlay), but
self-contained because this design's port lives on L4 (pcb_patch_build.add_feed_port).

Run:  python pcb_optimize.py
Out:  pcb_patch_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import pcb_patch_build as p

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
TARGET_DB        = -10.0
STOP_MARGIN_DB   = -10.7

# all knobs of the via-fed driven patch + air-gap parasitic: (init, lo, hi, step) mm
PARAMS = {
    'L':       (p.L,       16.0, 26.0, 0.1),
    'W':       (p.W,       18.0, 34.0, 0.2),
    'Lp':      (p.Lp,      16.0, 30.0, 0.1),
    'Wp':      (p.Wp,      18.0, 40.0, 0.2),
    'fv':      (p.fv,       1.0, 10.0, 0.1),
    'antipad': (p.antipad,  0.3,  1.2, 0.05),
    'h_air':   (p.h_air,    2.0, 12.0, 0.1),   # air gap -> bandwidth (back to tunable)
}

WARM_START_FROM = os.path.join(os.getcwd(), 'pcb_patch_opt', 'optimized_params.json')
MAX_EVALS = 150

EVAL_NRTS         = 50000          # thin feed substrate -> small cells; allow more steps
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121
EVAL_FPAD         = 0.20e9

FINAL_CHARACTERIZE = True
FINAL_NRTS         = 70000
FINAL_ENDCRITERIA  = 1e-4

# patch on top, feed on bottom -> absorb on both z faces
BOUNDARY = ['MUR', 'MUR', 'MUR', 'MUR', 'PML_8', 'PML_8']

sim_path = os.path.join(os.getcwd(), 'pcb_patch_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    """Overwrite tuned values on pcb_patch_build and refresh its derived geometry."""
    for name, val in values.items():
        setattr(p, name, float(val))
    p._derive()                      # recompute xv, x_port, feed_x, ground extents


def _rmtree_retry(path, tries=5):
    for _ in range(tries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def simulate_s11(values, run_dir):
    set_params(values)
    os.makedirs(run_dir, exist_ok=True)
    FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
    FDTD.SetGaussExcite(p.f0, p.fc)
    FDTD.SetBoundaryCond(BOUNDARY)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    p.build_antenna(CSX, FDTD)
    port = p.add_feed_port(FDTD)
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)
    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=p.feed_R)
    s11_dB = 20 * np.log10(np.abs(port.uf_ref / port.uf_inc))
    _rmtree_retry(run_dir)
    return f, s11_dB


def worst_in_band(f, s11_dB):
    band = (f >= BAND_LO) & (f <= BAND_HI)
    return float(np.max(s11_dB[band]))


_recent = deque(maxlen=LIVE_LAST)
_live   = {'fig': None, 'ax': None}


def update_live_plot(eval_no, f, s11_dB, obj, is_best):
    global LIVE_PLOT
    if not LIVE_PLOT:
        return
    _recent.append((eval_no, f, s11_dB, obj, is_best))
    try:
        if _live['fig'] is None:
            if _gui_ok:
                plt.ion()
            _live['fig'], _live['ax'] = plt.subplots(figsize=(8, 5))
        ax = _live['ax']; ax.clear()
        for i, (n, ff, ss, ob, bf) in enumerate(_recent):
            newest = (i == len(_recent) - 1)
            ax.plot(ff/1e9, ss, lw=2.4 if newest else 1.3, alpha=1.0 if newest else 0.5,
                    label='eval %d: %+.2f dB%s' % (n, ob, '  (best)' if bf else ''))
        ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        ax.set(title='Stacked via-fed patch: S11 - last %d of %d evals' % (len(_recent), eval_no),
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True)
        _live['fig'].tight_layout()
        if _gui_ok:
            _live['fig'].canvas.draw_idle(); _live['fig'].canvas.flush_events(); plt.pause(0.001)
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

history = []
best    = {'obj': np.inf, 'x': x0.copy(), 'f': None, 's11': None}
_cache  = {}


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
    f, s11_dB = simulate_s11(values, run_dir)
    obj = worst_in_band(f, s11_dB)
    _cache[key] = obj
    history.append((len(history) + 1, values, obj))
    if obj < best['obj']:
        best.update(obj=obj, x=x.copy(), f=f, s11=s11_dB)
    is_best = obj == best['obj']
    print('[eval %2d] %s -> worst S11 = %+6.2f dB%s'
          % (len(history), '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             obj, '  <-- best' if is_best else ''))
    update_live_plot(len(history), f, s11_dB, obj, is_best)
    if obj <= STOP_MARGIN_DB:
        print('Target met (worst %.2f <= %.2f dB) - stopping.' % (obj, STOP_MARGIN_DB))
        raise _Stop()
    return obj


def run_optimization():
    print('===== STACKED VIA-FED PATCH: minimize worst S11 over %.2f-%.2f GHz ====='
          % (BAND_LO/1e9, BAND_HI/1e9))
    print('Tuning:', ', '.join(names))
    print('Start: ', '  '.join('%s=%.3f' % (n, v) for n, v in zip(names, x0)), '\n')
    try:
        minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                 options={'xatol': float(steps.min()), 'fatol': 0.2,
                          'maxfev': 10 * MAX_EVALS, 'maxiter': 10 * MAX_EVALS})
    except _Stop:
        pass


def report():
    bx = dict(zip(names, best['x']))
    set_params(bx)
    print('\n=========================== BEST DESIGN ===========================')
    print('Worst |S11| in %.2f-%.2f GHz: %+.2f dB  (%s)'
          % (BAND_LO/1e9, BAND_HI/1e9, best['obj'],
             'PASS' if best['obj'] < TARGET_DB else 'FAIL - widen bounds / band, or add coplanar parasitic'))
    for n in names:
        print('    %-8s = %7.3f mm   (was %.3f)' % (n, bx[n], PARAMS[n][0]))

    print('\n--- paste into pcb_patch_build.py DESIGN PARAMETERS ---')
    for n in names:
        print('%-8s = %.2f' % (n, bx[n]))
    print('-------------------------------------------------------\n')

    with open(os.path.join(sim_path, 'optimized_params.json'), 'w') as fh:
        json.dump({'objective_worst_S11_dB': best['obj'],
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'params_mm': bx,
                   'fixed_mm': {'t_core': p.t_core, 't_prepreg': p.t_prepreg,
                                'h_sub2': p.h_sub2, 'a_via': p.a_via, 'Wf': p.Wf,
                                'Lf': p.Lf, 'eps_r': p.eps_r}}, fh, indent=2)

    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['worst_S11_dB'])
        for i, vals, obj in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names] + ['%.3f' % obj])

    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8)
        ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target band')
        ax.set(title='Optimized stacked via-fed patch  (worst in band %+.2f dB)' % best['obj'],
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved: optimized_params.json, optimization_log.csv, optimized_s11.png')


def characterize_best():
    set_params(dict(zip(names, best['x'])))
    run_dir = os.path.join(sim_path, 'best_full')
    os.makedirs(run_dir, exist_ok=True)
    FDTD = openEMS(NrTS=FINAL_NRTS, EndCriteria=FINAL_ENDCRITERIA)
    FDTD.SetGaussExcite(p.f0, p.fc)
    FDTD.SetBoundaryCond(BOUNDARY)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    p.build_antenna(CSX, FDTD)
    port = p.add_feed_port(FDTD)
    nf2ff = FDTD.CreateNF2FFBox()
    print('\nRunning full characterization of the best design...')
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(p.f0 - p.fc, p.f0 + p.fc, 601)
    port.CalcPort(run_dir, f, ref_impedance=p.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))
    Zin    = port.uf_tot / port.if_tot

    center = [0, 0, (p.z_top / 2) * p.unit]
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
    ax[0, 0].plot(f/1e9, s11_dB); ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target')
    ax[0, 0].set(title='Reflection coefficient', xlabel='Frequency (GHz)',
                 ylabel='|S11| (dB)'); ax[0, 0].legend(); ax[0, 0].grid(True)
    ax[0, 1].plot(f/1e9, np.real(Zin), label='Re'); ax[0, 1].plot(f/1e9, np.imag(Zin), label='Im')
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
    fig.suptitle('Optimized stacked via-fed patch (4-layer PCB)  '
                 '(worst in-band |S11| = %+.2f dB)' % worst_in_band(f, s11_dB), y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'optimized_results.png')
    fig.savefig(out, dpi=130)
    print('Saved full characterization plot to', out)
    _rmtree_retry(run_dir)


if __name__ == '__main__':
    run_optimization()
    report()
    if FINAL_CHARACTERIZE and best['f'] is not None:
        characterize_best()
    if LIVE_PLOT and _gui_ok:
        plt.ioff()
        plt.show()
