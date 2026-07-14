"""
Optimizer for the slotted inset patch (inset_slot_build): minimize worst |S11|
over 3.1-3.4 GHz. Tunes the patch/parasitic AND the U-slot, so the search can use
the slot's extra resonance if it helps - or shrink it away if it doesn't.

Tunes (9 knobs): L, W (driven), y0 (match), h_air (gap), Lp, Wp (parasitic),
                 slot_len, slot_w, slot_x (the U-slot).

Saves the best design AND the eval log INCREMENTALLY (every new best / every eval),
so a Ctrl-C or crash never loses progress.

Run:  python inset_slot_optimize.py
Out:  inset_slot_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import inset_slot_build as p1

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
TARGET_DB        = -12.0
STOP_MARGIN_DB   = -12.5

PARAMS = {
    'L':        (p1.L,        22.0, 32.0, 0.1),
    'W':        (p1.W,        26.0, 42.0, 0.2),
    'y0':       (p1.y0,        2.0, 14.0, 0.1),
    'h_air':    (p1.h_air,     2.0, 12.0, 0.1),
    'Lp':       (p1.Lp,       24.0, 36.0, 0.1),
    'Wp':       (p1.Wp,       28.0, 48.0, 0.2),
    'slot_len': (p1.slot_len,  2.0, 16.0, 0.2),   # slot arm length (2 mm ~ off)
    'slot_w':   (p1.slot_w,    2.0, 24.0, 0.2),   # slot tongue width
    'slot_x':   (p1.slot_x,  -12.0,  8.0, 0.2),   # slot position (down to the feed edge)
}

WARM_START_FROM = os.path.join(os.getcwd(), 'inset_slot_opt', 'optimized_params.json')
MAX_EVALS = 500

EVAL_NRTS         = 80000
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121
EVAL_FPAD         = 0.20e9

FINAL_CHARACTERIZE = False        # slot experiment: keep it to S11 (fast); use part2 later

sim_path = os.path.join(os.getcwd(), 'inset_slot_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    for name, val in values.items():
        setattr(p1, name, float(val))
    # refresh the inset derived geometry (slot geometry is computed inside build)
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
    p1.z_stk_patch = p1.h_sub + p1.h_air + p1.h_sub2


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
    FDTD.SetGaussExcite(p1.f0, p1.fc)
    FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    p1.build_antenna(CSX, FDTD)
    port = FDTD.AddLumpedPort(1, p1.feed_R,
                              [p1.feed_x, p1.feed_y, 0], [p1.feed_x, p1.feed_y, p1.h_sub],
                              'z', 1.0, priority=5, edges2grid='xy')
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)
    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
    s11_dB = 20 * np.log10(np.abs(port.uf_ref / port.uf_inc))
    _rmtree_retry(run_dir)
    return f, s11_dB


def worst_in_band(f, s11_dB):
    band = (f >= BAND_LO) & (f <= BAND_HI)
    return float(np.max(s11_dB[band]))


# ---- feature-aware guidance: pull a deep-but-misplaced resonance into the band ----
GUIDE_W = 2.0                     # dB of cost per unit normalized resonance offset


def resonance_offset(f, s11_dB):
    """Deepest S11 dip: its frequency and its normalized distance from band centre
    (0 = centred, 1 = at a band edge). Gives the optimizer a gradient even when the
    dip is out of band and `worst_in_band` alone is flat/uninformative."""
    f_res = float(f[int(np.argmin(s11_dB))])
    f_c   = 0.5 * (BAND_LO + BAND_HI)
    return abs(f_res - f_c) / (0.5 * (BAND_HI - BAND_LO)), f_res


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
        ax.set(title='Slotted inset patch: S11 - last %d of %d evals' % (len(_recent), eval_no),
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
            for k, v in data.get('params_mm', {}).items():
                if k in init:
                    init[k] = float(v)
            print('Warm-starting from', WARM_START_FROM)
        except Exception as e:
            print('warm-start read failed (%s)' % e)
    return init


_init = initial_values()
x0 = snap(np.array([_init[n] for n in names], dtype=float))

history = []
best    = {'obj': np.inf, 'x': x0.copy(), 'f': None, 's11': None}
_cache  = {}


def save_best_json():
    """Persist the current best immediately (atomic) - crash/Ctrl-C safe."""
    if not np.isfinite(best['obj']):
        return
    bx = dict(zip(names, best['x']))
    tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump({'objective_worst_S11_dB': best['obj'],
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'evals_so_far': len(history),
                   'params_mm': bx}, fh, indent=2)
    os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))


def save_log():
    """Rewrite the full eval log every eval - survives a crash."""
    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['worst_S11_dB'])
        for i, vals, obj in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names] + ['%.3f' % obj])


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
    worst = worst_in_band(f, s11_dB)                 # the TRUE metric (best/stop/log)
    f_off, f_res = resonance_offset(f, s11_dB)
    cost = worst + GUIDE_W * f_off                   # SHAPED cost the optimizer minimizes
    _cache[key] = cost
    history.append((len(history) + 1, values, worst))
    new_best = worst < best['obj']
    if new_best:
        best.update(obj=worst, x=x.copy(), f=f, s11=s11_dB)
        save_best_json()             # persist immediately on every new best (true worst)
    save_log()                       # keep the CSV current every eval
    is_best = worst == best['obj']
    print('[eval %2d] %s -> worst=%+6.2f dB  (res %.3f GHz, cost %+.2f)%s'
          % (len(history), '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             worst, f_res/1e9, cost, '  <-- best' if is_best else ''))
    update_live_plot(len(history), f, s11_dB, worst, is_best)
    if worst <= STOP_MARGIN_DB:
        print('Target met (worst %.2f <= %.2f dB) - stopping.' % (worst, STOP_MARGIN_DB))
        raise _Stop()
    return cost


def run_optimization():
    print('===== SLOTTED INSET PATCH: minimize worst S11 over %.2f-%.2f GHz ====='
          % (BAND_LO/1e9, BAND_HI/1e9))
    print('Tuning:', ', '.join(names))
    print('Start: ', '  '.join('%s=%.3f' % (n, v) for n, v in zip(names, x0)), '\n')
    # Explicit initial simplex: perturb each knob by a MEANINGFUL amount
    # (max of 8% of its range and 5 grid steps), clipped to bounds. scipy's default
    # perturbs a zero-valued knob by only 2.5e-4 - which snap() rounds back to 0, so
    # a knob starting at 0 (e.g. slot_x) would never move. This guarantees every knob
    # is explored from the start.
    n = len(x0)
    simplex = np.tile(x0, (n + 1, 1))
    for i in range(n):
        step = max(0.08 * (hi[i] - lo[i]), 5 * steps[i])
        cand = x0[i] + step
        if cand > hi[i]:
            cand = x0[i] - step
        simplex[i + 1, i] = np.clip(cand, lo[i], hi[i])
    try:
        minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                 options={'initial_simplex': simplex, 'xatol': float(steps.min()),
                          'fatol': 0.2, 'maxfev': 10 * MAX_EVALS, 'maxiter': 10 * MAX_EVALS})
    except _Stop:
        pass


def report():
    bx = dict(zip(names, best['x']))
    set_params(bx)
    fit, xb, xo, wi, sw = p1._slot_geom()
    print('\n=========================== BEST DESIGN ===========================')
    print('Worst |S11| in %.2f-%.2f GHz: %+.2f dB  (%s)'
          % (BAND_LO/1e9, BAND_HI/1e9, best['obj'],
             'PASS' if best['obj'] < TARGET_DB else 'best found'))
    print('U-slot active: %s  (len=%.1f w=%.1f x=%.1f)' % (fit, bx['slot_len'], bx['slot_w'], bx['slot_x']))
    for n in names:
        print('    %-8s = %7.3f mm   (was %.3f)' % (n, bx[n], PARAMS[n][0]))
    print('\n--- paste into inset_slot_build.py DESIGN PARAMETERS ---')
    for n in names:
        print('%-8s = %.2f' % (n, bx[n]))
    print('-------------------------------------------------------\n')

    save_best_json()                 # already saved each best; ensure final
    save_log()

    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8)
        ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target band')
        ax.set(title='Slotted inset patch S11 (worst in band %+.2f dB)' % best['obj'],
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved: optimized_params.json, optimization_log.csv, optimized_s11.png')


if __name__ == '__main__':
    run_optimization()
    report()
    if LIVE_PLOT and _gui_ok:
        plt.ioff()
        plt.show()
