"""
Part 5 - Maximize the realized boresight gain (the "realized forward power") of
the inset-fed parasitic patch across 3.1-3.4 GHz.

Realized gain folds match AND radiation into one number:

    G_realized = D_broadside * eta_rad * (1 - |S11|^2)
                 [    gain G     ]   [ mismatch eff. ]

i.e. gain de-rated by the mismatch loss (1-|S11|^2) - NOT S11 times gain. It is
the forward power actually radiated per unit available input power, relative to
isotropic. A bad match (high |S11|) bleeds realized gain even if raw directivity
is high, so this single objective trades off match and gain automatically with no
hard constraint.

Objective: maximize the WORST (minimum) realized boresight gain across the band,
so forward power stays high everywhere in 3.1-3.4 GHz - not just at one frequency.

Each evaluation is a full FDTD solve plus an NF2FF far-field (broadside only, so
cheap), so it costs about as much as a Part 4 eval. Warm-starts from the
S11-optimized design.

Run:  python part5_optimize_realized_gain.py
Out:  inset_patch_gain_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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

import inset_patch_build as p1          # `p1` is just the geometry-module handle

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
NGAIN            = 5               # in-band frequencies at which realized gain is scored
GAIN_TARGET      = None           # optional early stop, e.g. 8.0 (dBi); None = full budget

# Six geometry knobs. Wider W/Wp bounds than Part 3/4 - a bigger aperture buys
# directivity, which is part of what realized gain rewards.
PARAMS = {
    'L':     (p1.L,     18.0, 28.0, 0.1),
    'W':     (p1.W,     22.0, 40.0, 0.2),
    'y0':    (p1.y0,     3.0, 11.0, 0.1),
    'h_air': (p1.h_air,  2.0,  9.0, 0.1),
    'Lp':    (p1.Lp,    18.0, 28.0, 0.1),
    'Wp':    (p1.Wp,    22.0, 40.0, 0.2),
}

WARM_START_FROM = os.path.join(os.getcwd(), 'inset_patch_opt', 'optimized_params.json')

MAX_EVALS = 120

EVAL_NRTS         = 40000
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121
EVAL_FPAD         = 0.20e9

FINAL_CHARACTERIZE = True
FINAL_NRTS         = 60000
FINAL_ENDCRITERIA  = 1e-4

sim_path = os.path.join(os.getcwd(), 'inset_patch_gain_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    """Overwrite tuned values on the geometry module + refresh derived globals."""
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


def simulate(values, run_dir):
    """One FDTD solve. Returns (f, s11_dB, f_g, Gr_dBi, worst_Gr_dBi).
    Realized boresight gain Gr(f) = D_broadside * eta_rad * (1-|S11|^2)."""
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
    nf2ff = FDTD.CreateNF2FFBox()

    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))

    # port-side quantities on the full grid (real arrays -> safe to interpolate)
    Pacc_f   = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))   # accepted power
    s11mag_f = np.abs(s11)

    # broadside directivity + radiated power at the in-band scoring frequencies
    f_g    = np.linspace(BAND_LO, BAND_HI, NGAIN)
    center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
    nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)

    Pacc_g = np.interp(f_g, f, Pacc_f)
    s11_g  = np.interp(f_g, f, s11mag_f)
    Gr_dBi = np.empty(NGAIN)
    for i in range(NGAIN):
        D_broad   = nf.Dmax[i]                              # broadside directivity (lin)
        eta_rad   = np.clip(nf.Prad[i] / Pacc_g[i], 0, 1)   # radiation efficiency
        eta_match = max(1.0 - s11_g[i]**2, 0.0)             # mismatch efficiency
        Gr        = D_broad * eta_rad * eta_match           # realized gain (lin)
        Gr_dBi[i] = 10 * np.log10(max(Gr, 1e-6))

    _rmtree_retry(run_dir)
    return f, s11_dB, f_g, Gr_dBi, float(Gr_dBi.min())


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
        ax.axhline(-10, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        best_txt = ('best Gr = %.2f dBi' % best['Gr']) if np.isfinite(best['Gr']) else '...'
        ax.set(title='Max realized-gain search  (%s)  -  eval %d' % (best_txt, eval_no),
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

history = []                          # (eval#, params, worst_Gr)
best    = {'Gr': -np.inf, 'x': x0.copy(), 'f': None, 's11': None, 'f_g': None, 'Gr_dBi': None}
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
    f, s11_dB, f_g, Gr_dBi, worst_Gr = simulate(values, run_dir)

    obj = -worst_Gr                   # maximize worst-in-band realized gain
    _cache[key] = obj
    history.append((len(history) + 1, values, worst_Gr))

    if worst_Gr > best['Gr']:
        best.update(Gr=worst_Gr, x=x.copy(), f=f, s11=s11_dB, f_g=f_g, Gr_dBi=Gr_dBi)

    is_best = worst_Gr == best['Gr']
    label = 'Gr=%.2f dBi (band %.2f..%.2f)' % (worst_Gr, Gr_dBi.min(), Gr_dBi.max())
    print('[eval %2d] %s -> %s%s'
          % (len(history),
             '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             label, '  <-- best' if is_best else ''))

    update_live_plot(len(history), f, s11_dB, 'Gr=%.2f dBi' % worst_Gr, is_best)

    if GAIN_TARGET is not None and worst_Gr >= GAIN_TARGET:
        print('Realized-gain target met (%.2f >= %.2f dBi) - stopping.' % (worst_Gr, GAIN_TARGET))
        raise _Stop()
    return obj


def run_optimization():
    print('========= MAXIMIZING worst-in-band REALIZED BORESIGHT GAIN over '
          '%.2f-%.2f GHz =========' % (BAND_LO/1e9, BAND_HI/1e9))
    print('Tuning:', ', '.join(names))
    print('Start: ', '  '.join('%s=%.3f' % (n, v) for n, v in zip(names, x0)), '\n')
    try:
        minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                 options={'xatol': float(steps.min()), 'fatol': 0.1,
                          'maxfev': 10 * MAX_EVALS, 'maxiter': 10 * MAX_EVALS})
    except _Stop:
        pass


def report():
    if not np.isfinite(best['Gr']):
        print('\nNo successful evaluation.')
        return
    bx = dict(zip(names, best['x']))
    set_params(bx)

    print('\n=========================== BEST DESIGN (REALIZED GAIN) ===========================')
    print('Worst-in-band realized boresight gain: %.2f dBi' % best['Gr'])
    if best['Gr_dBi'] is not None:
        print('  realized gain across band: ' +
              '  '.join('%.2fGHz=%.2f' % (fg/1e9, g)
                        for fg, g in zip(best['f_g'], best['Gr_dBi'])))
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
        json.dump({'objective': 'max worst-in-band realized boresight gain',
                   'worst_realized_gain_dBi': best['Gr'],
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'params_mm': bx,
                   'fixed_mm': {'h_sub': p1.h_sub, 'Wf': p1.Wf, 'g': p1.g,
                                'Lf': p1.Lf, 'margin': p1.margin}}, fh, indent=2)

    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['worst_realized_gain_dBi'])
        for i, vals, g in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names] + ['%.3f' % g])

    if best['f'] is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(best['f']/1e9, best['s11'], lw=1.8)
        ax[0].axhline(-10, color='r', ls='--', lw=0.8)
        ax[0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='band')
        ax[0].set(title='|S11| of best realized-gain design', xlabel='Frequency (GHz)',
                  ylabel='|S11| (dB)'); ax[0].legend(); ax[0].grid(True)
        ax[1].plot(best['f_g']/1e9, best['Gr_dBi'], 'o-')
        ax[1].set(title='Realized boresight gain (worst %.2f dBi)' % best['Gr'],
                  xlabel='Frequency (GHz)', ylabel='G_realized (dBi)'); ax[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved: optimized_params.json, optimization_log.csv, optimized_s11.png')


def characterize_best():
    """Full Part 2-style solve on the best design -> S11, Zin, broadside directivity,
    co-pol principal-plane pattern."""
    if not np.isfinite(best['Gr']):
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

    fig.suptitle('Max realized-gain inset-fed parasitic patch  '
                 '(worst-in-band Gr = %.2f dBi)' % best['Gr'], y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'optimized_results.png')
    fig.savefig(out, dpi=130)
    print('Saved full Part 2-style plot to', out)
    _rmtree_retry(run_dir)


if __name__ == '__main__':
    run_optimization()
    report()
    if FINAL_CHARACTERIZE and np.isfinite(best['Gr']):
        characterize_best()
    if LIVE_PLOT and _gui_ok:
        plt.ioff()
        plt.show()
