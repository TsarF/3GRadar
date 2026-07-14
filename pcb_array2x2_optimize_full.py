"""
Full-geometry realized-gain optimizer for the 2x2 via-fed PCB array
(pcb_array2x2_build). Like pcb_array2x2_optimize.py but with the patch dimensions
opened up as well.

Tunes 6 knobs:
    d         element spacing       -> array directivity / sidelobes / coupling
    fv        via offset            -> the match
    L,  W     driven patch L x W     -> freq + feed resistance + aperture (gain)
    Lp, Wp    parasitic patch L x W  -> 2nd resonance + bandwidth + aperture

Objective: maximize the worst-in-band REALIZED boresight gain over 3.1-3.4 GHz,
    G = D_broadside * eta_rad * (1 - |S11|^2).

W/Wp add aperture (more directivity) AND a second handle on the match alongside fv,
which is useful once mutual coupling has shifted the array's active impedance.

Warm-starts from this file's own result if present, else the d/fv/L run
(pcb_array2x2_opt), else the build defaults. Each eval is a full-array FDTD + NF2FF
(a few M cells) -> minutes; 6 knobs -> budget a long run.

Run:  python pcb_array2x2_optimize_full.py
Out:  pcb_array2x2_gain6_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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

import pcb_array2x2_build as a

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
NGAIN            = 5
GAIN_TARGET      = None

# (initial, lower, upper, step) in mm
PARAMS = {
    'd':  (a.d,  44.0, 80.0, 0.5),     # spacing
    'fv': (a.fv,  1.0, 10.0, 0.1),     # via offset -> match
    'L':  (a.L,  16.0, 26.0, 0.1),     # driven length -> freq
    'W':  (a.W,  14.0, 34.0, 0.2),     # driven width  -> feed R + aperture
    'Lp': (a.Lp, 16.0, 32.0, 0.1),     # parasitic length
    'Wp': (a.Wp, 18.0, 42.0, 0.2),     # parasitic width -> aperture
}

# warm-start: this file's own result first, then the d/fv/L run, then build defaults
WARM_CANDIDATES = [
    os.path.join(os.getcwd(), 'pcb_array2x2_gain6_opt', 'optimized_params.json'),
    os.path.join(os.getcwd(), 'pcb_array2x2_opt', 'optimized_params.json'),
]
MAX_EVALS = 120

EVAL_NRTS         = 100000         # array is high-Q; needs the timesteps to settle
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121
EVAL_FPAD         = 0.20e9

FINAL_CHARACTERIZE = True
FINAL_NRTS         = 120000
FINAL_ENDCRITERIA  = 1e-4

BOUNDARY = ['MUR', 'MUR', 'MUR', 'MUR', 'PML_8', 'PML_8']
sim_path = os.path.join(os.getcwd(), 'pcb_array2x2_gain6_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    for name, val in values.items():
        setattr(a, name, float(val))
    a._derive()


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
    """Returns (f, s11_dB, f_g, Gr_dBi, worst_Gr_dBi)."""
    set_params(values)
    os.makedirs(run_dir, exist_ok=True)
    FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
    FDTD.SetGaussExcite(a.f0, a.fc)
    FDTD.SetBoundaryCond(BOUNDARY)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    a.build_antenna(CSX, FDTD)
    port = a.add_feed_port(FDTD)
    nf2ff = FDTD.CreateNF2FFBox()
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=a.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))
    Pacc_f   = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
    s11mag_f = np.abs(s11)

    f_g    = np.linspace(BAND_LO, BAND_HI, NGAIN)
    center = [0, 0, (a.z_top / 2) * a.unit]
    nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
    Pacc_g = np.interp(f_g, f, Pacc_f)
    s11_g  = np.interp(f_g, f, s11mag_f)
    Gr_dBi = np.empty(NGAIN)
    for i in range(NGAIN):
        eta_rad   = np.clip(nf.Prad[i] / Pacc_g[i], 0, 1)
        eta_match = max(1.0 - s11_g[i]**2, 0.0)
        Gr_dBi[i] = 10 * np.log10(max(nf.Dmax[i] * eta_rad * eta_match, 1e-6))

    _rmtree_retry(run_dir)
    return f, s11_dB, f_g, Gr_dBi, float(Gr_dBi.min())


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
        ax = _live['ax']; ax.clear()
        for i, (n, ff, ss, lab, bf) in enumerate(_recent):
            newest = (i == len(_recent) - 1)
            ax.plot(ff/1e9, ss, lw=2.4 if newest else 1.3, alpha=1.0 if newest else 0.5,
                    label='eval %d: %s%s' % (n, lab, '  (best)' if bf else ''))
        ax.axhline(-10, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        bt = ('best Gr = %.2f dBi' % best['Gr']) if np.isfinite(best['Gr']) else '...'
        ax.set(title='2x2 array (6-knob) realized-gain search  (%s)  eval %d' % (bt, eval_no),
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
    """Warm-start from the candidate with the BEST recorded objective (not just the
    first that exists), so a stale partial never shadows a better prior result."""
    init = {n: PARAMS[n][0] for n in names}
    cands = []
    for path in WARM_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
                cands.append((data.get('worst_realized_gain_dBi', -np.inf), path, data))
            except Exception as e:
                print('warm-start read failed (%s)' % e)
    if cands:
        obj, path, data = max(cands, key=lambda c: c[0])
        for section in ('params_mm', 'fixed_mm'):
            for k, v in data.get(section, {}).items():
                if k in init:
                    init[k] = float(v)
        print('Warm-starting from %s (best Gr=%.2f dBi of %d candidate(s))'
              % (path, obj, len(cands)))
    return init


_init = initial_values()
x0 = snap(np.array([_init[n] for n in names], dtype=float))

history = []
best    = {'Gr': -np.inf, 'x': x0.copy(), 'f': None, 's11': None, 'f_g': None, 'Gr_dBi': None}
_cache  = {}


def save_best_json():
    """Persist the current best immediately (called on every new best) so a crash
    or Ctrl-C mid-run never loses progress. Atomic write via a temp file."""
    if not np.isfinite(best['Gr']):
        return
    bx = dict(zip(names, best['x']))
    tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump({'objective': 'max worst-in-band realized boresight gain (2x2, 6 knobs)',
                   'worst_realized_gain_dBi': best['Gr'],
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'evals_so_far': len(history),
                   'params_mm': bx}, fh, indent=2)
    os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))


def save_log():
    """Rewrite the full eval log (called every eval, so the CSV survives a crash)."""
    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['worst_realized_gain_dBi'])
        for i, vals, gr in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names] + ['%.3f' % gr])


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
    obj = -worst_Gr
    _cache[key] = obj
    history.append((len(history) + 1, values, worst_Gr))
    new_best = worst_Gr > best['Gr']
    if new_best:
        best.update(Gr=worst_Gr, x=x.copy(), f=f, s11=s11_dB, f_g=f_g, Gr_dBi=Gr_dBi)
        save_best_json()                 # persist immediately on every new best
    save_log()                           # keep the CSV current every eval
    is_best = worst_Gr == best['Gr']
    print('[eval %2d] %s -> Gr=%.2f dBi (band %.2f..%.2f)%s'
          % (len(history), '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             worst_Gr, Gr_dBi.min(), Gr_dBi.max(), '  <-- best' if is_best else ''))
    update_live_plot(len(history), f, s11_dB, 'Gr=%.2f dBi' % worst_Gr, is_best)
    if GAIN_TARGET is not None and worst_Gr >= GAIN_TARGET:
        print('Target met (%.2f >= %.2f dBi) - stopping.' % (worst_Gr, GAIN_TARGET))
        raise _Stop()
    return obj


def run_optimization():
    print('===== 2x2 ARRAY (6-knob): maximize worst-in-band REALIZED GAIN over '
          '%.2f-%.2f GHz =====' % (BAND_LO/1e9, BAND_HI/1e9))
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
    print('\n=========================== BEST ARRAY DESIGN (6-knob) ===========================')
    print('Worst-in-band realized boresight gain: %.2f dBi' % best['Gr'])
    if best['Gr_dBi'] is not None:
        print('  across band: ' + '  '.join('%.2fGHz=%.2f' % (fg/1e9, g)
                                             for fg, g in zip(best['f_g'], best['Gr_dBi'])))
    for n in names:
        print('    %-4s = %7.3f mm   (was %.3f)' % (n, bx[n], PARAMS[n][0]))
    print('\n--- paste: d -> pcb_array2x2_build.py ; the rest -> pcb_patch_build.py ---')
    print('d  = %.2f' % bx['d'])
    print('fv = %.2f' % bx['fv'])
    print('L  = %.2f' % bx['L'])
    print('W  = %.2f' % bx['W'])
    print('Lp = %.2f' % bx['Lp'])
    print('Wp = %.2f' % bx['Wp'])
    print('---------------------------------------------------------------------------\n')

    save_best_json()                     # already saved each new best; ensure final
    save_log()

    if best['f'] is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(best['f']/1e9, best['s11'], lw=1.8); ax[0].axhline(-10, color='r', ls='--', lw=0.8)
        ax[0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='band')
        ax[0].set(title='|S11| of best array', xlabel='Frequency (GHz)',
                  ylabel='|S11| (dB)'); ax[0].legend(); ax[0].grid(True)
        ax[1].plot(best['f_g']/1e9, best['Gr_dBi'], 'o-')
        ax[1].set(title='Realized boresight gain (worst %.2f dBi)' % best['Gr'],
                  xlabel='Frequency (GHz)', ylabel='G_realized (dBi)'); ax[1].grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved: optimized_params.json, optimization_log.csv, optimized_s11.png')


def characterize_best():
    set_params(dict(zip(names, best['x'])))
    run_dir = os.path.join(sim_path, 'best_full')
    os.makedirs(run_dir, exist_ok=True)
    FDTD = openEMS(NrTS=FINAL_NRTS, EndCriteria=FINAL_ENDCRITERIA)
    FDTD.SetGaussExcite(a.f0, a.fc)
    FDTD.SetBoundaryCond(BOUNDARY)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    a.build_antenna(CSX, FDTD)
    port = a.add_feed_port(FDTD)
    nf2ff = FDTD.CreateNF2FFBox()
    print('\nRunning full characterization of the best array...')
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(a.f0 - a.fc, a.f0 + a.fc, 601)
    port.CalcPort(run_dir, f, ref_impedance=a.feed_R)
    s11 = port.uf_ref / port.uf_inc; s11_dB = 20 * np.log10(np.abs(s11)); Zin = port.uf_tot / port.if_tot
    center = [0, 0, (a.z_top / 2) * a.unit]
    f_pat = np.linspace(BAND_LO, BAND_HI, 7)
    d_res = nf2ff.CalcNF2FF(run_dir, f_pat, np.array([0.0]), np.array([0.0]), center=center)
    Dbroad = 10 * np.log10(np.array([d_res.Dmax[i] for i in range(len(f_pat))]))
    f_mid = 0.5 * (BAND_LO + BAND_HI)
    theta = np.arange(-180, 180.5, 1.0)
    e_cut = nf2ff.CalcNF2FF(run_dir, f_mid, theta, np.array([0.0]),  center=center)
    h_cut = nf2ff.CalcNF2FF(run_dir, f_mid, theta, np.array([90.0]), center=center)

    def _norm_db(res):
        E = np.sqrt(np.abs(res.E_theta[0][:, 0])**2 + np.abs(res.E_phi[0][:, 0])**2)
        return 20 * np.log10(E / np.max(E) + 1e-12)
    Eplane, Hplane = _norm_db(e_cut), _norm_db(h_cut)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(f/1e9, s11_dB); ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target')
    ax[0, 0].set(title='Array S11', xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
    ax[0, 0].legend(); ax[0, 0].grid(True)
    ax[0, 1].plot(f/1e9, np.real(Zin), label='Re'); ax[0, 1].plot(f/1e9, np.imag(Zin), label='Im')
    ax[0, 1].axhline(50, color='k', ls=':', lw=0.8)
    ax[0, 1].set(title='Input impedance', xlabel='Frequency (GHz)', ylabel='Z (ohm)')
    ax[0, 1].legend(); ax[0, 1].grid(True)
    ax[1, 0].plot(f_pat/1e9, Dbroad, 'o-')
    ax[1, 0].set(title='Broadside directivity', xlabel='Frequency (GHz)', ylabel='D (dBi)')
    ax[1, 0].grid(True)
    ax[1, 1].plot(theta, Eplane, label='E-plane (phi=0)'); ax[1, 1].plot(theta, Hplane, '--', label='H-plane (phi=90)')
    ax[1, 1].set(title='Co-pol pattern @ %.2f GHz' % (f_mid/1e9), xlabel='Theta (deg)',
                 ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
    ax[1, 1].legend(); ax[1, 1].grid(True)
    fig.suptitle('Optimized 2x2 array, 6 knobs (worst-in-band Gr = %.2f dBi, d=%.1f mm)'
                 % (best['Gr'], best['x'][names.index('d')]), y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'optimized_results.png')
    fig.savefig(out, dpi=130)
    print('Saved full characterization plot to', out)
    _rmtree_retry(run_dir)


if __name__ == '__main__':
    run_optimization()
    report()
    if FINAL_CHARACTERIZE and np.isfinite(best['Gr']):
        characterize_best()
    if LIVE_PLOT and _gui_ok:
        plt.ioff()
        plt.show()
