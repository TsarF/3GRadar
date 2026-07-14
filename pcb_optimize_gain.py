"""
Realized-gain optimizer for the via-fed driven patch + air-gap parasitic on the
JLCPCB 4-layer stackup (pcb_patch_build).

Maximizes the worst-in-band REALIZED boresight gain over 3.1-3.4 GHz:

    G_realized = D_broadside * eta_rad * (1 - |S11|^2)
                 [ directivity x radiation eff. ] x [ mismatch eff. ]

i.e. the directivity actually delivered forward per unit available input power -
match, dielectric/conductor loss, and pattern all in one number. Maximizing the
WORST in-band value keeps the forward power high across the whole band.

Warm-starts from the current design (pcb_patch_opt/optimized_params.json if a prior
S11 run exists, else pcb_patch_build's defaults). Tunes all 7 knobs.

Run:  python pcb_optimize_gain.py
Out:  pcb_patch_gain_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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

import pcb_patch_build as p

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
NGAIN            = 5               # in-band frequencies scored for realized gain
GAIN_TARGET      = None            # optional early stop (dBi); None = full budget

# all 7 knobs of the air-gap PCB design: (initial, lower, upper, step) in mm
PARAMS = {
    'L':       (p.L,       16.0, 26.0, 0.1),
    'W':       (p.W,       18.0, 36.0, 0.2),
    'Lp':      (p.Lp,      16.0, 32.0, 0.1),
    'Wp':      (p.Wp,      18.0, 42.0, 0.2),
    'fv':      (p.fv,       1.0, 10.0, 0.1),
    'antipad': (p.antipad,  0.3,  1.2, 0.05),
    'h_air':   (p.h_air,    2.0, 12.0, 0.1),
}

# warm-start from the matched S11 design (or build defaults if none yet)
WARM_START_FROM = os.path.join(os.getcwd(), 'pcb_patch_opt', 'optimized_params.json')
MAX_EVALS = 150

EVAL_NRTS         = 50000
EVAL_ENDCRITERIA  = 1e-3
EVAL_NFREQ        = 121
EVAL_FPAD         = 0.20e9

FINAL_CHARACTERIZE = True
FINAL_NRTS         = 70000
FINAL_ENDCRITERIA  = 1e-4

BOUNDARY = ['MUR', 'MUR', 'MUR', 'MUR', 'PML_8', 'PML_8']

sim_path = os.path.join(os.getcwd(), 'pcb_patch_gain_opt')
os.makedirs(sim_path, exist_ok=True)
# ==========================================================================


def set_params(values):
    for name, val in values.items():
        setattr(p, name, float(val))
    p._derive()


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
    """One FDTD solve. Returns (f, s11_dB, f_g, Gr_dBi, worst_Gr_dBi)."""
    set_params(values)
    os.makedirs(run_dir, exist_ok=True)
    FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
    FDTD.SetGaussExcite(p.f0, p.fc)
    FDTD.SetBoundaryCond(BOUNDARY)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    p.build_antenna(CSX, FDTD)
    port = p.add_feed_port(FDTD)
    nf2ff = FDTD.CreateNF2FFBox()
    CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
    FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=8)

    f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
    port.CalcPort(run_dir, f, ref_impedance=p.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))

    Pacc_f   = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
    s11mag_f = np.abs(s11)

    f_g    = np.linspace(BAND_LO, BAND_HI, NGAIN)
    center = [0, 0, (p.z_top / 2) * p.unit]
    nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)

    Pacc_g = np.interp(f_g, f, Pacc_f)
    s11_g  = np.interp(f_g, f, s11mag_f)
    Gr_dBi = np.empty(NGAIN)
    for i in range(NGAIN):
        D_broad   = nf.Dmax[i]
        eta_rad   = np.clip(nf.Prad[i] / Pacc_g[i], 0, 1)
        eta_match = max(1.0 - s11_g[i]**2, 0.0)
        Gr        = D_broad * eta_rad * eta_match
        Gr_dBi[i] = 10 * np.log10(max(Gr, 1e-6))

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
        best_txt = ('best Gr = %.2f dBi' % best['Gr']) if np.isfinite(best['Gr']) else '...'
        ax.set(title='Realized-gain search (4-layer + air parasitic)  (%s)  eval %d'
               % (best_txt, eval_no), xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
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
    obj = -worst_Gr
    _cache[key] = obj
    history.append((len(history) + 1, values, worst_Gr))
    if worst_Gr > best['Gr']:
        best.update(Gr=worst_Gr, x=x.copy(), f=f, s11=s11_dB, f_g=f_g, Gr_dBi=Gr_dBi)
    is_best = worst_Gr == best['Gr']
    print('[eval %2d] %s -> Gr=%.2f dBi (band %.2f..%.2f)%s'
          % (len(history), '  '.join('%s=%6.3f' % (n, v) for n, v in values.items()),
             worst_Gr, Gr_dBi.min(), Gr_dBi.max(), '  <-- best' if is_best else ''))
    update_live_plot(len(history), f, s11_dB, 'Gr=%.2f dBi' % worst_Gr, is_best)
    if GAIN_TARGET is not None and worst_Gr >= GAIN_TARGET:
        print('Realized-gain target met (%.2f >= %.2f dBi) - stopping.' % (worst_Gr, GAIN_TARGET))
        raise _Stop()
    return obj


def run_optimization():
    print('===== 4-LAYER + AIR PARASITIC: maximize worst-in-band REALIZED GAIN '
          'over %.2f-%.2f GHz =====' % (BAND_LO/1e9, BAND_HI/1e9))
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
        print('  across band: ' + '  '.join('%.2fGHz=%.2f' % (fg/1e9, g)
                                             for fg, g in zip(best['f_g'], best['Gr_dBi'])))
    for n in names:
        print('    %-8s = %7.3f mm   (was %.3f)' % (n, bx[n], PARAMS[n][0]))

    print('\n--- paste into pcb_patch_build.py DESIGN PARAMETERS ---')
    for n in names:
        print('%-8s = %.2f' % (n, bx[n]))
    print('-------------------------------------------------------\n')

    with open(os.path.join(sim_path, 'optimized_params.json'), 'w') as fh:
        json.dump({'objective': 'max worst-in-band realized boresight gain',
                   'worst_realized_gain_dBi': best['Gr'],
                   'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'params_mm': bx,
                   'fixed_mm': {'t_core': p.t_core, 't_prepreg': p.t_prepreg,
                                'h_sub2': p.h_sub2, 'a_via': p.a_via, 'Wf': p.Wf,
                                'Lf': p.Lf, 'eps_r': p.eps_r}}, fh, indent=2)

    with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['eval'] + names + ['worst_realized_gain_dBi'])
        for i, vals, gr in history:
            w.writerow([i] + ['%.4f' % vals[n] for n in names] + ['%.3f' % gr])

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
    if not np.isfinite(best['Gr']):
        return
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
    fig.suptitle('Max realized-gain via-fed patch + air parasitic (4-layer PCB)  '
                 '(worst-in-band Gr = %.2f dBi)' % best['Gr'], y=1.0)
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
