"""
Impedance diagnostic for the current best fpc3 design (feed + PRS + RCM).  Fixes the
cavity/RCM geometry to the DE best (read from fpc3_gain_de_opt/optimized_params.json)
and runs ONE FDTD solve to extract the full picture needed to DESIGN a feed match:

    * input impedance  Z(f) = R(f) + jX(f)   at the feed port
    * |S11|(f)
    * broadside directivity(f)
    * Smith chart (Gamma trajectory)

Saves f, Z, S11, D to fpc3_impedance/imp_data.npz + a CSV, and a 2x2 plot.  Use the
R/X curves to pick a matching topology (quarter-wave transformer sqrt(50*R) if X~0 at
resonance; feed-line length shift to a point where X=0; series/shunt stub from the Smith
trajectory).  Then modify the feed in fpc3_build.py and re-run to validate S11 < -10 dB.

Run:            python fpc3_impedance.py
Replot only:    python fpc3_impedance.py --replot
"""

import os
import sys
import json
import time
import shutil
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc3_build as p1
p1.RCM_ON = True
p1.mesh_res = (p1.C0 / (p1.f0 + p1.fc)) / p1.unit / 22.0     # moderate mesh for accurate Z

BAND_LO, BAND_HI = 3.10e9, 3.40e9
NRTS, ENDC, NFREQ = 90000, 5e-5, 601
sim_path = os.path.join(os.getcwd(), 'fpc3_impedance')
os.makedirs(sim_path, exist_ok=True)
data_npz = os.path.join(sim_path, 'imp_data.npz')


USE_BUILD_DEFAULTS = True          # True -> use fpc3_build defaults as-is (mesh-PRS test)


def load_best():
    """Set fpc3_build to the DE best (or the module defaults if no JSON)."""
    if USE_BUILD_DEFAULTS:
        p1._recompute()
        print('using fpc3_build defaults: PRS_MESH=%s mesh_N=%d h_cav=%.1f RCM_MULTI=%s'
              % (p1.PRS_MESH, p1.mesh_N, p1.h_cav, p1.RCM_MULTI))
        return
    jp = os.path.join(os.getcwd(), 'fpc3_gain_de_opt', 'optimized_params.json')
    if os.path.exists(jp):
        with open(jp) as f:
            params = json.load(f).get('params_mm', {})
        for k, v in params.items():
            if hasattr(p1, k):
                setattr(p1, k, float(v))
        print('loaded best:', params)
    p1._recompute()


@contextlib.contextmanager
def _redirect(path):
    sys.stdout.flush(); sys.stderr.flush()
    fout = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    s1, s2 = os.dup(1), os.dup(2)
    try:
        os.dup2(fout, 1); os.dup2(fout, 2); yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(s1, 1); os.dup2(s2, 2); os.close(s1); os.close(s2); os.close(fout)


def _rmtree(path):
    for _ in range(5):
        try:
            shutil.rmtree(path); return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def solve():
    load_best()
    run_dir = os.path.join(sim_path, 'run')
    log = os.path.join(sim_path, 'openems.log')
    print('solving impedance (output -> %s) ...' % log)
    t0 = time.time()
    os.makedirs(run_dir, exist_ok=True)
    with _redirect(log):
        FDTD = openEMS(NrTS=NRTS, EndCriteria=ENDC)
        FDTD.SetGaussExcite(p1.f0, p1.fc)
        FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])
        CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
        p1.build_antenna(CSX, FDTD)
        port = FDTD.AddLumpedPort(1, p1.feed_R, [p1.feed_x, p1.feed_y, 0],
                                  [p1.feed_x, p1.feed_y, p1.h_sub], 'z', 1.0, priority=5, edges2grid='xy')
        nf2ff = FDTD.CreateNF2FFBox()
        CSX.Write2XML(os.path.join(run_dir, 'a.xml'))
        FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=8)
        f = np.linspace(BAND_LO - 0.25e9, BAND_HI + 0.25e9, NFREQ)
        port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
        Z = port.uf_tot / port.if_tot                       # input impedance R + jX
        s11 = port.uf_ref / port.uf_inc
        center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
        f_g = np.linspace(BAND_LO, BAND_HI, 7)
        nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
        D = 10 * np.log10(np.array([nf.Dmax[i] for i in range(7)]))
    _rmtree(run_dir)
    print('done in %.0f s' % (time.time() - t0))
    np.savez(data_npz, f=f, Z=Z, s11=s11, f_g=f_g, D=D, Zref=p1.feed_R)
    return dict(f=f, Z=Z, s11=s11, f_g=f_g, D=D, Zref=p1.feed_R)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if '--replot' in sys.argv:
        if not os.path.exists(data_npz):
            sys.exit('no imp_data.npz - run without --replot first')
        d = dict(np.load(data_npz)); d['Zref'] = float(d['Zref'])
    else:
        d = solve()

    for bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(bk, force=True); break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    f, Z, s11, Zref = d['f'], d['Z'], d['s11'], d['Zref']
    R, X = Z.real, Z.imag
    s11_dB = 20 * np.log10(np.abs(s11))
    band = (f >= BAND_LO) & (f <= BAND_HI)
    i0 = int(np.argmin(np.abs(f - 3.25e9)))
    ires = int(np.argmin(np.abs(s11)))
    print('\n==================== IMPEDANCE @ feed port ====================')
    for tag, k in (('3.10', int(np.argmin(np.abs(f - BAND_LO)))), ('3.25', i0),
                   ('3.40', int(np.argmin(np.abs(f - BAND_HI))))):
        print('  %s GHz : Z = %6.1f %+6.1fj ohm | S11 = %+5.2f dB' % (tag, R[k], X[k], s11_dB[k]))
    print('  best match: %.3f GHz, Z = %.1f %+.1fj, S11 = %.2f dB' % (f[ires]/1e9, R[ires], X[ires], s11_dB[ires]))
    print('  worst in-band S11 = %+.2f dB' % float(np.max(s11_dB[band])))
    print('===============================================================\n')

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(f/1e9, R, label='R (real)'); ax[0, 0].plot(f/1e9, X, label='X (imag)')
    ax[0, 0].axhline(Zref, color='g', ls=':', lw=0.8, label='%.0f ohm' % Zref)
    ax[0, 0].axhline(0, color='k', lw=0.5); ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.08)
    ax[0, 0].set(title='Input impedance Z = R + jX', xlabel='GHz', ylabel='ohm'); ax[0, 0].legend()
    ax[0, 1].plot(f/1e9, s11_dB); ax[0, 1].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 1].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.08)
    ax[0, 1].set(title='|S11|', xlabel='GHz', ylabel='dB', ylim=(-25, 0))
    ax[1, 0].plot(d['f_g']/1e9, d['D'], 'o-'); ax[1, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.08)
    ax[1, 0].set(title='Broadside directivity', xlabel='GHz', ylabel='dBi')
    # Smith chart (Gamma trajectory)
    G = (Z - Zref) / (Z + Zref)
    th = np.linspace(0, 2*np.pi, 200)
    ax[1, 1].plot(np.cos(th), np.sin(th), 'k', lw=0.8)
    for rr in (0.2, 0.5, 1, 2, 5):                          # constant-R circles
        c = rr/(1+rr); rad = 1/(1+rr)
        ax[1, 1].plot(c + rad*np.cos(th), rad*np.sin(th), color='0.8', lw=0.5)
    ax[1, 1].plot(G[band].real, G[band].imag, 'b', lw=2, label='in-band')
    ax[1, 1].plot(G.real, G.imag, 'b', lw=0.6, alpha=0.4)
    ax[1, 1].plot(G[i0].real, G[i0].imag, 'ro', label='3.25 GHz')
    ax[1, 1].plot(0, 0, 'g+', ms=10)
    ax[1, 1].set(title='Smith (Gamma), ref %.0f ohm' % Zref, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax[1, 1].set_aspect('equal'); ax[1, 1].legend(fontsize=8)
    for a_ in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a_.grid(True)
    fig.suptitle('fpc3 best design — impedance / match diagnostic', y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'impedance.png')
    fig.savefig(out, dpi=130)
    # also dump a CSV for offline matching design
    np.savetxt(os.path.join(sim_path, 'impedance.csv'),
               np.column_stack([f/1e9, R, X, s11_dB]), delimiter=',',
               header='f_GHz,R_ohm,X_ohm,S11_dB', comments='')
    print('Saved', out, '+ impedance.csv + imp_data.npz')
    plt.show()


if __name__ == '__main__':
    main()
