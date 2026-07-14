"""
Lean FDTD sweep for the RCM design point (fpc3_build), guided by the bempp screen.
For each (h_cav, rcm_s, h3) it does ONE cavity solve and reports broadside directivity
at f0, worst-in-band realized gain, and the match - no patterns, coarser mesh, so each
point is ~20-30 min instead of ~1 hr.  bempp said the RCM adds ~+5 dB at rcm_s~17-19,
h3~42-46 (bempp units); scaled for the dielectric the FDTD target is ~rcm_s 17, h3 36.

Run:  python fpc3_sweep.py
"""

import os
import sys
import time
import shutil
import contextlib
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc3_build as p1
p1.RCM_ON = True
p1.mesh_res = (p1.C0 / (p1.f0 + p1.fc)) / p1.unit / 20.0     # coarser for speed

BAND_LO, BAND_HI = 3.10e9, 3.40e9
NRTS = 70000
ENDC = 1e-3
sim_path = os.path.join(os.getcwd(), 'fpc3_sweep')
os.makedirs(sim_path, exist_ok=True)

# (h_cav, rcm_s, h3, y0): push toward the directivity ceiling (bigger rcm_s / h3)
POINTS = [(45.0, 17.0, 48.0, 9.5), (45.0, 19.0, 45.0, 9.5), (45.0, 19.0, 48.0, 9.5)]


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


def evaluate(h_cav, rcm_s, h3, y0=9.5):
    p1.h_cav = h_cav; p1.rcm_s = rcm_s; p1.h3 = h3; p1.y0 = y0; p1._recompute()
    run_dir = os.path.join(sim_path, 'run')
    log = os.path.join(sim_path, 'openems.log')
    with _redirect(log):
        FDTD = openEMS(NrTS=NRTS, EndCriteria=ENDC)
        FDTD.SetGaussExcite(p1.f0, p1.fc)
        FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])
        CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
        p1.build_antenna(CSX, FDTD)
        port = FDTD.AddLumpedPort(1, p1.feed_R, [p1.feed_x, p1.feed_y, 0],
                                  [p1.feed_x, p1.feed_y, p1.h_sub], 'z', 1.0, priority=5, edges2grid='xy')
        nf2ff = FDTD.CreateNF2FFBox()
        os.makedirs(run_dir, exist_ok=True)
        CSX.Write2XML(os.path.join(run_dir, 'a.xml'))
        FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=8)
        f = np.linspace(BAND_LO - 0.15e9, BAND_HI + 0.15e9, 61)
        port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
        s11 = port.uf_ref / port.uf_inc
        Pacc = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
        band = (f >= BAND_LO) & (f <= BAND_HI)
        worst_s11 = float(np.max(20*np.log10(np.abs(s11))[band]))
        center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
        f_g = np.linspace(BAND_LO, BAND_HI, 5)
        nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
        Pg = np.interp(f_g, f, Pacc); s11g = np.interp(f_g, f, np.abs(s11))
        Gr = np.array([10*np.log10(max(nf.Dmax[i]*np.clip(nf.Prad[i]/Pg[i], 0, 1)*max(1-s11g[i]**2, 0), 1e-6))
                       for i in range(5)])
        Dpk = 10*np.log10(max(nf.Dmax[i] for i in range(5)))
    _rmtree(run_dir)
    return Dpk, float(Gr.min()), float(Gr.max()), worst_s11


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print('=== fpc3 FDTD RCM sweep (RCM_ON, coarser mesh) ===')
    print('  h_cav rcm_s  h3   y0 | peakD  Gr(min..max)  worstS11 |  s')
    best = None
    for hc, rs, h3, y0 in POINTS:
        t0 = time.time()
        try:
            Dpk, grmin, grmax, ws = evaluate(hc, rs, h3, y0)
        except Exception as e:
            print('  %4.0f %5.1f %4.0f %4.1f | FAILED %s' % (hc, rs, h3, y0, repr(e)[:80])); continue
        print('  %4.0f %5.1f %4.0f %4.1f | %5.2f  %5.2f..%5.2f  %+6.2f | %.0f'
              % (hc, rs, h3, y0, Dpk, grmin, grmax, ws, time.time() - t0))
        if best is None or grmax > best[0]:
            best = (grmax, hc, rs, h3, y0)
    if best:
        print('\nbest so far: Gr=%.2f dBi at h_cav=%.0f rcm_s=%.1f h3=%.0f y0=%.1f'
              % best)
    print('compare: single-PRS fpc2 ~11 dBi | bempp predicted RCM ~+5 dB')


if __name__ == '__main__':
    main()
