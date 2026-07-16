"""
Mesh-convergence check for the fpc3 co-optimizer search fidelity.

The metal-edge meshing (edge_res=0.4, min_cell=0.05) makes 0.083 mm cells that Courant-
limit the timestep to ~2e-13 s -> ~1200 steps/RF-period (Nyquist needs ~2), which is why
each eval takes ~3 h. Those cells are 3-4x finer than the smallest real feature (the 0.6 mm
PRS wire). This script quantifies how much we can coarsen the SEARCH mesh without moving the
answer: it runs ONE design at a high-fidelity "TRUTH" mesh and several candidate coarse
meshes, and compares worst-in-band realized gain and worst-in-band |S11|.

Decision rule: adopt the coarsest mesh whose worst-in-band gain AND |S11| stay within ~0.3 dB
of TRUTH. That mesh becomes the DE search fidelity (final design still validated on TRUTH).

Same physics as the optimizer: PML all sides, finite-conductivity copper, dense-|S11| worst-
in-band metric, -30 dB EndCriteria (each mesh stops when settled, so coarse meshes -- bigger
timestep -- also take fewer steps; the speedup compounds).

Design tested = fpc3_build defaults (the current seed). Override BEST below to test your EC2
winner instead.

Run (m8g.16xlarge, 64 vCPU):
    FPC_WORKERS=5 FPC_THREADS=12 nohup python3 fpc3_mesh_convergence.py > mesh_conv.out 2>&1 &
    tail -f mesh_conv.out
Out: fpc3_meshconv/mesh_convergence.png | mesh_convergence.csv
"""

import os
import re
import sys
import time
import shutil
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ProcessPoolExecutor

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc3_build as p1

BAND_LO, BAND_HI = 3.10e9, 3.40e9
NRTS, ENDC, NFREQ, NGAIN = 200000, 1e-3, 121, 9
WORKERS = int(os.environ.get('FPC_WORKERS', '5'))
THREADS = int(os.environ.get('FPC_THREADS', '12'))
sim_path = os.path.join(os.getcwd(), 'fpc3_meshconv')
os.makedirs(sim_path, exist_ok=True)

# Design under test (default = build's current seed). Set to your EC2 winner to check it.
BEST = None   # None -> use fpc3_build defaults; else dict(h_cav=..,h3=..,rcm_gap=..,y0=..,L=..,W=..)

# name -> (mesh_div, edge_res, min_cell). Reference is the current search mesh (div16/edge0.4)
# -- the mesh we're deciding whether to coarsen FROM. Set FPC_TRUTH=1 to also run the ~12M-cell
# div22 mesh (confirms div16 itself is converged), but that ~doubles the runtime (long pole).
INCLUDE_TRUTH = os.environ.get('FPC_TRUTH', '0') == '1'
CONFIGS = {}
if INCLUDE_TRUTH:
    CONFIGS['TRUTH_d22'] = (22, 0.40, 0.05)
CONFIGS.update({
    'SEARCH_d16':  (16, 0.40, 0.05),   # current search mesh (reference)
    'COARSE_e0.8': (16, 0.80, 0.15),   # ~3x faster
    'COARSE_e1.0': (16, 1.00, 0.20),   # ~4x faster
    'COARSE_e1.2': (16, 1.20, 0.25),   # ~7x faster
})
REF = 'TRUTH_d22' if INCLUDE_TRUTH else 'SEARCH_d16'


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


def run_config(item):
    name, (div, edge_res, min_cell) = item
    log = os.path.join(sim_path, 'openems_%s.log' % name)
    run_dir = os.path.join(sim_path, 'run_%s' % name)
    try:
        if BEST:
            for k, v in BEST.items():
                setattr(p1, k, float(v))
        p1.RCM_ON = True
        p1.edge_res = edge_res
        p1.min_cell = min_cell
        p1.mesh_res = (p1.C0 / (p1.f0 + p1.fc)) / p1.unit / div
        p1._recompute()
        os.makedirs(run_dir, exist_ok=True)
        t0 = time.time()
        if os.path.exists(log):
            os.remove(log)
        with _redirect(log):
            FDTD = openEMS(NrTS=NRTS, EndCriteria=ENDC)
            FDTD.SetGaussExcite(p1.f0, p1.fc)
            FDTD.SetBoundaryCond(['PML_8'] * 6)
            CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
            mesh = p1.build_antenna(CSX, FDTD)
            ncells = int(np.prod([len(mesh.GetLines(a)) for a in 'xyz']))
            port = FDTD.AddLumpedPort(1, p1.feed_R, [p1.feed_x, p1.feed_y, 0],
                                      [p1.feed_x, p1.feed_y, p1.h_sub], 'z', 1.0,
                                      priority=5, edges2grid='xy')
            nf2ff = FDTD.CreateNF2FFBox()
            CSX.Write2XML(os.path.join(run_dir, 'a.xml'))
            FDTD.Run(run_dir, verbose=0, cleanup=False, numThreads=THREADS)
            f = np.linspace(BAND_LO - 0.2e9, BAND_HI + 0.2e9, NFREQ)
            port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
            s11 = np.abs(port.uf_ref / port.uf_inc)
            Pacc = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
            f_g = np.linspace(BAND_LO, BAND_HI, NGAIN)
            center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
            nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
            Dmax_g = np.array([nf.Dmax[i] for i in range(NGAIN)])
            Prad_g = np.array([nf.Prad[i] for i in range(NGAIN)])
            inb = (f >= BAND_LO) & (f <= BAND_HI)
            D_d = np.interp(f, f_g, Dmax_g)[inb]
            Prad_d = np.interp(f, f_g, Prad_g)[inb]
            etar_d = np.clip(Prad_d / Pacc[inb], 0, 1)
            etam_d = np.clip(1.0 - s11[inb]**2, 0, 1)
            Gr_dense = 10 * np.log10(np.maximum(D_d * etar_d * etam_d, 1e-6))
            worst_gr = float(Gr_dense.min())
            worst_s11 = float(20 * np.log10(s11[inb]).max())
            Dmax = float(10 * np.log10(Dmax_g.max()))
        # convergence: last energy-decay dB and final timestep
        last_db, last_ts = 0.0, 0
        with open(log) as lh:
            for line in lh:
                m = re.search(r'\(-\s*([0-9.]+)dB\)', line)
                if m:
                    last_db = float(m.group(1))
                mt = re.search(r'Timestep:\s*(\d+)', line)
                if mt:
                    last_ts = int(mt.group(1))
        _rmtree(run_dir)
        return dict(name=name, div=div, edge=edge_res, mincell=min_cell, cells=ncells,
                    f=f, s11_dB=20*np.log10(s11), f_g=f_g, worst_gr=worst_gr,
                    worst_s11=worst_s11, Dmax=Dmax, last_db=last_db, ts=last_ts,
                    dt=time.time()-t0, err=None)
    except Exception as e:
        return dict(name=name, err=repr(e))


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print('mesh-convergence: design=%s' % ('EC2 winner ' + str(BEST) if BEST else 'build defaults (seed)'))
    print('configs=%s | %d workers x %d threads | NRTS<=%d\n' % (list(CONFIGS), WORKERS, THREADS, NRTS))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = {r['name']: r for r in ex.map(run_config, list(CONFIGS.items()))}

    ref = res.get(REF)
    print('\n===================== MESH CONVERGENCE =====================')
    print('%-12s | cells | worst-Gr | dGr(ref) | worst-S11 | dS11 | Dmax | conv | steps | %ss'
          % ('config', 'time'))
    order = [n for n in ['TRUTH_d22', 'SEARCH_d16', 'COARSE_e0.8', 'COARSE_e1.0', 'COARSE_e1.2']
             if n in CONFIGS]
    rows = []
    for name in order:
        r = res.get(name)
        if r is None or r.get('err'):
            print('%-12s | ERROR: %s' % (name, r and r.get('err')))
            continue
        dgr = '' if ref.get('err') else '%+.2f' % (r['worst_gr'] - ref['worst_gr'])
        ds11 = '' if ref.get('err') else '%+.2f' % (r['worst_s11'] - ref['worst_s11'])
        conv = 'yes' if r['last_db'] >= 27.0 else 'NO(-%.0f)' % r['last_db']
        print('%-12s | %4.1fM | %+7.2f | %7s  | %+7.2f  | %5s| %4.1f | %-8s| %5d | %4.0f'
              % (name, r['cells']/1e6, r['worst_gr'], dgr, r['worst_s11'], ds11,
                 r['Dmax'], conv, r['ts'], r['dt']))
        rows.append(r)
    print('============================================================')
    print('Adopt the coarsest config with |dGr|<0.3 and |dS11|<~0.5 dB vs TRUTH.\n')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for r in rows:
        ax[0].axhline(r['worst_gr'], lw=0.4, alpha=0)  # keep colors in sync
        ax[1].plot(r['f']/1e9, r['s11_dB'], label='%s (%.1fM)' % (r['name'], r['cells']/1e6))
    names = [r['name'] for r in rows]
    ax[0].bar(names, [r['worst_gr'] for r in rows])
    ax[0].set(title='worst-in-band realized gain', ylabel='dBi')
    ax[0].tick_params(axis='x', rotation=30)
    ax[1].axhline(-10, color='k', ls='--', lw=0.7)
    ax[1].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.08)
    ax[1].set(title='|S11|', xlabel='GHz', ylabel='dB', ylim=(-30, 0)); ax[1].legend(fontsize=7); ax[1].grid(True)
    fig.suptitle('fpc3 mesh convergence — search-mesh fidelity check')
    fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'mesh_convergence.png'), dpi=130)
    with open(os.path.join(sim_path, 'mesh_convergence.csv'), 'w') as fh:
        fh.write('config,cells,worst_gr_dBi,worst_s11_dB,Dmax_dBi,converged_dB,steps,walltime_s\n')
        for r in rows:
            fh.write('%s,%d,%.3f,%.3f,%.3f,%.1f,%d,%.0f\n'
                     % (r['name'], r['cells'], r['worst_gr'], r['worst_s11'], r['Dmax'],
                        r['last_db'], r['ts'], r['dt']))
    print('Saved fpc3_meshconv/mesh_convergence.png + mesh_convergence.csv')


if __name__ == '__main__':
    main()
