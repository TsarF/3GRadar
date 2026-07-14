"""
Validate the DE-optimized slotted feed in the FULL FPC cavity (feed + PRS + RCM).
One high-fidelity FDTD solve -> in-band |S11|, input Z, and broadside directivity(f).
Headless (Agg, no plt.show), so it is safe to launch detached.

The feed-only DE matched the bare slotted patch to -12.65 dB worst-in-band over 9%;
this checks whether that match SURVIVES the PRS/RCM cavity loading (the previous,
un-slotted feed matched only ~4.6%).  Directivity is expected to stay ~14-15 dBi.

Run:  python fpc3_validate.py
Out:  fpc3_validate/validate.png | validate.csv | validate.npz
"""

import os
import sys
import time
import shutil
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc3_build as p1
p1.FEED_ONLY = False
p1.RCM_ON = True
p1.mesh_res = (p1.C0 / (p1.f0 + p1.fc)) / p1.unit / 22.0     # accurate mesh
p1._recompute()

BAND_LO, BAND_HI = 3.10e9, 3.40e9
NRTS, ENDC, NFREQ, NGAIN = 90000, 5e-5, 601, 13
sim_path = os.path.join(os.getcwd(), 'fpc3_validate')
os.makedirs(sim_path, exist_ok=True)


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


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    run_dir = os.path.join(sim_path, 'run')
    log = os.path.join(sim_path, 'openems.log')
    print('full-cavity validation: y0=%.1f L=%.1f slot(%g,%g,x=%g) h_cav=%.1f h3=%.1f'
          % (p1.y0, p1.L, p1.slot_len, p1.slot_w, p1.slot_x, p1.h_cav, p1.h3))
    print('solving (openEMS output -> %s) ...' % log)
    t0 = time.time()
    os.makedirs(run_dir, exist_ok=True)
    with _redirect(log):
        FDTD = openEMS(NrTS=NRTS, EndCriteria=ENDC)
        FDTD.SetGaussExcite(p1.f0, p1.fc)
        FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])
        CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
        p1.build_antenna(CSX, FDTD)
        port = FDTD.AddLumpedPort(1, p1.feed_R, [p1.feed_x, p1.feed_y, 0],
                                  [p1.feed_x, p1.feed_y, p1.h_sub], 'z', 1.0,
                                  priority=5, edges2grid='xy')
        nf2ff = FDTD.CreateNF2FFBox()
        CSX.Write2XML(os.path.join(run_dir, 'a.xml'))
        FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=8)
        f = np.linspace(BAND_LO - 0.25e9, BAND_HI + 0.25e9, NFREQ)
        port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
        Z = port.uf_tot / port.if_tot
        s11 = port.uf_ref / port.uf_inc
        center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
        f_g = np.linspace(BAND_LO, BAND_HI, NGAIN)
        nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
        D = 10 * np.log10(np.array([nf.Dmax[i] for i in range(NGAIN)]))
    _rmtree(run_dir)
    dt = time.time() - t0

    s11_dB = 20 * np.log10(np.abs(s11))
    band = (f >= BAND_LO) & (f <= BAND_HI)
    worst = float(np.max(s11_dB[band]))
    # realized gain = D * (1-|S11|^2), sampled at the gain frequencies
    s11_at_g = np.interp(f_g, f, np.abs(s11))
    G_real = D + 10 * np.log10(np.clip(1 - s11_at_g**2, 1e-6, 1))

    np.savez(os.path.join(sim_path, 'validate.npz'),
             f=f, Z=Z, s11=s11, f_g=f_g, D=D, G_real=G_real, Zref=p1.feed_R)
    np.savetxt(os.path.join(sim_path, 'validate.csv'),
               np.column_stack([f_g/1e9, D, G_real]), delimiter=',',
               header='f_GHz,directivity_dBi,realized_gain_dBi', comments='')

    print('\n==================== FULL-CAVITY VALIDATION (%.0f s) ====================' % dt)
    print('  worst in-band |S11|      = %+.2f dB' % worst)
    gb = (f_g >= BAND_LO) & (f_g <= BAND_HI)
    print('  in-band directivity      = %.2f .. %.2f dBi (min..max)' % (D[gb].min(), D[gb].max()))
    print('  in-band realized gain    = %.2f .. %.2f dBi (min..max)' % (G_real[gb].min(), G_real[gb].max()))
    for tag, fk in (('3.10', BAND_LO), ('3.25', 3.25e9), ('3.40', BAND_HI)):
        k = int(np.argmin(np.abs(f - fk))); kg = int(np.argmin(np.abs(f_g - fk)))
        print('  %s GHz: S11=%+6.2f dB | Z=%6.1f%+6.1fj | D=%5.2f | Grealized=%5.2f dBi'
              % (tag, s11_dB[k], Z.real[k], Z.imag[k], D[kg], G_real[kg]))
    print('=======================================================================\n')

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(f/1e9, s11_dB, lw=1.8); ax[0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[0].set(title='|S11| (worst %.2f dB)' % worst, xlabel='GHz', ylabel='dB', ylim=(-30, 0))
    ax[1].plot(f/1e9, Z.real, label='R'); ax[1].plot(f/1e9, Z.imag, label='X')
    ax[1].axhline(p1.feed_R, color='g', ls=':', lw=0.8); ax[1].axhline(0, color='k', lw=0.5)
    ax[1].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[1].set(title='Input Z', xlabel='GHz', ylabel='ohm'); ax[1].legend()
    ax[2].plot(f_g/1e9, D, 'o-', label='directivity')
    ax[2].plot(f_g/1e9, G_real, 's--', label='realized gain')
    ax[2].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[2].set(title='Directivity / realized gain', xlabel='GHz', ylabel='dBi'); ax[2].legend()
    for a_ in ax:
        a_.grid(True)
    fig.suptitle('fpc3 full-cavity validation — slotted broadband feed', y=1.02)
    fig.tight_layout()
    out = os.path.join(sim_path, 'validate.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('Saved', out, '+ validate.csv + validate.npz')


if __name__ == '__main__':
    main()
