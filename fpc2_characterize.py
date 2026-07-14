"""
Full-wave characterization of the dual-layer-PRS FP antenna (fpc2_build): one FDTD
solve, then S11, realized boresight gain vs frequency, broadside directivity vs
frequency, and the E-/H-plane patterns at the operating frequency.

Realized gain = D_broadside * eta_rad * (1 - |S11|^2).  Watch the gain across
3.1-3.4 GHz: with a high-|Gamma| PRS the peak is tall but can be narrow, so this
tells you whether to trade a little reflectivity (larger slot sb) for flatter gain.

Run:            python fpc2_characterize.py           # solve + plot
Replot only:    python fpc2_characterize.py --replot   # reuse saved data
openEMS output goes to fpc2_char/openems.log so the console stays readable.
"""

import os
import sys
import time
import shutil
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc2_build as p1

# ============================ config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
CHAR_NRTS        = 120000
CHAR_ENDCRITERIA = 1e-4
NFREQ            = 601
NGAIN            = 21
THREADS          = 8

sim_path = os.path.join(os.getcwd(), 'fpc2_char')
os.makedirs(sim_path, exist_ok=True)
data_npz = os.path.join(sim_path, 'char_data.npz')
# ===============================================================


@contextlib.contextmanager
def _redirect_fds(path):
    sys.stdout.flush(); sys.stderr.flush()
    fout = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    s1, s2 = os.dup(1), os.dup(2)
    try:
        os.dup2(fout, 1); os.dup2(fout, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(s1, 1); os.dup2(s2, 2)
        os.close(s1); os.close(s2); os.close(fout)


def _rmtree_retry(path, tries=5):
    for _ in range(tries):
        try:
            shutil.rmtree(path); return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def _norm_db(res):
    E = np.sqrt(np.abs(res.E_theta[0][:, 0])**2 + np.abs(res.E_phi[0][:, 0])**2)
    return 20 * np.log10(E / np.max(E) + 1e-12)


def solve():
    run_dir = os.path.join(sim_path, 'run')
    log = os.path.join(sim_path, 'openems.log')
    print('Solving FPC-II (output -> %s) ...' % log)
    t0 = time.time()
    os.makedirs(run_dir, exist_ok=True)
    with _redirect_fds(log):
        FDTD = openEMS(NrTS=CHAR_NRTS, EndCriteria=CHAR_ENDCRITERIA)
        FDTD.SetGaussExcite(p1.f0, p1.fc)
        FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])
        CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
        p1.build_antenna(CSX, FDTD)
        port = FDTD.AddLumpedPort(1, p1.feed_R,
                                  [p1.feed_x, p1.feed_y, 0], [p1.feed_x, p1.feed_y, p1.h_sub],
                                  'z', 1.0, priority=5, edges2grid='xy')
        nf2ff = FDTD.CreateNF2FFBox()
        CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
        FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=THREADS)

        f = np.linspace(p1.f0 - p1.fc, p1.f0 + p1.fc, NFREQ)
        port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
        s11 = port.uf_ref / port.uf_inc
        s11_dB = 20 * np.log10(np.abs(s11))
        Pacc_f = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
        band = (f >= BAND_LO) & (f <= BAND_HI)
        f_op = float(f[band][np.argmin(np.abs(s11)[band])])

        center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
        f_g = np.linspace(BAND_LO - 0.05e9, BAND_HI + 0.05e9, NGAIN)
        gg = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
        Dbroad = 10 * np.log10(np.array([gg.Dmax[i] for i in range(NGAIN)]))
        Pacc_g = np.interp(f_g, f, Pacc_f)
        s11_g  = np.interp(f_g, f, np.abs(s11))
        Gr = np.empty(NGAIN)
        for i in range(NGAIN):
            eta_rad   = np.clip(gg.Prad[i] / Pacc_g[i], 0, 1)
            eta_match = max(1.0 - s11_g[i]**2, 0.0)
            Gr[i] = 10 * np.log10(max(gg.Dmax[i] * eta_rad * eta_match, 1e-6))

        theta = np.arange(-180, 180.5, 1.0)                     # DEGREES for CalcNF2FF
        ec = nf2ff.CalcNF2FF(run_dir, f_op, theta, np.array([0.0]),  center=center)
        hc = nf2ff.CalcNF2FF(run_dir, f_op, theta, np.array([90.0]), center=center)
        Eplane, Hplane = _norm_db(ec), _norm_db(hc)
    _rmtree_retry(run_dir)
    print('done in %.0f s' % (time.time() - t0))

    np.savez(data_npz, f=f, s11_dB=s11_dB, f_op=f_op, f_g=f_g, Dbroad=Dbroad,
             Gr=Gr, theta=theta, Eplane=Eplane, Hplane=Hplane)
    return dict(f=f, s11_dB=s11_dB, f_op=f_op, f_g=f_g, Dbroad=Dbroad, Gr=Gr,
                theta=theta, Eplane=Eplane, Hplane=Hplane)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    replot = '--replot' in sys.argv
    if replot:
        if not os.path.exists(data_npz):
            sys.exit('no saved data at %s - run without --replot first' % data_npz)
        d = dict(np.load(data_npz))
        d['f_op'] = float(d['f_op'])
    else:
        d = solve()

    for bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(bk, force=True); break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    band = (d['f'] >= BAND_LO) & (d['f'] <= BAND_HI)
    worst = float(np.max(d['s11_dB'][band]))
    gmin, gmax = float(np.min(d['Gr'])), float(np.max(d['Gr']))
    print('\n==================== FPC-II RESULTS ====================')
    print('worst in-band S11 : %+6.2f dB' % worst)
    print('peak directivity  : %5.2f dBi' % float(np.max(d['Dbroad'])))
    print('realized gain     : %.2f .. %.2f dBi (in band)' % (gmin, gmax))
    print('f_op (best match) : %.3f GHz' % (d['f_op'] / 1e9))
    print('=======================================================\n')

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(d['f']/1e9, d['s11_dB'])
    ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[0, 0].set(title='|S11| (worst in-band %.2f dB)' % worst,
                 xlabel='Frequency (GHz)', ylabel='|S11| (dB)', ylim=(-40, 0))

    ax[0, 1].plot(d['f_g']/1e9, d['Dbroad'], 'o-', label='directivity')
    ax[0, 1].plot(d['f_g']/1e9, d['Gr'], 's-', label='realized gain')
    ax[0, 1].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[0, 1].set(title='Broadside D & realized gain', xlabel='Frequency (GHz)',
                 ylabel='dBi'); ax[0, 1].legend()

    ax[1, 0].plot(d['theta'], d['Eplane'])
    ax[1, 0].set(title='E-plane co-pol @ %.3f GHz' % (d['f_op']/1e9), xlabel='Theta (deg)',
                 ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
    ax[1, 1].plot(d['theta'], d['Hplane'])
    ax[1, 1].set(title='H-plane co-pol @ %.3f GHz' % (d['f_op']/1e9), xlabel='Theta (deg)',
                 ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
    for a_ in ax.flat:
        a_.grid(True)
    fig.suptitle('Dual-layer-PRS Fabry-Perot antenna (3.25 GHz)', y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'fpc2_characterization.png')
    fig.savefig(out, dpi=130)
    print('Saved', out)
    plt.show()


if __name__ == '__main__':
    main()
