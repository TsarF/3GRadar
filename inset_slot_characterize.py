"""
Compare the inset patch WITH vs WITHOUT the U-slot - two FDTD solves run in
PARALLEL (one per process, 4 threads each), then S11 and directivity plotted for
both. Uses the current dimensions in inset_slot_build; the only difference between
the two runs is the slot (enabled vs disabled).

Plots (with-slot vs no-slot overlaid):
    * |S11|                          * broadside directivity vs frequency
    * E-plane co-pol pattern @ f_op  * H-plane co-pol pattern @ f_op

Run:  python inset_slot_characterize.py
Note: two full 3-D solves at once (a few M cells total) - minutes. openEMS output
goes to inset_slot_char/openems_logs/ so the console stays readable.
"""

import os
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

import inset_slot_build as p1

# ============================ config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
SLOT_LEN_WITH    = p1.slot_len       # "with slot" arm length (build value; bump for a real slot)
SLOT_LEN_OFF     = 0.0               # "without slot" - disables the slot (fit=False)
THREADS_EACH     = 4                 # 2 solves x 4 threads ~= 8 cores
CHAR_NRTS        = 60000
CHAR_ENDCRITERIA = 1e-4

sim_path = os.path.join(os.getcwd(), 'inset_slot_char')
os.makedirs(sim_path, exist_ok=True)
# ===============================================================


@contextlib.contextmanager
def _redirect_fds(path):
    sys.stdout.flush(); sys.stderr.flush()
    fout = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
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


def eval_variant(payload):
    """One full characterization solve for a given slot length. Returns a dict."""
    slot_len_val, label, run_dir, nthreads = payload
    logdir = os.path.join(sim_path, 'openems_logs'); os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, 'variant_%s.log' % label.replace(' ', '_'))
    try:
        p1.slot_len = float(slot_len_val)      # only the slot differs between runs
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
            FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=nthreads)

            f = np.linspace(p1.f0 - p1.fc, p1.f0 + p1.fc, 601)
            port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
            s11 = port.uf_ref / port.uf_inc
            s11_dB = 20 * np.log10(np.abs(s11))
            band = (f >= BAND_LO) & (f <= BAND_HI)
            f_op = float(f[band][np.argmin(np.abs(s11)[band])])
            worst = float(np.max(s11_dB[band]))

            center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
            f_dir = np.linspace(BAND_LO - 0.1e9, BAND_HI + 0.1e9, 13)
            dres = nf2ff.CalcNF2FF(run_dir, f_dir, np.array([0.0]), np.array([0.0]), center=center)
            Dbroad = 10 * np.log10(np.array([dres.Dmax[i] for i in range(len(f_dir))]))

            theta = np.arange(-180, 180.5, 1.0)          # DEGREES for CalcNF2FF
            ec = nf2ff.CalcNF2FF(run_dir, f_op, theta, np.array([0.0]),  center=center)
            hc = nf2ff.CalcNF2FF(run_dir, f_op, theta, np.array([90.0]), center=center)
            Eplane, Hplane = _norm_db(ec), _norm_db(hc)

            sph = nf2ff.CalcNF2FF(run_dir, np.array([f_op]), np.array([0.0]), np.array([0.0]), center=center)
            Dmax = float(sph.Dmax[0])
            Pacc = 0.5 * np.real(np.interp(f_op, f, port.uf_tot) * np.conj(np.interp(f_op, f, port.if_tot)))
            eff_rad   = float(np.clip(sph.Prad[0] / Pacc, 0, 1))
            eff_match = float(1 - np.interp(f_op, f, np.abs(s11))**2)
            realized_g = float(10 * np.log10(Dmax * eff_rad * eff_match))

        _rmtree_retry(run_dir)
        return dict(label=label, slot_len=slot_len_val, f=f, s11_dB=s11_dB, worst=worst,
                    f_op=f_op, f_dir=f_dir, Dbroad=Dbroad, theta=theta, Eplane=Eplane,
                    Hplane=Hplane, Dmax_dBi=10*np.log10(Dmax), realized_g=realized_g,
                    eff_rad=eff_rad, eff_match=eff_match)
    except Exception as e:
        return dict(label=label, error=str(e))


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    for bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(bk, force=True); break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    payloads = [
        (SLOT_LEN_WITH, 'with slot', os.path.join(sim_path, 'run_with'), THREADS_EACH),
        (SLOT_LEN_OFF,  'no slot',   os.path.join(sim_path, 'run_off'),  THREADS_EACH),
    ]
    print('Running 2 solves in parallel: slot_len=%.1f (with) vs %.1f (off)...'
          % (SLOT_LEN_WITH, SLOT_LEN_OFF))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(eval_variant, payloads))
    print('done in %.0f s' % (time.time() - t0))

    for r in res:
        if 'error' in r:
            print('  [%s] FAILED: %s' % (r['label'], r['error']))
    res = [r for r in res if 'error' not in r]
    if not res:
        sys.exit('both solves failed - check inset_slot_char/openems_logs/')

    print('\n================ WITH vs WITHOUT SLOT ================')
    for r in res:
        print('%-10s | worst S11 %+6.2f dB | peak D %5.2f dBi | realized gain %5.2f dBi (@ %.3f GHz)'
              % (r['label'], r['worst'], r['Dmax_dBi'], r['realized_g'], r['f_op']/1e9))
    print('=====================================================\n')

    colors = {'with slot': 'C0', 'no slot': 'C1'}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for r in res:
        c = colors.get(r['label'], None)
        ax[0, 0].plot(r['f']/1e9, r['s11_dB'], color=c, label=r['label'])
        ax[0, 1].plot(r['f_dir']/1e9, r['Dbroad'], 'o-', color=c, label=r['label'])
        ax[1, 0].plot(r['theta'], r['Eplane'], color=c, label=r['label'])
        ax[1, 1].plot(r['theta'], r['Hplane'], color=c, label=r['label'])

    ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
    ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[0, 0].set(title='|S11|', xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
    ax[0, 1].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.10)
    ax[0, 1].set(title='Broadside directivity', xlabel='Frequency (GHz)', ylabel='D (dBi)')
    ax[1, 0].set(title='E-plane co-pol @ f_op', xlabel='Theta (deg)',
                 ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
    ax[1, 1].set(title='H-plane co-pol @ f_op', xlabel='Theta (deg)',
                 ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
    for a in ax.flat:
        a.legend(); a.grid(True)
    fig.suptitle('Inset patch: WITH slot (%.1f mm) vs WITHOUT slot' % SLOT_LEN_WITH, y=1.0)
    fig.tight_layout()
    out = os.path.join(sim_path, 'slot_compare.png')
    fig.savefig(out, dpi=130)
    print('Saved comparison plot to', out)
    plt.show()


if __name__ == '__main__':
    main()
