"""
Parallel Differential-Evolution optimizer for the dual-layer-PRS FP antenna
(fpc2_build), maximizing worst-in-band REALIZED boresight gain over 3.1-3.4 GHz:
G = D_broadside * eta_rad * (1 - |S11|^2).

Tunes the feed (L, W, y0), the cavity height (h_cav), and the PRS slot (sb, which
trades reflectivity for aperture illumination).  Each eval is a full cavity FDTD +
NF2FF (~2.7 M cells), so this is an overnight-scale run: fewer parallel workers with
more threads each, and looser eval accuracy than the final characterization.

Run:  python fpc2_optimize_de_gain.py
Out:  fpc2_gain_de_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
Re-validate the winner with fpc2_characterize.py (tighter EndCriteria).
"""

import os
import sys
import csv
import json
import time
import shutil
import random
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ProcessPoolExecutor, as_completed

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc2_build as p1

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
NGAIN            = 5
GAIN_TARGET      = None
GUIDE_W          = 2.0

WORKERS            = 4                     # 2 cavity solves at once...
THREADS_PER_WORKER = 4                     # ...x4 threads = 8 cores

NP        = 16
F_MUT     = 0.7
CR        = 0.9
MAX_EVALS = 300
SEED      = 12345

# Stage B: feed L/W/y0 pre-tuned in Stage A (W fixed at 29.4); focus on the cavity.
PARAMS = {
    'h_cav': (p1.h_cav, 42.0, 50.0, 0.2),   # cavity resonance (main gain knob)
    'L':     (p1.L,     25.5, 28.0, 0.1),   # narrow: re-land resonance after cavity pull
    'y0':    (p1.y0,     7.0, 12.0, 0.1),   # re-match the feed in the cavity
    'sb':    (p1.sb,    12.0, 19.5, 0.2),   # PRS slot: bigger -> lower |Gamma|
}

WARM_START_FROM = os.path.join(os.getcwd(), 'fpc2_gain_de_opt', 'optimized_params.json')

EVAL_NRTS        = 70000
EVAL_ENDCRITERIA = 1e-3
EVAL_NFREQ       = 61
EVAL_FPAD        = 0.15e9

sim_path = os.path.join(os.getcwd(), 'fpc2_gain_de_opt')
os.makedirs(sim_path, exist_ok=True)

names  = list(PARAMS.keys())
lo     = np.array([PARAMS[n][1] for n in names])
hi     = np.array([PARAMS[n][2] for n in names])
steps  = np.array([PARAMS[n][3] for n in names])
D      = len(names)
# ==========================================================================


def snap(x):
    x = np.clip(np.asarray(x, float), lo, hi)
    return np.clip(np.round(x / steps) * steps, lo, hi)


def set_params(values):
    for name, val in values.items():
        setattr(p1, name, float(val))
    p1._recompute()


def resonance_offset(f, s11_dB):
    f_res = float(f[int(np.argmin(s11_dB))])
    return abs(f_res - 0.5*(BAND_LO + BAND_HI)) / (0.5*(BAND_HI - BAND_LO))


def _rmtree_retry(path, tries=5):
    for _ in range(tries):
        try:
            shutil.rmtree(path); return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


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


def eval_worker(payload):
    """One cavity FDTD + NF2FF solve. Returns (f, s11_dB, f_g, Gr_dBi) or None."""
    x, run_dir, nthreads = payload
    logdir = os.path.join(sim_path, 'openems_logs'); os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, 'worker_%d.log' % os.getpid())
    try:
        set_params(dict(zip(names, x)))
        os.makedirs(run_dir, exist_ok=True)
        with _redirect_fds(log):
            FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
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

            f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
            port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
            s11 = port.uf_ref / port.uf_inc
            s11_dB = 20 * np.log10(np.abs(s11))
            Pacc_f = 0.5 * np.real(port.uf_tot * np.conj(port.if_tot))
            s11mag = np.abs(s11)

            f_g = np.linspace(BAND_LO, BAND_HI, NGAIN)
            center = [0, 0, (p1.z_stk_patch / 2) * p1.unit]
            nf = nf2ff.CalcNF2FF(run_dir, f_g, np.array([0.0]), np.array([0.0]), center=center)
            Pacc_g = np.interp(f_g, f, Pacc_f)
            s11_g  = np.interp(f_g, f, s11mag)
            Gr_dBi = np.empty(NGAIN)
            for i in range(NGAIN):
                eta_rad   = np.clip(nf.Prad[i] / Pacc_g[i], 0, 1)
                eta_match = max(1.0 - s11_g[i]**2, 0.0)
                Gr_dBi[i] = 10 * np.log10(max(nf.Dmax[i] * eta_rad * eta_match, 1e-6))
        _rmtree_retry(run_dir)
        return f, s11_dB, f_g, Gr_dBi
    except Exception as e:
        return None


def initial_population():
    rng = random.Random(SEED)
    init = {n: PARAMS[n][0] for n in names}
    if os.path.exists(WARM_START_FROM):
        try:
            with open(WARM_START_FROM) as fh:
                data = json.load(fh)
            for k, v in data.get('params_mm', {}).items():
                if k in init:
                    init[k] = float(v)
            print('Warm-starting population seed from', WARM_START_FROM)
        except Exception as e:
            print('warm-start read failed (%s)' % e)
    pop = [snap(np.array([init[n] for n in names], float))]
    for _ in range(NP - 1):
        pop.append(snap(lo + np.array([rng.random() for _ in range(D)]) * (hi - lo)))
    return pop, rng


def make_trial(i, pop, best_x, rng):
    idxs = [j for j in range(NP) if j != i]
    a, b = (pop[k] for k in rng.sample(idxs, 2))
    v = pop[i] + F_MUT * (best_x - pop[i]) + F_MUT * (a - b)
    u = pop[i].copy()
    jr = rng.randrange(D)
    for j in range(D):
        if rng.random() < CR or j == jr:
            u[j] = v[j]
    return snap(u)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _gui = False
    for bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(bk, force=True); _gui = True; break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    pop, rng = initial_population()
    cache = {}
    history = []
    best = {'gain': -np.inf, 'x': pop[0].copy(), 'f': None, 's11': None, 'f_g': None, 'Gr': None}
    n_eval = [0]
    recent = []

    def key(x):
        return tuple(np.round(x, 6))

    def save_best():
        if not np.isfinite(best['gain']):
            return
        bx = dict(zip(names, best['x']))
        tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
        with open(tmp, 'w') as fh:
            json.dump({'objective': 'max worst-in-band realized boresight gain', 'optimizer': 'DE',
                       'worst_realized_gain_dBi': best['gain'],
                       'band_GHz': [BAND_LO/1e9, BAND_HI/1e9], 'evals_so_far': n_eval[0],
                       'params_mm': bx}, fh, indent=2)
        os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))

    def save_log():
        with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['eval'] + names + ['worst_realized_gain_dBi'])
            for i, x, g in history:
                w.writerow([i] + ['%.4f' % v for v in x] + ['%.3f' % g])

    def live_plot(gen):
        if not _gui or best['f'] is None:
            return
        try:
            if not hasattr(live_plot, 'ax'):
                plt.ion(); live_plot.fig, live_plot.ax = plt.subplots(figsize=(8, 5))
            ax = live_plot.ax; ax.clear()
            for (g, ff, ss, gg) in recent[-3:]:
                ax.plot(ff/1e9, ss, lw=1.3, alpha=0.5, label='gen %d Gr %.2f dBi' % (g, gg))
            ax.plot(best['f']/1e9, best['s11'], lw=2.4, color='k',
                    label='best Gr %.2f dBi' % best['gain'])
            ax.axhline(-10, color='r', ls='--', lw=0.8)
            ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
            ax.set(title='FPC DE realized-gain | gen %d | %d evals | best %.2f dBi'
                   % (gen, n_eval[0], best['gain']),
                   xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
            ax.legend(loc='upper right', fontsize=8); ax.grid(True)
            live_plot.fig.tight_layout()
            live_plot.fig.canvas.draw_idle(); live_plot.fig.canvas.flush_events(); plt.pause(0.001)
            live_plot.fig.savefig(os.path.join(sim_path, 'live_s11.png'), dpi=110)
        except Exception as e:
            print('live plot disabled (%s)' % e)

    def consider(x, tup):
        cost, gain, f, s11, f_g, Gr = tup
        if f is not None and gain > best['gain']:
            best.update(gain=gain, x=x.copy(), f=f, s11=s11, f_g=f_g, Gr=Gr)
            save_best()
            return True
        return False

    def evaluate(cands, executor, gen):
        out = [None] * len(cands)
        todo = [(i, x) for i, x in enumerate(cands) if key(x) not in cache]
        for i, x in enumerate(cands):
            if key(x) in cache:
                out[i] = cache[key(x)]
        n_cached = len(cands) - len(todo)
        if n_cached:
            print('  (gen %d: %d new evals, %d cached/skipped)' % (gen, len(todo), n_cached))
        futs = {}
        for j, x in todo:
            n_eval[0] += 1
            rd = os.path.join(sim_path, 'e_%04d' % n_eval[0])
            futs[executor.submit(eval_worker, (x, rd, THREADS_PER_WORKER))] = (j, x)
        for fut in as_completed(futs):
            j, x = futs[fut]
            res = fut.result()
            if res is None:
                tup = (1e9, -np.inf, None, None, None, None)
            else:
                ff, s11, f_g, Gr = res
                gain = float(Gr.min())
                cost = -gain + GUIDE_W * resonance_offset(ff, s11)
                tup = (cost, gain, ff, s11, f_g, Gr)
            cache[key(x)] = tup
            out[j] = tup
            history.append((len(history) + 1, x, tup[1]))
            is_best = consider(x, tup)
            save_log()
            if res is None:
                print('  [eval %3d | gen %d] FAILED' % (len(history), gen))
            else:
                print('  [eval %3d | gen %d] Gr=%+6.2f dBi%s'
                      % (len(history), gen, tup[1], '  <-- NEW BEST' if is_best else ''))
        return out

    print('===== FPC DE realized-gain (%d workers x %d threads) over %.2f-%.2f GHz ====='
          % (WORKERS, THREADS_PER_WORKER, BAND_LO/1e9, BAND_HI/1e9))
    print('Population %d, F=%.2f, CR=%.2f, max %d evals\n' % (NP, F_MUT, CR, MAX_EVALS))

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        print('Evaluating initial population (%d designs)...' % NP)
        res = evaluate(pop, executor, 0)
        pop_cost = [r[0] for r in res]
        gen = 0
        stagnant = 0
        GEN_CAP = 100                 # backstop so the loop can never spin forever
        while (GAIN_TARGET is None or best['gain'] < GAIN_TARGET) and n_eval[0] < MAX_EVALS and gen < GEN_CAP:
            gen += 1
            before = n_eval[0]
            trials = [make_trial(i, pop, best['x'], rng) for i in range(NP)]
            tres = evaluate(trials, executor, gen)
            gb = (-np.inf, None, None, None)
            for i in range(NP):
                cost, gain, f, s11, f_g, Gr = tres[i]
                if cost <= pop_cost[i]:               # DE selection (on shaped cost)
                    pop[i] = trials[i]; pop_cost[i] = cost
                if f is not None and gain > gb[0]:
                    gb = (gain, f, s11, trials[i])
            if gb[1] is not None:
                recent.append((gen, gb[1], gb[2], gb[0]))
            live_plot(gen)
            new = n_eval[0] - before
            print('[gen %2d done] evals=%3d  new=%2d  gen-best=%+6.2f dBi  overall-best=%+6.2f dBi'
                  % (gen, n_eval[0], new, gb[0] if gb[1] is not None else float('nan'), best['gain']))
            if GAIN_TARGET is not None and best['gain'] >= GAIN_TARGET:
                print('Gain target met (%.2f >= %.2f dBi) - stopping.' % (best['gain'], GAIN_TARGET))
                break
            stagnant = stagnant + 1 if new == 0 else 0
            if stagnant >= 3:
                print('Converged (population collapsed, all candidates cached) - stopping.'); break
        if gen >= GEN_CAP:
            print('Hit generation cap (%d) - stopping.' % GEN_CAP)

    # ---- report ----
    print('\n=========================== BEST DESIGN (FPC DE realized gain) ===========================')
    print('Worst-in-band realized gain: %+.2f dBi  (%d evals)' % (best['gain'], n_eval[0]))
    if best['Gr'] is not None:
        print('  across band: ' + '  '.join('%.2fGHz=%.2f' % (fg/1e9, g)
                                             for fg, g in zip(best['f_g'], best['Gr'])))
    print('\n--- paste into fpc2_build.py DESIGN PARAMETERS ---')
    for n, v in zip(names, best['x']):
        print('%-8s = %.2f' % (n, v))
    print('--------------------------------------------------\n')
    save_best(); save_log()
    if best['f'] is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(best['f']/1e9, best['s11'], lw=1.8); ax[0].axhline(-10, color='r', ls='--', lw=0.8)
        ax[0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='band')
        ax[0].set(title='|S11| of best realized-gain design', xlabel='Frequency (GHz)',
                  ylabel='|S11| (dB)'); ax[0].legend(); ax[0].grid(True)
        ax[1].plot(best['f_g']/1e9, best['Gr'], 'o-')
        ax[1].set(title='Realized boresight gain (worst %.2f dBi)' % best['gain'],
                  xlabel='Frequency (GHz)', ylabel='G_realized (dBi)'); ax[1].grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved optimized_params.json, optimization_log.csv, optimized_s11.png')
    if _gui:
        plt.ioff(); plt.show()


if __name__ == '__main__':
    main()
