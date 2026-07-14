"""
STAGE A: fast standalone tuning of the FPC feed patch (fpc2_build with
FEED_ONLY=True -> bare inset patch on the 1.52 mm feed board, no PRS/cavity).
Minimizes worst-in-band |S11| over 3.1-3.4 GHz by tuning L, W, y0.

This just SIZES the patch (locks L, W, y0) and confirms it resonates in band.  The
final match is set later with the cavity present (the PRS re-tunes the feed), so
don't over-read the standalone S11 number - Stage B (fpc2_optimize_de_gain.py) takes
these L, W as a warm start and finishes the job over h_cav + y0.

Eval is ~0.34 M cells (no cavity, no NF2FF), so this is minutes, not hours.

Run:  python fpc2_feed_optimize.py
Out:  fpc2_feed_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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
p1.FEED_ONLY = True                       # bare patch, no PRS/cavity

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
FEED_F         = 3.25e9                    # match the patch HERE (band center); the cavity,
                                           # not the feed, provides the band coverage in Stage B
TARGET_DB      = -15.0                     # stop early if the match at FEED_F beats this
STOP_MARGIN_DB = -15.5
GUIDE_W        = 2.0

WORKERS            = 4                      # small sim -> more parallelism
THREADS_PER_WORKER = 4

NP        = 16
F_MUT     = 0.7
CR        = 0.9
MAX_EVALS = 250
SEED      = 12345

PARAMS = {
    'L':  (p1.L,  24.0, 29.0, 0.1),
    'W':  (p1.W,  28.0, 37.0, 0.2),
    'y0': (p1.y0,  3.0, 12.0, 0.1),
}

WARM_START_FROM = os.path.join(os.getcwd(), 'fpc2_feed_opt', 'optimized_params.json')

EVAL_NRTS        = 30000
EVAL_ENDCRITERIA = 1e-3
EVAL_NFREQ       = 101
EVAL_FPAD        = 0.25e9

sim_path = os.path.join(os.getcwd(), 'fpc2_feed_opt')
os.makedirs(sim_path, exist_ok=True)

names = list(PARAMS.keys())
lo    = np.array([PARAMS[n][1] for n in names])
hi    = np.array([PARAMS[n][2] for n in names])
steps = np.array([PARAMS[n][3] for n in names])
D     = len(names)
# ==========================================================================


def snap(x):
    x = np.clip(np.asarray(x, float), lo, hi)
    return np.clip(np.round(x / steps) * steps, lo, hi)


def set_params(values):
    for name, val in values.items():
        setattr(p1, name, float(val))
    p1.FEED_ONLY = True
    p1._recompute()


def center_match(f, s11_dB):
    # A single patch can't be <-10 dB across a 9% band (patch BW ~2-4%), so match it at
    # band center and let the CAVITY provide band coverage in Stage B.
    return float(np.interp(FEED_F, f, s11_dB))


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
    """One bare-patch FDTD solve. Returns (f, s11_dB) or None."""
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
            CSX.Write2XML(os.path.join(run_dir, 'antenna.xml'))
            FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=nthreads)
            f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
            port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
        s11_dB = 20 * np.log10(np.abs(port.uf_ref / port.uf_inc))
        _rmtree_retry(run_dir)
        return f, s11_dB
    except Exception:
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
            print('Warm-starting from', WARM_START_FROM)
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
    best = {'m': np.inf, 'x': pop[0].copy(), 'f': None, 's11': None}
    n_eval = [0]

    def key(x):
        return tuple(np.round(x, 6))

    def save_best():
        if not np.isfinite(best['m']):
            return
        tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
        with open(tmp, 'w') as fh:
            json.dump({'objective': 'min |S11| at %.3f GHz (bare feed patch)' % (FEED_F/1e9),
                       'optimizer': 'DE', 'match_S11_dB': best['m'], 'match_freq_GHz': FEED_F/1e9,
                       'evals_so_far': n_eval[0], 'params_mm': dict(zip(names, best['x']))}, fh, indent=2)
        os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))

    def save_log():
        with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['eval'] + names + ['match_S11_dB'])
            for i, x, wv in history:
                w.writerow([i] + ['%.4f' % v for v in x] + ['%.3f' % wv])

    def consider(x, tup):
        cost, m, f, s11 = tup
        if f is not None and m < best['m']:
            best.update(m=m, x=x.copy(), f=f, s11=s11)
            save_best()
            return True
        return False

    def evaluate(cands, executor, gen):
        out = [None] * len(cands)
        todo = [(i, x) for i, x in enumerate(cands) if key(x) not in cache]
        for i, x in enumerate(cands):
            if key(x) in cache:
                out[i] = cache[key(x)]
        futs = {}
        for j, x in todo:
            n_eval[0] += 1
            rd = os.path.join(sim_path, 'e_%04d' % n_eval[0])
            futs[executor.submit(eval_worker, (x, rd, THREADS_PER_WORKER))] = (j, x)
        for fut in as_completed(futs):
            j, x = futs[fut]
            res = fut.result()
            if res is None:
                tup = (1e9, np.inf, None, None)
            else:
                f, s11 = res
                m = center_match(f, s11)
                cost = m + GUIDE_W * resonance_offset(f, s11)
                tup = (cost, m, f, s11)
            cache[key(x)] = tup
            out[j] = tup
            history.append((len(history) + 1, x, tup[1]))
            is_best = consider(x, tup)
            save_log()
            print('  [eval %3d | gen %d] match@%.2fGHz=%+6.2f dB%s'
                  % (len(history), gen, FEED_F/1e9, tup[1] if np.isfinite(tup[1]) else float('nan'),
                     '  <-- NEW BEST' if is_best else ''))
        return out

    print('===== STAGE-A feed patch DE (%d x %d) over %.2f-%.2f GHz, target %.1f dB =====\n'
          % (WORKERS, THREADS_PER_WORKER, BAND_LO/1e9, BAND_HI/1e9, TARGET_DB))

    GEN_CAP = 100                 # backstop so the loop can never spin forever
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        res = evaluate(pop, executor, 0)
        pop_cost = [r[0] for r in res]
        gen = 0
        stagnant = 0
        while best['m'] > STOP_MARGIN_DB and n_eval[0] < MAX_EVALS and gen < GEN_CAP:
            gen += 1
            before = n_eval[0]
            trials = [make_trial(i, pop, best['x'], rng) for i in range(NP)]
            tres = evaluate(trials, executor, gen)
            for i in range(NP):
                if tres[i][0] <= pop_cost[i]:
                    pop[i] = trials[i]; pop_cost[i] = tres[i][0]
            new = n_eval[0] - before
            print('[gen %2d] evals=%3d  new=%2d  best match@%.2fGHz=%+6.2f dB'
                  % (gen, n_eval[0], new, FEED_F/1e9, best['m']))
            if best['m'] <= STOP_MARGIN_DB:
                print('Target met - stopping.'); break
            stagnant = stagnant + 1 if new == 0 else 0
            if stagnant >= 3:
                print('Converged (population collapsed, all candidates cached) - stopping.'); break
        if gen >= GEN_CAP:
            print('Hit generation cap (%d) - stopping.' % GEN_CAP)

    print('\n===================== BEST FEED PATCH =====================')
    print('|S11| at %.3f GHz = %+.2f dB  (%d evals)' % (FEED_F/1e9, best['m'], n_eval[0]))
    print('\n--- paste into fpc2_build.py (feed section) ---')
    for n, v in zip(names, best['x']):
        print('%-4s = %.2f' % (n, v))
    print('----------------------------------------------\n')
    print('Next: warm-start fpc2_optimize_de_gain.py with these L, W and tune h_cav + y0.')
    save_best(); save_log()
    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8)
        ax.axhline(-10, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        ax.axvline(FEED_F/1e9, color='b', ls=':', lw=0.8)
        ax.set(title='Bare feed patch |S11| (%.2f dB @ %.2f GHz)' % (best['m'], FEED_F/1e9),
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)'); ax.grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved optimized_params.json, optimization_log.csv, optimized_s11.png')
    if _gui:
        plt.show()


if __name__ == '__main__':
    main()
