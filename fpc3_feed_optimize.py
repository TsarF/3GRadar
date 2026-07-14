"""
Optimize the U-slot on the fpc3 feed patch for a BROADBAND match (feed-only, no cavity
-> fast, ~0.3 M cells).  A plain patch can't match across 9%, but the U-slot adds a 2nd
resonance so a slotted patch CAN -> objective is worst-in-band |S11| over 3.1-3.4 GHz.

This sizes the slot (slot_len, slot_w, slot_x) + feed (y0, L) for the widest bare-feed
match; the winner is then validated in the full cavity (the cavity re-tunes it, but a
broadband slotted feed is the right starting point).  Eval is seconds, so a full DE runs
in minutes-to-an-hour.

Run:  python fpc3_feed_optimize.py
Out:  fpc3_feed_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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

import fpc3_build as p1
p1.FEED_ONLY = True
p1.SLOT_ON = True

# ============================ config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
TARGET_DB      = -12.0
STOP_MARGIN_DB = -12.5
GUIDE_W        = 1.0

WORKERS, THREADS_PER_WORKER = 4, 2
NP, F_MUT, CR, MAX_EVALS, SEED = 20, 0.7, 0.9, 300, 12345

PARAMS = {
    'slot_len': (p1.slot_len,  4.0, 14.0, 0.2),
    'slot_w':   (p1.slot_w,    4.0, 20.0, 0.2),
    'slot_x':   (p1.slot_x,    0.0,  9.0, 0.2),
    'y0':       (p1.y0,        4.0, 13.0, 0.1),
    'L':        (p1.L,        24.0, 29.0, 0.1),
}
WARM_START_FROM = os.path.join(os.getcwd(), 'fpc3_feed_opt', 'optimized_params.json')
EVAL_NRTS, EVAL_ENDCRITERIA, EVAL_NFREQ, EVAL_FPAD = 30000, 1e-3, 101, 0.25e9

sim_path = os.path.join(os.getcwd(), 'fpc3_feed_opt')
os.makedirs(sim_path, exist_ok=True)
names = list(PARAMS.keys())
lo = np.array([PARAMS[n][1] for n in names]); hi = np.array([PARAMS[n][2] for n in names])
steps = np.array([PARAMS[n][3] for n in names]); D = len(names)
# ===============================================================


def snap(x):
    x = np.clip(np.asarray(x, float), lo, hi)
    return np.clip(np.round(x / steps) * steps, lo, hi)


def set_params(values):
    for k, v in values.items():
        setattr(p1, k, float(v))
    p1.FEED_ONLY = True; p1.SLOT_ON = True
    p1._recompute()


def worst_in_band(f, s11_dB):
    b = (f >= BAND_LO) & (f <= BAND_HI)
    return float(np.max(s11_dB[b]))


def resonance_offset(f, s11_dB):
    fr = float(f[int(np.argmin(s11_dB))])
    return abs(fr - 0.5*(BAND_LO+BAND_HI)) / (0.5*(BAND_HI-BAND_LO))


def _rmtree(path):
    for _ in range(5):
        try:
            shutil.rmtree(path); return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


@contextlib.contextmanager
def _redirect(path):
    sys.stdout.flush(); sys.stderr.flush()
    fout = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    s1, s2 = os.dup(1), os.dup(2)
    try:
        os.dup2(fout, 1); os.dup2(fout, 2); yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(s1, 1); os.dup2(s2, 2); os.close(s1); os.close(s2); os.close(fout)


def eval_worker(payload):
    x, run_dir, nthreads = payload
    logdir = os.path.join(sim_path, 'openems_logs'); os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, 'worker_%d.log' % os.getpid())
    try:
        set_params(dict(zip(names, x)))
        os.makedirs(run_dir, exist_ok=True)
        with _redirect(log):
            FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
            FDTD.SetGaussExcite(p1.f0, p1.fc)
            FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])
            CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
            p1.build_antenna(CSX, FDTD)
            port = FDTD.AddLumpedPort(1, p1.feed_R, [p1.feed_x, p1.feed_y, 0],
                                      [p1.feed_x, p1.feed_y, p1.h_sub], 'z', 1.0, priority=5, edges2grid='xy')
            CSX.Write2XML(os.path.join(run_dir, 'a.xml'))
            FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=nthreads)
            f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
            port.CalcPort(run_dir, f, ref_impedance=p1.feed_R)
        s11_dB = 20 * np.log10(np.abs(port.uf_ref / port.uf_inc))
        _rmtree(run_dir)
        return f, s11_dB
    except Exception:
        return None


def initial_population():
    rng = random.Random(SEED)
    init = {n: PARAMS[n][0] for n in names}
    if os.path.exists(WARM_START_FROM):
        try:
            data = json.load(open(WARM_START_FROM))
            for k, v in data.get('params_mm', {}).items():
                if k in init:
                    init[k] = float(v)
            print('Warm-starting from', WARM_START_FROM)
        except Exception as e:
            print('warm-start failed (%s)' % e)
    pop = [snap(np.array([init[n] for n in names], float))]
    for _ in range(NP - 1):
        pop.append(snap(lo + np.array([rng.random() for _ in range(D)]) * (hi - lo)))
    return pop, rng


def make_trial(i, pop, best_x, rng):
    idxs = [j for j in range(NP) if j != i]
    a, b = (pop[k] for k in rng.sample(idxs, 2))
    v = pop[i] + F_MUT * (best_x - pop[i]) + F_MUT * (a - b)
    u = pop[i].copy(); jr = rng.randrange(D)
    for j in range(D):
        if rng.random() < CR or j == jr:
            u[j] = v[j]
    return snap(u)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    import matplotlib.pyplot as plt
    pop, rng = initial_population()
    cache, history = {}, []
    best = {'worst': np.inf, 'x': pop[0].copy(), 'f': None, 's11': None}
    n_eval = [0]

    def key(x):
        return tuple(np.round(x, 6))

    def save_best():
        if not np.isfinite(best['worst']):
            return
        tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
        json.dump({'objective': 'min worst-in-band |S11| (slotted feed, feed-only)',
                   'worst_S11_dB': best['worst'], 'band_GHz': [BAND_LO/1e9, BAND_HI/1e9],
                   'evals': n_eval[0], 'params_mm': dict(zip(names, best['x']))},
                  open(tmp, 'w'), indent=2)
        os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))

    def save_log():
        with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['eval'] + names + ['worst_S11_dB'])
            for i, x, wv in history:
                w.writerow([i] + ['%.3f' % v for v in x] + ['%.3f' % wv])

    def consider(x, tup):
        cost, worst, f, s11 = tup
        if f is not None and worst < best['worst']:
            best.update(worst=worst, x=x.copy(), f=f, s11=s11); save_best(); return True
        return False

    def evaluate(cands, ex, gen):
        out = [None] * len(cands)
        todo = [(i, x) for i, x in enumerate(cands) if key(x) not in cache]
        for i, x in enumerate(cands):
            if key(x) in cache:
                out[i] = cache[key(x)]
        futs = {}
        for j, x in todo:
            n_eval[0] += 1
            futs[ex.submit(eval_worker, (x, os.path.join(sim_path, 'e_%04d' % n_eval[0]), THREADS_PER_WORKER))] = (j, x)
        for fut in as_completed(futs):
            j, x = futs[fut]; res = fut.result()
            if res is None:
                tup = (1e9, np.inf, None, None)
            else:
                f, s11 = res; worst = worst_in_band(f, s11)
                tup = (worst + GUIDE_W * resonance_offset(f, s11), worst, f, s11)
            cache[key(x)] = tup; out[j] = tup
            history.append((len(history) + 1, x, tup[1]))
            nb = consider(x, tup); save_log()
            print('  [eval %3d gen %d] worst=%+6.2f dB%s'
                  % (len(history), gen, tup[1] if np.isfinite(tup[1]) else float('nan'),
                     '  <-- NEW BEST' if nb else ''))
        return out

    print('=== fpc3 SLOTTED-FEED DE (%dx%d), broadband match over %.2f-%.2f GHz, target %.1f ===\n'
          % (WORKERS, THREADS_PER_WORKER, BAND_LO/1e9, BAND_HI/1e9, TARGET_DB))
    GEN_CAP = 100
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = evaluate(pop, ex, 0); pop_cost = [r[0] for r in res]
        gen, stag = 0, 0
        while best['worst'] > STOP_MARGIN_DB and n_eval[0] < MAX_EVALS and gen < GEN_CAP:
            gen += 1; before = n_eval[0]
            trials = [make_trial(i, pop, best['x'], rng) for i in range(NP)]
            tr = evaluate(trials, ex, gen)
            for i in range(NP):
                if tr[i][0] <= pop_cost[i]:
                    pop[i] = trials[i]; pop_cost[i] = tr[i][0]
            new = n_eval[0] - before
            print('[gen %2d] evals=%3d new=%2d best worst-in-band=%+6.2f dB' % (gen, n_eval[0], new, best['worst']))
            if best['worst'] <= STOP_MARGIN_DB:
                print('Target met.'); break
            stag = stag + 1 if new == 0 else 0
            if stag >= 3:
                print('Converged.'); break

    print('\n===== BEST SLOTTED FEED =====')
    print('worst-in-band |S11| = %+.2f dB (%d evals)' % (best['worst'], n_eval[0]))
    for n, v in zip(names, best['x']):
        print('  %-8s = %.2f' % (n, v))
    print('Next: set these into fpc3_build + validate in the full cavity.')
    save_best(); save_log()
    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8); ax.axhline(-10, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
        ax.set(title='Slotted feed |S11| (worst %.2f dB, feed-only)' % best['worst'],
               xlabel='GHz', ylabel='dB'); ax.grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved optimized_params.json, optimization_log.csv, optimized_s11.png')


if __name__ == '__main__':
    main()
