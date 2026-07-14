"""
Parallel Differential-Evolution optimizer for the slotted inset patch
(inset_slot_build): minimize worst |S11| over 3.1-3.4 GHz.

Why DE instead of Nelder-Mead: NM is a single simplex that collapses into the first
local basin it finds (the -6.3 dB plateau). DE evolves a POPULATION using difference
vectors, so it explores many basins at once and climbs out of local minima. The
population is also embarrassingly parallel - each generation's candidates run
concurrently in a process pool (WORKERS solves at once, 8/WORKERS threads each).

Keeps: shaped cost (resonance-centering guidance), incremental best-save + live log,
dedup cache, warm-start seeding, live S11 plot (per generation).

Run:  python inset_slot_optimize_de.py
Out:  inset_slot_de_opt/optimized_params.json | optimization_log.csv | optimized_s11.png
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
matplotlib.use('Agg')                    # safe in worker subprocesses; main may switch
from concurrent.futures import ProcessPoolExecutor, as_completed

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import inset_slot_build as p1

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
TARGET_DB        = -12.0
STOP_MARGIN_DB   = -12.5
GUIDE_W          = 2.0                    # resonance-centering guidance weight

# parallelism: WORKERS solves at once, THREADS_PER_WORKER each (product ~= cores)
WORKERS            = 4
THREADS_PER_WORKER = 4

# DE controls
NP        = 20                            # population size
F_MUT     = 0.7                           # differential weight
CR        = 0.9                           # crossover probability
MAX_EVALS = 800                           # hard cap on FDTD solves
SEED      = 12345

PARAMS = {
    'L':        (p1.L,        22.0, 32.0, 0.1),
    'W':        (p1.W,        26.0, 42.0, 0.2),
    'y0':       (p1.y0,        2.0, 14.0, 0.1),
    'h_air':    (p1.h_air,     2.0, 12.0, 0.1),
    'Lp':       (p1.Lp,       24.0, 36.0, 0.1),
    'Wp':       (p1.Wp,       28.0, 48.0, 0.2),
    'slot_len': (p1.slot_len,  2.0, 16.0, 0.2),
    'slot_w':   (p1.slot_w,    2.0, 24.0, 0.2),
    'slot_x':   (p1.slot_x,  -12.0,  8.0, 0.2),
}

WARM_START_FROM = os.path.join(os.getcwd(), 'inset_slot_de_opt', 'optimized_params.json')

EVAL_NRTS        = 80000
EVAL_ENDCRITERIA = 1e-3
EVAL_NFREQ       = 121
EVAL_FPAD        = 0.20e9

sim_path = os.path.join(os.getcwd(), 'inset_slot_de_opt')
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
    p1.hf     = p1.Wf / 2.0
    p1.x_in   = -p1.L/2 + p1.y0
    p1.x_port = -p1.L/2 - p1.Lf
    p1.feed_x, p1.feed_y = p1.x_port, 0.0
    p1.gnd_x0 = p1.x_port - p1.margin
    p1.gnd_x1 =  p1.L/2   + p1.margin
    p1.gnd_y0 = -p1.W/2   - p1.margin
    p1.gnd_y1 =  p1.W/2   + p1.margin
    p1.z_gnd       = 0.0
    p1.z_drv_patch = p1.h_sub
    p1.z_stk_bot   = p1.h_sub + p1.h_air
    p1.z_stk_patch = p1.h_sub + p1.h_air + p1.h_sub2


def worst_in_band(f, s11_dB):
    band = (f >= BAND_LO) & (f <= BAND_HI)
    return float(np.max(s11_dB[band]))


def resonance_offset(f, s11_dB):
    f_res = float(f[int(np.argmin(s11_dB))])
    f_c   = 0.5 * (BAND_LO + BAND_HI)
    return abs(f_res - f_c) / (0.5 * (BAND_HI - BAND_LO))


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
    """Redirect OS-level stdout+stderr (fd 1/2) to `path`, so openEMS's C/C++ chatter
    goes to a log file instead of the console. Python-level redirect_stdout can't do
    this - the engine writes below Python. Restores the fds on exit."""
    sys.stdout.flush(); sys.stderr.flush()
    fout = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    saved1, saved2 = os.dup(1), os.dup(2)
    try:
        os.dup2(fout, 1); os.dup2(fout, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved1, 1); os.dup2(saved2, 2)
        os.close(saved1); os.close(saved2); os.close(fout)


def eval_worker(payload):
    """Run one S11 FDTD solve in a worker process. Returns (f, s11_dB) or None."""
    x, run_dir, nthreads = payload
    logdir = os.path.join(sim_path, 'openems_logs')
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, 'worker_%d.log' % os.getpid())
    try:
        set_params(dict(zip(names, x)))
        os.makedirs(run_dir, exist_ok=True)
        with _redirect_fds(log):        # openEMS chatter -> per-worker log file
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
    except Exception as e:
        return None


# ----------------------------- DE core -----------------------------
def initial_population():
    """First member = warm-start (best-so-far / build defaults); rest random."""
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
    """DE/current-to-best/1 + binomial crossover."""
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
    global _plt
    try:
        sys.stdout.reconfigure(line_buffering=True)   # flush each line -> live output
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
    cache = {}                            # snapped tuple -> (cost, worst, f, s11)
    history = []                          # (eval#, x, worst)
    best = {'worst': np.inf, 'x': pop[0].copy(), 'cost': np.inf, 'f': None, 's11': None}
    n_eval = [0]
    recent = []

    def key(x):
        return tuple(np.round(x, 6))

    def save_best():
        if not np.isfinite(best['worst']):
            return
        bx = dict(zip(names, best['x']))
        tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
        with open(tmp, 'w') as fh:
            json.dump({'objective_worst_S11_dB': best['worst'], 'optimizer': 'DE',
                       'band_GHz': [BAND_LO/1e9, BAND_HI/1e9], 'evals_so_far': n_eval[0],
                       'params_mm': bx}, fh, indent=2)
        os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))

    def save_log():
        with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['eval'] + names + ['worst_S11_dB'])
            for i, x, wo in history:
                w.writerow([i] + ['%.4f' % v for v in x] + ['%.3f' % wo])

    def live_plot(gen):
        if not _gui or best['f'] is None:
            return
        try:
            if not hasattr(live_plot, 'ax'):
                plt.ion(); live_plot.fig, live_plot.ax = plt.subplots(figsize=(8, 5))
            ax = live_plot.ax; ax.clear()
            for (g, ff, ss, wo) in recent[-3:]:
                ax.plot(ff/1e9, ss, lw=1.3, alpha=0.5, label='gen %d best %.2f dB' % (g, wo))
            ax.plot(best['f']/1e9, best['s11'], lw=2.4, color='k', label='overall best %.2f dB' % best['worst'])
            ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
            ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
            ax.set(title='DE gen %d | %d evals | best %.2f dB' % (gen, n_eval[0], best['worst']),
                   xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
            ax.legend(loc='upper right', fontsize=8); ax.grid(True)
            live_plot.fig.tight_layout()
            live_plot.fig.canvas.draw_idle(); live_plot.fig.canvas.flush_events(); plt.pause(0.001)
            live_plot.fig.savefig(os.path.join(sim_path, 'live_s11.png'), dpi=110)
        except Exception as e:
            print('live plot disabled (%s)' % e)

    def consider(x, tup, gen):
        """Update global best (by true worst) + persist immediately."""
        cost, wo, f, s11 = tup
        if f is not None and wo < best['worst']:
            best.update(worst=wo, x=x.copy(), cost=cost, f=f, s11=s11)
            save_best()
            return True
        return False

    def evaluate(cands, executor, gen):
        """Evaluate candidates; returns list of (cost, worst, f, s11). Uncached ones
        run in the pool and are logged + printed AS EACH finishes (live progress)."""
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
                tup = (1e9, 0.0, None, None)
            else:
                ff, s11 = res
                wo = worst_in_band(ff, s11)
                tup = (wo + GUIDE_W * resonance_offset(ff, s11), wo, ff, s11)
            cache[key(x)] = tup
            out[j] = tup
            history.append((len(history) + 1, x, tup[1]))
            is_best = consider(x, tup, gen)
            save_log()                        # log + best persisted every eval
            if res is None:
                print('  [eval %3d | gen %d] FAILED' % (len(history), gen))
            else:
                print('  [eval %3d | gen %d] worst=%+6.2f dB%s'
                      % (len(history), gen, tup[1], '  <-- NEW BEST' if is_best else ''))
        return out

    print('===== DE (%d workers x %d threads): minimize worst S11 over %.2f-%.2f GHz ====='
          % (WORKERS, THREADS_PER_WORKER, BAND_LO/1e9, BAND_HI/1e9))
    print('Population %d, F=%.2f, CR=%.2f, max %d evals\n' % (NP, F_MUT, CR, MAX_EVALS))

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        print('Evaluating initial population (%d designs)...' % NP)
        res = evaluate(pop, executor, 0)      # initial pop (logs/prints per eval)
        pop_cost = [r[0] for r in res]
        gen = 0
        while best['worst'] > STOP_MARGIN_DB and n_eval[0] < MAX_EVALS:
            gen += 1
            trials = [make_trial(i, pop, best['x'], rng) for i in range(NP)]
            tres = evaluate(trials, executor, gen)
            gen_best = (np.inf, None, None, None)
            for i in range(NP):
                cost, wo, f, s11 = tres[i]
                if cost <= pop_cost[i]:               # DE selection (on shaped cost)
                    pop[i] = trials[i]; pop_cost[i] = cost
                if f is not None and wo < gen_best[0]:
                    gen_best = (wo, f, s11, trials[i])
            if gen_best[1] is not None:
                recent.append((gen, gen_best[1], gen_best[2], gen_best[0]))
            live_plot(gen)
            print('[gen %2d done] evals=%3d  gen-best=%+6.2f dB  overall-best=%+6.2f dB'
                  % (gen, n_eval[0], gen_best[0] if gen_best[1] is not None else float('nan'), best['worst']))
            if best['worst'] <= STOP_MARGIN_DB:
                print('Target met (%.2f <= %.2f dB) - stopping.' % (best['worst'], STOP_MARGIN_DB))
                break

    # ---- report ----
    set_params(dict(zip(names, best['x'])))
    fit, xb, xo, wi, sw = p1._slot_geom()
    print('\n=========================== BEST DESIGN (DE) ===========================')
    print('Worst |S11| in %.2f-%.2f GHz: %+.2f dB  (%d evals)'
          % (BAND_LO/1e9, BAND_HI/1e9, best['worst'], n_eval[0]))
    print('U-slot active: %s' % fit)
    print('\n--- paste into inset_slot_build.py DESIGN PARAMETERS ---')
    for n, v in zip(names, best['x']):
        print('%-8s = %.2f' % (n, v))
    print('-------------------------------------------------------\n')
    save_best(); save_log()
    if best['f'] is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(best['f']/1e9, best['s11'], lw=1.8)
        ax.axhline(TARGET_DB, color='r', ls='--', lw=0.8)
        ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target band')
        ax.set(title='DE best: slotted inset patch (worst %+.2f dB)' % best['worst'],
               xlabel='Frequency (GHz)', ylabel='|S11| (dB)')
        ax.legend(); ax.grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'optimized_s11.png'), dpi=130)
        print('Saved optimized_params.json, optimization_log.csv, optimized_s11.png')
    if _gui:
        plt.ioff(); plt.show()


if __name__ == '__main__':
    main()
