"""
Self-driving DE optimizer for the 3-layer FP antenna (fpc3_build: feed + PRS + RCM),
maximizing worst-in-band REALIZED boresight gain over 3.1-3.4 GHz.  It CO-TUNES the feed
(y0, L, W) WITH the cavity (h_cav, h3) and the subdivided-RCM sub-square spacing (rcm_gap).

Co-tuning is essential: the cavity strongly loads the feed (feed-only matching does NOT
transfer -- see the feed-only slot DE, which matched -12.65 dB in isolation but -1.1 dB in
the cavity).  Realized gain = directivity * eta_rad * (1-|S11|^2), so maximizing worst-in-
band realized gain simultaneously centers/broadens the match AND keeps directivity high.

Runs on the current FAITHFUL geometry: 8x8 wire-grid PRS + subdivided RCM (RCM_MAP, tile
footprint rcm_s=41 held fixed = 2 PRS cells).  rcm_gap sets the sub-square sizes, which
stagger the RCM resonances for bandwidth.

Designed to run UNATTENDED for days AND survive spot/preemptible interruption: it writes a
FULL DE-state checkpoint (population, costs, generation, RNG, eval cache, best, history) to
de_state.pkl atomically after EVERY eval, and resumes mid-search on relaunch (prints
'>>> RESUMED ... gen=N'). Losing the box costs at most the few evals in flight (cache-hits
on resume). Also writes the best design to optimized_params.json on every improvement.

Search opt-mesh /16 (~6.9 M cells on the faithful geometry); default 2 workers x 4 threads.
Parallelism + run-identity are ENV-CONFIGURABLE (no code edits) so the SAME script runs on
this box, a Threadripper, or EC2 M8g spot:
  FPC_WORKERS, FPC_THREADS, FPC_NP, FPC_SEED   -- parallelism / DE size / trial stream
  FPC_TAG   -- output dir suffix (fpc3_gain_de_opt<TAG>), isolate parallel "island" runs
  FPC_WARM  -- warm-start seed json (default: own dir's optimized_params.json)
  FPC_CKPT  -- checkpoint path (point at durable/S3-synced storage on spot)
See fpc3_THREADRIPPER.md and fpc3_EC2.md for the multi-machine / spot recipes.

Run:  python fpc3_optimize_de_gain.py
Out:  fpc3_gain_de_opt<TAG>/optimized_params.json | optimization_log.csv | live_s11.png | de_state.pkl
"""

import os
import re
import sys
import csv
import traceback
import json
import time
import pickle
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
p1.RCM_ON = True
# Search mesh: /16 (~5.9 M cells) on the faithful 8x8-PRS + subdivided-RCM geometry (which
# is ~3x heavier than the old geometry). Coarse enough for design RANKING; the winner is
# re-validated at /22 in fpc3_validate.py.
p1.mesh_res = (p1.C0 / (p1.f0 + p1.fc)) / p1.unit / 16.0

# ============================ optimiser config ============================
BAND_LO, BAND_HI = 3.10e9, 3.40e9
NGAIN            = 9      # nf2ff sample freqs (directivity); match term is evaluated densely
GAIN_TARGET      = None
GUIDE_W          = 2.0

# Soft match penalty: realized gain barely punishes a bad match (a -5 dB match only costs
# 1.65 dB of gain), so the DE happily trades match for directivity. We subtract a GRADED
# penalty for worst-in-band |S11| above the target: score = gain - MATCH_K*max(0, S11-target).
# MATCH_K=1 proved TOO WEAK -- the DE maximized gain (to +14 dBi) and left S11 at ~-2 dB, using
# a big slot/tall cavity to pump gain instead of matching. MATCH_K>=3 makes closing the match
# gap worth more than the whole gain range, so the DE gets matched FIRST, then maximizes gain
# among matched designs (and reveals the best achievable match if -10 dB is out of reach).
MATCH_TARGET_DB  = float(os.environ.get('FPC_MATCH_TARGET_DB', '-10.0'))
MATCH_K          = float(os.environ.get('FPC_MATCH_K', '3.0'))   # dB of score per dB of S11 over target
NONCONV_PENALTY  = 15.0  # score penalty when the field didn't settle (untrustworthy far-field)

# --- parallelism / run-identity are ENV-CONFIGURABLE so the SAME script + checkpoint format
#     runs on this box AND a big many-core machine without editing code. openEMS FDTD is
#     memory-bandwidth bound, so throughput comes from MORE WORKERS, not more threads/sim.
#     Threadripper 7960X (24c/48t, 4x 6-core CCDs): FPC_WORKERS=4 FPC_THREADS=6 pins one
#     worker per CCD (cache/NUMA-local). Unset -> the local 2x4 defaults (local run untouched).
WORKERS            = int(os.environ.get('FPC_WORKERS', '2'))
THREADS_PER_WORKER = int(os.environ.get('FPC_THREADS', '4'))
SEED               = int(os.environ.get('FPC_SEED', '12345'))
NP                 = int(os.environ.get('FPC_NP', '16'))
F_MUT     = 0.7
CR        = 0.9
MAX_EVALS = 600                            # large: runs for days, checkpoints as it goes

# Output/checkpoint dir: set FPC_TAG (e.g. "_TR") to run an independent ISLAND into its own
# dir on another machine, so the two runs never clobber each other's files. Same geometry +
# mesh -> the two optimized_params.json are directly comparable; keep whichever wins.
_TAG     = os.environ.get('FPC_TAG', '')
sim_path = os.path.join(os.getcwd(), 'fpc3_gain_de_opt' + _TAG)
os.makedirs(sim_path, exist_ok=True)

PARAMS = {
    'h_cav':    (p1.h_cav,   42.0, 52.0, 0.5),  # PRS cavity height (Trentini gain-peak freq)
    'h3':       (p1.h3,      36.0, 58.0, 0.5),  # PRS -> RCM gap
    'rcm_gap':  (p1.rcm_gap,  0.5,  8.0, 0.5),  # RCM sub-square spacing (staggers resonances -> BW)
    'y0':       (p1.y0,       5.0, 13.0, 0.2),  # feed inset (match: input R / coupling)
    'L':        (p1.L,       24.5, 28.5, 0.1),  # patch length (match: centers/deepens S11 dip)
    'W':        (p1.W,       26.0, 33.0, 0.2),  # patch width (match: input impedance / Q)
    # U-slot on the feed patch = broadbanding lever (adds a 2nd resonance) to break the ~-8 dB
    # match plateau. Co-tuned WITH the cavity here (feed-only tuning did NOT transfer).
    'slot_len': (8.0,         3.0, 16.0, 0.5),  # U arm length (x)
    'slot_w':   (10.0,        4.0, 22.0, 0.5),  # U arm separation / tongue width (y)
    'slot_x':   (3.0,         1.0, 10.0, 0.5),  # U base x-position
    # PRS reflectivity = cavity Q knobs. The slot alone couldn't match 9% because the fixed
    # high-|Gamma| PRS makes Q~22 (Bode-Fano: ~4.5% match BW). LOWER r1 / thinner mesh_wt ->
    # more transmissive PRS -> lower Q -> wider match. With the K match penalty, the DE lowers
    # Q just enough to hit -10 dB, then keeps gain as high as possible -> optimal gain-vs-BW.
    # r1 bounds are env-overridable (FPC_R1_LO/HI) so you can FORCE a transmissive, low-Q PRS
    # (e.g. FPC_R1_HI=6.5) to test whether opening the cavity actually improves the match --
    # the free search collapses to high r1 and never tests the low-Q regime on its own.
    'r1':       (p1.r1, float(os.environ.get('FPC_R1_LO', '4.0')),
                        float(os.environ.get('FPC_R1_HI', '10.5')), 0.25),  # PRS disc radius
    'mesh_wt':  (p1.mesh_wt,  0.3,  2.0, 0.1),  # PRS bottom wire-mesh trace width
    # RCM tile footprint on the fixed 43 mm grid -> sets the INTER-TILE gap (P_RCM - rcm_s),
    # which was frozen at 2 mm. Smaller rcm_s = bigger gaps between tiles (more transmissive
    # RCM / shifts its resonances); sub-squares scale with the footprint.
    'rcm_s':    (p1.rcm_s,   33.0, 42.5, 0.5),  # inter-tile gap = 43 - rcm_s (was fixed 2 mm)
}

# Warm-start seed: own checkpoint by default; set FPC_WARM to a path (e.g. the other
# machine's optimized_params.json) to seed an island run from the current global best.
WARM_START_FROM = os.environ.get('FPC_WARM', os.path.join(sim_path, 'optimized_params.json'))

# FULL DE-STATE checkpoint (population, costs, generation, RNG, eval cache, best, history)
# -> a spot/preemptible instance can be reclaimed at any moment and lose ZERO progress:
# relaunch and it resumes mid-search. Written atomically after every eval. Point FPC_CKPT at
# durable storage (persistent EBS or an S3-synced dir) so it survives instance replacement.
CKPT_PATH = os.environ.get('FPC_CKPT', os.path.join(sim_path, 'de_state.pkl'))

EVAL_NRTS        = 150000         # safety ceiling; with PML sides most evals hit the -30 dB
EVAL_ENDCRITERIA = 1e-3           # EndCriteria (~77k steps for Q~22) well before this cap
EVAL_NFREQ       = 121   # dense |S11| grid for the ungameable worst-in-band match term
EVAL_FPAD        = 0.15e9

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
    p1.RCM_ON = True
    p1.SLOT_ON = True                 # slot_len/slot_w/slot_x are now DE knobs
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
    """One 3-layer cavity FDTD + NF2FF solve. Returns (f, s11_dB, f_g, Gr_dBi) or None."""
    x, run_dir, nthreads = payload
    if os.environ.get('FPC_FAKE'):                # test hook: synthetic eval, no FDTD
        time.sleep(float(os.environ.get('FPC_FAKE_SLEEP', '0')))
        f = np.linspace(BAND_LO - EVAL_FPAD, BAND_HI + EVAL_FPAD, EVAL_NFREQ)
        f_g = np.linspace(BAND_LO, BAND_HI, NGAIN)
        # smooth deterministic surrogate so the DE has a real gradient to climb (knob-agnostic)
        c = 0.5 * (lo + hi)
        pen = float(np.sum(((np.asarray(x) - c) / ((hi - lo) / 2.0))**2))
        Gr = np.full(NGAIN, 14.0 - pen)
        s11 = np.full(EVAL_NFREQ, -8.0 - 4.0 * np.exp(-pen))
        return f, s11, f_g, Gr, float(Gr.min()), float(s11.max()), True
    logdir = os.path.join(sim_path, 'openems_logs'); os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, 'worker_%d.log' % os.getpid())
    try:
        set_params(dict(zip(names, x)))
        os.makedirs(run_dir, exist_ok=True)
        log_pos = os.path.getsize(log) if os.path.exists(log) else 0   # for convergence check
        with _redirect_fds(log):
            FDTD = openEMS(NrTS=EVAL_NRTS, EndCriteria=EVAL_ENDCRITERIA)
            FDTD.SetGaussExcite(p1.f0, p1.fc)
            # PML on ALL sides (was MUR on the 4 lateral walls). MUR reflects the grazing
            # parallel-plate mode trapped between the big PEC planes, so domain energy never
            # decayed to EndCriteria and ~82% of evals hit the NRTS cap TRUNCATED (unreliable
            # S11/gain). PML absorbs that mode -> ring-down follows the true antenna Q.
            FDTD.SetBoundaryCond(['PML_8'] * 6)
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
            Gr_dBi = np.empty(NGAIN)                     # sparse, for the live plot
            for i in range(NGAIN):
                eta_rad   = np.clip(nf.Prad[i] / Pacc_g[i], 0, 1)
                eta_match = max(1.0 - s11_g[i]**2, 0.0)
                Gr_dBi[i] = 10 * np.log10(max(nf.Dmax[i] * eta_rad * eta_match, 1e-6))
            # OBJECTIVE = worst-in-band realized gain on the DENSE |S11| grid. Evaluating the
            # match at only NGAIN nf2ff points lets the DE game it: a high-Q narrowband design
            # dips |S11| at exactly those points while the match is garbage in between, scoring
            # a fake ~+16 dBi. Directivity/eta_rad vary smoothly -> interpolate them onto the
            # dense grid; the sharp, gameable match term uses the dense |S11| directly.
            Dmax_g = np.array([nf.Dmax[i] for i in range(NGAIN)])
            Prad_g = np.array([nf.Prad[i] for i in range(NGAIN)])
            inb = (f >= BAND_LO) & (f <= BAND_HI)
            D_d    = np.interp(f, f_g, Dmax_g)[inb]
            Prad_d = np.interp(f, f_g, Prad_g)[inb]
            etar_d = np.clip(Prad_d / Pacc_f[inb], 0, 1)
            etam_d = np.clip(1.0 - s11mag[inb]**2, 0, 1)
            worst_dense = float(np.min(10 * np.log10(np.maximum(D_d * etar_d * etam_d, 1e-6))))
            worst_s11 = float(np.max(20 * np.log10(s11mag[inb])))   # worst in-band |S11| (dB)
        # Convergence flag: did the field ring down to the -30 dB EndCriteria before the NRTS
        # cap? If not, the NF2FF directivity was computed on unsettled fields (inflated, can
        # exceed the aperture ceiling) and the design is high-Q. The penalty is applied in main.
        last_db = 0.0
        try:
            with open(log) as lh:
                lh.seek(log_pos)
                for line in lh:
                    m = re.search(r'\(-\s*([0-9.]+)dB\)', line)
                    if m:
                        last_db = float(m.group(1))
        except Exception:
            pass
        _rmtree_retry(run_dir)
        return f, s11_dB, f_g, Gr_dBi, worst_dense, worst_s11, (last_db >= 27.0)
    except Exception:
        try:    # record WHY it failed instead of silently returning -inf
            fdir = os.path.join(sim_path, 'failures'); os.makedirs(fdir, exist_ok=True)
            with open(os.path.join(fdir, 'fail_%d.log' % os.getpid()), 'a') as fh:
                fh.write('\n===== params: %s =====\n' % dict(zip(names, np.round(x, 3))))
                traceback.print_exc(file=fh)
        except Exception:
            pass
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
    import matplotlib.pyplot as plt

    cache = {}
    history = []
    n_eval = [0]

    def key(x):
        return tuple(np.round(x, 6))

    # ---- full-state checkpoint (survives spot/preemptible interruption) ----
    # SIG guards against resuming a state built with different knobs/mesh/NP/bounds.
    SIG = {'names': tuple(names), 'mesh_div': 16, 'NP': NP,
           'lo': tuple(map(float, lo)), 'hi': tuple(map(float, hi)),
           'nrts': EVAL_NRTS, 'bc': 'PML6', 'metal': 'Cu',
           'obj': 'dense-worst+matchpen', 'mt': MATCH_TARGET_DB, 'mk': MATCH_K}  # invalidates old cache
    # ST holds the gen-BOUNDARY snapshot (pop, costs, gen, rng). cache/history/best/n_eval
    # are updated every eval; ST['rng'] only at gen boundaries, so a mid-gen interruption
    # regenerates that gen's identical trials -> its completed evals are cache-hits on resume.
    ST = {'pop': None, 'pop_cost': None, 'gen': -1, 'rng': None}

    def save_state():
        tmp = CKPT_PATH + '.tmp'
        try:
            with open(tmp, 'wb') as fh:
                pickle.dump({'sig': SIG, 'pop': ST['pop'], 'pop_cost': ST['pop_cost'],
                             'gen': ST['gen'], 'rng': ST['rng'], 'best': best,
                             'cache': cache, 'history': history, 'n_eval': n_eval[0]},
                            fh, protocol=4)
            os.replace(tmp, CKPT_PATH)
        except Exception as e:
            print('  [checkpoint write failed: %s]' % e)

    def load_state():
        if not os.path.exists(CKPT_PATH):
            return None
        try:
            with open(CKPT_PATH, 'rb') as fh:
                st = pickle.load(fh)
        except Exception as e:
            print('checkpoint unreadable (%s) - starting fresh' % e)
            return None
        if st.get('sig') != SIG:
            print('checkpoint signature differs (knobs/mesh/NP/bounds changed) - starting fresh')
            return None
        return st

    st = load_state()
    if st is not None:
        pop = [np.array(p, float) for p in st['pop']]
        cache.update(st['cache']); history.extend(st['history'])
        n_eval[0] = st['n_eval']
        best = st['best']
        rng = random.Random(); rng.setstate(st['rng'])
        ST.update(pop=pop, pop_cost=st['pop_cost'], gen=st['gen'], rng=st['rng'])
        print('>>> RESUMED from %s: gen=%d, %d evals, best score=%+.2f (gain %+.2f dBi, S11 %+.1f dB)'
              % (os.path.basename(CKPT_PATH), st['gen'], n_eval[0],
                 best.get('score', float('nan')), best.get('gain', float('nan')),
                 best.get('s11_worst', float('nan'))))
    else:
        pop, rng = initial_population()
        best = {'score': -np.inf, 'gain': -np.inf, 's11_worst': np.inf,
                'x': pop[0].copy(), 'f': None, 's11': None, 'f_g': None, 'Gr': None}
        ST.update(pop=pop, pop_cost=None, gen=-1, rng=rng.getstate())

    def save_best():
        if not np.isfinite(best['score']):
            return
        bx = dict(zip(names, best['x']))
        tmp = os.path.join(sim_path, 'optimized_params.json.tmp')
        with open(tmp, 'w') as fh:
            json.dump({'objective': 'max [worst-in-band realized gain - MATCH_K*max(0, worstS11 - target)]',
                       'optimizer': 'DE', 'match_target_dB': MATCH_TARGET_DB, 'match_K': MATCH_K,
                       'score': best['score'],
                       'worst_realized_gain_dBi': best['gain'],
                       'worst_in_band_S11_dB': best['s11_worst'],
                       'gain_across_band': None if best['Gr'] is None else list(map(float, best['Gr'])),
                       'band_GHz': [BAND_LO/1e9, BAND_HI/1e9], 'evals_so_far': n_eval[0],
                       'fixed_mm': {'sb': p1.sb, 'N_PRS': p1.N_PRS, 'P': p1.P, 'P_RCM': p1.P_RCM,
                                    'RCM_MAP': p1.RCM_MAP},
                       'params_mm': bx}, fh, indent=2)
        os.replace(tmp, os.path.join(sim_path, 'optimized_params.json'))

    def save_log():
        with open(os.path.join(sim_path, 'optimization_log.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['eval'] + names + ['worst_gain_dBi', 'worst_S11_dB', 'score'])
            for i, x, gv, s11v, sc in history:
                w.writerow([i] + ['%.3f' % v for v in x] + ['%.3f' % gv, '%.3f' % s11v, '%.3f' % sc])

    def live_plot():
        if best['f'] is None:
            return
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax[0].plot(best['f']/1e9, best['s11'], lw=1.8); ax[0].axhline(-10, color='r', ls='--', lw=0.8)
            ax[0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
            ax[0].set(title='best |S11| (worst %.1f dB)' % best['s11_worst'], xlabel='GHz', ylabel='dB')
            ax[0].grid(True)
            ax[1].plot(best['f_g']/1e9, best['Gr'], 'o-')
            ax[1].set(title='realized gain (worst %.2f dBi, score %.2f)' % (best['gain'], best['score']),
                      xlabel='GHz', ylabel='dBi')
            ax[1].grid(True)
            fig.tight_layout(); fig.savefig(os.path.join(sim_path, 'live_s11.png'), dpi=110)
            plt.close(fig)
        except Exception:
            pass

    def consider(x, tup):
        cost, gain, f, s11, f_g, Gr, s11_worst, score = tup
        if f is not None and score > best['score']:
            best.update(score=score, gain=gain, s11_worst=s11_worst,
                        x=x.copy(), f=f, s11=s11, f_g=f_g, Gr=Gr)
            save_best(); live_plot()
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
                tup = (1e9, -np.inf, None, None, None, None, np.inf, -np.inf)
            else:
                ff, s11, f_g, Gr, worst_dense, worst_s11, converged = res
                gain = worst_dense                     # dense worst-in-band realized gain (reported)
                match_pen = MATCH_K * max(0.0, worst_s11 - MATCH_TARGET_DB)   # soft, graded
                conv_pen = 0.0 if converged else NONCONV_PENALTY
                score = gain - match_pen - conv_pen    # what the DE actually maximizes
                cost = -score + GUIDE_W * resonance_offset(ff, s11)
                tup = (cost, gain, ff, s11, f_g, Gr, worst_s11, score)
            cache[key(x)] = tup
            out[j] = tup
            history.append((len(history) + 1, x, tup[1], tup[6], tup[7]))
            is_best = consider(x, tup)
            save_log()
            save_state()                          # checkpoint after EVERY eval (cache is now current)
            tag = '  <-- NEW BEST' if is_best else ''
            print('[eval %3d | gen %d] gain=%+6.2f dBi  S11=%+5.1f dB  score=%+6.2f  x=%s%s'
                  % (len(history), gen, tup[1], tup[6], tup[7], np.round(x, 1), tag))
        return out

    print('===== fpc3 DE realized-gain (%d workers x %d threads) over %.2f-%.2f GHz | max %d evals ====='
          % (WORKERS, THREADS_PER_WORKER, BAND_LO/1e9, BAND_HI/1e9, MAX_EVALS))
    print('sim_path=%s | NP=%d SEED=%d | warm_start=%s'
          % (os.path.basename(sim_path), NP, SEED, os.path.relpath(WARM_START_FROM)))
    print('knobs:', names, '\n')

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        if ST['gen'] == -1:                       # initial population not yet fully evaluated
            res = evaluate(pop, executor, 0)      # (on resume, completed gen-0 evals are cache-hits)
            pop_cost = [r[0] for r in res]
            ST.update(pop=pop, pop_cost=pop_cost, gen=0)   # rng unchanged: gen-0 uses no rng
            save_state()
        else:
            pop_cost = ST['pop_cost']
        gen = ST['gen']
        stagnant = 0
        GEN_CAP = 100
        while (GAIN_TARGET is None or best['gain'] < GAIN_TARGET) and n_eval[0] < MAX_EVALS and gen < GEN_CAP:
            gen += 1
            before = n_eval[0]
            trials = [make_trial(i, pop, best['x'], rng) for i in range(NP)]
            tres = evaluate(trials, executor, gen)
            for i in range(NP):
                if tres[i][0] <= pop_cost[i]:
                    pop[i] = trials[i]; pop_cost[i] = tres[i][0]
            # new gen boundary: persist advanced rng so a resume regenerates gen+1's trials
            ST.update(pop=pop, pop_cost=pop_cost, gen=gen, rng=rng.getstate())
            save_state()
            new = n_eval[0] - before
            print('[gen %2d done] evals=%3d  new=%2d  best score=%+6.2f (gain %+.2f dBi, S11 %+.1f dB)'
                  % (gen, n_eval[0], new, best['score'], best['gain'], best['s11_worst']))
            stagnant = stagnant + 1 if new == 0 else 0
            if stagnant >= 3:
                print('Converged (population collapsed) - stopping.'); break
        if gen >= GEN_CAP:
            print('Hit generation cap (%d) - stopping.' % GEN_CAP)

    print('\n=========== BEST 3-layer DESIGN ===========')
    print('score %+.2f  |  worst-in-band realized gain %+.2f dBi  |  worst-in-band S11 %+.1f dB  (%d evals)'
          % (best['score'], best['gain'], best['s11_worst'], n_eval[0]))
    if best['Gr'] is not None:
        print('  gain across band: ' + '  '.join('%.2fGHz=%.2f' % (fg/1e9, gg)
                                                 for fg, gg in zip(best['f_g'], best['Gr'])))
    print('  params:', dict(zip(names, np.round(best['x'], 2))))
    save_best(); save_log()


if __name__ == '__main__':
    main()
