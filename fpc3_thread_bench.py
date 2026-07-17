"""
Thread-scaling benchmark for ONE fpc3 eval: openEMS FDTD throughput (MC/s) vs numThreads,
run SOLO (one sim at a time) so the per-sim scaling is clean.

FDTD is memory-bandwidth bound, so throughput scales SUBLINEARLY with threads. This tells
you the best workers x threads split: if 16 threads is only ~1.5x over 4, then on a 64-vCPU
box 16 workers x 4 threads gives far more TOTAL throughput than 4 workers x 16 threads --
and the DE runs one eval per worker, so more workers = more parallel evals (up to NP).

Uses the same physics as the real search (PML, copper, mesh/16, slot on) so cell count and
MC/s are representative. Short fixed NRTS with no early stop -- we only want steady-state
timestep speed, not a full ring-down. openEMS's own "Speed: X MC/s" lines are parsed and
the warm-up discarded, so setup/operator time doesn't pollute the number.

Run:  python fpc3_thread_bench.py                       # 4,8,16
      FPC_THREAD_LIST=4,8,12,16,32 python fpc3_thread_bench.py
"""

import os
import re
import sys
import time
import shutil
import contextlib
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS

import fpc3_build as p1

BENCH_NRTS     = int(os.environ.get('FPC_BENCH_NRTS', '3000'))     # steady-state is reached fast
THREAD_LIST    = [int(t) for t in os.environ.get('FPC_THREAD_LIST', '4,8,16').split(',')]
RINGDOWN_STEPS = int(os.environ.get('FPC_RINGDOWN_STEPS', '111000'))  # observed steps to -30 dB
sim_path = os.path.join(os.getcwd(), 'fpc3_threadbench')
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


def bench(threads):
    log = os.path.join(sim_path, 'bench_%dthr.log' % threads)
    run_dir = os.path.join(sim_path, 'run_%dthr' % threads)
    p1.RCM_ON = True
    p1.SLOT_ON = True
    p1.mesh_res = (p1.C0 / (p1.f0 + p1.fc)) / p1.unit / 16.0
    p1._recompute()
    os.makedirs(run_dir, exist_ok=True)
    with _redirect(log):
        FDTD = openEMS(NrTS=BENCH_NRTS, EndCriteria=1e-9)      # tiny -> never early-stops
        FDTD.SetGaussExcite(p1.f0, p1.fc)
        FDTD.SetBoundaryCond(['PML_8'] * 6)
        CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
        mesh = p1.build_antenna(CSX, FDTD)
        ncells = int(np.prod([len(mesh.GetLines(a)) for a in 'xyz']))
        FDTD.AddLumpedPort(1, p1.feed_R, [p1.feed_x, p1.feed_y, 0],
                           [p1.feed_x, p1.feed_y, p1.h_sub], 'z', 1.0, priority=5, edges2grid='xy')
        CSX.Write2XML(os.path.join(run_dir, 'a.xml'))
        t0 = time.time()
        FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=threads)
        wall = time.time() - t0
    mcs = [float(m) for m in re.findall(r'Speed:\s*([0-9.]+)\s*MC/s', open(log).read())]
    # discard warm-up (first ~third) and take the median of the steady-state readings
    steady = float(np.median(mcs[max(2, len(mcs)//3):])) if len(mcs) >= 4 else (
             float(np.median(mcs)) if mcs else float('nan'))
    _rmtree(run_dir)
    return ncells, steady, wall


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print('thread-scaling benchmark | NRTS=%d | threads=%s | (solo, one sim at a time)\n'
          % (BENCH_NRTS, THREAD_LIST))
    rows = []
    for t in THREAD_LIST:
        nc, mcs, wall = bench(t)
        rows.append((t, nc, mcs, wall))
        print('  %2d threads -> %6.1f MC/s   (%.2fM cells, %.0fs wall for %d steps)'
              % (t, mcs, nc/1e6, wall, BENCH_NRTS))

    base_t, base_nc, base_mcs, _ = rows[0]
    print('\n==================== THREAD SCALING ====================')
    print('threads | MC/s  | speedup | efficiency | est. eval (%dk steps) | total on 64 vCPU*'
          % (RINGDOWN_STEPS // 1000))
    for t, nc, mcs, wall in rows:
        speedup = mcs / base_mcs
        eff = speedup / (t / base_t)                       # 100% = perfect scaling
        eval_min = nc * RINGDOWN_STEPS / (mcs * 1e6) / 60.0
        workers = max(1, 64 // t)
        agg = workers * mcs                                # naive aggregate (ignores contention)
        print('  %3d   | %5.1f | %5.2fx  |   %3.0f%%    |      %5.1f min       | %2dw x %d MC/s = %.0f'
              % (t, mcs, speedup, eff * 100, eval_min, workers, t, agg))
    print('=======================================================')
    print('* naive aggregate = (64/threads) workers x per-sim MC/s. Real parallel throughput is')
    print('  LOWER (workers share memory bandwidth), so the true optimum is usually even fewer')
    print('  threads/worker than this suggests. Pick the smallest thread count whose per-sim')
    print('  eval-time is still acceptable, and run WORKERS = 64/threads (>= NP=16 is ideal).')


if __name__ == '__main__':
    main()
