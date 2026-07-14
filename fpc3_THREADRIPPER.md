# Running the FPC co-optimizer on the Threadripper 7960X

The optimizer (`fpc3_optimize_de_gain.py`) is one script, env-configurable. On the big box
you run it as an **independent island**: its own output dir (`FPC_TAG=_TR`), warm-started
from the current global best, a different `FPC_SEED` so it explores different trials. Both
machines use the *same geometry + mesh (/16)*, so the two `optimized_params.json` are
directly comparable — at the end you keep whichever design has the higher worst-in-band
realized gain. The local run is untouched (defaults preserve 2 workers × 4 threads).

## Why these worker/thread numbers
openEMS FDTD is **memory-bandwidth bound** — a single sim stops scaling past ~6-8 threads.
Throughput on many cores comes from running **more parallel workers**, not fatter sims.
The 7960X has 24 cores across **4 CCDs of 6 cores** (quad-channel DDR5). So:

| Config              | env                                  | notes                              |
|---------------------|--------------------------------------|------------------------------------|
| **4×6 (start here)**| `FPC_WORKERS=4 FPC_THREADS=6`        | one worker per CCD — cache/NUMA-local |
| 6×4                 | `FPC_WORKERS=6 FPC_THREADS=4`        | more parallel evals, crosses CCDs  |
| 8×3                 | `FPC_WORKERS=8 FPC_THREADS=3`        | max parallelism; may saturate DRAM BW |

RAM: ~6.9 M cells/worker ≈ ~2 GB/worker → even 8 workers ≈ 16 GB. Not the limiter.
**Benchmark before committing** (see below) — pick the config with the best *evals/hour*,
which is workers ÷ median-eval-time, not just the most workers.

## 1. Get the repo + deps on the TR
Copy the whole `3GRadar` folder over **including** `fpc3_gain_de_opt/optimized_params.json`
(that's the seed). Install openEMS with Python bindings (`openEMS`, `CSXCAD` importable),
plus `numpy`, `matplotlib`.

Quick check the model builds there:
```bash
python -c "import fpc3_build, openEMS, CSXCAD; print('ok')"
```

## 2. Seed the island from the current global best
```bash
# Linux/macOS
mkdir -p fpc3_gain_de_opt_TR
cp fpc3_gain_de_opt/optimized_params.json fpc3_gain_de_opt_TR/optimized_params.json
```
```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force fpc3_gain_de_opt_TR | Out-Null
Copy-Item fpc3_gain_de_opt\optimized_params.json fpc3_gain_de_opt_TR\optimized_params.json
```
(Alternatively skip the copy and point `FPC_WARM` at the local best directly.)

## 3. Benchmark one eval per config (optional, ~15 min, recommended)
Times a single full-cavity eval so you can compare evals/hour:
```bash
for W in "4 6" "6 4" "8 3"; do set -- $W; \
  echo "=== $1 workers x $2 threads ==="; \
  FPC_WORKERS=$1 FPC_THREADS=$2 FPC_TAG=_bench FPC_NP=$1 python - <<'PY'
import os,time,fpc3_optimize_de_gain as o
from concurrent.futures import ProcessPoolExecutor
x=o.snap(o.np.array([o.PARAMS[n][0] for n in o.names]))
t=time.time()
with ProcessPoolExecutor(max_workers=o.WORKERS) as ex:
    list(ex.map(o.eval_worker,[(x,os.path.join(o.sim_path,'b%d'%i),o.THREADS_PER_WORKER) for i in range(o.WORKERS)]))
dt=time.time()-t
print("  %d evals in %.0fs -> %.1f evals/hour"%(o.WORKERS,dt,o.WORKERS*3600/dt))
PY
done; rm -rf fpc3_gain_de_opt_bench
```
Keep the config with the highest evals/hour.

## 4. Launch the island (detached)
```bash
# Linux/macOS — replace 4 6 with your benchmark winner
FPC_WORKERS=4 FPC_THREADS=6 FPC_TAG=_TR FPC_SEED=777 \
  nohup python fpc3_optimize_de_gain.py > fpc3_gain_de_opt_TR/driver.out 2>&1 &
echo $! > fpc3_gain_de_opt_TR/driver.pid
```
```powershell
# Windows PowerShell
$env:FPC_WORKERS=4; $env:FPC_THREADS=6; $env:FPC_TAG='_TR'; $env:FPC_SEED=777
$p = Start-Process python 'fpc3_optimize_de_gain.py' -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput fpc3_gain_de_opt_TR\driver.out -RedirectStandardError fpc3_gain_de_opt_TR\driver.err
$p.Id | Out-File -Encoding utf8 fpc3_gain_de_opt_TR\driver.pid
```

Watch progress:
```bash
tail -f fpc3_gain_de_opt_TR/driver.out          # new-bests + gen summaries
cat  fpc3_gain_de_opt_TR/optimized_params.json  # current best (checkpointed live)
```

## 5. Take the winner back
When done (or when you return the TR), copy `fpc3_gain_de_opt_TR/optimized_params.json`
back into the repo. Compare `worst_realized_gain_dBi` between it and the local
`fpc3_gain_de_opt/optimized_params.json`; set the better design's params into
`fpc3_build.py` and validate at full fidelity with `python fpc3_validate.py`.

**Crash-safe:** the island checkpoints its best every improvement and warm-starts from it,
so if the process dies just relaunch step 4 — no progress lost.
