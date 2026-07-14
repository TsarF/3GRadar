# Running the FPC co-optimizer on EC2 M8g spot

`fpc3_optimize_de_gain.py` now writes a **full DE-state checkpoint** (population, per-member
costs, generation, RNG state, eval cache, best, history) atomically after *every* eval, and
resumes from it on launch. So a spot reclaim costs at most the few evals in flight — they
re-run as cache-hits. Validated: interrupt → relaunch → `>>> RESUMED ... gen=N` and continue.

## Region
Best **M8g spot price + capacity**: **us-east-2 (Ohio)** or **us-east-1 (N. Virginia)** —
deepest Graviton capacity and typically the cheapest spot. `us-west-2` / `eu-west-1` are
fine alternates. This is a batch job, so latency to you is irrelevant — optimize for price
and interruption rate. Before launching, check **EC2 Console → Spot Requests → Spot
placement score** for `m8g.16xlarge` in your candidate regions/AZs and pick the highest.

## Instance size — match it to the DE, not "biggest"
DE is synchronous: within a generation only **NP** trials can run in parallel (default
`NP=16`). Beyond 16 concurrent evals there's nothing to run until the generation closes. So
the sweet spot evaluates one whole generation at once:

**→ `m8g.16xlarge` (64 vCPU, 256 GiB): `FPC_WORKERS=16 FPC_THREADS=4`.**
One generation = one parallel batch ≈ one eval-time. Graviton4 is 1 vCPU = 1 real core (no
SMT) with DDR5-5600, so 64 vCPU = 64 real cores and the numbers are honest. RAM is a
non-issue (~6.9 M cells ≈ ~2 GB/worker → ~32 GB of 256).

- Want more exploration? `FPC_NP=24` on **`m8g.24xlarge`** (96 vCPU) → `FPC_WORKERS=24 FPC_THREADS=4`.
- **Don't** jump to `m8g.48xlarge` (192 vCPU) with NP=16 — 176 vCPU sit idle. Only worth it
  if you raise NP to ~48 (slower per-gen, more thorough) or run multiple islands.
- openEMS is memory-bandwidth bound; if a single eval is slow, try `FPC_THREADS=4` vs `8`
  with correspondingly fewer workers and keep whichever gives more **evals/hour** (the
  benchmark loop in `fpc3_THREADRIPPER.md` §3 works here too).

## Setup (ARM64 / aarch64)
The AMI must be ARM64 (Amazon Linux 2023 arm64 or Ubuntu 24.04 arm64). openEMS has no ARM
wheels — build once from source and bake an AMI so respawns are instant:
```bash
sudo dnf groupinstall -y "Development Tools" || sudo apt-get install -y build-essential
# deps: cmake, boost, hdf5, CGAL, tinyxml, vtk, qt; then build openEMS + python bindings
#   https://docs.openems.de/install/  (build AppCSXCAD/CSXCAD/openEMS, pip install its python/)
python -c "import fpc3_build, openEMS, CSXCAD; print('ok')"   # must pass
```
Bake this into a custom AMI so a reclaimed-then-relaunched spot boots ready.

## Spot fault tolerance — durable checkpoint
The checkpoint file must **survive instance termination**. Two options:

**A) S3 sync (recommended — survives full instance loss).** Give the instance an IAM role
with access to a bucket, then run a background sync alongside the optimizer:
```bash
export FPC_CKPT=/data/fpc/de_state.pkl        # local path the optimizer writes
mkdir -p /data/fpc
aws s3 cp s3://YOUR_BUCKET/fpc/de_state.pkl "$FPC_CKPT" 2>/dev/null || true   # restore if exists
# push checkpoint + best to S3 every 60s in the background:
( while true; do
    aws s3 cp "$FPC_CKPT" s3://YOUR_BUCKET/fpc/de_state.pkl --only-show-errors || true
    aws s3 cp fpc3_gain_de_opt_EC2/optimized_params.json s3://YOUR_BUCKET/fpc/ --only-show-errors || true
    sleep 60
  done ) &
```
On a fresh instance the `aws s3 cp ... "$FPC_CKPT"` restore line pulls the latest state and
the optimizer resumes. (Checkpoint is <1 MB, so 60 s sync is cheap. Optionally also trigger
a sync on the 2-min interruption notice — poll
`http://169.254.169.254/latest/meta-data/spot/instance-action`.)

**B) Persistent EBS.** Put the repo + `FPC_CKPT` on a data EBS volume with
`DeleteOnTermination=false`; after a reclaim, launch a new spot, attach the volume, relaunch.

**Auto-relaunch:** request a **persistent** spot request (not one-time) with interruption
behavior `stop` or `terminate`; it re-provisions when capacity returns. Put the launch
command (below) in user-data so a respawn restores from S3 and continues unattended.

## Launch (detached, resumable)
```bash
cd /path/to/3GRadar
# seed the island from your current global best (first launch only):
mkdir -p fpc3_gain_de_opt_EC2
aws s3 cp s3://YOUR_BUCKET/fpc/optimized_params.json fpc3_gain_de_opt_EC2/ 2>/dev/null \
  || cp fpc3_gain_de_opt/optimized_params.json fpc3_gain_de_opt_EC2/ 2>/dev/null || true

FPC_WORKERS=16 FPC_THREADS=4 FPC_TAG=_EC2 FPC_SEED=2025 \
FPC_CKPT=/data/fpc/de_state.pkl \
  nohup python fpc3_optimize_de_gain.py > fpc3_gain_de_opt_EC2/driver.out 2>&1 &
echo $! > fpc3_gain_de_opt_EC2/driver.pid
tail -f fpc3_gain_de_opt_EC2/driver.out
```
If reclaimed and relaunched with the **same env**, it prints `>>> RESUMED ... gen=N` and
continues. `FPC_TAG=_EC2` keeps it isolated from the local run so nothing clobbers.

## Cost sanity
`m8g.16xlarge` on-demand ≈ small-single-digit $/hr; spot typically ~60-70% off. If the
search needs ~20 generations and a generation ≈ one eval-time, budget by benchmarking one
eval on the instance first (`evals/hour` loop) — you'll know $/run before committing.

## Take the winner back
Copy `fpc3_gain_de_opt_EC2/optimized_params.json` (or the S3 copy) into the repo, compare
`worst_realized_gain_dBi` against the local/TR bests, set the winner into `fpc3_build.py`,
and validate at full fidelity: `python fpc3_validate.py`.
```
