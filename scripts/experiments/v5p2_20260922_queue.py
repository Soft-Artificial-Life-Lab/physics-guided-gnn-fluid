"""
v5p2 experiment queue - seed repeats + wall-offset sweep on Water-3D.

Version ID : v5p2_20260922
Purpose    : close the two top-tier gaps identified from the ICLR pattern
             analysis (paper/iclr/REPORT.md):
               (1) single-seed results   -> seeds 43, 44 for every headline
                   configuration (seed 42 runs already exist);
               (2) three-point boundary story -> a wall-offset sweep that turns
                   "misplaced vs corrected" into a regime curve.
Reuses the existing training/eval drivers unchanged (flag-driven):
  scripts/training/v5_20260820_022909_train_water3d.py
  scripts/training/v5p1_20260903_train_gns.py
  scripts/validation/v5_20260820_022909_rollout_water3d.py
  scripts/validation/v5p1_20260903_rollout_gns.py

Runs strictly one job at a time (GPU memory near 24 GB has silently killed
training processes before) and waits while another process holds the GPU.
Idempotent: a job whose two evaluation JSONs exist is skipped, so the queue can
be relaunched after an interruption.

Wall offset convention: --wall-margin pushes the metadata box (0.125..0.875)
outward. Measured GT extent is ~0.116..0.882, so margin 0.009 is the true
container; 0.010 is the published "corrected" setting.

Usage:
  nohup python scripts/experiments/v5p2_20260922_queue.py > logs/v5p2_20260922/queue_stdout.log 2>&1 &
  python scripts/experiments/v5p2_20260922_queue.py --list
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V5P2_VERSION_ID = "v5p2_20260922"
LOG_DIR = ROOT / "logs" / V5P2_VERSION_ID
PROGRESS = LOG_DIR / "queue_progress.log"

HYB_TRAIN = ROOT / "scripts/training/v5_20260820_022909_train_water3d.py"
HYB_EVAL = ROOT / "scripts/validation/v5_20260820_022909_rollout_water3d.py"
HYB_VID = "v5_20260820_022909"
GNS_TRAIN = ROOT / "scripts/training/v5p1_20260903_train_gns.py"
GNS_EVAL = ROOT / "scripts/validation/v5p1_20260903_rollout_gns.py"
GNS_VID = "v5p1_20260903"

COMMON = ["--epochs", "100", "--stride", "10"]
WALL2 = ["--wall-margin", "0.010", "--wall-layers", "2"]


def job(tag, kind, extra):
    return {"tag": tag, "kind": kind, "extra": extra}


# Ordered by value: headline comparison seeds first, then the sweep that
# produces the diagnostic figure, then the remaining configuration seeds.
JOBS = [
    job("w3d_wall2_e100_s43", "hybrid", WALL2 + ["--seed", "43"]),
    job("gns_e100_s43", "gns", ["--seed", "43"]),
    job("w3d_wall2_e100_s44", "hybrid", WALL2 + ["--seed", "44"]),
    job("gns_e100_s44", "gns", ["--seed", "44"]),
    # wall-offset sweep, 2 layers, seed 42 (margin 0.010 = existing run)
    job("w3d_sweep_m000_e100", "hybrid", ["--wall-margin", "0.000", "--wall-layers", "2"]),
    job("w3d_sweep_m005_e100", "hybrid", ["--wall-margin", "0.005", "--wall-layers", "2"]),
    job("w3d_sweep_m015_e100", "hybrid", ["--wall-margin", "0.015", "--wall-layers", "2"]),
    job("w3d_sweep_m025_e100", "hybrid", ["--wall-margin", "0.025", "--wall-layers", "2"]),
    job("w3d_sweep_m040_e100", "hybrid", ["--wall-margin", "0.040", "--wall-layers", "2"]),
    # remaining configuration seeds for the 5-way table
    job("w3d_wall2_bb_e100_s43", "hybrid", WALL2 + ["--no-attention", "--no-type-weighting", "--seed", "43"]),
    job("w3d_wall2_bb_e100_s44", "hybrid", WALL2 + ["--no-attention", "--no-type-weighting", "--seed", "44"]),
    job("w3d_nowall_e100_s43", "hybrid", ["--boundary", "none", "--seed", "43"]),
    job("w3d_nowall_e100_s44", "hybrid", ["--boundary", "none", "--seed", "44"]),
    job("w3d_wall_e100_s43", "hybrid", ["--seed", "43"]),
    job("w3d_wall_e100_s44", "hybrid", ["--seed", "44"]),
]


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return int(out[0].strip())
    except Exception:
        return -1


def wait_for_gpu(limit_mib=8000, poll_s=300):
    warned = False
    while True:
        used = gpu_used_mib()
        if 0 <= used < limit_mib:
            return
        if not warned:
            log("WAIT gpu busy (%s MiB used) - polling every %ds" % (used, poll_s))
            warned = True
        time.sleep(poll_s)


def eval_paths(tag):
    return [LOG_DIR / ("rollout_%s_%dstep.json" % (tag, s)) for s in (77, 400)]


def find_ckpt(j):
    prefix, vid = ("gns", GNS_VID) if j["kind"] == "gns" else ("hybrid", HYB_VID)
    hits = sorted((ROOT / "models").glob("%s_%s_20*_%s.pt" % (prefix, j["tag"], vid)),
                  key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def run(cmd, log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ %s\n" % " ".join(str(c) for c in cmd))
        f.flush()
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(ROOT)).returncode


def summarize(path):
    try:
        s = json.loads(path.read_text(encoding="utf-8"))["summary"]
        return ("escape=%.2f%% |KE-1|=%.3g vel=%.3g pos=%.3g mom=%.3g"
                % (s["domain_frac"]["mean"] * 100, s["ke_dev"]["mean"],
                   s["vel_mse"]["mean"], s["pos_mse"]["mean"], s["mom_drift"]["mean"]))
    except Exception as e:
        return "(summary unavailable: %s)" % e


def do_job(i, n, j):
    tag = j["tag"]
    e77, e400 = eval_paths(tag)
    if e77.exists() and e400.exists():
        log("JOB %d/%d SKIP %s (evaluations exist)" % (i, n, tag))
        return True

    train_log = LOG_DIR / ("train_%s.log" % tag)
    ckpt = find_ckpt(j)
    trained = ckpt is not None and train_log.exists() and \
        "Best ckpt:" in train_log.read_text(encoding="utf-8", errors="ignore")

    if not trained:
        wait_for_gpu()
        log("JOB %d/%d START %s (%s) %s" % (i, n, tag, j["kind"], " ".join(j["extra"])))
        t0 = time.time()
        script = GNS_TRAIN if j["kind"] == "gns" else HYB_TRAIN
        rc = run([sys.executable, script, "--tag", tag] + COMMON + j["extra"], train_log)
        if rc != 0:
            log("JOB %d/%d FAIL %s train rc=%d (see %s)" % (i, n, tag, rc, train_log.name))
            return False
        ckpt = find_ckpt(j)
        if ckpt is None:
            log("JOB %d/%d FAIL %s no checkpoint found" % (i, n, tag))
            return False
        log("JOB %d/%d TRAIN_DONE %s in %.0f min" % (i, n, tag, (time.time() - t0) / 60))
    else:
        log("JOB %d/%d RESUME %s (already trained, evaluating)" % (i, n, tag))

    eval_script = GNS_EVAL if j["kind"] == "gns" else HYB_EVAL
    for steps, out in ((77, e77), (400, e400)):
        if out.exists():
            continue
        wait_for_gpu()
        rc = run([sys.executable, eval_script, "--ckpt", ckpt, "--steps", str(steps),
                  "--out", out], train_log)
        if rc != 0 or not out.exists():
            log("JOB %d/%d FAIL %s eval%d rc=%d" % (i, n, tag, steps, rc))
            return False
        log("JOB %d/%d EVAL%d %s %s" % (i, n, steps, tag, summarize(out)))
    log("JOB %d/%d DONE %s" % (i, n, tag))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the job plan and exit")
    ap.add_argument("--only", nargs="*", default=None, help="run only these tags")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in JOBS if not args.only or j["tag"] in args.only]
    if args.list:
        for i, j in enumerate(jobs, 1):
            done = all(p.exists() for p in eval_paths(j["tag"]))
            print("%2d. %-26s %-6s %s %s" % (i, j["tag"], j["kind"], " ".join(j["extra"]),
                                             "[done]" if done else ""))
        return

    log("QUEUE START %d jobs (version %s)" % (len(jobs), V5P2_VERSION_ID))
    ok = sum(do_job(i, len(jobs), j) for i, j in enumerate(jobs, 1))
    log("QUEUE COMPLETE %d/%d ok" % (ok, len(jobs)))


if __name__ == "__main__":
    main()
